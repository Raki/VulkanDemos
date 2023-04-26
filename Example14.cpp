#include "CommonHeaders.h"
#include "Utility.h"
#include "VKUtility.h"
#include "VKBackend.h"
#include "Colors.h"
#include <imgui.h>
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>
#include <filesystem>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#pragma region vars
const int MAX_FRAMES_IN_FLIGHT = 2;
const int WIN_WIDTH = 1024;
const int WIN_HEIGHT = 1024;

#define VERT_SHADER_SPV "shaders/simpleMatBVH.vert.spv"
#define FRAG_SHADER_SPV "shaders/simpleMat.frag.spv"
#define VERT_SHADER_GLSL "shaders/simpleMatBVH.vert.glsl"
#define FRAG_SHADER_GLSL "shaders/simpleMat.frag.glsl"

GLFWwindow* window;
auto closeWindow = false;

struct UniformBufferObject
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 nrmlMat;
};

struct UBOFrag
{
    glm::vec4 position;
    glm::vec4 color;
};

struct VKMesh
{
    glm::vec3 rOrigin=glm::vec3(0);
    std::vector<uint16_t> indices;
    glm::mat4 rMatrix=glm::mat4(1);
    glm::mat4 tMatrix=glm::mat4(1);
    float th = 0;
    bool isInterleaved = true;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    VkBuffer posBuffer;
    VkDeviceMemory posBufferMemory;

    VkBuffer normBuffer;
    VkDeviceMemory normBufferMemory;

    VkBuffer uvBuffer;
    VkDeviceMemory uvBufferMemory;

    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    
    std::shared_ptr<VKUtility::Mesh> meshData;

    void createBuffers(const VkDevice& device)
    {
        //size of buffers
        VkDeviceSize vBuffSize = sizeof(VKUtility::Vertex) * meshData->vData.size();
        VkDeviceSize iBuffSize = sizeof(uint16_t) * meshData->iData.size();

        //staging buffer
        VkBuffer vStageBuff, iStageBuff;
        VkDeviceMemory vStageBuffMemory, iStageBuffMemory;

        VKBackend::createBuffer(vBuffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vStageBuff, vStageBuffMemory);
        void* data;
        vkMapMemory(device, vStageBuffMemory, 0, vBuffSize, 0, &data);
        memcpy(data, meshData->vData.data(), (size_t)vBuffSize);
        vkUnmapMemory(device, vStageBuffMemory);

        VKBackend::createBuffer(iBuffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, iStageBuff, iStageBuffMemory);
        void* iData;
        vkMapMemory(device, iStageBuffMemory, 0, iBuffSize, 0, &iData);
        memcpy(iData, meshData->iData.data(), (size_t)iBuffSize);
        vkUnmapMemory(device, iStageBuffMemory);

        //create device memory backed buffer
        VKBackend::createBuffer(vBuffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        VKBackend::createBuffer(iBuffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        //transfer memory from staging to device memory backed buffer

        VkCommandBuffer commandBuffer = VKBackend::beginSingleTimeCommands();

        VkBufferCopy copyRegion{}, copyRegionIndex{};
        copyRegion.size = vBuffSize;
        copyRegionIndex.size = iBuffSize;

        vkCmdCopyBuffer(commandBuffer, vStageBuff, vertexBuffer, 1, &copyRegion);
        vkCmdCopyBuffer(commandBuffer, iStageBuff, indexBuffer, 1, &copyRegionIndex);

        VKBackend::endSingleTimeCommands(commandBuffer);

        vkDestroyBuffer(device, vStageBuff, nullptr);
        vkFreeMemory(device, vStageBuffMemory, nullptr);

        vkDestroyBuffer(device, iStageBuff, nullptr);
        vkFreeMemory(device, iStageBuffMemory, nullptr);
    }

    void createBuffNonInterleaved(const VkDevice& device)
    {
        isInterleaved = false;

        std::vector<glm::vec3> posArr;
        std::vector<glm::vec3> normArr;
        std::vector<glm::vec2> uvArr;

        for (auto& vert : meshData->vData)
        {
            posArr.push_back(vert.position);
            normArr.push_back(vert.normal);
            uvArr.push_back(vert.uv);
        }

        enum class CType { POS, NORM, UV, INDEX };

        struct Container
        {
            VkDeviceSize buffSize;
            void* data;
            CType type;
            VkBuffer stageBuff;
            VkDeviceMemory stageBuffMemory;
        };

        std::vector<Container> cInfos;
        Container vCont;
        vCont.buffSize = sizeof(glm::vec3) * posArr.size();
        vCont.data = posArr.data();
        vCont.type = CType::POS;
        cInfos.push_back(vCont);

        Container nCont;
        nCont.buffSize = sizeof(glm::vec3) * normArr.size();
        nCont.data = normArr.data();
        nCont.type = CType::NORM;
        cInfos.push_back(nCont);

        Container uvCont;
        uvCont.buffSize = sizeof(glm::vec2) * uvArr.size();
        uvCont.data = uvArr.data();
        uvCont.type = CType::UV;
        cInfos.push_back(uvCont);

        Container iCont;
        iCont.buffSize = sizeof(uint16_t) * meshData->iData.size();
        iCont.data = meshData->iData.data();
        iCont.type = CType::INDEX;
        cInfos.push_back(iCont);

        for (size_t i = 0; i < cInfos.size(); i++)
        {
            VKBackend::createBuffer(cInfos.at(i).buffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, cInfos.at(i).stageBuff, cInfos.at(i).stageBuffMemory);
            vkMapMemory(device, cInfos.at(i).stageBuffMemory, 0, cInfos.at(i).buffSize, 0, &cInfos.at(i).data);
            
            switch (cInfos.at(i).type)
            {
            case CType::POS:
                memcpy(cInfos.at(i).data, posArr.data(), (size_t)cInfos.at(i).buffSize);
                break;
            case CType::NORM:
                memcpy(cInfos.at(i).data, normArr.data(), (size_t)cInfos.at(i).buffSize);
                break;
            case CType::UV:
                memcpy(cInfos.at(i).data, uvArr.data(), (size_t)cInfos.at(i).buffSize);
                break;
            case CType::INDEX:
                memcpy(cInfos.at(i).data, meshData->iData.data(), (size_t)cInfos.at(i).buffSize);
                break;
            }
            
            vkUnmapMemory(device, cInfos.at(i).stageBuffMemory);
        }

        //create device memory backed buffer
        VKBackend::createBuffer(cInfos.at(0).buffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, posBuffer, posBufferMemory);
        VKBackend::createBuffer(cInfos.at(1).buffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, normBuffer, normBufferMemory);
        VKBackend::createBuffer(cInfos.at(2).buffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, uvBuffer, uvBufferMemory);
        VKBackend::createBuffer(cInfos.at(3).buffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        //transfer memory from staging to device memory backed buffer

        VkCommandBuffer commandBuffer = VKBackend::beginSingleTimeCommands();

        for (size_t i=0;i<cInfos.size();i++)
        {
            VkBufferCopy copyRegion{};
            copyRegion.size = cInfos.at(i).buffSize;
            switch (cInfos.at(i).type)
            {
            case CType::POS:
                vkCmdCopyBuffer(commandBuffer, cInfos.at(i).stageBuff, posBuffer, 1, &copyRegion);
                break;
            case CType::NORM:
                vkCmdCopyBuffer(commandBuffer, cInfos.at(i).stageBuff, normBuffer, 1, &copyRegion);
                break;
            case CType::UV:
                vkCmdCopyBuffer(commandBuffer, cInfos.at(i).stageBuff, uvBuffer, 1, &copyRegion);
                break;
            case CType::INDEX:
                vkCmdCopyBuffer(commandBuffer, cInfos.at(i).stageBuff, indexBuffer, 1, &copyRegion);
                break;
            }
            
        }

        VKBackend::endSingleTimeCommands(commandBuffer);

        for (size_t i=0;i<cInfos.size();i++)
        {
            vkDestroyBuffer(device, cInfos.at(i).stageBuff, nullptr);
            vkFreeMemory(device, cInfos.at(i).stageBuffMemory, nullptr);
        }
    }
};

struct VKMesh3D : public VKMesh
{
    std::vector<VKUtility::VDPosNorm> vertices;
    void createBuffers(VkDevice device)
    {
        //size of buffers
        VkDeviceSize vBuffSize = sizeof(VKUtility::VDPosNorm) * vertices.size();
        VkDeviceSize iBuffSize = sizeof(uint16_t) * indices.size();

        //staging buffer
        VkBuffer vStageBuff, iStageBuff;
        VkDeviceMemory vStageBuffMemory, iStageBuffMemory;

        VKBackend::createBuffer(vBuffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vStageBuff, vStageBuffMemory);
        void* data;
        vkMapMemory(device, vStageBuffMemory, 0, vBuffSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)vBuffSize);
        vkUnmapMemory(device, vStageBuffMemory);

        VKBackend::createBuffer(iBuffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, iStageBuff, iStageBuffMemory);
        void* iData;
        vkMapMemory(device, iStageBuffMemory, 0, iBuffSize, 0, &iData);
        memcpy(iData, indices.data(), (size_t)iBuffSize);
        vkUnmapMemory(device, iStageBuffMemory);

        //create device memory backed buffer
        VKBackend::createBuffer(vBuffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        VKBackend::createBuffer(iBuffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        //transfer memory from staging to device memory backed buffer

        VkCommandBuffer commandBuffer = VKBackend::beginSingleTimeCommands();

        VkBufferCopy copyRegion{}, copyRegionIndex{};
        copyRegion.size = vBuffSize;
        copyRegionIndex.size = iBuffSize;

        vkCmdCopyBuffer(commandBuffer, vStageBuff, vertexBuffer, 1, &copyRegion);
        vkCmdCopyBuffer(commandBuffer, iStageBuff, indexBuffer, 1, &copyRegionIndex);

        VKBackend::endSingleTimeCommands(commandBuffer);

        vkDestroyBuffer(device, vStageBuff, nullptr);
        vkFreeMemory(device, vStageBuffMemory, nullptr);

        vkDestroyBuffer(device, iStageBuff, nullptr);
        vkFreeMemory(device, iStageBuffMemory, nullptr);
    }

    void createBuffNonInterleaved(const VkDevice& device)  
    {
        isInterleaved = false;

        std::vector<glm::vec3> posArr;
        std::vector<glm::vec3> normArr;

        for (auto& vert : meshData->vDataOL)
        {
            posArr.push_back(vert.position);
            normArr.push_back(vert.normal);
        }

        enum class CType { POS, NORM, UV, INDEX };

        struct Container
        {
            VkDeviceSize buffSize;
            void* data;
            CType type;
            VkBuffer stageBuff;
            VkDeviceMemory stageBuffMemory;
        };

        std::vector<Container> cInfos;
        Container vCont;
        vCont.buffSize = sizeof(glm::vec3) * posArr.size();
        vCont.data = posArr.data();
        vCont.type = CType::POS;
        cInfos.push_back(vCont);

        Container nCont;
        nCont.buffSize = sizeof(glm::vec3) * normArr.size();
        nCont.data = normArr.data();
        nCont.type = CType::NORM;
        cInfos.push_back(nCont);

        Container iCont;
        iCont.buffSize = sizeof(uint16_t) * meshData->iData.size();
        iCont.data = meshData->iData.data();
        iCont.type = CType::INDEX;
        cInfos.push_back(iCont);

        for (size_t i = 0; i < cInfos.size(); i++)
        {
            VKBackend::createBuffer(cInfos.at(i).buffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, cInfos.at(i).stageBuff, cInfos.at(i).stageBuffMemory);
            vkMapMemory(device, cInfos.at(i).stageBuffMemory, 0, cInfos.at(i).buffSize, 0, &cInfos.at(i).data);

            switch (cInfos.at(i).type)
            {
            case CType::POS:
                memcpy(cInfos.at(i).data, posArr.data(), (size_t)cInfos.at(i).buffSize);
                break;
            case CType::NORM:
                memcpy(cInfos.at(i).data, normArr.data(), (size_t)cInfos.at(i).buffSize);
                break;
            case CType::UV:
                
                break;
            case CType::INDEX:
                memcpy(cInfos.at(i).data, meshData->iData.data(), (size_t)cInfos.at(i).buffSize);
                break;
            }

            vkUnmapMemory(device, cInfos.at(i).stageBuffMemory);
        }

        //create device memory backed buffer
        VKBackend::createBuffer(cInfos.at(0).buffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, posBuffer, posBufferMemory);
        VKBackend::createBuffer(cInfos.at(1).buffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, normBuffer, normBufferMemory);
        VKBackend::createBuffer(cInfos.at(2).buffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        //transfer memory from staging to device memory backed buffer

        VkCommandBuffer commandBuffer = VKBackend::beginSingleTimeCommands();

        for (size_t i = 0; i < cInfos.size(); i++)
        {
            VkBufferCopy copyRegion{};
            copyRegion.size = cInfos.at(i).buffSize;
            switch (cInfos.at(i).type)
            {
            case CType::POS:
                vkCmdCopyBuffer(commandBuffer, cInfos.at(i).stageBuff, posBuffer, 1, &copyRegion);
                break;
            case CType::NORM:
                vkCmdCopyBuffer(commandBuffer, cInfos.at(i).stageBuff, normBuffer, 1, &copyRegion);
                break;
            case CType::UV:
                
                break;
            case CType::INDEX:
                vkCmdCopyBuffer(commandBuffer, cInfos.at(i).stageBuff, indexBuffer, 1, &copyRegion);
                break;
            }

        }

        VKBackend::endSingleTimeCommands(commandBuffer);

        for (size_t i = 0; i < cInfos.size(); i++)
        {
            vkDestroyBuffer(device, cInfos.at(i).stageBuff, nullptr);
            vkFreeMemory(device, cInfos.at(i).stageBuffMemory, nullptr);
        }
    }
};

struct PushConstant
{
    glm::mat4 tMat;
};

struct Buffer
{
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBufferMemories;
    VkDeviceSize range=0;
    std::vector<VkDescriptorBufferInfo> bufferInfo;
    bool isDirtry = true;
    Buffer()
    {

    }
    Buffer(const Buffer& buff)
    {
        uniformBuffers.clear();
        uniformBuffers.insert(uniformBuffers.begin(), buff.uniformBuffers.begin(), buff.uniformBuffers.end());
        uniformBufferMemories.clear();
        uniformBufferMemories.insert(uniformBufferMemories.begin(), buff.uniformBufferMemories.begin(), buff.uniformBufferMemories.end());
        bufferInfo.clear();
        bufferInfo.insert(bufferInfo.begin(), buff.bufferInfo.begin(), buff.bufferInfo.end());
        range = buff.range;
        isDirtry = buff.isDirtry;
    }

};

struct Image
{
    std::shared_ptr<VKBackend::VKTexture> texContainer;
    VkDescriptorImageInfo imageInfo;
};

struct Descriptor
{
    VkDescriptorSetLayoutBinding layout;
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<Image> image;
};

struct DescSetBuildResult
{
    std::vector<Descriptor> descriptors;
    std::vector<VkDescriptorSet> descSets;
    VkDescriptorSetLayout descSetLayout;
    std::vector<VkVertexInputAttributeDescription> vertIPAttribDesc;
    std::vector<VkVertexInputBindingDescription> vertIPBindDesc;
    VkShaderModule triangleVS;
    VkShaderModule triangleFS;
};

Buffer uboFrag,uboVert;
UBOFrag lightInfo;
//std::vector<VkBuffer> uniformBuffers;
//std::vector<VkDeviceMemory> uniformBufferMemories;

std::vector<VKUtility::Vertex> vertices;
std::vector<uint16_t> indices;

VkImage depthImage;
VkDeviceMemory depthImageMemory;
VkImageView depthImageView;

VkPipelineCache pipelineCache;
VkPipeline wireframePipeline=VK_NULL_HANDLE;
VkPipelineLayout wireframePipelineLayout;

std::shared_ptr<VKBackend::VKRenderTarget> msColorAttch;
std::vector<std::shared_ptr<VKMesh>> cubes;
std::vector<std::shared_ptr<VKMesh3D>> wireFrameObjs;
std::chrono::system_clock::time_point lastTime{};

std::shared_ptr<VKBackend::VKTexture> texture,redTexture;

VkDescriptorSetLayout descLayoutFlat;
std::vector<Descriptor> descriptors, descriptorsFlat;
std::vector<VkDescriptorSet> descriptorSetsFlat;

struct FiveColors
{
    /*
    * e63946
    * f1faee
    * a8dadc
    * 457b9d
    * 1d3557
    */
    glm::vec3 col1;
    glm::vec3 col2;
    glm::vec3 col3;
    glm::vec3 col4;
    glm::vec3 col5;
};

//BVH
using uint = unsigned int;
constexpr size_t N = 50;

struct Tri
{
    glm::vec3 v0, v1, v2;
    glm::vec3 centroid;
};

Tri tris[N];
uint trisIdx[N];

struct Ray
{
    glm::vec3 O;
    glm::vec3 D;
    float t = 1e30f;
};


__declspec(align(32)) struct BVHNode
{
    glm::vec3 aabbMin, aabbMax;
    uint leftFirst, //left node if triCount=0 otherwise firstTriIndex
        triCount;
    bool isLeaf()
    {
        return triCount > 0;
    }

};

// For N primitives BVH will max of 2N=1 NODES  
BVHNode bvhNodes[2 * N - 1];
uint rootNodeIdx = 0, nodesUsed = 1;

#pragma endregion vars

#pragma region prototypes
void createWindow();
void initVulkan();
void buildDescriptorsFor(std::string fsPath, std::string vsPath,DescSetBuildResult &result, std::shared_ptr<Image> image=nullptr);
void updateFrame();
void compileShaders();
void destroyVulkan();

VkSurfaceKHR createSurface(GLFWwindow* window, VkInstance instace);
void createUniformBuffers();
void createDescriptorSets(const VkDevice device, const std::vector<Descriptor>& descriptors);
void createDescriptorSets(const VkDevice device, const std::vector<Descriptor>& descriptors,std::vector<VkDescriptorSet> &descSets);
void createDescriptorSets(const VkDevice device, const std::vector<Descriptor>& descriptors, std::vector<VkDescriptorSet>& descSets,VkDescriptorSetLayout descLayout);
void createPipelineCache(const VkDevice device);
VkPipeline createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkShaderModule vsModule, VkShaderModule fsModule);
//ToDo: Any chance to improve this?
VkPipeline createGraphicsPipeline(VkDevice device,VkPipelineCache pipelineCache , VkRenderPass renderPass, VkShaderModule vsModule, VkShaderModule fsModule,
    std::vector<VkVertexInputAttributeDescription>& vertIPAttribDesc, 
    std::vector<VkVertexInputBindingDescription>& vertIPBindDesc,
    const VkPipelineLayout pipelineLayout,
    const VkPolygonMode polygonMode,
    const VkPrimitiveTopology primToplogy);
void createFramebuffers();
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
void updateUniformBuffer(uint32_t currentImage);
void createDepthResources();
void createColorResource();
void fillCube(float width, float height, float depth,glm::mat4 tMat, std::vector<VKUtility::VDPosNorm>& verts, std::vector<uint16_t>& indices);
void setupScene();
void setupRandomTris();
void loadGlbModel(std::string filePath);

//BVH
void buildBVH();
void intersectTri(const Tri& tri, Ray& ray);
void updateNodeBounds(const uint& nodeIdx);
void subdivide(uint nodeIdx);
void intersectBVH(Ray& ray, const uint nodeIndex);
bool intersectAABB(const Ray& ray, const glm::vec3 bMin, const glm::vec3 bMax);
inline glm::vec3 minOf2(const glm::vec3& v1, const glm::vec3& v2);
inline glm::vec3 maxOf2(const glm::vec3& v1, const glm::vec3& v2);
inline void swap(Tri& tri1, Tri& tri2);
inline void swap(uint& v1, uint& v2);
//

#pragma endregion prototypes

#pragma region functions
static void error_callback(int error, const char* description)
{
    fmt::print(stderr, "Error: {}\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
    {
        lightInfo.position.x -= 1;
        uboFrag.isDirtry = true;
    }
    else if (key == GLFW_KEY_RIGHT&& action == GLFW_PRESS)
    {
        lightInfo.position.x += 1;
        uboFrag.isDirtry = true;
    }
    else if (key == GLFW_KEY_W&& action == GLFW_PRESS)
    {
        
    }
}

void createWindow()
{
    if (GLFW_TRUE != glfwInit())
    {
        fmt::print("Failed to init glfw \n");
    }
    else
        fmt::print("glfw init success \n");

    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, "Vulkan Example6", NULL, NULL);
    glfwSetKeyCallback(window, key_callback);
}
void initVulkan()
{
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    VKBackend::winWidth = width;
    VKBackend::winHeight = height;
    VKBackend::vkInstance = VKBackend::createVulkanInstance();
    VKBackend::surface = createSurface(window, VKBackend::vkInstance);
    VKBackend::physicalDevice = VKBackend::pickPhysicalDevice(VKBackend::vkInstance);
    assert(VKBackend::physicalDevice);

    auto queFamilyIndices = VKBackend::pickDeviceQueueFamily(VKBackend::physicalDevice, VKBackend::surface);

    bool bindlessResources = VKBackend::supportForDescriptorIndexing(VKBackend::physicalDevice);

    if(bindlessResources)
        fmt::print("GPU supports descriptor indexing\n");

    VKBackend::device = VKBackend::createDevice(VKBackend::physicalDevice);

    auto swapChainSupportDetails = VKBackend::querySwapChainSupport(VKBackend::physicalDevice);

    VKBackend::swapchain = VKBackend::createSwapchain(VKBackend::device, VKBackend::surface);

    VKBackend::createSwapchainImageViews();

    VKBackend::renderPass = VKBackend::createRenerPass(VKBackend::device);
    VKBackend::commandPool = VKBackend::createCommandPool(VKBackend::device);
    VKBackend::createCommandBuffers(VKBackend::device, VKBackend::commandPool);

    createUniformBuffers();

    // create descriptor pool
    std::vector<VkDescriptorPoolSize> poolsizes;

    {
        VkDescriptorPoolSize poolSize;
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        //why x10?
        poolSize.descriptorCount = static_cast<uint32_t>(VKBackend::swapchainMinImageCount * 10);
        poolsizes.push_back(poolSize);
        poolsizes.push_back(poolSize);
    }
    {
        VkDescriptorPoolSize poolSize;
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        //why x10?
        poolSize.descriptorCount = static_cast<uint32_t>(VKBackend::swapchainMinImageCount * 10);
        poolsizes.push_back(poolSize);
    }
    VKBackend::createDescriptorPool(VKBackend::device, poolsizes);

    texture = VKBackend::createVKTexture("img/sample.jpg");

    auto image = std::make_shared<Image>();
    image->texContainer = texture;
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = texture->textureImageView;
    imageInfo.sampler = texture->textureSampler;
    image->imageInfo = imageInfo;


    DescSetBuildResult result,resultFlat;

    buildDescriptorsFor(FRAG_SHADER_SPV, VERT_SHADER_SPV, result, image);
    buildDescriptorsFor("shaders/simpleMatFlat.frag.spv", "shaders/simpleMatBVHFlat.vert.spv", resultFlat, nullptr);

    VKBackend::descriptorSetLayout = result.descSetLayout;
    VKBackend::descriptorSets = result.descSets;
    descriptors = result.descriptors;

    descriptorSetsFlat = resultFlat.descSets;
    descriptorsFlat = resultFlat.descriptors;
    descLayoutFlat = resultFlat.descSetLayout;


    VkPushConstantRange pushConstant;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(PushConstant);
    pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    std::vector<VkPushConstantRange> pushConstants = { pushConstant };
    std::vector<VkDescriptorSetLayout> descriptorLayouts = { VKBackend::descriptorSetLayout };
    std::vector<VkDescriptorSetLayout> descriptorLayoutsWF = { descLayoutFlat };

    VKBackend::pipelineLayout = VKBackend::createPipelineLayout(descriptorLayouts, pushConstants);
    wireframePipelineLayout = VKBackend::createPipelineLayout(descriptorLayoutsWF, pushConstants);

    createPipelineCache(VKBackend::device);
    VKBackend::graphicsPipeline = createGraphicsPipeline(VKBackend::device,pipelineCache, VKBackend::renderPass, result.triangleVS, result.triangleFS,
        result.vertIPAttribDesc,result.vertIPBindDesc, VKBackend::pipelineLayout,VK_POLYGON_MODE_FILL, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    wireframePipeline = createGraphicsPipeline(VKBackend::device, pipelineCache, VKBackend::renderPass, resultFlat.triangleVS, resultFlat.triangleFS,
        resultFlat.vertIPAttribDesc, resultFlat.vertIPBindDesc, wireframePipelineLayout,VK_POLYGON_MODE_LINE, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

    vkDestroyShaderModule(VKBackend::device, result.triangleVS, nullptr);
    vkDestroyShaderModule(VKBackend::device, result.triangleFS, nullptr);

    vkDestroyShaderModule(VKBackend::device, resultFlat.triangleVS, nullptr);
    vkDestroyShaderModule(VKBackend::device, resultFlat.triangleFS, nullptr);

    createColorResource();
    createDepthResources();
    createFramebuffers();

    VKBackend::createSyncObjects();

    setupScene();
    
}
//ToDo :  make this function generic and move to VKBackend
void buildDescriptorsFor(const std::string fsPath,const std::string vsPath, DescSetBuildResult& result,std::shared_ptr<Image> image)
{
    const auto vsFileContent = Utility::readBinaryFileContents(vsPath);
    const auto fsFileContent = Utility::readBinaryFileContents(fsPath);

    result.triangleVS = VKBackend::loadShader(VKBackend::device, vsFileContent);
    assert(result.triangleVS);
    result.triangleFS = VKBackend::loadShader(VKBackend::device, fsFileContent);
    assert(result.triangleFS);

    auto setsV = VKBackend::getDescriptorSetLayoutDataFromSpv(vsFileContent);
    auto setsF = VKBackend::getDescriptorSetLayoutDataFromSpv(fsFileContent);

    //std::vector<VkVertexInputAttributeDescription> vertIPAttribDesc;
    //std::vector<VkVertexInputBindingDescription> vertIPBindDesc;
    VKBackend::getInputInfoFromSpv(vsFileContent, result.vertIPAttribDesc, result.vertIPBindDesc, false);

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

    for (const auto& set : setsV)
    {
        layoutBindings.insert(layoutBindings.end(), set.bindings.begin(), set.bindings.end());
    }

    for (const auto& set : setsF)
    {
        layoutBindings.insert(layoutBindings.end(), set.bindings.begin(), set.bindings.end());
    }

    result.descSetLayout = VKBackend::getDescriptorSetLayout(layoutBindings);

    for (size_t l = 0; l < layoutBindings.size(); l++)
    {
        auto lb = layoutBindings.at(l);
        Descriptor descriptor;
        switch (lb.binding)
        {
        case 0: // v shader ubo
            descriptor.buffer = std::make_shared<Buffer>(uboVert);
            descriptor.layout = lb;
            break;
        case 1: // f shader ubo
            descriptor.buffer = std::make_shared<Buffer>(uboFrag);
            descriptor.layout = lb;
            break;
        case 2: // f shader sampler
            descriptor.image = image;
            descriptor.layout = lb;
            break;
        default:
            assert(false);
        }

        result.descriptors.push_back(descriptor);
    }
    
    createDescriptorSets(VKBackend::device, result.descriptors, result.descSets, result.descSetLayout);
}
void updateFrame()
{
    auto now = std::chrono::system_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(lastTime.time_since_epoch()).count()==0)
    {
        lastTime = now;
    }
    
   
}
void compileShaders()
{
    //ToDo : Compile only file content is changed
    auto res = system("glslangValidator.exe -V ./shaders/simpleMat.frag.glsl -o ./shaders/simpleMat.frag.spv");
    assert(res == 0);
    res = system("glslangValidator.exe -V ./shaders/simpleMatBVH.vert.glsl -o ./shaders/simpleMatBVH.vert.spv");
    assert(res == 0);

    res = system("glslangValidator.exe -V ./shaders/simpleMatFlat.frag.glsl -o ./shaders/simpleMatFlat.frag.spv");
    assert(res == 0);
    res = system("glslangValidator.exe -V ./shaders/simpleMatBVHFlat.vert.glsl -o ./shaders/simpleMatBVHFlat.vert.spv");
    assert(res == 0);
}
void destroyVulkan()
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(VKBackend::device, VKBackend::renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(VKBackend::device, VKBackend::imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(VKBackend::device, VKBackend::inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(VKBackend::device, VKBackend::commandPool, nullptr);

    for (auto framebuffer : VKBackend::swapChainFramebuffers) {
        vkDestroyFramebuffer(VKBackend::device, framebuffer, nullptr);
    }

    vkDestroyPipelineCache(VKBackend::device, pipelineCache, nullptr);

    if(wireframePipeline!=VK_NULL_HANDLE)
        vkDestroyPipeline(VKBackend::device, wireframePipeline, nullptr);

    vkDestroyPipeline(VKBackend::device, VKBackend::graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(VKBackend::device, VKBackend::pipelineLayout, nullptr);
    vkDestroyPipelineLayout(VKBackend::device, wireframePipelineLayout, nullptr);
    vkDestroyRenderPass(VKBackend::device, VKBackend::renderPass, nullptr);

    for (auto imageView : VKBackend::swapChainImageViews) {
        vkDestroyImageView(VKBackend::device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(VKBackend::device, VKBackend::swapchain, nullptr);

    //if (circle != nullptr)
    for(auto shape : cubes)
    {
        if (shape->isInterleaved)
        {
            vkDestroyBuffer(VKBackend::device, shape->vertexBuffer, nullptr);
            vkFreeMemory(VKBackend::device, shape->vertexBufferMemory, nullptr);
        }
        else
        {
            vkDestroyBuffer(VKBackend::device, shape->posBuffer, nullptr);
            vkFreeMemory(VKBackend::device, shape->posBufferMemory, nullptr);
            vkDestroyBuffer(VKBackend::device, shape->normBuffer, nullptr);
            vkFreeMemory(VKBackend::device, shape->normBufferMemory, nullptr);
            vkDestroyBuffer(VKBackend::device, shape->uvBuffer, nullptr);
            vkFreeMemory(VKBackend::device, shape->uvBufferMemory, nullptr);
        }
        vkDestroyBuffer(VKBackend::device, shape->indexBuffer, nullptr);
        vkFreeMemory(VKBackend::device, shape->indexBufferMemory, nullptr);
    }

    for (auto shape : wireFrameObjs)
    {
        if (shape->isInterleaved)
        {
            vkDestroyBuffer(VKBackend::device, shape->vertexBuffer, nullptr);
            vkFreeMemory(VKBackend::device, shape->vertexBufferMemory, nullptr);
        }
        else
        {
            vkDestroyBuffer(VKBackend::device, shape->posBuffer, nullptr);
            vkFreeMemory(VKBackend::device, shape->posBufferMemory, nullptr);
            vkDestroyBuffer(VKBackend::device, shape->normBuffer, nullptr);
            vkFreeMemory(VKBackend::device, shape->normBufferMemory, nullptr);
            vkDestroyBuffer(VKBackend::device, shape->uvBuffer, nullptr);
            vkFreeMemory(VKBackend::device, shape->uvBufferMemory, nullptr);
        }
        vkDestroyBuffer(VKBackend::device, shape->indexBuffer, nullptr);
        vkFreeMemory(VKBackend::device, shape->indexBufferMemory, nullptr);
    }

    vkDestroyImageView(VKBackend::device, depthImageView, nullptr);
    vkDestroyImage(VKBackend::device, depthImage, nullptr);
    vkFreeMemory(VKBackend::device, depthImageMemory, nullptr);

    vkDestroyImageView(VKBackend::device,msColorAttch->colorImageView, nullptr);
    vkDestroyImage(VKBackend::device, msColorAttch->colorImage, nullptr);
    vkFreeMemory(VKBackend::device, msColorAttch->colorImageMemory, nullptr);

    for (size_t i=0;i<descriptors.size();i++)
    {
        auto desc = descriptors.at(i);
        if (desc.layout.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        {
            for (size_t u = 0; u < desc.buffer->uniformBuffers.size(); u++)
            {
                vkDestroyBuffer(VKBackend::device, desc.buffer->uniformBuffers.at(u), nullptr);
                vkFreeMemory(VKBackend::device, desc.buffer->uniformBufferMemories.at(u), nullptr);
            }
        }
        else if (desc.layout.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
        {
            vkDestroyImageView(VKBackend::device, desc.image->texContainer->textureImageView, nullptr);
            vkDestroyImage(VKBackend::device, desc.image->texContainer->textureImage, nullptr);
            vkFreeMemory(VKBackend::device, desc.image->texContainer->textureImageMemory, nullptr);
            vkDestroySampler(VKBackend::device, desc.image->texContainer->textureSampler, nullptr);
        }
    }

    vkDestroyDescriptorPool(VKBackend::device, VKBackend::descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(VKBackend::device, VKBackend::descriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(VKBackend::device, descLayoutFlat, nullptr);

    vkDestroyDevice(VKBackend::device, nullptr);
    vkDestroySurfaceKHR(VKBackend::vkInstance, VKBackend::surface, nullptr);
    vkDestroyInstance(VKBackend::vkInstance,nullptr);
}
VkSurfaceKHR createSurface(GLFWwindow* window, VkInstance instance)
{
    VkSurfaceKHR surface;
    glfwCreateWindowSurface(instance, window, nullptr, &surface);
    return surface;
}
void createUniformBuffers()
{
    uboFrag.range = sizeof(UBOFrag);
    uboFrag.uniformBuffers.resize(VKBackend::swapchainMinImageCount);
    uboFrag.uniformBufferMemories.resize(VKBackend::swapchainMinImageCount);
    uboFrag.bufferInfo.resize(VKBackend::swapchainMinImageCount);
    
    uboVert.range = sizeof(UniformBufferObject);
    uboVert.uniformBuffers.resize(VKBackend::swapchainMinImageCount);
    uboVert.uniformBufferMemories.resize(VKBackend::swapchainMinImageCount);
    uboVert.bufferInfo.resize(VKBackend::swapchainMinImageCount);

    for (size_t i = 0; i < VKBackend::swapchainMinImageCount; i++)
    {
       VkDeviceSize bufferSize = sizeof(UniformBufferObject);
       VKBackend::createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboVert.uniformBuffers.at(i), uboVert.uniformBufferMemories.at(i));
       uboVert.bufferInfo.at(i).buffer = uboVert.uniformBuffers.at(i);
       uboVert.bufferInfo.at(i).offset = 0;
       uboVert.bufferInfo.at(i).range = uboVert.range;

       bufferSize = sizeof(UBOFrag);
       VKBackend::createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboFrag.uniformBuffers.at(i), uboFrag.uniformBufferMemories.at(i));
       uboFrag.bufferInfo.at(i).buffer = uboFrag.uniformBuffers.at(i);
       uboFrag.bufferInfo.at(i).offset = 0;
       uboFrag.bufferInfo.at(i).range = uboFrag.range;
    }
}
void createDescriptorSets(const VkDevice device, const std::vector<Descriptor>& descriptors)
{
    std::vector<VkDescriptorSetLayout> layouts(VKBackend::swapchainMinImageCount, VKBackend::descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocateInfo.descriptorPool = VKBackend::descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(3);
    allocateInfo.pSetLayouts = layouts.data();;

    VKBackend::descriptorSets.resize(VKBackend::swapchainMinImageCount);

    if (vkAllocateDescriptorSets(device, &allocateInfo, VKBackend::descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < VKBackend::swapchainMinImageCount; i++)
    {
        std::vector<VkWriteDescriptorSet> descriptorWrites(descriptors.size());

        for (size_t d = 0; d < descriptors.size(); d++)
        {
            descriptorWrites.at(d).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(d).dstSet = VKBackend::descriptorSets[i];
            descriptorWrites.at(d).dstBinding = descriptors.at(d).layout.binding;
            descriptorWrites.at(d).dstArrayElement = 0;
            descriptorWrites.at(d).descriptorType = descriptors.at(d).layout.descriptorType;
            descriptorWrites.at(d).descriptorCount = descriptors.at(d).layout.descriptorCount;
            if (descriptors.at(d).layout.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            {
                descriptors.at(d).buffer->bufferInfo.at(i).buffer = descriptors.at(d).buffer->uniformBuffers.at(i);
                descriptors.at(d).buffer->bufferInfo.at(i).offset = 0;
                descriptors.at(d).buffer->bufferInfo.at(i).range = descriptors.at(d).buffer->range;
                descriptorWrites.at(d).pBufferInfo = &descriptors.at(d).buffer->bufferInfo.at(i);
            }
            else if (descriptors.at(d).layout.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            {
                descriptorWrites.at(d).pImageInfo = &descriptors.at(d).image->imageInfo;
            }
            
        }

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
void createDescriptorSets(const VkDevice device, const std::vector<Descriptor>& descriptors, std::vector<VkDescriptorSet>& descSets)
{
    std::vector<VkDescriptorSetLayout> layouts(VKBackend::swapchainMinImageCount, VKBackend::descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocateInfo.descriptorPool = VKBackend::descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(3);
    allocateInfo.pSetLayouts = layouts.data();;

    descSets.resize(VKBackend::swapchainMinImageCount);

    if (vkAllocateDescriptorSets(device, &allocateInfo, descSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < VKBackend::swapchainMinImageCount; i++)
    {
        std::vector<VkWriteDescriptorSet> descriptorWrites(descriptors.size());

        for (size_t d = 0; d < descriptors.size(); d++)
        {
            descriptorWrites.at(d).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(d).dstSet = descSets[i];
            descriptorWrites.at(d).dstBinding = descriptors.at(d).layout.binding;
            descriptorWrites.at(d).dstArrayElement = 0;
            descriptorWrites.at(d).descriptorType = descriptors.at(d).layout.descriptorType;
            descriptorWrites.at(d).descriptorCount = descriptors.at(d).layout.descriptorCount;
            if (descriptors.at(d).layout.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            {
                descriptors.at(d).buffer->bufferInfo.at(i).buffer = descriptors.at(d).buffer->uniformBuffers.at(i);
                descriptors.at(d).buffer->bufferInfo.at(i).offset = 0;
                descriptors.at(d).buffer->bufferInfo.at(i).range = descriptors.at(d).buffer->range;
                descriptorWrites.at(d).pBufferInfo = &descriptors.at(d).buffer->bufferInfo.at(i);
            }
            else if (descriptors.at(d).layout.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            {
                descriptorWrites.at(d).pImageInfo = &descriptors.at(d).image->imageInfo;
            }

        }

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
void createDescriptorSets(const VkDevice device, const std::vector<Descriptor>& descriptors, std::vector<VkDescriptorSet>& descSets, VkDescriptorSetLayout descLayout)
{
    std::vector<VkDescriptorSetLayout> layouts(VKBackend::swapchainMinImageCount, descLayout);
    VkDescriptorSetAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocateInfo.descriptorPool = VKBackend::descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(3);
    allocateInfo.pSetLayouts = layouts.data();;

    descSets.resize(VKBackend::swapchainMinImageCount);

    if (vkAllocateDescriptorSets(device, &allocateInfo, descSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < VKBackend::swapchainMinImageCount; i++)
    {
        std::vector<VkWriteDescriptorSet> descriptorWrites(descriptors.size());

        for (size_t d = 0; d < descriptors.size(); d++)
        {
            descriptorWrites.at(d).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(d).dstSet = descSets[i];
            descriptorWrites.at(d).dstBinding = descriptors.at(d).layout.binding;
            descriptorWrites.at(d).dstArrayElement = 0;
            descriptorWrites.at(d).descriptorType = descriptors.at(d).layout.descriptorType;
            descriptorWrites.at(d).descriptorCount = descriptors.at(d).layout.descriptorCount;
            if (descriptors.at(d).layout.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            {
                descriptors.at(d).buffer->bufferInfo.at(i).buffer = descriptors.at(d).buffer->uniformBuffers.at(i);
                descriptors.at(d).buffer->bufferInfo.at(i).offset = 0;
                descriptors.at(d).buffer->bufferInfo.at(i).range = descriptors.at(d).buffer->range;
                descriptorWrites.at(d).pBufferInfo = &descriptors.at(d).buffer->bufferInfo.at(i);
            }
            else if (descriptors.at(d).layout.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            {
                descriptorWrites.at(d).pImageInfo = &descriptors.at(d).image->imageInfo;
            }

        }

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
void createPipelineCache(const VkDevice device)
{
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    if (vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline cache!");
    }
}
VkPipeline createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkShaderModule vsModule, VkShaderModule fsModule)
{
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = VKBackend::getPipelineShaderStage(VK_SHADER_STAGE_VERTEX_BIT,vsModule);
    stages[1] = VKBackend::getPipelineShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fsModule);
    

    auto vertBindingDesc = VKUtility::Vertex::getBindingDescription();
    auto vertAttribDescs = VKUtility::Vertex::getAttributeDescriptions();

    auto vertexInputInfo = VKBackend::getPipelineVertexInputState(1, &vertBindingDesc,static_cast<uint32_t>(vertAttribDescs.size()),
        vertAttribDescs.data());
    auto inputAssembly = VKBackend::getPipelineInputAssemblyState(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE);
    auto viewportState = VKBackend::getPipelineViewportState(1, 1);
    auto rasterizer = VKBackend::getPipelineRasterState(VK_POLYGON_MODE_FILL, 1.0f);
    //rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    //rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    //rasterizer.depthBiasEnable = VK_FALSE;
    auto multisampling = VKBackend::getPipelineMultisampleState(VK_FALSE, VKBackend::msaaSamples);
    auto depthStencil = VKBackend::getPipelineDepthStencilState(VK_TRUE,VK_TRUE,VK_COMPARE_OP_LESS,VK_FALSE,0.0f,1.0f,VK_FALSE);

    VkPipelineColorBlendAttachmentState colorBlendAttachment = VKBackend::getPipelineColorBlendAttachState(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT, VK_FALSE);
    const float blendConsts[4] = {0,0,0,0};
    auto colorBlending = VKBackend::getPipelineColorBlendState(VK_FALSE, VK_LOGIC_OP_COPY, 1, &colorBlendAttachment, blendConsts);

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    auto dynamicState = VKBackend::getPipelineDynamicState(dynamicStates);

    VkPushConstantRange pushConstant;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(PushConstant);
    pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    std::vector<VkPushConstantRange> pushConstants = {pushConstant};
    std::vector<VkDescriptorSetLayout> descriptorLayouts = { VKBackend::descriptorSetLayout };

    VKBackend::pipelineLayout = VKBackend::createPipelineLayout(descriptorLayouts, pushConstants);

    VkPipeline graphicsPipeline;
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.layout = VKBackend::pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    return graphicsPipeline;
}
VkPipeline createGraphicsPipeline(VkDevice device, VkPipelineCache pipelineCache, VkRenderPass renderPass, VkShaderModule vsModule, VkShaderModule fsModule,
    std::vector<VkVertexInputAttributeDescription>& vertIPAttribDesc, 
    std::vector<VkVertexInputBindingDescription>& vertIPBindDesc,
    const VkPipelineLayout pipelineLayout,
    const VkPolygonMode polygonMode,
    const VkPrimitiveTopology primToplogy)
{
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = VKBackend::getPipelineShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vsModule);
    stages[1] = VKBackend::getPipelineShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fsModule);

    auto vertexInputInfo = VKBackend::getPipelineVertexInputState(static_cast<uint32_t>(vertIPBindDesc.size()), vertIPBindDesc.data(), static_cast<uint32_t>(vertIPAttribDesc.size()),
        vertIPAttribDesc.data());
    auto inputAssembly = VKBackend::getPipelineInputAssemblyState(primToplogy, VK_FALSE);
    auto viewportState = VKBackend::getPipelineViewportState(1, 1);
    auto rasterizer = VKBackend::getPipelineRasterState(polygonMode, 1.0f);
    auto multisampling = VKBackend::getPipelineMultisampleState(VK_FALSE, VKBackend::msaaSamples);
    auto depthStencil = VKBackend::getPipelineDepthStencilState(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS, VK_FALSE, 0.0f, 1.0f, VK_FALSE);

    VkPipelineColorBlendAttachmentState colorBlendAttachment = VKBackend::getPipelineColorBlendAttachState(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT, VK_FALSE);
    const float blendConsts[4] = { 0,0,0,0 };
    auto colorBlending = VKBackend::getPipelineColorBlendState(VK_FALSE, VK_LOGIC_OP_COPY, 1, &colorBlendAttachment, blendConsts);

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    auto dynamicState = VKBackend::getPipelineDynamicState(dynamicStates);

    VkPipeline graphicsPipeline;
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    //rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
    ////https://github.com/KhronosGroup/Vulkan-LoaderAndValidationLayers/issues/785
    //pipelineInfo.pRasterizationState = &rasterizer;

    //if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &wireframePipeline) != VK_SUCCESS) {
    //    throw std::runtime_error("failed to create graphics pipeline for wireframe !");
    //}

    return graphicsPipeline;
}
void createFramebuffers()
{
    VKBackend::swapChainFramebuffers.resize(VKBackend::swapChainImageViews.size());

    for (size_t i = 0; i < VKBackend::swapChainImageViews.size(); i++) {
        /*VkImageView attachments[] = {
            swapChainImageViews[i]
        };*/

        std::array<VkImageView, 3> attachments = {
            msColorAttch->colorImageView,
            depthImageView,
            VKBackend::swapChainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = VKBackend::renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());;
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = VKBackend::swapChainExtent.width;
        framebufferInfo.height = VKBackend::swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(VKBackend::device, &framebufferInfo, nullptr, &VKBackend::swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = VKBackend::beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    VKBackend::endSingleTimeCommands(commandBuffer);
}
void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = VKBackend::renderPass;
    renderPassInfo.framebuffer = VKBackend::swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = VKBackend::swapChainExtent;

    VkClearColorValue color = { 0.1f,0.2f,0.3f,1.0 };
    VkClearValue clearColor = { {{0.1f, 0.2f, 0.3f, 1.0f}} };

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { {0.f, 0.f, 0.f, 1.0f} };
    clearValues[1].depthStencil = { 1.0f, 0 };


    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VKBackend::graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0;
    viewport.width = (float)VKBackend::swapChainExtent.width;
    viewport.height = (float)VKBackend::swapChainExtent.height;
    /*viewport.x = 0.0f;
    viewport.y = (float)swapChainExtent.height;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = -(float)swapChainExtent.height;*/
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = VKBackend::swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VKBackend::pipelineLayout, 0, 1, &VKBackend::descriptorSets[imageIndex], 0, nullptr);
    for (const auto& shape : cubes)
    {
        //std::shared_ptr<VKMesh3D> shape = cube;

        std::vector<VkBuffer> vertexBuffers;// [] = { shape->vertexBuffer };
        std::vector<VkDeviceSize> offsets;
        if (shape->isInterleaved)
        {
            vertexBuffers.push_back(shape->vertexBuffer);
            offsets.push_back(0);
        }
        else
        {
            vertexBuffers.push_back(shape->posBuffer);
            vertexBuffers.push_back(shape->normBuffer);
            vertexBuffers.push_back(shape->uvBuffer);
            offsets.push_back(0);
            offsets.push_back(0);
            offsets.push_back(0);
        }
        

        PushConstant pConstant;
        pConstant.tMat = shape->rMatrix* shape->tMatrix;
        //https://www.reddit.com/r/vulkan/comments/hszobo/noob_question_utilizing_multiple_vkbuffers_and/
        //https://gist.github.com/SaschaWillems/428d15ed4b5d71ead462bc63adffa93a
        vkCmdBindVertexBuffers(commandBuffer, 0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(), &offsets.at(0));
        vkCmdBindIndexBuffer(commandBuffer, shape->indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdPushConstants(commandBuffer, VKBackend::pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant), &pConstant);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(shape->meshData->iData.size()), 1, 0, 0, 0);
    }


    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframePipelineLayout, 0, 1, &descriptorSetsFlat[imageIndex], 0, nullptr);
    for (const auto& shape : wireFrameObjs)
    {
        //std::shared_ptr<VKMesh3D> shape = cube;

        std::vector<VkBuffer> vertexBuffers;// [] = { shape->vertexBuffer };
        std::vector<VkDeviceSize> offsets;
        if (shape->isInterleaved)
        {
            vertexBuffers.push_back(shape->vertexBuffer);
            offsets.push_back(0);
        }
        else
        {
            vertexBuffers.push_back(shape->posBuffer);
            vertexBuffers.push_back(shape->normBuffer);
            //vertexBuffers.push_back(shape->uvBuffer);
            offsets.push_back(0);
            offsets.push_back(0);
            //offsets.push_back(0);
        }


        PushConstant pConstant;
        pConstant.tMat = shape->rMatrix * shape->tMatrix;
        //https://www.reddit.com/r/vulkan/comments/hszobo/noob_question_utilizing_multiple_vkbuffers_and/
        //https://gist.github.com/SaschaWillems/428d15ed4b5d71ead462bc63adffa93a
        vkCmdBindVertexBuffers(commandBuffer, 0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(), &offsets.at(0));
        vkCmdBindIndexBuffer(commandBuffer, shape->indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdPushConstants(commandBuffer, VKBackend::pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant), &pConstant);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(shape->meshData->iData.size()), 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}
void updateUniformBuffer(uint32_t currentImage)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo;
    //ubo.model = glm::mat4(1.0f);
    ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(time*5), glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.view = glm::lookAt(glm::vec3(0.0f, 5.f, 20.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), VKBackend::swapChainExtent.width / (float)VKBackend::swapChainExtent.height, 0.1f, 500.0f);

    auto mv = ubo.model;
    ubo.nrmlMat = glm::transpose(glm::inverse(mv));
    ubo.proj[1][1] *= -1;

    void* data;
    vkMapMemory(VKBackend::device, uboVert.uniformBufferMemories[currentImage], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(VKBackend::device, uboVert.uniformBufferMemories[currentImage]);

    if (uboFrag.isDirtry)
    {
        uboFrag.isDirtry = false;
        for (size_t ci=0;ci<VKBackend::swapChainImageViews.size();ci++)
        {
            void* uboData;
            
            vkMapMemory(VKBackend::device, uboFrag.uniformBufferMemories[ci], 0, sizeof(lightInfo), 0, &uboData);
            memcpy(uboData, &lightInfo, sizeof(lightInfo));
            vkUnmapMemory(VKBackend::device, uboFrag.uniformBufferMemories[ci]);
        }
    }
}
void createDepthResources()
{
    VkFormat depthFormat = VKBackend::findDepthFormat();

    VKBackend::createImage(VKBackend::swapChainExtent.width, VKBackend::swapChainExtent.height, VKBackend::msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
    depthImageView = VKBackend::createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
}
void createColorResource()
{
    VkFormat colorFormat = VKBackend::swapChainImageFormat;
    msColorAttch = std::make_shared<VKBackend::VKRenderTarget>();

    VKBackend::createImage(VKBackend::swapChainExtent.width, VKBackend::swapChainExtent.height, VKBackend::msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,msColorAttch->colorImage, msColorAttch->colorImageMemory);
    msColorAttch->colorImageView = VKBackend::createImageView(msColorAttch->colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
}

void fillCube(float width, float height, float depth,glm::mat4 tMat, std::vector<VKUtility::VDPosNorm>& verts, std::vector<uint16_t>& indices)
{
    auto hasTex = true;
    glm::vec3 bbMin, bbMax;
    bbMax.x = width / 2;
    bbMax.y = height / 2;
    bbMax.z = depth / 2;
    bbMin.x = -width / 2;
    bbMin.y = -height / 2;
    bbMin.z = -depth / 2;


    std::vector<glm::vec3> vArr, nArr;
    std::vector<glm::vec2> uvArr;

    //top
    glm::vec3 t1 = glm::vec3(bbMin.x, bbMax.y, bbMin.z);
    glm::vec3 t2 = glm::vec3(bbMax.x, bbMax.y, bbMin.z);
    glm::vec3 t3 = glm::vec3(bbMax.x, bbMax.y, bbMax.z);
    glm::vec3 t4 = glm::vec3(bbMin.x, bbMax.y, bbMax.z);

    //bottom
    glm::vec3 b1 = glm::vec3(bbMin.x, bbMin.y, bbMin.z);
    glm::vec3 b2 = glm::vec3(bbMax.x, bbMin.y, bbMin.z);
    glm::vec3 b3 = glm::vec3(bbMax.x, bbMin.y, bbMax.z);
    glm::vec3 b4 = glm::vec3(bbMin.x, bbMin.y, bbMax.z);

    // front			back
    //		t4--t3			t2--t1
    //		|    |			|	|
    //		b4--b3			b2--b1
    // left			right
    //		t1--t4			t3--t2
    //		|    |			|	|
    //		b1--b4			b3--b2
    // top			bottom
    //		t1--t2			b4--b3
    //		|    |			|	|
    //		t4--t3			b1--b2
    //front
    vArr.push_back(b4);		vArr.push_back(b3);		vArr.push_back(t3);
    vArr.push_back(b4);		vArr.push_back(t3);		vArr.push_back(t4);
    {
        auto n = VKUtility::getNormal(b4, b3, t3);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        if (hasTex)
        {
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 1));
        }
        else
        {
            for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
        }
    }


    //back
    vArr.push_back(b2);		vArr.push_back(b1);		vArr.push_back(t1);
    vArr.push_back(b2);		vArr.push_back(t1);		vArr.push_back(t2);
    {
        auto n = VKUtility::getNormal(b2, b1, t1);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        if (hasTex)
        {
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 1));
        }
        else
        {
            for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
        }
    }

    //left
    vArr.push_back(b1);		vArr.push_back(b4);		vArr.push_back(t4);
    vArr.push_back(b1);		vArr.push_back(t4);		vArr.push_back(t1);
    {
        auto n = VKUtility::getNormal(b1, b4, t4);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        if (hasTex)
        {
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 1));
        }
        else
        {
            for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
        }
    }

    //right
    vArr.push_back(b3);		vArr.push_back(b2);		vArr.push_back(t2);
    vArr.push_back(b3);		vArr.push_back(t2);		vArr.push_back(t3);
    {
        auto n = VKUtility::getNormal(b3, b2, t2);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        if (hasTex)
        {
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 1));
        }
        else
        {
            for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
        }
    }

    //top
    vArr.push_back(t4);		vArr.push_back(t3);		vArr.push_back(t2);
    vArr.push_back(t4);		vArr.push_back(t2);		vArr.push_back(t1);
    {
        auto n = VKUtility::getNormal(t4, t3, t2);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        if (hasTex)
        {
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 1));
        }
        else
        {
            for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
        }
    }

    //bottom
    vArr.push_back(b1);		vArr.push_back(b2);		vArr.push_back(b3);
    vArr.push_back(b1);		vArr.push_back(b3);		vArr.push_back(b4);
    {
        auto n = VKUtility::getNormal(b1, b2, b3);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        nArr.push_back(n);
        if (hasTex)
        {
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 0));
            uvArr.push_back(glm::vec2(1, 1));
            uvArr.push_back(glm::vec2(0, 1));
        }
        else
        {
            for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
        }
    }

    auto totalVerts = vArr.size();
    for (auto i = 0; i < totalVerts; i++)
    {
        auto v = glm::vec3(tMat*glm::vec4(vArr.at(i),1));
        auto n = glm::normalize(nArr.at(i));
        auto uv = uvArr.at(i);
        verts.push_back({ v,n });
        indices.push_back((uint16_t)indices.size());
    }
   
}
void setupScene()
{
   /* cube = std::make_shared<VKMesh3D>();

    std::vector<VKUtility::VDPosNorm> verts;
    std::vector<uint16_t> inds;

    const int rows = 1;
    const int cols = 1;

    glm::vec3 origin = glm::vec3(-rows / 2, 0, -cols / 2);
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int ind = (row * rows) + col;
            float noise = (rand() % 100) / 100.0f;
            float h = 0.8f + (noise * 2);
            auto trans = glm::vec3(origin.x + (row), origin.y + (h / 2), origin.z + (col));
            auto tMat = glm::translate(glm::mat4(1), trans);
            fillCube(0.8f, h, 0.8f, tMat, verts, inds);
        }
    }
    cube->vertices = verts;
    cube->indices = inds;
    cube->createBuffers(VKBackend::device);
    cube->tMatrix = glm::mat4(1);

    shapes.push_back(cube);*/
    
    setupRandomTris();
    //loadGlbModel("models/Avocado.glb");

    lightInfo.position = glm::vec4(0, 20, 0, 0);
    lightInfo.color = glm::vec4(0.5, 0.5, 1.f, 1.0f);
    
}
void setupRandomTris()
{
    
    std::vector<VKUtility::Vertex> interleavedArr;
    std::vector<uint16_t> iArr;

    const auto uv1 = glm::vec2(0, 0);
    const auto uv2 = glm::vec2(1, 0);
    const auto uv3 = glm::vec2(0.5, 0.5);
    for (size_t i=0;i<N;i++)
    {
        glm::vec3 r0((rand() % 100) / 100.0f, (rand() % 100) / 100.0f, (rand() % 100) / 100.0f);
        glm::vec3 r1((rand() % 100) / 100.0f, (rand() % 100) / 100.0f, (rand() % 100) / 100.0f);
        glm::vec3 r2((rand() % 100) / 100.0f, (rand() % 100) / 100.0f, (rand() % 100) / 100.0f);
        
        glm::vec3 p1 = r0 * 9.0f - glm::vec3(5);
        glm::vec3 p2 = p1 + r1;
        glm::vec3 p3 = p1 + r2;

        tris[i].v0 = p1;
        tris[i].v1 = p2;
        tris[i].v2 = p3;
        
        auto n = VKUtility::getNormal(p1, p2, p3);
        VKUtility::Vertex v1(p1, n, uv1);
        VKUtility::Vertex v2(p2, n, uv2);
        VKUtility::Vertex v3(p3, n, uv3);
        interleavedArr.push_back(v1);
        interleavedArr.push_back(v2);
        interleavedArr.push_back(v3);

        iArr.push_back(static_cast<uint16_t>(iArr.size()));
        iArr.push_back(static_cast<uint16_t>(iArr.size()));
        iArr.push_back(static_cast<uint16_t>(iArr.size()));
    }

    auto triMesh = std::make_shared<VKUtility::Mesh>(interleavedArr, iArr);
    triMesh->name = "tri mesh";


    auto cubeMeshOL = VKUtility::getCube(3.5, 3.5, 3.5);
    
    auto cube = std::make_shared<VKMesh>();
    cube->meshData = triMesh;
    //cube->createBuffers(VKBackend::device);
    cube->createBuffNonInterleaved(VKBackend::device);
    cube->tMatrix = glm::mat4(1);

    cubes.push_back(cube);

    buildBVH();

    std::vector<BVHNode*> stack;
    BVHNode* root = &bvhNodes[0];
    stack.push_back(root);

    while (stack.size()>0)
    {
        auto node = stack.back();
        stack.pop_back();

        auto tst = (node->aabbMin + node->aabbMax) / 2.0f;

        auto mesh = VKUtility::getCubeOutline(node->aabbMin, node->aabbMax);
        auto cubeWF = std::make_shared<VKMesh3D>();
        cubeWF->meshData = mesh;
        //cube->createBuffers(VKBackend::device);
        cubeWF->createBuffNonInterleaved(VKBackend::device);
        //test case
        cubeWF->tMatrix = glm::translate(glm::mat4(1),(node->aabbMin+node->aabbMax)/2.0f);

        wireFrameObjs.push_back(cubeWF);

        if (!node->isLeaf())
        {
            stack.push_back(&bvhNodes[node->leftFirst]);
            stack.push_back(&bvhNodes[node->leftFirst + 1]);
        }
        //break;
    }
    
}
void loadGlbModel(std::string filePath)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;

    bool res = loader.LoadBinaryFromFile(&model, &err, &warn, filePath);
    if (!warn.empty())
    {
        fmt::print("WARN: {}\n",warn);
    }

    if (!err.empty())
    {
        fmt::print("ERR: {}\n", err);
    }

    if (!res)
        fmt::print("Failed to load glTF: {}\n", filePath);
    else
        fmt::print("Loaded glTF: {}\n", filePath);

    const tinygltf::Scene& scene = model.scenes[model.defaultScene];
    for (size_t i = 0; i < scene.nodes.size(); ++i) 
    {
        assert((scene.nodes[i] >= 0) && (scene.nodes[i] < model.nodes.size()));
        
        auto &node = model.nodes[scene.nodes[i]];

        if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
            //bindMesh(vbos, model, model.meshes[node.mesh]);
            auto& mesh = model.meshes[node.mesh];
            if (mesh.primitives.size() > 0)
            {
                auto& primitive = mesh.primitives.at(0);
                auto& posAccrInd = primitive.attributes["POSITION"];
                auto& nrmAccrInd = primitive.attributes["NORMAL"];
                auto& uvAccrInd = primitive.attributes["TEXCOORD_0"];

                auto& posAccr = model.accessors.at(posAccrInd);
                auto& posBuffView = model.bufferViews.at(posAccr.bufferView);
                auto& posBuff = model.buffers.at(posBuffView.buffer);
                
            }
        }
    }
}

//BVH
void buildBVH()
{
    for (size_t i = 0; i < N; i++)
    {
        tris[i].centroid = (tris[i].v0 + tris[i].v1 + tris[i].v2) * 0.3333f;
        trisIdx[i] = static_cast<uint>(i);
    }

    //assign all triangles to root Node
    BVHNode& root = bvhNodes[rootNodeIdx];
    root.leftFirst = 0;
    root.triCount = N;

    updateNodeBounds(rootNodeIdx);
    subdivide(rootNodeIdx);
}

void updateNodeBounds(const uint& nodeIdx)
{
    BVHNode& node = bvhNodes[nodeIdx];
    node.aabbMin = glm::vec3(1e30f);
    node.aabbMax = glm::vec3(-1e30f);

    for (size_t first = node.leftFirst, i = 0; i < node.triCount; i++)
    {
        auto leafTriIndex = trisIdx[first + i];
        Tri& leafTri = tris[leafTriIndex];
        node.aabbMin = minOf2(node.aabbMin, leafTri.v0);
        node.aabbMin = minOf2(node.aabbMin, leafTri.v1);
        node.aabbMin = minOf2(node.aabbMin, leafTri.v2);
        node.aabbMax = maxOf2(node.aabbMax, leafTri.v0);
        node.aabbMax = maxOf2(node.aabbMax, leafTri.v1);
        node.aabbMax = maxOf2(node.aabbMax, leafTri.v2);
    }
}

void subdivide(uint nodeIdx)
{
    BVHNode& node = bvhNodes[nodeIdx];
    //terminate recursion
    if (node.triCount <= 2)
        return;

    //split plane axis and position
    auto extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;

    int i = node.leftFirst;
    int j = i + node.triCount - 1;

    while (i <= j)
    {
        if (tris[trisIdx[i]].centroid[axis] < splitPos)
        {
            i++;
        }
        else
        {
            swap(trisIdx[i], trisIdx[j--]);
        }
    }

    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.triCount) return;
    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;

    bvhNodes[leftChildIdx].leftFirst = node.leftFirst;
    bvhNodes[leftChildIdx].triCount = leftCount;

    bvhNodes[rightChildIdx].leftFirst = i;
    bvhNodes[rightChildIdx].triCount = node.triCount - leftCount;
    node.triCount = 0;
    node.leftFirst = leftChildIdx;

    updateNodeBounds(leftChildIdx);
    updateNodeBounds(rightChildIdx);

    //recurse
    subdivide(leftChildIdx);
    subdivide(rightChildIdx);
}

inline void swap(Tri& tri1, Tri& tri2)
{
    Tri temp = tri1;
    tri1 = tri2;
    tri2 = temp;
}

inline void swap(uint& v1, uint& v2)
{
    auto temp = v1;
    v1 = v2;
    v2 = temp;
}

inline glm::vec3 minOf2(const glm::vec3& v1, const glm::vec3& v2)
{
    auto x = (v1.x < v2.x) ? v1.x : v2.x;
    auto y = (v1.y < v2.y) ? v1.y : v2.y;
    auto z = (v1.z < v2.z) ? v1.z : v2.z;
    return glm::vec3(x, y, z);
}

inline glm::vec3 maxOf2(const glm::vec3& v1, const glm::vec3& v2)
{
    auto x = (v1.x > v2.x) ? v1.x : v2.x;
    auto y = (v1.y > v2.y) ? v1.y : v2.y;
    auto z = (v1.z > v2.z) ? v1.z : v2.z;
    return glm::vec3(x, y, z);
}

#pragma endregion functions

/*
* BVH experiment
* 
* PSO
* https://www.youtube.com/watch?v=o1cdo3d2FQk&ab_channel=Vulkan
*/

int main()
{
    compileShaders();
    createWindow();
    initVulkan();

    uint32_t currentFrame = 0;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        updateFrame();
        vkWaitForFences(VKBackend::device, 1, &VKBackend::inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        vkResetFences(VKBackend::device, 1, &VKBackend::inFlightFences[currentFrame]);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(VKBackend::device, VKBackend::swapchain, UINT64_MAX, VKBackend::imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        vkResetCommandBuffer(VKBackend::commandBuffers[currentFrame], 0);
        recordCommandBuffer(VKBackend::commandBuffers[currentFrame], imageIndex);

        updateUniformBuffer(imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { VKBackend::imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &VKBackend::commandBuffers[currentFrame];
        VkSemaphore signalSemaphores[] = { VKBackend::renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(VKBackend::graphicsQueue, 1, &submitInfo, VKBackend::inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = { VKBackend::swapchain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // Optional

        vkQueuePresentKHR(VKBackend::presentQueue, &presentInfo);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    vkDeviceWaitIdle(VKBackend::device);

    glfwTerminate();
    destroyVulkan();
    exit(EXIT_SUCCESS);
}
