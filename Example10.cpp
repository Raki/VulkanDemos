// VulkanDemos.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "CommonHeaders.h"
#include "Utility.h"
#include "VKUtility.h"
#include "VKBackend.h"
#include "Colors.h"
#include <imgui.h>
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#pragma region vars
const int MAX_FRAMES_IN_FLIGHT = 2;
const int WIN_WIDTH = 1024;
const int WIN_HEIGHT = 1024;
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
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
};

struct VKMesh3D : VKMesh
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
    bool isDirtry = true;
};

struct Image
{
    VkDeviceMemory imageMemory;
    VkImageView imageView;
};

struct Descriptor
{
    VkDescriptorSetLayoutBinding layout;
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<Image> image;
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

std::shared_ptr<VKBackend::VKRenderTarget> msColorAttch;
std::vector<std::shared_ptr<VKMesh3D>> shapes;
std::shared_ptr<VKMesh3D> cube;
std::chrono::system_clock::time_point lastTime{};

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
#pragma endregion vars

#pragma region prototypes
void createWindow();
void initVulkan();
void updateFrame();
void compileShaders();
void destroyVulkan();

VkSurfaceKHR createSurface(GLFWwindow* window, VkInstance instace);
void createUniformBuffers();
void createDescriptorSets(VkDevice device);
void createDescriptorSets(const VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& descSetLayoutBindings, const std::vector<Buffer>& uboBuffers);
VkPipeline createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkShaderModule vsModule, VkShaderModule fsModule);
void createFramebuffers();
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
void updateUniformBuffer(uint32_t currentImage);
void createDepthResources();
void createColorResource();
void fillCube(float width, float height, float depth,glm::mat4 tMat, std::vector<VKUtility::VDPosNorm>& verts, std::vector<uint16_t>& indices);
void setupScene();
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
        fmt::print("GPU supports descriptor indexing");

    VKBackend::device = VKBackend::createDevice(VKBackend::physicalDevice);

    auto swapChainSupportDetails = VKBackend::querySwapChainSupport(VKBackend::physicalDevice);

    VKBackend::swapchain = VKBackend::createSwapchain(VKBackend::device, VKBackend::surface);

    VKBackend::createSwapchainImageViews();

    VKBackend::renderPass = VKBackend::createRenerPass(VKBackend::device);


    auto vsFileContent = Utility::readBinaryFileContents("shaders/solidShapes3D.vert.spv");
    auto fsFileContent = Utility::readBinaryFileContents("shaders/solidShapes3D.frag.spv");

    auto triangleVS = VKBackend::loadShader(VKBackend::device, vsFileContent);
    assert(triangleVS);
    auto triangleFS = VKBackend::loadShader(VKBackend::device, fsFileContent);
    assert(triangleFS);

    auto setsV = VKBackend::getDescriptorSetLayoutDataFromSpv(vsFileContent);
    auto setsF = VKBackend::getDescriptorSetLayoutDataFromSpv(fsFileContent);

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

    for (const auto& set : setsV)
    {
        layoutBindings.insert(layoutBindings.end(), set.bindings.begin(), set.bindings.end());
    }

    for (const auto& set : setsF)
    {
        layoutBindings.insert(layoutBindings.end(), set.bindings.begin(), set.bindings.end());
    }

    VKBackend::commandPool = VKBackend::createCommandPool(VKBackend::device);
    VKBackend::createCommandBuffers(VKBackend::device, VKBackend::commandPool);
    
    VKBackend::createDescriptorSetLayout(layoutBindings);
    
    createUniformBuffers();

    std::vector<VkDescriptorPoolSize> poolsizes;

    for (const auto& descLayoutBinding : layoutBindings)
    {
        VkDescriptorPoolSize poolSize;
        poolSize.type = descLayoutBinding.descriptorType;
        poolSize.descriptorCount = static_cast<uint32_t>(VKBackend::swapchainMinImageCount);
        poolsizes.push_back(poolSize);
    }

    VKBackend::createDescriptorPool(VKBackend::device,poolsizes);
    //createDescriptorSets(VKBackend::device);
    std::vector<Buffer> ubos = {uboVert,uboFrag};
    createDescriptorSets(VKBackend::device,layoutBindings,ubos);

    VKBackend::graphicsPipeline = createGraphicsPipeline(VKBackend::device, VKBackend::renderPass, triangleVS, triangleFS);

    vkDestroyShaderModule(VKBackend::device, triangleVS, nullptr);
    vkDestroyShaderModule(VKBackend::device, triangleFS, nullptr);

    createColorResource();
    createDepthResources();
    createFramebuffers();

    VKBackend::createSyncObjects();

    setupScene();
    
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
    auto res = system("glslangValidator.exe -V ./shaders/solidShapes3D.frag.glsl -o ./shaders/solidShapes3D.frag.spv");
    assert(res == 0);
    res = system("glslangValidator.exe -V ./shaders/solidShapes3D.vert.glsl -o ./shaders/solidShapes3D.vert.spv");
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

    vkDestroyPipeline(VKBackend::device, VKBackend::graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(VKBackend::device, VKBackend::pipelineLayout, nullptr);
    vkDestroyRenderPass(VKBackend::device, VKBackend::renderPass, nullptr);

    for (auto imageView : VKBackend::swapChainImageViews) {
        vkDestroyImageView(VKBackend::device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(VKBackend::device, VKBackend::swapchain, nullptr);

    /*if(polyline!=nullptr)
    {
        vkDestroyBuffer(VKBackend::device, polyline->vertexBuffer, nullptr);
        vkFreeMemory(VKBackend::device, polyline->vertexBufferMemory, nullptr);
        vkDestroyBuffer(VKBackend::device, polyline->indexBuffer, nullptr);
        vkFreeMemory(VKBackend::device, polyline->indexBufferMemory, nullptr);
    }*/

    //if (circle != nullptr)
    for(auto shape : shapes)
    {
        vkDestroyBuffer(VKBackend::device, shape->vertexBuffer, nullptr);
        vkFreeMemory(VKBackend::device, shape->vertexBufferMemory, nullptr);
        vkDestroyBuffer(VKBackend::device, shape->indexBuffer, nullptr);
        vkFreeMemory(VKBackend::device, shape->indexBufferMemory, nullptr);
    }

    /*vkDestroySampler(VKBackend::device, texture0->textureSampler, nullptr);
    vkDestroyImageView(VKBackend::device, texture0->textureImageView, nullptr);
    vkDestroyImage(VKBackend::device, texture0->textureImage, nullptr);
    vkFreeMemory(VKBackend::device, texture0->textureImageMemory, nullptr);

    vkDestroySampler(VKBackend::device, texture->textureSampler, nullptr);
    vkDestroyImageView(VKBackend::device, texture->textureImageView, nullptr);
    vkDestroyImage(VKBackend::device, texture->textureImage, nullptr);
    vkFreeMemory(VKBackend::device, texture->textureImageMemory, nullptr);*/

    vkDestroyImageView(VKBackend::device, depthImageView, nullptr);
    vkDestroyImage(VKBackend::device, depthImage, nullptr);
    vkFreeMemory(VKBackend::device, depthImageMemory, nullptr);

    vkDestroyImageView(VKBackend::device,msColorAttch->colorImageView, nullptr);
    vkDestroyImage(VKBackend::device, msColorAttch->colorImage, nullptr);
    vkFreeMemory(VKBackend::device, msColorAttch->colorImageMemory, nullptr);

    for (size_t i = 0; i < VKBackend::swapchainMinImageCount; i++)
    {
        vkDestroyBuffer(VKBackend::device, uboVert.uniformBuffers.at(i), nullptr);
        vkFreeMemory(VKBackend::device, uboVert.uniformBufferMemories.at(i), nullptr);

        vkDestroyBuffer(VKBackend::device, uboFrag.uniformBuffers.at(i), nullptr);
        vkFreeMemory(VKBackend::device, uboFrag.uniformBufferMemories.at(i), nullptr);
    }


    vkDestroyDescriptorPool(VKBackend::device, VKBackend::descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(VKBackend::device, VKBackend::descriptorSetLayout, nullptr);

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
    
    uboVert.range = sizeof(UniformBufferObject);
    uboVert.uniformBuffers.resize(VKBackend::swapchainMinImageCount);
    uboVert.uniformBufferMemories.resize(VKBackend::swapchainMinImageCount);

    for (size_t i = 0; i < VKBackend::swapchainMinImageCount; i++)
    {
       VkDeviceSize bufferSize = sizeof(UniformBufferObject);
       VKBackend::createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboVert.uniformBuffers.at(i), uboVert.uniformBufferMemories.at(i));

       bufferSize = sizeof(UBOFrag);
       VKBackend::createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uboFrag.uniformBuffers.at(i), uboFrag.uniformBufferMemories.at(i));
    }
}

/*
* This is incomplete. On supports descriptors of type buffers.
*/
void createDescriptorSets(const VkDevice device,const std::vector<VkDescriptorSetLayoutBinding> &descSetLayoutBindings,
    const std::vector<Buffer> &uboBuffers)
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
        std::vector<VkDescriptorBufferInfo> bufferInfos(descSetLayoutBindings.size());
        std::vector<VkWriteDescriptorSet> descriptorWrites(descSetLayoutBindings.size());
        for (size_t b = 0; b < descSetLayoutBindings.size(); b++)
        {
            bufferInfos.at(b) = {};
            bufferInfos.at(b).buffer = uboBuffers.at(b).uniformBuffers.at(i);
            bufferInfos.at(b).offset = 0;
            bufferInfos.at(b).range = uboBuffers.at(b).range;

            descriptorWrites.at(b).sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites.at(b).dstSet = VKBackend::descriptorSets[i];
            descriptorWrites.at(b).dstBinding = descSetLayoutBindings.at(b).binding;
            descriptorWrites.at(b).dstArrayElement = 0;
            descriptorWrites.at(b).descriptorType = descSetLayoutBindings.at(b).descriptorType;
            descriptorWrites.at(b).descriptorCount = descSetLayoutBindings.at(b).descriptorCount;
            descriptorWrites.at(b).pBufferInfo = &bufferInfos.at(b);
        }

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
void createDescriptorSets(VkDevice device)
{
    std::vector<VkDescriptorSetLayout> layouts(VKBackend::swapchainMinImageCount, VKBackend::descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocateInfo.descriptorPool = VKBackend::descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(3);
    allocateInfo.pSetLayouts = layouts.data();;

    VKBackend::descriptorSets.resize(VKBackend::swapchainMinImageCount);

    if (vkAllocateDescriptorSets(device, &allocateInfo, VKBackend::descriptorSets.data())!=VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < VKBackend::swapchainMinImageCount; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uboVert.uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorBufferInfo bufferInfoFrag{};
        bufferInfoFrag.buffer = uboFrag.uniformBuffers[i];
        bufferInfoFrag.offset = 0;
        bufferInfoFrag.range = sizeof(UBOFrag);

        /*VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = texture0->textureImageView;
        imageInfo.sampler = texture0->textureSampler;

        VkDescriptorImageInfo imageInfo2{};
        imageInfo2.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo2.imageView = texture->textureImageView;
        imageInfo2.sampler = texture->textureSampler;*/

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = VKBackend::descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = VKBackend::descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &bufferInfoFrag;

       /* descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = VKBackend::descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = VKBackend::descriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pImageInfo = &imageInfo2;*/

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
VkPipeline createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkShaderModule vsModule, VkShaderModule fsModule)
{
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0] = VKBackend::getPipelineShaderStage(VK_SHADER_STAGE_VERTEX_BIT,vsModule);
    stages[1] = VKBackend::getPipelineShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fsModule);
    

    auto vertBindingDesc = VKUtility::VDPosNorm::getBindingDescription();
    auto vertAttribDescs = VKUtility::VDPosNorm::getAttributeDescriptions();

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
    //for (const auto shape : shapes)
    {
        std::shared_ptr<VKMesh3D> shape = cube;
        VkBuffer vertexBuffers[] = { shape->vertexBuffer };
        VkDeviceSize offsets[] = { 0 };

        PushConstant pConstant;
        pConstant.tMat = shape->rMatrix*shape->tMatrix;

        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, shape->indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VKBackend::pipelineLayout, 0, 1, &VKBackend::descriptorSets[imageIndex], 0, nullptr);
        vkCmdPushConstants(commandBuffer, VKBackend::pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant), &pConstant);
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(shape->indices.size()), 1, 0, 0, 0);
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
    ubo.view = glm::lookAt(glm::vec3(0.0f, 5.f, 10.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
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
    cube = std::make_shared<VKMesh3D>();

    std::vector<VKUtility::VDPosNorm> verts;
    std::vector<uint16_t> inds;

    const int rows = 2;
    const int cols = 2;

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

    shapes.push_back(cube);

    lightInfo.position = glm::vec4(0, 20, 0, 0);
    lightInfo.color = glm::vec4(0.5, 0.5, 1.f, 1.0f);
    
}

#pragma endregion functions


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
