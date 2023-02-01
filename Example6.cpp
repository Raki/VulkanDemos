// VulkanDemos.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "CommonHeaders.h"
#include "Utility.h"
#include "VKUtility.h"
#include "VKBackend.h"
#include "Colors.h"

#pragma region vars
const int MAX_FRAMES_IN_FLIGHT = 2;
const int WIN_WIDTH = 1024;
const int WIN_HEIGHT = 1024;
GLFWwindow* window;
auto closeWindow = false;




//struct Vertex
//{
//    Vertex(glm::vec3 pos,glm::vec3 norm,glm::vec2 texcoords)
//    {
//        position = pos;
//        normal = norm;
//        uv = texcoords;
//    }
//    glm::vec3 position;
//    glm::vec3 normal;
//    glm::vec2 uv;
//
//    static VkVertexInputBindingDescription getBindingDescription()
//    {
//        VkVertexInputBindingDescription description;
//        description.binding = 0;
//        description.stride = sizeof(Vertex);
//        description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
//
//        return description;
//    }
//
//    static VkVertexInputAttributeDescription getAttribDescription()
//    {
//        VkVertexInputAttributeDescription description;
//        description.binding = 0;
//        description.location = 0;
//        description.format = VK_FORMAT_R32G32B32_SFLOAT;
//        description.offset = 0;
//
//        return description;
//    }
//
//    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
//        VkVertexInputAttributeDescription pos, norm,uv;
//        
//        pos.binding = 0;
//        pos.location = 0;
//        pos.format = VK_FORMAT_R32G32B32_SFLOAT;
//        pos.offset = 0;
//
//        norm.binding = 0;
//        norm.location = 1;
//        norm.format = VK_FORMAT_R32G32B32_SFLOAT;
//        norm.offset = sizeof(glm::vec3);
//
//        uv.binding = 0;
//        uv.location = 2;
//        uv.format = VK_FORMAT_R32G32_SFLOAT;
//        uv.offset = sizeof(glm::vec3)*2;
//
//        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = { pos,norm,uv };
//
//        return attributeDescriptions;
//    }
//    
//};

struct VDPosColor
{
    VDPosColor(glm::vec3 pos, glm::vec3 colr)
    {
        position = pos;
        color = colr;
    }
    glm::vec3 position;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription description;
        description.binding = 0;
        description.stride = sizeof(VDPosColor);
        description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return description;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        VkVertexInputAttributeDescription pos, colr;

        pos.binding = 0;
        pos.location = 0;
        pos.format = VK_FORMAT_R32G32B32_SFLOAT;
        pos.offset = 0;

        colr.binding = 0;
        colr.location = 1;
        colr.format = VK_FORMAT_R32G32B32_SFLOAT;
        colr.offset = sizeof(glm::vec3);

        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = { pos,colr };

        return attributeDescriptions;
    }
};

struct UniformBufferObject
{
    glm::mat4 proj;
};




std::vector<VkBuffer> uniformBuffers;
std::vector<VkDeviceMemory> uniformBufferMemories;
std::vector<VKUtility::Vertex> vertices;
std::vector<VDPosColor> verticesSolid;
std::vector<uint16_t> indices;
VkBuffer vertexBuffer;
VkDeviceMemory vertexBufferMemory;
VkBuffer indexBuffer;
VkDeviceMemory indexBufferMemory;

//std::shared_ptr<VKBackend::VKTexture> texture,texture0;

VkImage depthImage;
VkDeviceMemory depthImageMemory;
VkImageView depthImageView;

std::shared_ptr<VKBackend::VKRenderTarget> msColorAttch;
#pragma endregion vars

#pragma region prototypes
void createWindow();
void initVulkan();
void destroyVulkan();

VkSurfaceKHR createSurface(GLFWwindow* window, VkInstance instace);
void createUniformBuffers();
void createDescriptorSets(VkDevice device);
VkPipeline createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkShaderModule vsModule, VkShaderModule fsModule);
void createFramebuffers();
void creatVertexAndIndexBuffers(VkDevice device, VkBuffer &vBuff,VkBuffer &iBuff);
void creatSSVertexAndIndexBuffers(VkDevice device, VkBuffer& vBuff, VkBuffer& iBuff);
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
void updateUniformBuffer(uint32_t currentImage);
void createDepthResources();
void createColorResource();
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

    VKBackend::device = VKBackend::createDevice(VKBackend::physicalDevice);

    auto swapChainSupportDetails = VKBackend::querySwapChainSupport(VKBackend::physicalDevice);

    VKBackend::swapchain = VKBackend::createSwapchain(VKBackend::device, VKBackend::surface);

    VKBackend::createSwapchainImageViews();

    VKBackend::renderPass = VKBackend::createRenerPass(VKBackend::device);

    auto triangleVS = VKBackend::loadShader(VKBackend::device, "shaders/solidShapesBasic.vert.spv");
    assert(triangleVS);
    auto triangleFS = VKBackend::loadShader(VKBackend::device, "shaders/solidShapes.frag.spv");
    assert(triangleFS);

    VKBackend::commandPool = VKBackend::createCommandPool(VKBackend::device);
    VKBackend::createCommandBuffers(VKBackend::device, VKBackend::commandPool);
    
    VkDescriptorSetLayoutBinding uboLayoutBinding;
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings = { uboLayoutBinding };
    VKBackend::createDescriptorSetLayout(layoutBindings);
    
    createUniformBuffers();
    
    VkDescriptorPoolSize poolSize;
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(VKBackend::swapchainMinImageCount);

    std::vector<VkDescriptorPoolSize> poolsizes = { poolSize };
    VKBackend::createDescriptorPool(VKBackend::device,poolsizes);
    createDescriptorSets(VKBackend::device);

    VKBackend::graphicsPipeline = createGraphicsPipeline(VKBackend::device, VKBackend::renderPass, triangleVS, triangleFS);

    vkDestroyShaderModule(VKBackend::device, triangleVS, nullptr);
    vkDestroyShaderModule(VKBackend::device, triangleFS, nullptr);

    createColorResource();
    createDepthResources();
    createFramebuffers();

    

    VKBackend::createSyncObjects();

    /*auto fsQuad = VKUtility::getFSQuad();
    vertices = fsQuad->vData;
    indices = fsQuad->iData;*/

    //Bezier::Bezier<2> cubicBezier({ {0.120, 0.160}, {0.035, 0.200}, {0.220, 0.260}/*, {0.220, 0.040}*/ });
    Bezier::Point b1{ 0.0157,0.1528 };
    Bezier::Point b2{ 0.0865,0.237 };
    Bezier::Point b3{ 0.220, 0.260 };
    Bezier::Point b4{ 0.249, 0.105 };
    const float scl = 4;
    const float wid = 0.03f;
    b1 = b1 * scl;
    b2 = b2 * scl;
    b3 = b3 * scl;
    b4 = b4 * scl;
    Bezier::Bezier<3> cubicBezier({ b1, b2, b3, b4});
    cubicBezier.translate({ -0.5,-0.5 });
    glm::vec3 colr = glm::vec3(0.1,0.2,0.3);
    for (float v = 0; v < 1.0; v += 0.01)
    {
        Bezier::Point p1;
        p1 = cubicBezier.valueAt(v);
        Bezier::Normal bn1 = cubicBezier.normalAt(v);
        glm::vec3 p1r = glm::vec3(p1[0],p1[1],0);
        glm::vec3 n1 = glm::normalize(glm::vec3(bn1[0], bn1[1], 0));
        auto v0 = p1r + (wid) * n1;
        auto v1 = p1r + (-wid) * n1;
        verticesSolid.push_back({v0,Color::getRandomColor()});
        verticesSolid.push_back({v1,Color::getRandomColor() });
    }

    for (uint16_t ind=0;ind<100-1;ind++)
    {
        uint16_t c0 = ind * 2;
        uint16_t c1 = (ind+1) * 2;
        indices.push_back(c0+1);
        indices.push_back(c1+1);
        indices.push_back(c1);

        indices.push_back(c0+1);
        indices.push_back(c1);
        indices.push_back(c0);
    }

    creatSSVertexAndIndexBuffers(VKBackend::device, vertexBuffer, indexBuffer);
    
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

    vkDestroyBuffer(VKBackend::device, vertexBuffer, nullptr);
    vkFreeMemory(VKBackend::device, vertexBufferMemory, nullptr);
    vkDestroyBuffer(VKBackend::device, indexBuffer, nullptr);
    vkFreeMemory(VKBackend::device, indexBufferMemory, nullptr);

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
        vkDestroyBuffer(VKBackend::device, uniformBuffers.at(i), nullptr);
        vkFreeMemory(VKBackend::device, uniformBufferMemories.at(i), nullptr);
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
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(VKBackend::swapchainMinImageCount);
    uniformBufferMemories.resize(VKBackend::swapchainMinImageCount);

    for (size_t i = 0; i < VKBackend::swapchainMinImageCount; i++)
    {
       VKBackend::createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers.at(i), uniformBufferMemories.at(i));
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
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        /*VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = texture0->textureImageView;
        imageInfo.sampler = texture0->textureSampler;

        VkDescriptorImageInfo imageInfo2{};
        imageInfo2.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo2.imageView = texture->textureImageView;
        imageInfo2.sampler = texture->textureSampler;*/

        std::array<VkWriteDescriptorSet, 1> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = VKBackend::descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

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
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vsModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fsModule;
    stages[1].pName = "main";

    auto vertBindingDesc = VDPosColor::getBindingDescription();
    auto vertAttribDescs = VDPosColor::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &vertBindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertAttribDescs.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertAttribDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    //rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    //rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    //rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VKBackend::msaaSamples;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();


    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &VKBackend::descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &VKBackend::pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

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

void creatVertexAndIndexBuffers(VkDevice device,VkBuffer& vBuff, VkBuffer& iBuff)
{
    //size of buffers
    VkDeviceSize vBuffSize = sizeof(VKUtility::Vertex) * vertices.size();
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
void creatSSVertexAndIndexBuffers(VkDevice device, VkBuffer& vBuff, VkBuffer& iBuff)
{
    //size of buffers
    VkDeviceSize vBuffSize = sizeof(VDPosColor) * verticesSolid.size();
    VkDeviceSize iBuffSize = sizeof(uint16_t) * indices.size();

    //staging buffer
    VkBuffer vStageBuff, iStageBuff;
    VkDeviceMemory vStageBuffMemory, iStageBuffMemory;

    VKBackend::createBuffer(vBuffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vStageBuff, vStageBuffMemory);
    void* data;
    vkMapMemory(device, vStageBuffMemory, 0, vBuffSize, 0, &data);
    memcpy(data, verticesSolid.data(), (size_t)vBuffSize);
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
    //VkClearValue clearValues;
    //clearValues.color = color;
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

    VkBuffer vertexBuffers[] = { vertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, VKBackend::pipelineLayout, 0, 1, &VKBackend::descriptorSets[imageIndex], 0, nullptr);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

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
    ubo.proj = glm::mat4(1.0f);
    ubo.proj[1][1] *= -1;

    void* data;
    vkMapMemory(VKBackend::device, uniformBufferMemories[currentImage], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(VKBackend::device, uniformBufferMemories[currentImage]);
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
#pragma endregion functions

int main()
{
    createWindow();
    initVulkan();

    uint32_t currentFrame = 0;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

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
