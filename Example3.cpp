// VulkanDemos.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "CommonHeaders.h"
#include "Utility.h"
#include "VKUtility.h"

#pragma region vars
const int MAX_FRAMES_IN_FLIGHT = 2;
const int WIN_WIDTH = 1024;
const int WIN_HEIGHT = 1024;
GLFWwindow* window;
auto closeWindow = false;
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex
{
    Vertex(glm::vec3 pos,glm::vec3 norm,glm::vec2 texcoords)
    {
        position = pos;
        normal = norm;
        uv = texcoords;
    }
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription description;
        description.binding = 0;
        description.stride = sizeof(Vertex);
        description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return description;
    }

    static VkVertexInputAttributeDescription getAttribDescription()
    {
        VkVertexInputAttributeDescription description;
        description.binding = 0;
        description.location = 0;
        description.format = VK_FORMAT_R32G32B32_SFLOAT;
        description.offset = 0;

        return description;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        VkVertexInputAttributeDescription pos, norm,uv;
        
        pos.binding = 0;
        pos.location = 0;
        pos.format = VK_FORMAT_R32G32B32_SFLOAT;
        pos.offset = 0;

        norm.binding = 0;
        norm.location = 1;
        norm.format = VK_FORMAT_R32G32B32_SFLOAT;
        norm.offset = sizeof(glm::vec3);

        uv.binding = 0;
        uv.location = 2;
        uv.format = VK_FORMAT_R32G32_SFLOAT;
        uv.offset = sizeof(glm::vec3)*2;

        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = { pos,norm,uv };

        return attributeDescriptions;
    }
    
};

struct UniformBufferObject
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 nrmlMat;
};


VkInstance vkInstance;
VkSurfaceKHR surface;
VkPhysicalDevice physicalDevice;
VkDevice device;
VkQueue graphicsQueue;
VkQueue presentQueue;
VkSwapchainKHR swapchain;
std::vector<VkImage> swapChainImages;
std::vector<VkImageView> swapChainImageViews;
std::vector<VkFramebuffer> swapChainFramebuffers;
size_t swapchainMinImageCount = 0;
VkFormat swapChainImageFormat;
VkExtent2D swapChainExtent;

VkRenderPass renderPass;

VkDescriptorSetLayout descriptorSetLayout;
VkDescriptorPool descriptorPool;
std::vector<VkDescriptorSet> descriptorSets;
std::vector<VkBuffer> uniformBuffers;
std::vector<VkDeviceMemory> uniformBufferMemories;

VkPipeline graphicsPipeline;
VkPipelineLayout pipelineLayout;

std::vector<VkCommandBuffer> commandBuffers;
VkCommandPool commandPool;

std::vector<VkSemaphore> imageAvailableSemaphores;
std::vector<VkSemaphore> renderFinishedSemaphores;
std::vector<VkFence> inFlightFences;

std::vector<VKUtility::Vertex> vertices;
std::vector<uint16_t> indices;
VkBuffer vertexBuffer;
VkDeviceMemory vertexBufferMemory;
VkBuffer indexBuffer;
VkDeviceMemory indexBufferMemory;
VkImage textureImage;
VkDeviceMemory textureImageMemory;
VkImageView textureImageView;
VkSampler textureSampler;

VkImage depthImage;
VkDeviceMemory depthImageMemory;
VkImageView depthImageView;
#pragma endregion vars

#pragma region prototypes
void createWindow();
void initVulkan();
void destroyVulkan();

VkInstance createVulkanInstance();
VkSurfaceKHR createSurface(GLFWwindow* window, VkInstance instace);
VkPhysicalDevice pickPhysicalDevice(VkInstance instance);
QueueFamilyIndices pickDeviceQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
VkDevice createDevice(VkPhysicalDevice physicalDevice);
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
VkSwapchainKHR createSwapchain(VkDevice device, VkSurfaceKHR surface);
VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
void createSwapchainImageViews();
VkRenderPass createRenerPass(VkDevice device);
VkShaderModule loadShader(VkDevice device, std::string path);
void createDescriptorSetLayout();
void createDescriptorPool(VkDevice device);
void createUniformBuffers();
void createDescriptorSets(VkDevice device);

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlag, VkMemoryPropertyFlags props, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

VkPipeline createGraphicsPipeline(VkDevice device, VkRenderPass renderPass, VkShaderModule vsModule, VkShaderModule fsModule);

void createFramebuffers();

VkCommandPool createCommandPool(VkDevice device);
void createCommandBuffers(VkDevice device, VkCommandPool commandPool);

void createSyncObjects();

VkBuffer createVertexBuffer(VkDevice device);
VkBuffer createIndexBuffer(VkDevice device);
void creatVertexAndIndexBuffers(VkDevice device, VkBuffer &vBuff,VkBuffer &iBuff);
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
void updateUniformBuffer(uint32_t currentImage);

void createImageTexture(std::string filepath);
void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
VkCommandBuffer beginSingleTimeCommands();
void endSingleTimeCommands(VkCommandBuffer commandBuffer);
void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
void createTextureImageView();
VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
void createTextureSampler();
void createDepthResources();
VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
VkFormat findDepthFormat();
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

    window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, "Vulkan Example2", NULL, NULL);
    glfwSetKeyCallback(window, key_callback);
}
void initVulkan()
{
    vkInstance = createVulkanInstance();
    surface = createSurface(window, vkInstance);
    physicalDevice = pickPhysicalDevice(vkInstance);
    assert(physicalDevice);
    
    auto queFamilyIndices = pickDeviceQueueFamily(physicalDevice, surface);

    device = createDevice(physicalDevice);

    auto swapChainSupportDetails = querySwapChainSupport(physicalDevice);

    swapchain = createSwapchain(device, surface);

    createSwapchainImageViews();

    renderPass = createRenerPass(device);

    auto triangleVS = loadShader(device, "shaders/diffuseTex.vert.spv");
    assert(triangleVS);
    auto triangleFS = loadShader(device, "shaders/diffuseTex.frag.spv");
    assert(triangleFS);

    commandPool = createCommandPool(device);
    createCommandBuffers(device, commandPool);
    
    createDescriptorSetLayout();
    createDepthResources();
    createImageTexture("img/vulkan.jpg");
    createTextureImageView();
    createTextureSampler();
    createUniformBuffers();
    createDescriptorPool(device);
    createDescriptorSets(device);

    graphicsPipeline = createGraphicsPipeline(device, renderPass, triangleVS, triangleFS);

    vkDestroyShaderModule(device, triangleVS, nullptr);
    vkDestroyShaderModule(device, triangleFS, nullptr);

    createDepthResources();
    createFramebuffers();

    

    createSyncObjects();

    auto cube = VKUtility::getCube(1, 1, 1);
    vertices = cube->vData;
    indices = cube->iData;

    creatVertexAndIndexBuffers(device, vertexBuffer, indexBuffer);

    
}
void destroyVulkan()
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapchain, nullptr);

    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);

    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);

    for (size_t i = 0; i < swapchainMinImageCount; i++)
    {
        vkDestroyBuffer(device, uniformBuffers.at(i), nullptr);
        vkFreeMemory(device, uniformBufferMemories.at(i), nullptr);
    }


    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(vkInstance, surface, nullptr);
    vkDestroyInstance(vkInstance,nullptr);
}
VkInstance createVulkanInstance()
{
    VkInstance vkInstance;
    //Warn : In real world application need to check availability of
    // version using vkEnumerateInstanceVersion
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pNext = NULL;
    appInfo.pApplicationName = "Vulkan Program Template";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "LunarG SDK";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = VK_API_VERSION_1_3;

    /*
    * Bellow structure specifies layers and extensions vulkan will be
    * using, if they any one is not present then vulkan instance won't be created
    */
    VkInstanceCreateInfo vkInstCreateInfo = {};
    vkInstCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    vkInstCreateInfo.pNext = NULL;
    vkInstCreateInfo.flags = 0;
    vkInstCreateInfo.pApplicationInfo = &appInfo;

#ifdef _DEBUG
    //validation layers
    const char* debugLayers[] = { "VK_LAYER_KHRONOS_validation" };
    vkInstCreateInfo.ppEnabledLayerNames = debugLayers;
    vkInstCreateInfo.enabledLayerCount = sizeof(debugLayers) / sizeof(debugLayers[0]);
#endif

    const char* extensions[] = { VK_KHR_SURFACE_EXTENSION_NAME
#ifdef VK_USE_PLATFORM_WIN32_KHR
        ,"VK_KHR_win32_surface"
#endif
        ,VK_EXT_DEBUG_REPORT_EXTENSION_NAME
    };

    vkInstCreateInfo.ppEnabledExtensionNames = extensions;
    vkInstCreateInfo.enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]);

    VK_CHECK(vkCreateInstance(&vkInstCreateInfo, 0, &vkInstance));

    return vkInstance;
}
VkSurfaceKHR createSurface(GLFWwindow* window, VkInstance instance)
{
    VkSurfaceKHR surface;
    glfwCreateWindowSurface(instance, window, nullptr, &surface);
    return surface;
}
VkPhysicalDevice pickPhysicalDevice(VkInstance instance)
{
    uint32_t phyicalDeviceCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &phyicalDeviceCount, nullptr));
    std::vector<VkPhysicalDevice> physicalDevices(phyicalDeviceCount);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &phyicalDeviceCount, physicalDevices.data()));

    VkPhysicalDeviceProperties physicalDevProps;
    for (size_t i = 0; i < phyicalDeviceCount; i++)
    {
        vkGetPhysicalDeviceProperties(physicalDevices[i], &physicalDevProps);
        if (physicalDevProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            fmt::print("Selecting discrete GPU {}\n",physicalDevProps.deviceName);
            return physicalDevices[i];
        }
    }

    if (phyicalDeviceCount > 0)
    {
        vkGetPhysicalDeviceProperties(physicalDevices[0], &physicalDevProps);
        fmt::print("Selecting device {}\n",physicalDevProps.deviceName);
        return physicalDevices[0];
    }

    return 0;
}
QueueFamilyIndices pickDeviceQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
    QueueFamilyIndices indices;
    uint32_t qFamilyPropCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qFamilyPropCount, nullptr);
    //ToDo : make this dynamic
    std::vector<VkQueueFamilyProperties> qFamProps(qFamilyPropCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qFamilyPropCount, qFamProps.data());

    for (uint32_t i = 0; i < qFamilyPropCount; i++)
    {
        VkQueueFamilyProperties props = qFamProps[i];
        if (props.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);

        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}
VkDevice createDevice(VkPhysicalDevice physicalDevice)
{
    auto queFamilyIndices = pickDeviceQueueFamily(physicalDevice, surface);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { queFamilyIndices.graphicsFamily.value(), queFamilyIndices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }



    const char* extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    VkDevice device;
    VkDeviceCreateInfo devCreatInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    devCreatInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
    devCreatInfo.pQueueCreateInfos = queueCreateInfos.data();
    devCreatInfo.ppEnabledExtensionNames = extensions;
    devCreatInfo.enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]);
    devCreatInfo.pEnabledFeatures = &deviceFeatures;

    VK_CHECK(vkCreateDevice(physicalDevice, &devCreatInfo, 0, &device));

    vkGetDeviceQueue(device, queFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, queFamilyIndices.presentFamily.value(), 0, &presentQueue);

    return device;
}
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
{
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}
VkSwapchainKHR createSwapchain(VkDevice device, VkSurfaceKHR surface)
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }
    swapchainMinImageCount = imageCount;
    int wWidth = 0, wHeight = 0;
    glfwGetWindowSize(window, &wWidth, &wHeight);
    VkSwapchainKHR swapchain;
    VkSwapchainCreateInfoKHR sChainCrtInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    sChainCrtInfo.surface = surface;
    sChainCrtInfo.minImageCount = imageCount;
    sChainCrtInfo.imageFormat = surfaceFormat.format;// VK_FORMAT_R8G8B8A8_UNORM; //Warn : some devices support BGRA
    sChainCrtInfo.imageColorSpace = surfaceFormat.colorSpace;// VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sChainCrtInfo.imageExtent.width = wWidth;
    sChainCrtInfo.imageExtent.height = wHeight;
    sChainCrtInfo.imageArrayLayers = 1;
    sChainCrtInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;// | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    QueueFamilyIndices indices = pickDeviceQueueFamily(physicalDevice, surface);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        sChainCrtInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        sChainCrtInfo.queueFamilyIndexCount = 2;
        sChainCrtInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        sChainCrtInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    //sChainCrtInfo.queueFamilyIndexCount = 1;
    //sChainCrtInfo.pQueueFamilyIndices= &qFamIndices;
    sChainCrtInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    sChainCrtInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sChainCrtInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sChainCrtInfo.clipped = VK_TRUE;
    sChainCrtInfo.oldSwapchain = VK_NULL_HANDLE;


    VK_CHECK(vkCreateSwapchainKHR(device, &sChainCrtInfo, 0, &swapchain));

    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;


    return swapchain;
}
VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_A8B8G8R8_UNORM_PACK32 && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}
VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != 0xffffffff) {
        return capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}
void createSwapchainImageViews()
{
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }
}
VkRenderPass createRenerPass(VkDevice device)
{
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    //dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    //dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };

    VkRenderPass renderPass;
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }

    return renderPass;
}
VkShaderModule loadShader(VkDevice device, std::string path)
{
    auto fileContent = Utility::readBinaryFileContents(path);

    //assert(fileContent.size() % 4 == 0);

    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = fileContent.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(fileContent.data());

    VK_CHECK(vkCreateShaderModule(device, &createInfo, 0, &shaderModule));

    return shaderModule;
}
void createDescriptorSetLayout()
{
    VkDescriptorSetLayoutCreateInfo createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };

    VkDescriptorSetLayoutBinding uboLayoutBinding;
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding samplerLayoutBinding;
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings = {uboLayoutBinding,samplerLayoutBinding};

    createInfo.bindingCount = 2;
    createInfo.pBindings = layoutBindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &createInfo, nullptr, &descriptorSetLayout));
}
void createDescriptorPool(VkDevice device)
{
    VkDescriptorPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };

    VkDescriptorPoolSize poolSize;
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(swapchainMinImageCount);

    VkDescriptorPoolSize poolSizeSampler;
    poolSizeSampler.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizeSampler.descriptorCount = static_cast<uint32_t>(swapchainMinImageCount);

    std::array<VkDescriptorPoolSize, 2> poolsizes = {poolSize,poolSizeSampler};

    createInfo.poolSizeCount = 2;
    createInfo.pPoolSizes = poolsizes.data();
    createInfo.maxSets = static_cast<uint32_t>(swapchainMinImageCount);

    VK_CHECK(vkCreateDescriptorPool(device, &createInfo, nullptr, &descriptorPool));
}
void createUniformBuffers()
{
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(swapchainMinImageCount);
    uniformBufferMemories.resize(swapchainMinImageCount);

    for (size_t i = 0; i < swapchainMinImageCount; i++)
    {
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers.at(i), uniformBufferMemories.at(i));
    }
}
void createDescriptorSets(VkDevice device)
{
    std::vector<VkDescriptorSetLayout> layouts(swapchainMinImageCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(3);
    allocateInfo.pSetLayouts = layouts.data();;

    descriptorSets.resize(swapchainMinImageCount);

    if (vkAllocateDescriptorSets(device, &allocateInfo, descriptorSets.data())!=VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < swapchainMinImageCount; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}
void createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlag, VkMemoryPropertyFlags props, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    createInfo.size = size;
    createInfo.usage = usageFlag;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(device, &createInfo, nullptr, &buffer));
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory));

    VK_CHECK(vkBindBufferMemory(device, buffer, bufferMemory, 0));
}
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties phyDevMemProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &phyDevMemProps);

    for (uint32_t i = 0; i < phyDevMemProps.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (phyDevMemProps.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
    return uint32_t();
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

    auto vertBindingDesc = Vertex::getBindingDescription();
    auto vertAttribDescs = Vertex::getAttributeDescriptions();

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
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
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
    pipelineInfo.layout = pipelineLayout;
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
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        /*VkImageView attachments[] = {
            swapChainImageViews[i]
        };*/

        std::array<VkImageView, 2> attachments = {
            swapChainImageViews[i],
            depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());;
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}
VkCommandPool createCommandPool(VkDevice device)
{
    QueueFamilyIndices queueFamilyIndices = pickDeviceQueueFamily(physicalDevice, surface);
    VkCommandPool commandPool = 0;
    VkCommandPoolCreateInfo cmdCrtInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cmdCrtInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmdCrtInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    VK_CHECK(vkCreateCommandPool(device, &cmdCrtInfo, 0, &commandPool));
    return commandPool;
}
void createCommandBuffers(VkDevice device, VkCommandPool commandPool)
{
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = (uint32_t)commandBuffers.size();
    VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, commandBuffers.data()));
}
void createSyncObjects()
{
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {

            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}
VkBuffer createVertexBuffer(VkDevice device)
{
    VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();


    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMem;

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMem);

    void* data;
    vkMapMemory(device, stagingBufferMem, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMem);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMem, nullptr);

    return vertexBuffer;
}
VkBuffer createIndexBuffer(VkDevice device)
{
    VkDeviceSize bufferSize = sizeof(uint16_t) * indices.size();


    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMem;

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMem);

    void* data;
    vkMapMemory(device, stagingBufferMem, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMem);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMem, nullptr);

    return indexBuffer;
}
void creatVertexAndIndexBuffers(VkDevice device,VkBuffer& vBuff, VkBuffer& iBuff)
{
    //size of buffers
    VkDeviceSize vBuffSize = sizeof(Vertex) * vertices.size();
    VkDeviceSize iBuffSize = sizeof(uint16_t) * indices.size();

    //staging buffer
    VkBuffer vStageBuff, iStageBuff;
    VkDeviceMemory vStageBuffMemory, iStageBuffMemory;

    createBuffer(vBuffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vStageBuff, vStageBuffMemory);
    void* data;
    vkMapMemory(device, vStageBuffMemory, 0, vBuffSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)vBuffSize);
    vkUnmapMemory(device, vStageBuffMemory);
    
    createBuffer(iBuffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, iStageBuff, iStageBuffMemory);
    void* iData;
    vkMapMemory(device, iStageBuffMemory, 0, iBuffSize, 0, &iData);
    memcpy(iData, indices.data(), (size_t)iBuffSize);
    vkUnmapMemory(device, iStageBuffMemory);

    //create device memory backed buffer
    createBuffer(vBuffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
    createBuffer(iBuffSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

    //transfer memory from staging to device memory backed buffer

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{}, copyRegionIndex{};
    copyRegion.size = vBuffSize;
    copyRegionIndex.size = iBuffSize;

    vkCmdCopyBuffer(commandBuffer, vStageBuff, vertexBuffer, 1, &copyRegion);
    vkCmdCopyBuffer(commandBuffer, iStageBuff, indexBuffer, 1, &copyRegionIndex);

    endSingleTimeCommands(commandBuffer);

    vkDestroyBuffer(device, vStageBuff, nullptr);
    vkFreeMemory(device, vStageBuffMemory, nullptr);

    vkDestroyBuffer(device, iStageBuff, nullptr);
    vkFreeMemory(device, iStageBuffMemory, nullptr);
}
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
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
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = swapChainExtent;

    VkClearColorValue color = { 0.1f,0.2f,0.3f,1.0 };
    //VkClearValue clearValues;
    //clearValues.color = color;
    VkClearValue clearColor = { {{0.1f, 0.2f, 0.3f, 1.0f}} };

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = { {0.1f, 0.2f, 0.3f, 1.0f} };
    clearValues[1].depthStencil = { 1.0f, 0 };


    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    /*viewport.x = 0.0f;
    viewport.y = (float)swapChainExtent.height;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = -(float)swapChainExtent.height;*/
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    VkBuffer vertexBuffers[] = { vertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, nullptr);

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
    //ubo.model = glm::mat4(1.0f);
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    ubo.view = glm::lookAt(glm::vec3(0.0f, 0.5f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

    auto mv = ubo.model;
    ubo.nrmlMat = glm::transpose(glm::inverse(mv));

    //Warn : need to remove gl_Positioin.y=-gl_Position.y from vertex shader.
    ubo.proj[1][1] *= -1;

    void* data;
    vkMapMemory(device, uniformBufferMemories[currentImage], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBufferMemories[currentImage]);
}
void createImageTexture(std::string filepath)
{
    int texWidth, texHeight, texChannels;
    auto pixels = VKUtility::getImageData(filepath, texWidth, texHeight, texChannels, 4);

    VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4;

    if (!pixels) {
        std::string error = fmt::format("Failed to laod tex image : {}",filepath);
        throw std::runtime_error(error);
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBufferMemory);

    VKUtility::freeImageData(pixels);

    createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}
VkCommandBuffer beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}
void endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}
void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    endSingleTimeCommands(commandBuffer);
}
void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = {
        width,
        height,
        1
    };

    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    endSingleTimeCommands(commandBuffer);
}
void createTextureImageView()
{
    textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
}
VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}
void createTextureSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}
void createDepthResources()
{
    VkFormat depthFormat = findDepthFormat();

    createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
    depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
}
VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}
VkFormat findDepthFormat()
{
    return findSupportedFormat(
        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
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

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);

        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        updateUniformBuffer(imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = { swapchain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // Optional

        vkQueuePresentKHR(presentQueue, &presentInfo);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    vkDeviceWaitIdle(device);

    glfwTerminate();
    destroyVulkan();
    exit(EXIT_SUCCESS);
}

