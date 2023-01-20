// VulkanDemos.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "CommonHeaders.h"

#pragma region vars
const int WIN_WIDTH = 1024;
const int WIN_HEIGHT = 1024;
GLFWwindow* window;
auto closeWindow = false;

VkInstance vkInstance;
#pragma endregion vars

#pragma region prototypes
void createWindow();
void initVulkan();
void destroyVulkan();

VkInstance createVulkanInstance();
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

    window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, "Vulkan Renderer", NULL, NULL);
    glfwSetKeyCallback(window, key_callback);
}
void initVulkan()
{
    vkInstance = createVulkanInstance();
}
void destroyVulkan()
{
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
#pragma endregion functions

int main()
{
    createWindow();
    initVulkan();

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
    }

    glfwTerminate();
    destroyVulkan();
    exit(EXIT_SUCCESS);
}

