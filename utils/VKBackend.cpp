#include "VKBackend.h"

namespace VKBackend
{
	VkInstance vkInstance;
	VkSurfaceKHR surface;
	VkPhysicalDevice physicalDevice;

	VkPhysicalDeviceProperties physicalDevProps{};
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


	VkPipeline graphicsPipeline;
	VkPipelineLayout pipelineLayout;

	std::vector<VkCommandBuffer> commandBuffers;
	VkCommandPool commandPool;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	
	
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
}