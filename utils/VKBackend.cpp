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

	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

	int winWidth;
	int winHeight;
	int MAX_FRAMES_IN_FLIGHT = 2;

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
	VkPhysicalDevice pickPhysicalDevice(VkInstance instance)
	{
		uint32_t phyicalDeviceCount = 0;
		VK_CHECK(vkEnumeratePhysicalDevices(instance, &phyicalDeviceCount, nullptr));
		std::vector<VkPhysicalDevice> physicalDevices(phyicalDeviceCount);
		VK_CHECK(vkEnumeratePhysicalDevices(instance, &phyicalDeviceCount, physicalDevices.data()));

		for (size_t i = 0; i < phyicalDeviceCount; i++)
		{
			vkGetPhysicalDeviceProperties(physicalDevices[i], &VKBackend::physicalDevProps);
			if (VKBackend::physicalDevProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			{
				fmt::print("Selecting discrete GPU {}\n", VKBackend::physicalDevProps.deviceName);
				msaaSamples = getMaxUsableSampleCount();
				return physicalDevices[i];
			}
		}

		if (phyicalDeviceCount > 0)
		{
			vkGetPhysicalDeviceProperties(physicalDevices[0], &VKBackend::physicalDevProps);
			fmt::print("Selecting device {}\n", VKBackend::physicalDevProps.deviceName);
			return physicalDevices[0];
		}

		return 0;
	}

	QueueFamilyIndices pickDeviceQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkQueueFlagBits qFamily)
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
			if (props.queueFlags & qFamily)
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
		auto queFamilyIndices = pickDeviceQueueFamily(physicalDevice, VKBackend::surface);

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

		vkGetDeviceQueue(device, queFamilyIndices.graphicsFamily.value(), 0, &VKBackend::graphicsQueue);
		vkGetDeviceQueue(device, queFamilyIndices.presentFamily.value(), 0, &VKBackend::presentQueue);

		return device;
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, VKBackend::surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, VKBackend::surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, VKBackend::surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, VKBackend::surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, VKBackend::surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	VkSwapchainKHR createSwapchain(VkDevice device, VkSurfaceKHR surface)
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(VKBackend::physicalDevice);
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}
		VKBackend::swapchainMinImageCount = imageCount;
		int wWidth = 0, wHeight = 0;
		//glfwGetWindowSize(window, &wWidth, &wHeight);
		wWidth = winWidth, wHeight = winHeight;
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

		QueueFamilyIndices indices = pickDeviceQueueFamily(VKBackend::physicalDevice, surface);
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
		VKBackend::swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapchain, &imageCount, VKBackend::swapChainImages.data());

		VKBackend::swapChainImageFormat = surfaceFormat.format;
		VKBackend::swapChainExtent = extent;


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
			//glfwGetFramebufferSize(window, &width, &height);
			width = winWidth; height = winHeight;

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
		VKBackend::swapChainImageViews.resize(VKBackend::swapChainImages.size());

		for (size_t i = 0; i < VKBackend::swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = VKBackend::swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = VKBackend::swapChainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			if (vkCreateImageView(VKBackend::device, &createInfo, nullptr, &VKBackend::swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}
		}
	}
	VkRenderPass createRenerPass(VkDevice device)
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = VKBackend::swapChainImageFormat;
		colorAttachment.samples = msaaSamples;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = msaaSamples;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		
		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 2;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
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
		subpass.pResolveAttachments = &colorAttachmentResolveRef;

		std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };

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
	VkRenderPass createRenderPass1Sample(VkDevice device)
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

	void createDescriptorSetLayout(std::vector <VkDescriptorSetLayoutBinding> layoutBindings)
	{
		VkDescriptorSetLayoutCreateInfo createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		createInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
		createInfo.pBindings = layoutBindings.data();
		VK_CHECK(vkCreateDescriptorSetLayout(VKBackend::device, &createInfo, nullptr, &VKBackend::descriptorSetLayout));
	}

	void createDescriptorPool(VkDevice device, std::vector<VkDescriptorPoolSize> poolsizes)
	{
		VkDescriptorPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };

		createInfo.poolSizeCount = static_cast<uint32_t>(poolsizes.size());;
		createInfo.pPoolSizes = poolsizes.data();
		createInfo.maxSets = static_cast<uint32_t>(VKBackend::swapchainMinImageCount);

		VK_CHECK(vkCreateDescriptorPool(device, &createInfo, nullptr, &VKBackend::descriptorPool));
	}

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
	{
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(VKBackend::physicalDevice, format, &props);

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
	VkCommandPool createCommandPool(VkDevice device)
	{
		VKBackend::QueueFamilyIndices queueFamilyIndices = VKBackend::pickDeviceQueueFamily(VKBackend::physicalDevice, VKBackend::surface);
		VkCommandPool commandPool = 0;
		VkCommandPoolCreateInfo cmdCrtInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
		cmdCrtInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cmdCrtInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
		VK_CHECK(vkCreateCommandPool(device, &cmdCrtInfo, 0, &commandPool));
		return commandPool;
	}
	void createCommandBuffers(VkDevice device, VkCommandPool commandPool)
	{
		VKBackend::commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
		allocateInfo.commandPool = commandPool;
		allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocateInfo.commandBufferCount = (uint32_t)VKBackend::commandBuffers.size();
		VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, VKBackend::commandBuffers.data()));
	}
	void createSyncObjects()
	{
		VKBackend::imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		VKBackend::renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		VKBackend::inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(VKBackend::device, &semaphoreInfo, nullptr, &VKBackend::imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(VKBackend::device, &semaphoreInfo, nullptr, &VKBackend::renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(VKBackend::device, &fenceInfo, nullptr, &VKBackend::inFlightFences[i]) != VK_SUCCESS) {

				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlag, VkMemoryPropertyFlags props, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
	{
		VkBufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		createInfo.size = size;
		createInfo.usage = usageFlag;
		createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VK_CHECK(vkCreateBuffer(VKBackend::device, &createInfo, nullptr, &buffer));
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(VKBackend::device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, props);
		VK_CHECK(vkAllocateMemory(VKBackend::device, &allocInfo, nullptr, &bufferMemory));

		VK_CHECK(vkBindBufferMemory(VKBackend::device, buffer, bufferMemory, 0));
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties phyDevMemProps;
		vkGetPhysicalDeviceMemoryProperties(VKBackend::physicalDevice, &phyDevMemProps);

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

	std::shared_ptr<VKTexture> createVKTexture(std::string filename)
	{
		auto texture = std::make_shared<VKTexture>();
		int texWidth, texHeight, texChannels;
		auto pixels = VKUtility::getImageData(filename, texWidth, texHeight, texChannels, 4);

		VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) * texHeight * 4;

		if (!pixels) {
			std::string error = fmt::format("Failed to laod tex image : {}", filename);
			throw std::runtime_error(error);
		}

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		VKBackend::createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
		void* data;
		vkMapMemory(VKBackend::device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(VKBackend::device, stagingBufferMemory);

		VKUtility::freeImageData(pixels);

		createImage(texWidth, texHeight, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture->textureImage, texture->textureImageMemory);

		transitionImageLayout(texture->textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, texture->textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		transitionImageLayout(texture->textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(VKBackend::device, stagingBuffer, nullptr);
		vkFreeMemory(VKBackend::device, stagingBufferMemory, nullptr);

		texture->textureImageView = createImageView(texture->textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);

		createTextureSampler(texture->textureSampler);

		return texture;
	}

	void createImage(uint32_t width, uint32_t height, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
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
		imageInfo.samples = numSamples;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(VKBackend::device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(VKBackend::device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = VKBackend::findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(VKBackend::device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(VKBackend::device, image, imageMemory, 0);
	}
	VkCommandBuffer beginSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = VKBackend::commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(VKBackend::device, &allocInfo, &commandBuffer);

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

		vkQueueSubmit(VKBackend::graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(VKBackend::graphicsQueue);

		vkFreeCommandBuffers(VKBackend::device, VKBackend::commandPool, 1, &commandBuffer);
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
		if (vkCreateImageView(VKBackend::device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}
	void createTextureSampler(VkSampler& textureSampler)
	{
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;

		samplerInfo.maxAnisotropy = VKBackend::physicalDevProps.limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(VKBackend::device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}
	VkSampleCountFlagBits getMaxUsableSampleCount()
	{
		VkSampleCountFlags counts = physicalDevProps.limits.framebufferColorSampleCounts & physicalDevProps.limits.framebufferDepthSampleCounts;
		if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
		if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
		if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
		if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
		if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
		if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

		return VK_SAMPLE_COUNT_1_BIT;
	}

	VkPipelineShaderStageCreateInfo getPipelineShaderStage(VkShaderStageFlagBits shaderStage, VkShaderModule shaderModule)
	{
		VkPipelineShaderStageCreateInfo stage = {};
		stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage = shaderStage;
		stage.module = shaderModule;
		stage.pName = "main";
		return stage;
	}

	VkPipelineVertexInputStateCreateInfo getPipelineVertexInputState(uint32_t vertexBindingDescriptionCount, VkVertexInputBindingDescription* pVertexBindingDescriptions,
		uint32_t vertexAttributeDescriptionCount, VkVertexInputAttributeDescription* pVertexAttributeDescriptions)
	{
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = vertexBindingDescriptionCount;
		vertexInputInfo.pVertexBindingDescriptions = pVertexBindingDescriptions;
		vertexInputInfo.vertexAttributeDescriptionCount = vertexAttributeDescriptionCount;
		vertexInputInfo.pVertexAttributeDescriptions = pVertexAttributeDescriptions;
		return vertexInputInfo;
	}
}