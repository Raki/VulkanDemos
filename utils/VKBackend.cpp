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
		colorAttachment.format = VKBackend::swapChainImageFormat;
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

		VkDescriptorSetLayoutBinding samplerLayoutBinding2;
		samplerLayoutBinding2.binding = 2;
		samplerLayoutBinding2.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding2.descriptorCount = 1;
		samplerLayoutBinding2.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		samplerLayoutBinding2.pImmutableSamplers = nullptr;

		std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings = { uboLayoutBinding,samplerLayoutBinding,samplerLayoutBinding2 };

		createInfo.bindingCount = 3;
		createInfo.pBindings = layoutBindings.data();
		VK_CHECK(vkCreateDescriptorSetLayout(VKBackend::device, &createInfo, nullptr, &VKBackend::descriptorSetLayout));
	}
	void createDescriptorPool(VkDevice device)
	{
		VkDescriptorPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };

		VkDescriptorPoolSize poolSize;
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = static_cast<uint32_t>(VKBackend::swapchainMinImageCount);

		VkDescriptorPoolSize poolSizeSampler;
		poolSizeSampler.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizeSampler.descriptorCount = static_cast<uint32_t>(VKBackend::swapchainMinImageCount);

		VkDescriptorPoolSize poolSizeSampler2;
		poolSizeSampler2.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizeSampler2.descriptorCount = static_cast<uint32_t>(VKBackend::swapchainMinImageCount);

		std::array<VkDescriptorPoolSize, 3> poolsizes = { poolSize,poolSizeSampler,poolSizeSampler2 };

		createInfo.poolSizeCount = 3;
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
}