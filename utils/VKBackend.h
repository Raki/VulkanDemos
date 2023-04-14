#pragma once
#ifndef VK_BACKEND_H
#define VK_BACKEND_H

#include "CommonHeaders.h"
#include "Utility.h"
#include "VKUtility.h"

namespace VKBackend
{
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

	struct VKTexture
	{
		VkImage textureImage;
		VkDeviceMemory textureImageMemory;
		VkImageView textureImageView;
		VkSampler textureSampler;
	};

	struct VKRenderTarget
	{
		VkImage colorImage;
		VkDeviceMemory colorImageMemory;
		VkImageView colorImageView;
	};

	struct DescriptorSetLayoutData {
		uint32_t setNumber;
		VkDescriptorSetLayoutCreateInfo createInfo;
		std::vector<VkDescriptorSetLayoutBinding> bindings;
	};


	struct Buffer
	{
		std::vector<VkBuffer> uniformBuffers;
		std::vector<VkDeviceMemory> uniformBufferMemories;
		VkDeviceSize range = 0;
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

	extern VkInstance vkInstance;
	extern VkSurfaceKHR surface;
	extern VkPhysicalDevice physicalDevice;
	extern VkPhysicalDeviceProperties physicalDevProps;
	extern VkDevice device;
	extern VkQueue graphicsQueue;
	extern VkQueue presentQueue;
	extern VkSwapchainKHR swapchain;
	extern std::vector<VkImage> swapChainImages;
	extern std::vector<VkImageView> swapChainImageViews;
	extern std::vector<VkFramebuffer> swapChainFramebuffers;
	extern size_t swapchainMinImageCount;
	extern VkFormat swapChainImageFormat;
	extern VkExtent2D swapChainExtent;

	extern VkRenderPass renderPass;

	extern VkDescriptorSetLayout descriptorSetLayout;
	extern VkDescriptorPool descriptorPool;
	extern std::vector<VkDescriptorSet> descriptorSets;


	extern VkPipeline graphicsPipeline;
	extern VkPipelineLayout pipelineLayout;

	extern std::vector<VkCommandBuffer> commandBuffers;
	extern VkCommandPool commandPool;

	extern std::vector<VkSemaphore> imageAvailableSemaphores;
	extern std::vector<VkSemaphore> renderFinishedSemaphores;
	extern std::vector<VkFence> inFlightFences;

	extern VkSampleCountFlagBits msaaSamples;

	extern int winWidth;
	extern int winHeight;
	extern int MAX_FRAMES_IN_FLIGHT;

	VkInstance createVulkanInstance();
	VkPhysicalDevice pickPhysicalDevice(VkInstance instance);
	QueueFamilyIndices pickDeviceQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,VkQueueFlagBits qFamily=VK_QUEUE_GRAPHICS_BIT);
	VkDevice createDevice(VkPhysicalDevice physicalDevice);
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
	VkSwapchainKHR createSwapchain(VkDevice device, VkSurfaceKHR surface);
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
	void createSwapchainImageViews();
	VkRenderPass createRenerPass(VkDevice device);
	VkRenderPass createRenderPass1Sample(VkDevice device);
	VkShaderModule loadShader(VkDevice device, std::string path);
	VkShaderModule loadShader(VkDevice device, const std::vector<unsigned char> &fileContent);

	std::vector<DescriptorSetLayoutData> getDescriptorSetLayoutDataFromSpv(const std::string path);
	std::vector<DescriptorSetLayoutData> getDescriptorSetLayoutDataFromSpv(const std::vector<unsigned char> &fileContent);
	uint32_t formatSize(VkFormat format);
	void getInputInfoFromSpv(const std::vector<unsigned char>& fileContent, std::vector<VkVertexInputAttributeDescription> &vertIPAttribDesc,
		std::vector<VkVertexInputBindingDescription> &vertIPBindDesc,bool interleaved=true);
	void createDescriptorSetLayout(std::vector <VkDescriptorSetLayoutBinding> layoutBindings);
	VkDescriptorSetLayout getDescriptorSetLayout(std::vector <VkDescriptorSetLayoutBinding> layoutBindings);
	void createDescriptorPool(VkDevice device, std::vector<VkDescriptorPoolSize> poolsizes);

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	VkFormat findDepthFormat();
	VkCommandPool createCommandPool(VkDevice device);
	void createCommandBuffers(VkDevice device, VkCommandPool commandPool);
	void createSyncObjects();
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlag, VkMemoryPropertyFlags props, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	std::shared_ptr<VKTexture> createVKTexture(std::string filename);
	void createImage(uint32_t width, uint32_t height, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
	VkCommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(VkCommandBuffer commandBuffer);
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
	void createTextureSampler(VkSampler& textureSampler);
	VkSampleCountFlagBits getMaxUsableSampleCount();


	VkPipelineLayout createPipelineLayout(std::vector<VkDescriptorSetLayout> &setLayouts,std::vector<VkPushConstantRange> &pushConstants);
	VkPipelineShaderStageCreateInfo getPipelineShaderStage(VkShaderStageFlagBits shaderStage,VkShaderModule shaderModule);
	VkPipelineVertexInputStateCreateInfo getPipelineVertexInputState(uint32_t vertexBindingDescriptionCount, VkVertexInputBindingDescription* pVertexBindingDescriptions,
	uint32_t vertexAttributeDescriptionCount, VkVertexInputAttributeDescription* pVertexAttributeDescriptions);
	VkPipelineInputAssemblyStateCreateInfo getPipelineInputAssemblyState(VkPrimitiveTopology topology,VkBool32 primitiveRestartEnable);
	VkPipelineViewportStateCreateInfo getPipelineViewportState(uint32_t viewportCount,uint32_t scissorCount);
	VkPipelineRasterizationStateCreateInfo getPipelineRasterState(VkPolygonMode polygonMode,float lineWidth);
	VkPipelineMultisampleStateCreateInfo getPipelineMultisampleState(VkBool32 sampleShadingEnable, VkSampleCountFlagBits rasterizationSamples);
	VkPipelineDepthStencilStateCreateInfo getPipelineDepthStencilState(VkBool32 depthTestEnable, VkBool32 depthWriteEnable,VkCompareOp depthCompareOp,VkBool32 depthBoundsTestEnable, float minDepthBounds,float maxDepthBounds,
		VkBool32 stencilTestEnable);
	VkPipelineColorBlendAttachmentState getPipelineColorBlendAttachState(VkColorComponentFlags colorWriteMask, VkBool32 blendEnable);
	VkPipelineColorBlendStateCreateInfo getPipelineColorBlendState(VkBool32 logicOpEnable, VkLogicOp logicOp,uint32_t attachmentCount,
		VkPipelineColorBlendAttachmentState* pAttachments,const float blendConsts[4]);
	VkPipelineDynamicStateCreateInfo getPipelineDynamicState(std::vector<VkDynamicState> &dynamicStates);

	void compileShader(const std::filesystem::path sPath);
	void compileShaders(const std::filesystem::path vsPath,const std::filesystem::path fsPath);

	bool supportForDescriptorIndexing(VkPhysicalDevice phyDevice);

}


#endif // !VK_BACKEND_H
