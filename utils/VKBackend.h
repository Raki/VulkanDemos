#pragma once
#ifndef VK_BACKEND_H
#define VK_BACKEND_H

#include "CommonHeaders.h"

namespace VKBackend
{
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


	VkInstance createVulkanInstance();
}

#endif // !VK_BACKEND_H
