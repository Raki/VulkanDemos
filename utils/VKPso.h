#pragma once
#include <vulkan/vulkan.h>
#include <VKBackend.h>

class VKPso
{
	//All components required to create pipeline object
	//descriptors
public:
	VKPso();
	VKPso(VkDevice dece,VkRenderPass rPass,VkPipelineCache pCache,VkDescriptorPool desPool);
	void prepareShaders(const std::vector<unsigned char> &vsFileContent, const std::vector<unsigned char>& fsFileContent);
	void addDescriptor(VKBackend::Descriptor descriptor);
private:
	VkDevice device;
	VkRenderPass renderPass;
	VkPipelineCache pipelineCache;
	VkDescriptorPool descPool;
	std::vector<VkDescriptorPoolSize> poolsizes;
};

