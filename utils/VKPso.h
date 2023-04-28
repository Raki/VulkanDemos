#pragma once
#include <vulkan/vulkan.h>
#include <VKBackend.h>

class VKPso
{
public:
	VKPso();

	VKPso& addShaderModules(VkShaderModule vsModule, VkShaderModule fsModule);
	VKPso& addPipelineVertexInputState(VkPipelineVertexInputStateCreateInfo cInfo);
	VKPso& addPipelineInputAssemblyState(VkPipelineInputAssemblyStateCreateInfo cInfo);
	VKPso& addPipelineViewportState(VkPipelineViewportStateCreateInfo cInfo);
	VKPso& addPipelineRasterState(VkPipelineRasterizationStateCreateInfo cInfo);
	VKPso& addPipelineMultisampleState(VkPipelineMultisampleStateCreateInfo cInfo);
	VKPso& addPipelineColorBlendState(VkPipelineColorBlendStateCreateInfo cInfo);
	VKPso& addPipelineDynamicState(VkPipelineDynamicStateCreateInfo cInfo);
	VKPso& addPipelineDepthStencilState(VkPipelineDepthStencilStateCreateInfo cInfo);
	VKPso& addPipelineLayout(const VkPipelineLayout pipelineLayout);
	VKPso& addRenderpass(const VkRenderPass renderPass);
	VKPso& addSubpass(const uint32_t subPass);
	VKPso& addBasePipelineHandle(const VkPipeline pipeline);
	VkPipeline build(const VkDevice device, const VkPipelineCache pipelineCache);
private:
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	std::vector<VkPipelineShaderStageCreateInfo> stages;
};

