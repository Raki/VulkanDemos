#pragma once
#include <vulkan/vulkan.h>
#include <VKBackend.h>

class VKPso
{
public:
	VKPso();

	VKPso& addShaderModules(const VkShaderModule vsModule,const VkShaderModule fsModule);
	VKPso& addPipelineVertexInputState(const VkPipelineVertexInputStateCreateInfo cInfo);
	VKPso& addPipelineInputAssemblyState(const VkPipelineInputAssemblyStateCreateInfo cInfo);
	VKPso& addPipelineViewportState(const VkPipelineViewportStateCreateInfo cInfo);
	VKPso& addPipelineRasterState(const VkPipelineRasterizationStateCreateInfo cInfo);
	VKPso& addPipelineMultisampleState(const VkPipelineMultisampleStateCreateInfo cInfo);
	VKPso& addPipelineColorBlendState(const VkPipelineColorBlendStateCreateInfo cInfo);
	VKPso& addPipelineDynamicState(const VkPipelineDynamicStateCreateInfo cInfo);
	VKPso& addPipelineDepthStencilState(const VkPipelineDepthStencilStateCreateInfo cInfo);
	VKPso& addPipelineLayout(const VkPipelineLayout pipelineLayout);
	VKPso& addRenderpass(const VkRenderPass renderPass);
	VKPso& addSubpass(const uint32_t subPass);
	VKPso& addBasePipelineHandle(const VkPipeline pipeline);
	VkPipeline build(const VkDevice device, const VkPipelineCache pipelineCache) const;
private:
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	std::vector<VkPipelineShaderStageCreateInfo> stages;
};

