#include "VKPso.h"

VKPso::VKPso() 
{
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
}

VKPso& VKPso::addShaderModules(const VkShaderModule vsModule,const VkShaderModule fsModule)
{
	// TODO: What if the pipeline has more stages ?
	//VkPipelineShaderStageCreateInfo stages[2] = {};
	stages.clear();
	stages.push_back(VKBackend::getPipelineShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vsModule));
	stages.push_back(VKBackend::getPipelineShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fsModule));

	pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
	pipelineInfo.pStages = stages.data();

	return *this;
}

VKPso& VKPso::addPipelineVertexInputState(const VkPipelineVertexInputStateCreateInfo cInfo)
{
	pipelineInfo.pVertexInputState = &cInfo;
	return *this;
}

VKPso& VKPso::addPipelineInputAssemblyState(const VkPipelineInputAssemblyStateCreateInfo cInfo)
{
	pipelineInfo.pInputAssemblyState = &cInfo;
	return *this;
}

VKPso& VKPso::addPipelineViewportState(const VkPipelineViewportStateCreateInfo cInfo)
{
	pipelineInfo.pViewportState = &cInfo;
	return *this;
}

VKPso& VKPso::addPipelineRasterState(const VkPipelineRasterizationStateCreateInfo cInfo)
{
	pipelineInfo.pRasterizationState = &cInfo;
	return *this;
}

VKPso& VKPso::addPipelineMultisampleState(const VkPipelineMultisampleStateCreateInfo cInfo)
{
	pipelineInfo.pMultisampleState = &cInfo;
	return *this;
}

VKPso& VKPso::addPipelineColorBlendState(const VkPipelineColorBlendStateCreateInfo cInfo)
{
	pipelineInfo.pColorBlendState = &cInfo;
	return *this;
}

VKPso& VKPso::addPipelineDynamicState(const VkPipelineDynamicStateCreateInfo cInfo)
{
	pipelineInfo.pDynamicState = &cInfo;
	return *this;
}

VKPso& VKPso::addPipelineDepthStencilState(const VkPipelineDepthStencilStateCreateInfo cInfo)
{
	pipelineInfo.pDepthStencilState = &cInfo;
	return *this;
}

VKPso& VKPso::addPipelineLayout(const VkPipelineLayout pipelineLayout)
{
	pipelineInfo.layout = pipelineLayout;
	return *this;
}

VKPso& VKPso::addRenderpass(const VkRenderPass renderPass)
{
	pipelineInfo.renderPass = renderPass;
	return *this;
}

VKPso& VKPso::addSubpass(const uint32_t subPass)
{
	pipelineInfo.subpass = subPass;
	return *this;
}

VKPso& VKPso::addBasePipelineHandle(const VkPipeline pipeline)
{
	pipelineInfo.basePipelineHandle = pipeline;
	return *this;
}

VkPipeline VKPso::build(const VkDevice device, const VkPipelineCache pipelineCache) const
{
	assert(device!=VK_NULL_HANDLE);

	VkPipeline pipeline;
	if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline!");
	}
	return pipeline;
}
