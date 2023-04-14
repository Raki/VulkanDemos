#include "VKPso.h"

VKPso::VKPso():device(VK_NULL_HANDLE), renderPass(VK_NULL_HANDLE), pipelineCache(VK_NULL_HANDLE),descPool(VK_NULL_HANDLE)
{
}

VKPso::VKPso(VkDevice dvc, VkRenderPass rPass, VkPipelineCache pCache,VkDescriptorPool desPool): device(dvc),renderPass(rPass),pipelineCache(pCache),descPool(desPool)
{
}

void VKPso::prepareShaders(const std::vector<unsigned char>& vsFileContent, const std::vector<unsigned char>& fsFileContent)
{
    auto triangleVS = VKBackend::loadShader(VKBackend::device, vsFileContent);
    assert(triangleVS);
    auto triangleFS = VKBackend::loadShader(VKBackend::device, fsFileContent);
    assert(triangleFS);

    auto setsV = VKBackend::getDescriptorSetLayoutDataFromSpv(vsFileContent);
    auto setsF = VKBackend::getDescriptorSetLayoutDataFromSpv(fsFileContent);

    std::vector<VkVertexInputAttributeDescription> vertIPAttribDesc;
    std::vector<VkVertexInputBindingDescription> vertIPBindDesc;
    VKBackend::getInputInfoFromSpv(vsFileContent, vertIPAttribDesc, vertIPBindDesc, false);

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

    for (const auto& set : setsV)
    {
        layoutBindings.insert(layoutBindings.end(), set.bindings.begin(), set.bindings.end());
    }

    for (const auto& set : setsF)
    {
        layoutBindings.insert(layoutBindings.end(), set.bindings.begin(), set.bindings.end());
    }

    auto descriptorsetLayout = VKBackend::getDescriptorSetLayout(layoutBindings);
    
}

void VKPso::addDescriptor(VKBackend::Descriptor descriptor)
{
}
