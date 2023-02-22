
#include "CommonHeaders.h"
#include "Utility.h"
#include "VKUtility.h"
#include "VKBackend.h"
#include "spirv_reflect.h"


/*
* Goal of this program is to explore SPIRV-Reflect
* https://github.com/KhronosGroup/SPIRV-Reflect
*/
int main()
{
    auto data = Utility::readBinaryFileContents("shaders/blend.vert.spv");

    if (data.size() == 0)
        return EXIT_FAILURE;

    auto s = sizeof(unsigned char);

    // Generate reflection data for a shader
    SpvReflectShaderModule module;
    SpvReflectResult result = spvReflectCreateShaderModule(data.size()*sizeof(unsigned char), data.data(), &module);
    assert(result == SPV_REFLECT_RESULT_SUCCESS);

    // Enumerate and extract shader's input variables
    uint32_t var_count = 0;
    result = spvReflectEnumerateInputVariables(&module, &var_count, NULL);
    assert(result == SPV_REFLECT_RESULT_SUCCESS);
    SpvReflectInterfaceVariable** input_vars =
        (SpvReflectInterfaceVariable**)malloc(var_count * sizeof(SpvReflectInterfaceVariable*));
    result = spvReflectEnumerateInputVariables(&module, &var_count, input_vars);
    assert(result == SPV_REFLECT_RESULT_SUCCESS);


    for (size_t i=0;i<var_count;i++)
    {
        auto ip = input_vars[i];
        fmt::print("{}\n",ip->name);
    }

    // Destroy the reflection data when no longer required.
    spvReflectDestroyShaderModule(&module);

    return EXIT_SUCCESS;
}
