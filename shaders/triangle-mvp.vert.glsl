#version 450
#extension GL_KHR_vulkan_glsl: enable

layout (binding=0) uniform UniformBufferObject
{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location =0) in vec3 position;
void main()
{
	gl_Position= ubo.proj*ubo.view*ubo.model*vec4(position,1.0);
}