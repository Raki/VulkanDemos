#version 450
#extension GL_KHR_vulkan_glsl: enable

layout (binding=0) uniform UniformBufferObject
{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 nrmlMat;
} ubo;

layout(location =0) in vec3 position;
layout(location =1) in vec3 normal;

layout(location =0)out vec3 nrml_out;
layout(location =1)out vec3 frag_pos;

void main()
{
	nrml_out = mat3(ubo.nrmlMat)*normal;
	frag_pos = vec3(ubo.model*vec4(position,1.0));
	gl_Position= ubo.proj*ubo.view*ubo.model*vec4(position,1.0);
}