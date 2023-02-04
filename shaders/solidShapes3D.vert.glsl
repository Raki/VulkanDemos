#version 450

layout (binding=0) uniform UniformBufferObject
{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 nrmlMat;
} ubo;

layout (push_constant) uniform constants
{
	mat4 tMat;
}PushConstant;

layout(location =0) in vec3 position;
layout(location =1) in vec3 normal;

layout(location =0)out vec3 normal_out;
layout(location =1)out vec3 fraPos_out;

void main()
{
	normal_out = normalize(mat3(ubo.nrmlMat)*normal);
	fraPos_out = vec3(ubo.model*vec4(position,1.0));
	gl_Position= ubo.proj*ubo.view*ubo.model*vec4(position,1.0);
}