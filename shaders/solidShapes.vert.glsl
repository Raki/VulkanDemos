#version 450

layout (binding=0) uniform UniformBufferObject
{
	mat4 proj;
} ubo;

layout (push_constant) uniform constants
{
	mat4 tMat;
}PushConstant;

layout(location =0) in vec3 position;
layout(location =1) in vec3 color;

layout(location =0)out vec3 color_out;

void main()
{
	color_out = color;
	gl_Position= ubo.proj*PushConstant.tMat*vec4(position,1.0);
}