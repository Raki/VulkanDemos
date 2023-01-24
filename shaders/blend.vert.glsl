#version 450

layout (binding=0) uniform UniformBufferObject
{
	mat4 proj;
} ubo;

layout(location =0) in vec3 position;
layout(location =1) in vec3 normal;
layout(location =2) in vec2 uv;

layout(location =0)out vec2 uv_out;

void main()
{
	uv_out = uv;
	gl_Position= ubo.proj*vec4(position,1.0);
}