#version 450
#extension GL_KHR_vulkan_glsl: enable
layout(location =0) in vec3 position;
void main()
{
	gl_Position=vec4(position,1.0);
	gl_Position.y=-gl_Position.y;
}