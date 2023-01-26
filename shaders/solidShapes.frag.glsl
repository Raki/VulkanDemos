#version 460
layout(location = 0) out vec4 outColor;

layout(location =0)in vec3 color_out;


void main()
{
    outColor = vec4(color_out,1.0);
}