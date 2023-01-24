#version 460
layout(location = 0) out vec4 outColor;

layout(location =0)in vec2 uv_out;

layout(binding = 1) uniform sampler2D texSampler1;
layout(binding = 2) uniform sampler2D texSampler2;

void main()
{
    outColor = mix(texture(texSampler1, uv_out),texture(texSampler2, uv_out),0.5);
}