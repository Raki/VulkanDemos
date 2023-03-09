#version 460
layout(location = 0) out vec4 outColor;

layout(location =0)in vec3 normal_out;
layout(location =1)in vec3 fraPos_out;
layout(location =2)in vec2 uv_out;

layout (binding=1) uniform LightInfo
{
	vec4 position;
	vec4 color;
}lInfo;


layout(binding = 2) uniform sampler2D texSampler;

float getDiffuseFactr()
{
	vec3 lightDir = normalize(lInfo.position.xyz-fraPos_out); 
	float diffuse = max(dot(normal_out,lightDir),0.);

	return diffuse;
}

vec3 getDiffuseColrVertex()
{
	vec4 texel = texture(texSampler,uv_out);
	return texel.xyz*getDiffuseFactr();
}

vec3 getAmbientColor()
{
	return vec3(0.2);
}

void main()
{
    outColor = vec4(getDiffuseColrVertex()+getAmbientColor(),1.0);
}