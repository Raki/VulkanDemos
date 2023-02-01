#version 460
layout(location = 0) out vec4 outColor;

layout(location =0)in vec3 normal_out;
layout(location =1)in vec3 fraPos_out;

vec3 lightPos = vec3(0,4,0);
vec3 color = vec3(0.5,0.5,1.0);

float getDiffuseFactr()
{
	vec3 lightDir = normalize(lightPos-fraPos_out); 
	float diffuse = max(dot(normal_out,lightDir),0.);

	return diffuse;
}

vec3 getDiffuseColrVertex()
{
	return color*getDiffuseFactr();
}

void main()
{
    outColor = vec4(getDiffuseColrVertex(),1.0);
}