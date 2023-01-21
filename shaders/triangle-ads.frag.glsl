#version 460
layout(location = 0) out vec4 outColor;

layout(location =0)in vec3 nrml_out;
layout(location =1)in vec3 frag_pos;

void main()
{
	vec3 lightDir=vec3(1);
	vec3 norm = normalize(nrml_out);
	float diff = max(dot(norm,lightDir),0.0);
	outColor=vec4(vec3(0.7,0,0)*diff,1.0);
}