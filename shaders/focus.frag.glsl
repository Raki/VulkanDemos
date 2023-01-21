#version 460
layout(location = 0) out vec4 outColor;

layout(location =0)in vec3 nrml_out;
layout(location =1)in vec3 frag_pos;

void main()
{
	vec3 viewPos = vec3(0,0.5,2.0);

	struct Material
	{
		float shininess;
		vec3 specular;
	}material;

    struct LightInfo
    {
        vec3 position;
        vec3 ambient;
        vec3 diffuse;
        vec3 specular;
        vec3 direction;
        float linear;
        float constant;
        float quadratic;
        float cutOff;
        float outerCutOff;
    }light;

	light.direction = vec3(0, -1, 0);
    light.position = vec3(0, 3, 0);
    light.ambient = vec3(0.2);
    light.diffuse = vec3(0.5);
    light.specular = vec3(1.0, 1.0, 1.0);
    light.constant = 1.0;
    light.linear = 0.045;
    light.quadratic = 0.0075;
    light.cutOff = 1.0472;
    light.outerCutOff = 1.22173;

	material.shininess = 64.0;
	material.specular = vec3(0.5);

	vec3 result;
	vec3 lightDir = normalize(light.position - frag_pos);  
	float theta = dot(lightDir,-light.direction);
	float epsilon   = light.cutOff - light.outerCutOff;
	float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);   


	//attenuation
	float dist = length(light.position-frag_pos); 
	float attenuation = 1.0/(light.constant+ light.linear*dist + light.quadratic*(dist*dist));

	//ambient
	vec3 ambient = light.ambient*vec3(0.7,0,0);//vec3(texture(material.diffuseSampler,uv_out));
	ambient*=attenuation;

	//diffuse
	vec3 norm = normalize(nrml_out);
	float diff = max(dot(norm,lightDir),0.0);
	vec3 diffuse = light.diffuse*(diff*vec3(0.7,0,0));
	diffuse*=attenuation;
	diffuse*=intensity;


	//specular
	vec3 viewDir = normalize(viewPos-frag_pos);
	vec3 reflectDir = reflect(-lightDir,norm);
	float spec = pow(max(dot(viewDir,reflectDir),0),material.shininess);
	vec3 specular = light.specular*(spec*material.specular);
	specular*=attenuation;
	specular*=intensity;

	result = ambient+diffuse+specular;

	outColor = vec4(result,1.0);
	
}