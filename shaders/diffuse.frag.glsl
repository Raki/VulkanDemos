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
    light.position = vec3(3, 3, 0);
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

	float ambientStrength = 0.1;
    vec3 lightColor = vec3(0.8,0.8,0.8);
    vec3 objectColor = vec3(0.7,0,0);
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(nrml_out);
    vec3 lightDir = normalize(light.position - frag_pos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
            
    vec3 result = (ambient + diffuse) * objectColor;
    outColor = vec4(result, 1.0);
	
}