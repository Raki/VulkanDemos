#ifndef VK_UTILITY_H
#define VK_UTILITY_H

#include "CommonHeaders.h"


namespace VKUtility
{
    struct Vertex
    {
        Vertex(glm::vec3 pos, glm::vec3 norm, glm::vec2 texcoords);
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions();
    };

	struct Mesh
	{
        std::vector<Vertex> vData;
        std::vector<uint16_t> iData;
        std::string name;
        Mesh(std::vector<Vertex> vData, std::vector<uint16_t> iData);
	};

    std::shared_ptr<Mesh> getCube(float width, float height, float depth);
	unsigned char* getImageData(std::string fileanme, int& width, int& height, int& nChannels,int reqChannels= 0);
	void freeImageData(unsigned char* data);
    glm::vec3 getNormal(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);
}


#endif // !VK_UTILITY_H