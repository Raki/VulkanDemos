#ifndef VK_UTILITY_H
#define VK_UTILITY_H

#include "CommonHeaders.h"
#include "bezier.h"
#include <poly2tri/poly2tri.h>

namespace VKUtility
{
    struct Vertex
    {
        Vertex(glm::vec3 pos, glm::vec3 norm, glm::vec2 texcoords);
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
        static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions();
        static VkVertexInputBindingDescription getBindingDescription();
    };

    struct VDPosNorm
    {
        VDPosNorm(glm::vec3 pos, glm::vec3 norm);
        glm::vec3 position;
        glm::vec3 normal;
        static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
        static VkVertexInputBindingDescription getBindingDescription();
    };

	struct Mesh
	{
        std::vector<Vertex> vData;
        std::vector<VDPosNorm> vDataOL;
        std::vector<uint16_t> iData;
        std::string name;
        Mesh(std::vector<Vertex> vData, std::vector<uint16_t> iData);
        Mesh(std::vector<VDPosNorm> vData, std::vector<uint16_t> iData);
	};

    std::shared_ptr<Mesh> getCube(float width, float height, float depth);
    std::shared_ptr<Mesh> getCube(const glm::vec3 min,const glm::vec3 max);
    std::shared_ptr<Mesh> getCubeOutline(glm::vec3 min,glm::vec3 max);
    std::shared_ptr<Mesh> getFSQuad();
	unsigned char* getImageData(std::string fileanme, int& width, int& height, int& nChannels,int reqChannels= 0);
	void freeImageData(unsigned char* data);
    glm::vec3 getNormal(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);
}


#endif // !VK_UTILITY_H