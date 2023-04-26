#include "VKUtility.h"

//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace VKUtility
{
	std::shared_ptr<Mesh> getCube(float width, float height, float depth)
	{
		auto hasTex = true;
		glm::vec3 bbMin, bbMax;
		bbMax.x = width / 2;
		bbMax.y = height / 2;
		bbMax.z = depth / 2;
		bbMin.x = -width / 2;
		bbMin.y = -height / 2;
		bbMin.z = -depth / 2;


		std::vector<glm::vec3> vArr, nArr;
		std::vector<glm::vec2> uvArr;

		//top
		glm::vec3 t1 = glm::vec3(bbMin.x, bbMax.y, bbMin.z);
		glm::vec3 t2 = glm::vec3(bbMax.x, bbMax.y, bbMin.z);
		glm::vec3 t3 = glm::vec3(bbMax.x, bbMax.y, bbMax.z);
		glm::vec3 t4 = glm::vec3(bbMin.x, bbMax.y, bbMax.z);

		//bottom
		glm::vec3 b1 = glm::vec3(bbMin.x, bbMin.y, bbMin.z);
		glm::vec3 b2 = glm::vec3(bbMax.x, bbMin.y, bbMin.z);
		glm::vec3 b3 = glm::vec3(bbMax.x, bbMin.y, bbMax.z);
		glm::vec3 b4 = glm::vec3(bbMin.x, bbMin.y, bbMax.z);

		// front			back
		//		t4--t3			t2--t1
		//		|    |			|	|
		//		b4--b3			b2--b1
		// left			right
		//		t1--t4			t3--t2
		//		|    |			|	|
		//		b1--b4			b3--b2
		// top			bottom
		//		t1--t2			b4--b3
		//		|    |			|	|
		//		t4--t3			b1--b2
		//front
		vArr.push_back(b4);		vArr.push_back(b3);		vArr.push_back(t3);
		vArr.push_back(b4);		vArr.push_back(t3);		vArr.push_back(t4);
		{
			auto n = getNormal(b4, b3, t3);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			if (hasTex)
			{
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 1));
			}
			else
			{
				for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
			}
		}


		//back
		vArr.push_back(b2);		vArr.push_back(b1);		vArr.push_back(t1);
		vArr.push_back(b2);		vArr.push_back(t1);		vArr.push_back(t2);
		{
			auto n = getNormal(b2, b1, t1);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			if (hasTex)
			{
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 1));
			}
			else
			{
				for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
			}
		}

		//left
		vArr.push_back(b1);		vArr.push_back(b4);		vArr.push_back(t4);
		vArr.push_back(b1);		vArr.push_back(t4);		vArr.push_back(t1);
		{
			auto n = getNormal(b1, b4, t4);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			if (hasTex)
			{
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 1));
			}
			else
			{
				for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
			}
		}

		//right
		vArr.push_back(b3);		vArr.push_back(b2);		vArr.push_back(t2);
		vArr.push_back(b3);		vArr.push_back(t2);		vArr.push_back(t3);
		{
			auto n = getNormal(b3, b2, t2);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			if (hasTex)
			{
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 1));
			}
			else
			{
				for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
			}
		}

		//top
		vArr.push_back(t4);		vArr.push_back(t3);		vArr.push_back(t2);
		vArr.push_back(t4);		vArr.push_back(t2);		vArr.push_back(t1);
		{
			auto n = getNormal(t4, t3, t2);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			if (hasTex)
			{
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 1));
			}
			else
			{
				for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
			}
		}

		//bottom
		vArr.push_back(b1);		vArr.push_back(b2);		vArr.push_back(b3);
		vArr.push_back(b1);		vArr.push_back(b3);		vArr.push_back(b4);
		{
			auto n = getNormal(b1, b2, b3);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			if (hasTex)
			{
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 0));
				uvArr.push_back(glm::vec2(1, 1));
				uvArr.push_back(glm::vec2(0, 1));
			}
			else
			{
				for (auto i = 0; i < 6; i++)uvArr.push_back(glm::vec2(-1, -1));
			}
		}

		std::vector<Vertex> interleavedArr;
		std::vector<uint16_t> iArr;

		auto totalVerts = vArr.size();
		for (auto i = 0; i < totalVerts; i++)
		{
			auto v = vArr.at(i);
			auto n = nArr.at(i);
			auto uv = uvArr.at(i);
			interleavedArr.push_back({ v,n,uv });
			iArr.push_back((uint16_t)iArr.size());
		}

		auto cubeMesh = std::make_shared<Mesh>(interleavedArr, iArr);
		cubeMesh->name = "Cube Mesh";

		return cubeMesh;
	}
	std::shared_ptr<Mesh> getCube(const glm::vec3 min, const glm::vec3 max)
	{
		const auto extent = max - min;
		return getCube(extent.x, extent.y, extent.z);
	}
	std::shared_ptr<Mesh> getCubeOutline(glm::vec3 min, glm::vec3 max)
	{
		auto extent = max - min;
		const auto width = extent.x;
		const auto height = extent.y;
		const auto depth = extent.z;

		
		glm::vec3 bbMin, bbMax;
		bbMax.x = width / 2;
		bbMax.y = height / 2;
		bbMax.z = depth / 2;
		bbMin.x = -width / 2;
		bbMin.y = -height / 2;
		bbMin.z = -depth / 2;


		std::vector<glm::vec3> vArr, nArr;
		std::vector<glm::vec2> uvArr;

		//top
		glm::vec3 t1 = glm::vec3(bbMin.x, bbMax.y, bbMin.z);
		glm::vec3 t2 = glm::vec3(bbMax.x, bbMax.y, bbMin.z);
		glm::vec3 t3 = glm::vec3(bbMax.x, bbMax.y, bbMax.z);
		glm::vec3 t4 = glm::vec3(bbMin.x, bbMax.y, bbMax.z);

		//bottom
		glm::vec3 b1 = glm::vec3(bbMin.x, bbMin.y, bbMin.z);
		glm::vec3 b2 = glm::vec3(bbMax.x, bbMin.y, bbMin.z);
		glm::vec3 b3 = glm::vec3(bbMax.x, bbMin.y, bbMax.z);
		glm::vec3 b4 = glm::vec3(bbMin.x, bbMin.y, bbMax.z);

		// front			back
		//		t4--t3			t2--t1
		//		|    |			|	|
		//		b4--b3			b2--b1
		// left			right
		//		t1--t4			t3--t2
		//		|    |			|	|
		//		b1--b4			b3--b2
		// top			bottom
		//		t1--t2			b4--b3
		//		|    |			|	|
		//		t4--t3			b1--b2
		//front
		vArr.push_back(b4);		vArr.push_back(b3);		
		vArr.push_back(b3);		vArr.push_back(t3);
		vArr.push_back(t3);		vArr.push_back(t4);
		vArr.push_back(t4);		vArr.push_back(b4);	
		{
			auto n = getNormal(b4, b3, t3);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
		}


		//back
		vArr.push_back(b2);		vArr.push_back(b1);
		vArr.push_back(b1);		vArr.push_back(t1);
		vArr.push_back(t1);		vArr.push_back(t2);
		vArr.push_back(t2);		vArr.push_back(b2);
		{
			auto n = getNormal(b2, b1, t1);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
		}

		//left
		vArr.push_back(b1);		vArr.push_back(b4);
		vArr.push_back(b4);		vArr.push_back(t4);
		vArr.push_back(t4);		vArr.push_back(t1);
		vArr.push_back(t1);		vArr.push_back(b1);
		{
			auto n = getNormal(b1, b4, t4);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
		}

		//right
		vArr.push_back(b3);		vArr.push_back(b2);
		vArr.push_back(b2);		vArr.push_back(t2);
		vArr.push_back(t2);		vArr.push_back(t3);
		vArr.push_back(t3);		vArr.push_back(b3);
		{
			auto n = getNormal(b3, b2, t2);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
		}

		//top
		vArr.push_back(t4);		vArr.push_back(t3);
		vArr.push_back(t3);		vArr.push_back(t2);
		vArr.push_back(t2);		vArr.push_back(t1);
		vArr.push_back(t1);		vArr.push_back(t4);
		{
			auto n = getNormal(t4, t3, t2);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
		}

		//bottom
		vArr.push_back(b1);		vArr.push_back(b2);
		vArr.push_back(b2);		vArr.push_back(b3);
		vArr.push_back(b3);		vArr.push_back(b4);
		vArr.push_back(b4);		vArr.push_back(b1);
		{
			auto n = getNormal(b1, b2, b3);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
			nArr.push_back(n);
		}

		std::vector<VDPosNorm> interleavedArr;
		std::vector<uint16_t> iArr;

		auto totalVerts = vArr.size();
		for (auto i = 0; i < totalVerts; i++)
		{
			auto v = vArr.at(i);
			auto n = nArr.at(i);
			interleavedArr.push_back({ v,n });
			iArr.push_back((uint16_t)iArr.size());
		}

		auto cubeMesh = std::make_shared<Mesh>(interleavedArr, iArr);
		cubeMesh->name = "Cube Mesh";

		return cubeMesh;
	}
	std::shared_ptr<Mesh> getFSQuad()
	{
		std::vector<Vertex> vData = {
			{glm::vec3(-1,-1,0),glm::vec3(0,0,1),glm::vec2(0,0)},
			{glm::vec3(1,-1,0),glm::vec3(0,0,1),glm::vec2(1,0)},
			{glm::vec3(1,1,0),glm::vec3(0,0,1),glm::vec2(1,1)},
			{glm::vec3(-1,1,0),glm::vec3(0,0,1),glm::vec2(0,1)}
		};
		std::vector<uint16_t> iData = { 0,1,2,0,2,3 };

		auto triMesh = std::make_shared<Mesh>(vData, iData);
		triMesh->name = "Full screen quad";

		return triMesh;
	}
	unsigned char* getImageData(std::string filename, int& width, int& height, int& nChannels,int reqChannels)
	{
		stbi_set_flip_vertically_on_load(1);
		unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nChannels, reqChannels);
		return data;
	}

	void freeImageData(unsigned char* data)
	{
		if (data != NULL)
			stbi_image_free(data);
	}

	glm::vec3 getNormal(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3)
	{
		auto p = v2 - v1;
		auto q = v3 - v1;
		return glm::cross(p, q);
	}

	Vertex::Vertex(glm::vec3 pos, glm::vec3 norm, glm::vec2 texcoords)
	{
		position = pos;
		normal = norm;
		uv = texcoords;
	}
	
	std::array<VkVertexInputAttributeDescription, 3> Vertex::getAttributeDescriptions() {
		VkVertexInputAttributeDescription pos, norm, uv;

		pos.binding = 0;
		pos.location = 0;
		pos.format = VK_FORMAT_R32G32B32_SFLOAT;
		pos.offset = 0;

		norm.binding = 0;
		norm.location = 1;
		norm.format = VK_FORMAT_R32G32B32_SFLOAT;
		norm.offset = sizeof(glm::vec3);

		uv.binding = 0;
		uv.location = 2;
		uv.format = VK_FORMAT_R32G32_SFLOAT;
		uv.offset = sizeof(glm::vec3) * 2;

		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = { pos,norm,uv };

		return attributeDescriptions;
	}

	VkVertexInputBindingDescription Vertex::getBindingDescription()
	{
		VkVertexInputBindingDescription description;
		description.binding = 0;
		description.stride = sizeof(Vertex);
		description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return description;
	}

	Mesh::Mesh(std::vector<Vertex> vData, std::vector<uint16_t> iData)
	{
		this->vData = vData;
		this->iData = iData;
	}

	Mesh::Mesh(std::vector<VDPosNorm> vData, std::vector<uint16_t> iData)
	{
		this->vDataOL = vData;
		this->iData = iData;
	}

	VDPosNorm::VDPosNorm(glm::vec3 pos, glm::vec3 norm)
	{
		position = pos;
		normal = norm;
	}
	std::array<VkVertexInputAttributeDescription, 2> VDPosNorm::getAttributeDescriptions()
	{
		VkVertexInputAttributeDescription pos, norm;

		pos.binding = 0;
		pos.location = 0;
		pos.format = VK_FORMAT_R32G32B32_SFLOAT;
		pos.offset = 0;

		norm.binding = 0;
		norm.location = 1;
		norm.format = VK_FORMAT_R32G32B32_SFLOAT;
		norm.offset = sizeof(glm::vec3);


		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = { pos,norm };

		return attributeDescriptions;
	}
	VkVertexInputBindingDescription VDPosNorm::getBindingDescription()
	{
		VkVertexInputBindingDescription description;
		description.binding = 0;
		description.stride = sizeof(VDPosNorm);
		description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return description;
	}
}
