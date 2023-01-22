#include "VKUtility.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace VKUtility
{
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
}
