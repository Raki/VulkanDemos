#ifndef VK_UTILITY_H
#define VK_UTILITY_H

#include "CommonHeaders.h"


namespace VKUtility
{
	unsigned char* getImageData(std::string fileanme, int& width, int& height, int& nChannels,int reqChannels= 0);
	void freeImageData(unsigned char* data);
}


#endif // !VK_UTILITY_H