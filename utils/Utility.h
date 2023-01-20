#pragma once
#include "CommonHeaders.h"

namespace Utility
{
	std::string readTxtFileContents(std::string filePath);
	std::vector<unsigned char> readBinaryFileContents(const std::string filename);
	void savePngFile(std::string filename, int w, int h, int comp, unsigned char* data);

	/*
	* @param str string with the content
	* @param delim string with delimiters
	*/
	std::vector<std::string> split(std::string str, std::string delim);

	
	/*
	* @param cAngle current angle in degrees
	* @param normal normal angle in degrees
	*/
	float getReflectionAngle(float cAngle,float normal);
}