#pragma once
#include <iostream>
#include <assert.h>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <set>
#include <optional>
#include <limits> 
#include <algorithm>
#include <chrono>
#include <map>
#include <filesystem>


#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <fmt/format.h>

#define VK_CHECK(call)\
do {\
	VkResult result = call; \
	assert(result == VK_SUCCESS); \
}while(0)

constexpr auto cendl = '\n';