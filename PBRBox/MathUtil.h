#pragma once
#include "glm\glm.hpp"


inline glm::vec4 sRGBToLinear(glm::vec4 rgb)
{
	glm::vec3 temp = rgb;
	temp = glm::pow(temp, glm::vec3(2.2));
	return glm::vec4(temp, rgb.a);
}
