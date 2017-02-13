#pragma once
#include "glm\glm.hpp"

#define M_PI_2 1.57079632679
#define M_PI 3.14156265

#define CLAMP( x, min, max ) ( (x)<(min) ? (min) : ( (x)>(max) ? (max) : (x) ) )
#define LERP(x0, x1, t) (x0 + ((t) * (x1 - x0)))

inline glm::vec4 sRGBToLinear(glm::vec4 rgb)
{
	glm::vec3 temp = glm::vec3(rgb.x, rgb.y, rgb.z);
	temp = glm::pow(temp, glm::vec3(2.2));
	return glm::vec4(temp, rgb.a);
}
