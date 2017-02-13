#pragma once
#include "glm\vec3.hpp"
#include "glm\vec2.hpp"

enum class LightType { Directional, Point, Spot };

class Light 
{
public:
	bool castShadows;

	glm::vec2 shadowMapSize;
	float shadowBias = 0;
	float shadowStrength;
	float shadowNormalBias;
	float shadowRadius;
	float shadowAngle;

	float radius = 1;
	//Camera camera;
	glm::vec3 color;
	float intensity;

	float spotAngle;
	
	float range;

//	float intensity;

	void GetShadowCamera() {}

	void renderGizmo()
	{
		//c_world = inverse(projection * view) * vec4(c_ncd, 1);
		//c_world = c_world*1.0 / c_world.w;
	}
};