#pragma once
#include "SceneObject.h"

class Light : public SceneObject
{
public:
	bool castShadows;

	glm::vec2 shadowMapSize;
	float bias = 0;
	float radius = 1;
	//Camera camera;
	glm::vec3 color;
	float intensity;

	void GetShadowCamera() {}

	void renderGizmo()
	{
		
		//c_world = inverse(projection * view) * vec4(c_ncd, 1);
		//c_world = c_world*1.0 / c_world.w;
	}
};