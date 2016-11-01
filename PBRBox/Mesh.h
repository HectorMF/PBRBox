#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <GL/glew.h>
#include "glm\glm.hpp"
#include "Geometry.h"
#include "Material.h"
#include "SceneObject.h"

class Mesh : public SceneObject
{
	friend class ModelLoader;
public:
	Mesh(Geometry &geometry, Material* material);
	~Mesh();
	
	void render();

//protected:
	Geometry m_geometry;
	Material* m_material;
};
