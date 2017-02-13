#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>

#include "SceneNode.h"
#include "Mesh.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Model : public ResourceBase
{
	friend class ModelLoader;
public:
	Model();

	void foo() {}
	virtual ~Model();

	const std::vector<Mesh*>& getMeshes() const { return m_meshes; }

	const SceneNode* getHierarchy() const { return m_hierarchy; }

	//gb::AABox3f getAABB() const { return m_AABB; }
//protected:
	//gb::AABox3f m_AABB;
	std::string m_fileName;
	std::vector<Mesh*> m_meshes;
	SceneNode* m_hierarchy;
};