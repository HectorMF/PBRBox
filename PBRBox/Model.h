#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>

#include "ModelNode.h"
#include "Mesh.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Model : public SceneObject
{
	friend class ModelLoader;
public:
	Model();
	Model(const std::string &filename);

	virtual ~Model();

	//load and save the resource object
	virtual void load(const std::string &file);
	virtual void save(const std::string &file);

	void render();
	void renderNode(const ModelNode* node, glm::mat4 transform);
	const std::vector<Mesh*>& getMeshes() const { return m_meshes; }

	const ModelNode* getHierarchy() const { return m_hierarchy; }
	ModelNode* copyHierarchy() const { return m_hierarchy->copyNode(nullptr); }

	//gb::AABox3f getAABB() const { return m_AABB; }
//protected:
	//gb::AABox3f m_AABB;
	std::string m_fileName;
	std::vector<Mesh*> m_meshes;
	ModelNode* m_hierarchy;
};