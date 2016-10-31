#pragma once

#include <memory>
#include <string>
#include <vector>
#include "glm\glm.hpp"

class ModelNode
{
public:

	ModelNode();
	~ModelNode();

	std::string m_name;
	//transformation relative to parent node
	glm::mat4 m_transform;

	//returns null if root node
	//GLMaterialObject* m_material;

	ModelNode* m_parent;
	std::vector<unsigned int> m_meshes;
	std::vector<ModelNode*> m_children;

	ModelNode* copyNode(ModelNode* parent)
	{
		ModelNode* node = new ModelNode();
		node->m_name = m_name;
		node->m_transform = m_transform;
		//node->m_material = m_material;
		node->m_parent = parent;
		for (int i = 0; i < getNumChildren(); i++)
			node->m_children.push_back(m_children[i]->copyNode(node));
		node->m_meshes = m_meshes;

		return node;
	}

	void addChild(ModelNode* child)
	{
		m_children.push_back(child);
	}

	ModelNode& getChild(int index) { return *(m_children[index]); }
	ModelNode* findNode(std::string name);
	ModelNode* getParent() { return m_parent; }
	void setParent(ModelNode* parent) { m_parent = parent; }

	void setTransform(glm::mat4 transform) { m_transform = transform; }
	glm::mat4& getTransform() { return m_transform; }

	int getNumMeshes() const { return m_meshes.size(); }
	int getNumChildren() const { return m_children.size(); }
};
