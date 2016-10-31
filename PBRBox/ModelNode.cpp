#include "ModelNode.h"

ModelNode::ModelNode()
{
}

ModelNode::~ModelNode()
{
	for (int i = 0; i < m_children.size(); i++)
		delete m_children[i];
	m_parent = 0x0;
	//m_material = 0x0;
	m_meshes.clear();
	m_children.clear();
}

ModelNode* ModelNode::findNode(std::string name)
{
	if (m_name == name)
		return this;

	for (int i = 0; i < m_children.size(); i++)
	{
		ModelNode* node = m_children[i]->findNode(name);
		if (node != NULL)
			return node;
	}

	return NULL;
}