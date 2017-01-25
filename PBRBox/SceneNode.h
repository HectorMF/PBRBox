#pragma once

#include "glm\glm.hpp"
#include <string>
#include <vector>

class SceneNode
{

private:
	static unsigned int counterID;

protected:
	bool m_dirty;

	Scene* m_scene;
	SceneNode* m_parent;
	std::vector<SceneNode*> m_children;

	glm::mat4 m_transform;
	unsigned int uID;

public:
	Mesh* mesh;

	SceneNode()
	{
		uID = counterID;
		counterID++;
	}

	~SceneNode()
	{
		for (int i = 0; i < m_children.size(); i++)
			delete m_children[i];
	}

	unsigned int getID() { return uID; }

	std::vector<SceneNode*> getChildren() { return m_children; }

	void addChild(SceneNode* node) { m_children.push_back(node); }

	void removeChild(unsigned int index) 
	{
		m_children.erase(m_children.begin() + index);
	}

	void removeChildByID(unsigned int id)
	{
		for (int i = 0; i < m_children.size(); i++)
			if (m_children[i]->getID() == id)
			{
				m_children.erase(m_children.begin() + i);
				break;
			}
	}

	SceneNode* getChild(unsigned int index) { return m_children[index]; }

	SceneNode* getChildByID(unsigned int id) 
	{ 
		for (int i = 0; i < m_children.size(); i++) 
			if (m_children[i]->getID() == id) 
				return m_children[i]; 
		return NULL;
	}

	SceneNode* getParent() { return m_parent; }
	Scene* getScene() { return m_scene; }

	unsigned int getChildCount() { return m_children.size(); }
	//right, up, forward matrices

	//localPosition;
	//localRotation;
	//localScale;

	//position;
	//rotation;

	glm::mat4 getTransformMatrix();

	//void setPosition(gb::Vec3f);
	//gb::Vec3f getPosition();

	//void setRotationEuler(gb::Vec3f);
	//gb::Vec3f getRotationEuler();

	//void setRotation(gb::Quatf);
	//gb::Quatf getRotation();


};