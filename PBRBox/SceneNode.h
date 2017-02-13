#pragma once

#include "glm\glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <string>
#include <vector>
#include "Mesh.h"

class Scene;
class SceneNode
{

private:
	static unsigned int counterID;

protected:
public:
	bool m_dirty;

	Scene* m_scene;
	SceneNode* m_parent;
	SceneNode* root;
	std::vector<SceneNode*> m_children;

	glm::mat4 m_localMatrix;
	glm::mat4 m_worldMatrix;
	std::string m_name;
	unsigned int uID;

	

	glm::vec3 position;
	glm::quat rotation;
	glm::vec3 scale = glm::vec3(1,1,1);

	Mesh* mesh;

	SceneNode()
	{
		uID = counterID;
		counterID++;
		mesh = nullptr;
	}

	~SceneNode()
	{
		for (int i = 0; i < m_children.size(); i++)
			delete m_children[i];
	}

	unsigned int getID() { return uID; }

	std::vector<SceneNode*> getChildren() { return m_children; }

	SceneNode* add(SceneNode* node) { 
		if (node->uID == uID)
			printf("Object ID: \'%d\' cannot be added as a child to itself.", uID);

		if (node->m_parent != nullptr)
			node->m_parent->remove(node);

		node->root = root;
		node->m_parent = this;

		m_children.push_back(node);
		return this;
	}

	void remove(SceneNode* obj)
	{
		for (int i = m_children.size() - 1; i >= 0; i--)
		{
			if (m_children[i]->uID == obj->uID)
			{
				obj->m_parent = nullptr;
				obj->root = nullptr;

				m_children.erase(m_children.begin() + i);
			}
		}
	}

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

	void updateWorldMatrix()
	{
		updateLocalMatrix();

		if (m_parent == nullptr)
			m_worldMatrix = m_localMatrix;
		else
			m_worldMatrix = m_parent->m_worldMatrix * m_localMatrix;

		for (auto& child : m_children)
		{
			child->updateWorldMatrix();
		}
	}

	void updateLocalMatrix()
	{
		m_localMatrix = glm::mat4();

		glm::mat4 t = glm::translate(glm::mat4(), position);
		glm::mat4 r = glm::mat4_cast(rotation);
		glm::mat4 s = glm::scale(glm::mat4(), scale);

		m_localMatrix = t * r * s;
		//m_localMatrix = glm::scale(m_localMatrix, scale);
		//m_localMatrix = glm::mat4_cast(rotation) * m_localMatrix;
		
		//m_localMatrix = glm::translate(m_localMatrix, position);
	}

	glm::mat4 getWorldMatrix()
	{
		return m_worldMatrix;
	}

	/*glm::quat getWorldRotation()
	{
		updateWorldMatrix();

		Vec3f p;
		Matrix4f r;
		Vec3f s;
		worldMatrix.decompose(&s, &r, &p);
		Quatf rq;
		r.getQuat(&rq);
		return rq;
	}

	glm::vec3 getWorldPosition()
	{
		updateWorldMatrix();

		Vec3f p;
		Matrix4f r;
		Vec3f s;
		worldMatrix.decompose(&s, &r, &p);
		return p;
	}

	glm::vec3 getWorldScale()
	{
		updateWorldMatrix();

		Vec3f p;
		Matrix4f r;
		Vec3f s;
		worldMatrix.decompose(&s, &r, &p);
		return s;
	}
	*/

	//gb::Vec3f getPosition();

	//void setRotationEuler(gb::Vec3f);
	//gb::Vec3f getRotationEuler();

	//void setRotation(gb::Quatf);
	//gb::Quatf getRotation();


};


