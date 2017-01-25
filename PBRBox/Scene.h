#pragma once
#include "Camera.h"
#include "SceneNode.h"
#include "Mesh.h"

#include <vector>
class Scene
{
public:
	SceneNode* root;
	Mesh* skybox;

	~Scene()
	{
		delete root;
		delete skybox;
	}

	//void add(Mesh* object)
	//{
	//	root->addChild(object);
	//}
};