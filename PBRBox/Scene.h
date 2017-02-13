#pragma once
#include "Camera.h"
#include "SceneNode.h"
#include "Mesh.h"
#include "Light.h"

#include <vector>
class Scene
{
public:
	SceneNode* root;

	ResourceHandle<Environment> environment;

	Mesh* skybox;

	std::vector<Light*> lights;

	//void init()
	//void start()
	//void update()
	//void end();
	//void quit();

	Scene()
	{
		root = new SceneNode();
	}

	void add(SceneNode* m)
	{
		root->add(m);
	}

	void add(Model* model)
	{
		
	}

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