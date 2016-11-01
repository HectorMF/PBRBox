#pragma once
#include "Camera.h"
#include "SceneObject.h"
#include "Mesh.h"

#include <vector>
class Scene
{
public:
	std::vector<Mesh*> sceneGraph;
	Mesh* skybox;

	~Scene()
	{
		for (int i = 0; i < sceneGraph.size(); i++)
			delete sceneGraph[i];
		delete skybox;
	}

	void add(Mesh* object)
	{
		sceneGraph.push_back(object);
	}
};