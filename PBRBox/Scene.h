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

	glm::vec4 clearColor;

	~Scene()
	{
		for (int i = 0; i < sceneGraph.size(); i++)
			delete sceneGraph[i];
		delete skybox;
	}

	void render(Camera& camera)
	{
		glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear all pixels

		if(skybox)
			skybox->render(camera);

		glClear(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		for (int i = 0; i < sceneGraph.size(); i++)
			sceneGraph[i]->render(camera);
	}

	void add(Mesh* object)
	{
		sceneGraph.push_back(object);
	}
};