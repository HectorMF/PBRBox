#pragma once
#include "Scene.h"
#include "Camera.h"
#include "RenderTarget.h"
#include "Material.h"

class Renderer
{
public:

	RenderTarget* renderTarget;
	Material* overrideMaterial;

	Renderer()
	{
		renderTarget = nullptr;
		overrideMaterial = nullptr;
	}

	~Renderer();

	void setOverrideMaterial(Material& material)
	{
		overrideMaterial = &material;
	}

	void clearOverideMaterial()
	{
		overrideMaterial = nullptr;
	}

	void setRenderTarget(RenderTarget& target)
	{
		renderTarget = &target;
	}

	void clearRenderTarget()
	{
		renderTarget = nullptr;
	}


	void render(Scene& scene, Camera& camera)
	{
		if (renderTarget)
			renderTarget->Bind();

		glClearColor(scene.clearColor.r, scene.clearColor.g, scene.clearColor.b, scene.clearColor.a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear all pixels

		if (scene.skybox)
			scene.skybox->render(camera);

		glClear(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		for (int i = 0; i < scene.sceneGraph.size(); i++)
		{
			Material* m = (!overrideMaterial)? (&scene.sceneGraph[i]->m_material): overrideMaterial;

			m->Bind();

			glm::vec3 lightInvDir = glm::vec3(3, 3, 3);
			glm::mat4 depthProjectionMatrix = glm::ortho<float>(-10, 10, -10, 10, -10, 20);
			glm::mat4 depthViewMatrix = glm::lookAt(lightInvDir, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
			glm::mat4 depthModelMatrix = scene.sceneGraph[i]->transform;
			glm::mat4 depthMVP = depthProjectionMatrix * depthViewMatrix * depthModelMatrix;

			glm::mat4 biasMatrix(
					0.5, 0.0, 0.0, 0.0,
					0.0, 0.5, 0.0, 0.0,
					0.0, 0.0, 0.5, 0.0,
					0.5, 0.5, 0.5, 1.0
					);

			glm::mat4 depthBiasMVP = biasMatrix*depthMVP;
			int t = glGetUniformLocation(m->shader, "uDepthBiasMatrix");
			glUniformMatrix4fv(t, 1, GL_FALSE, glm::value_ptr(depthBiasMVP));

			t = glGetUniformLocation(m->shader, "depthMVP");
			glUniformMatrix4fv(t, 1, GL_FALSE, glm::value_ptr(depthMVP));

			glm::mat4 model = scene.sceneGraph[i]->transform;
			glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.view, camera.up);
			glm::mat4 projection = glm::perspective(45.0f, (float)camera.resolution.x / (float)camera.resolution.y, 0.1f, 1000.0f);

			glm::mat4 normal = glm::transpose(glm::inverse(view * model));
			glm::mat4 invProjection = glm::inverse(projection);
			glm::mat4 transView = glm::transpose(view);
			glm::vec4 viewDir = view * model * glm::vec4(1, 0, 0, 0);

			GLint mm = glGetUniformLocation(m->shader, "camera.mModel");
			GLint v = glGetUniformLocation(m->shader, "camera.mView");
			GLint p = glGetUniformLocation(m->shader, "camera.mProjection");
			GLint n = glGetUniformLocation(m->shader, "camera.mNormal");
			GLint ii = glGetUniformLocation(m->shader, "camera.mInvView");
			GLint vd = glGetUniformLocation(m->shader, "camera.mViewDirection");

			glUniformMatrix4fv(mm, 1, GL_FALSE, glm::value_ptr(model));
			glUniformMatrix4fv(v, 1, GL_FALSE, glm::value_ptr(view));
			glUniformMatrix4fv(p, 1, GL_FALSE, glm::value_ptr(projection));
			glUniformMatrix4fv(n, 1, GL_FALSE, glm::value_ptr(normal));
			glUniformMatrix4fv(ii, 1, GL_FALSE, glm::value_ptr(glm::inverse(view)));
			glUniform3fv(vd, 1, glm::value_ptr(viewDir));

			scene.sceneGraph[i]->render(camera);

			m->Unbind();
		}

		if (renderTarget)
			renderTarget->Unbind();
	}
};