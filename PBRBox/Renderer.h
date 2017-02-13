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
	glm::vec4 clearColor;
	float clearDepth;
	glm::vec2 windowSize;

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

	void renderShadow(Scene& scene)
	{
		overrideMaterial->bind();
		renderTarget->Bind();

		glViewport(0, 0, 2048, 2048);
		glClearDepth(1.0);
		glClear(GL_DEPTH_BUFFER_BIT);
		//glEnable(GL_DEPTH_TEST);
		
		GLfloat near_plane = 1.0f, far_plane = 7.5f;
		glm::mat4 lightProjection = glm::ortho(-3.0f, 3.0f, -3.0f, 3.0f, near_plane, far_plane);
		glm::mat4 lightView = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
			glm::vec3(0.0f, 0.0f, 0.0f),
			glm::vec3(0.0f, 1.0f, 0.0f));

		glm::mat4 lightSpaceMatrix = lightProjection * lightView;

		int t = glGetUniformLocation(overrideMaterial->shader.getProgram(), "lightSpaceMatrix");
		glUniformMatrix4fv(t, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

		//renderNode(SceneNode* node, Camera& camera);
		//for (int i = 0; i < scene.root->getChildCount(); i++)
		/*if(scene.root)
		{
			t = glGetUniformLocation(overrideMaterial->shader.getProgram(), "model");
			glUniformMatrix4fv(t, 1, GL_FALSE, glm::value_ptr(scene.root->getWorldMatrix()));

			(scene.root)->mesh->render();
		}*/
		
		renderTarget->Unbind();
		overrideMaterial->unbind();
	}

	void render(Scene& scene, Camera& camera)
	{
		glViewport(0, 0, windowSize.x, windowSize.y);

		if (renderTarget)
			renderTarget->Bind();

		glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
		//glClearDepth(clearDepth);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear all pixels

		if (scene.skybox)
		{
			Material* m = (!overrideMaterial) ? (scene.skybox->m_material) : overrideMaterial;
			m->bind();

			glm::mat4 model;
			glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.view, camera.up);
			glm::mat4 projection = glm::perspective(45.0f, (float)camera.resolution.x / (float)camera.resolution.y, 0.1f, 1000.0f);

			glm::mat4 normal = glm::transpose(glm::inverse(view * model));
			glm::mat4 invProjection = glm::inverse(projection);
			glm::mat4 transView = glm::transpose(view);
			glm::vec4 viewDir = view * model * glm::vec4(1, 0, 0, 0);

			view = glm::mat4(glm::mat3(view));


			m->shader.setUniform("camera.position", camera.position);
			m->shader.setUniform("camera.viewDirection", camera.view);

			m->shader.setUniform("camera.mModel", model);
			m->shader.setUniform("camera.mView", view);
			m->shader.setUniform("camera.mProjection", projection);
			m->shader.setUniform("camera.mNormal", normal);
			m->shader.setUniform("camera.mInvView", glm::inverse(view));

			scene.skybox->render();
			m->unbind();
		}

		glClear(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		renderNode(scene.root, camera);

		if (renderTarget)
			renderTarget->Unbind();
	}

	void renderNode(SceneNode* node, Camera& camera)
	{
		if (node->mesh != nullptr)
		{
			Material* m = (!overrideMaterial) ? (node->mesh->m_material) : overrideMaterial;

			m->bind();

			glm::mat4 biasMatrix(
				0.5, 0.0, 0.0, 0.0,
				0.0, 0.5, 0.0, 0.0,
				0.0, 0.0, 0.5, 0.0,
				0.5, 0.5, 0.5, 1.0
				);

			//glm::mat4 depthBiasMVP = biasMatrix*depthMVP;
			//int t = glGetUniformLocation(m->shader, "uDepthBiasMatrix");
			//glUniformMatrix4fv(t, 1, GL_FALSE, glm::value_ptr(depthBiasMVP));

			GLfloat near_plane = 1.0f, far_plane = 7.5f;
			glm::mat4 lightProjection = glm::ortho(-3.0f, 3.0f, -3.0f, 3.0f, near_plane, far_plane);
			glm::mat4 lightView = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
				glm::vec3(0.0f, 0.0f, 0.0f),
				glm::vec3(0.0f, 1.0f, 0.0f));

			glm::mat4 lightSpaceMatrix = lightProjection * lightView;

			int t = glGetUniformLocation(m->shader.getProgram(), "lightSpaceMatrix");
			glUniformMatrix4fv(t, 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

			glm::mat4 model = node->getWorldMatrix();
			glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.view, camera.up);
			glm::mat4 projection = glm::perspective(45.0f, (float)camera.resolution.x / (float)camera.resolution.y, 0.1f, 1000.0f);

			glm::mat4 normal = glm::transpose(glm::inverse(view * model));
			glm::mat4 invProjection = glm::inverse(projection);
			glm::mat4 transView = glm::transpose(view);
			glm::vec4 viewDir = view * model * glm::vec4(1, 0, 0, 0);


			GLint vp = glGetUniformLocation(m->shader.getProgram(), "camera.position");
			GLint vd = glGetUniformLocation(m->shader.getProgram(), "camera.viewDirection");
			GLint mm = glGetUniformLocation(m->shader.getProgram(), "camera.mModel");
			GLint v = glGetUniformLocation(m->shader.getProgram(), "camera.mView");
			GLint p = glGetUniformLocation(m->shader.getProgram(), "camera.mProjection");
			GLint n = glGetUniformLocation(m->shader.getProgram(), "camera.mNormal");
			GLint ii = glGetUniformLocation(m->shader.getProgram(), "camera.mInvView");


			glUniformMatrix4fv(mm, 1, GL_FALSE, glm::value_ptr(model));
			glUniformMatrix4fv(v, 1, GL_FALSE, glm::value_ptr(view));
			glUniformMatrix4fv(p, 1, GL_FALSE, glm::value_ptr(projection));
			glUniformMatrix4fv(n, 1, GL_FALSE, glm::value_ptr(normal));
			glUniformMatrix4fv(ii, 1, GL_FALSE, glm::value_ptr(glm::inverse(view)));

			glUniform3fv(vd, 1, glm::value_ptr(camera.view));
			glUniform3fv(vp, 1, glm::value_ptr(camera.position));

			node->mesh->render();

			m->unbind();
		}

		for (int i = 0; i < node->getChildCount(); i++)
			renderNode(node->getChild(i), camera);
	}
};