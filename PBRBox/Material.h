#pragma once
#include "Shader.h"
#include "Texture.h"
#include "glm\glm.hpp"
#include "Environment.h"
#include "ResourceBase.h"

class Material : public ResourceBase
{
public:
	void foo(){}

	virtual void bind()
	{
		shader.bind();

		glUniform3f(glGetUniformLocation(shader.getProgram(), "uLightPos"), 2.0, 2.0, 2.0);

		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		//GLenum drawMode = GL_TRIANGLES;
		//Wireframe mode
		/*if (displayMode == 1)
		{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		}
		//Point rendering mode
		if (displayMode == 2)
		{
		glPointSize(2.0f);
		drawMode = GL_POINTS;
		}*/
	}

	virtual void unbind()
	{
		shader.unbind();
	}

	void setEnvironment(ResourceHandle<Environment> env)
	{
		m_environment = env;
	}

	ResourceHandle<Environment> m_environment;


	Shader shader;
};