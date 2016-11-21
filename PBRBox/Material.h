#pragma once
#include "Shader.h"
#include "Texture.h"
#include "glm\glm.hpp"

enum TextureType { Diffuse, Environment };

class Material
{
public:
	virtual void bind()
	{
		shader.bind();

		glUniform3f(glGetUniformLocation(shader.getProgram(), "uLightPos"), 2.0, 2.0, 2.0);

		GLint d = glGetUniformLocation(shader.getProgram(), "skybox");

		glUniform1i(d, 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, environment);

		glUniform1i(glGetUniformLocation(shader.getProgram(), "uShadowMap"), 1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, shadowTex);

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

	GLuint environment;
	GLuint shadowTex;

	Shader shader;
};