#pragma once
#include "Shader.h"
#include "Texture.h"
#include "glm\glm.hpp"

enum TextureType { Diffuse, Environment };

class Material
{
public:
	void Bind()
	{
		shader.Bind();

		glUniform3f(glGetUniformLocation(shader, "uLightPos"), 10.0, 10.0, 10.0);

		GLint d = glGetUniformLocation(shader, "material.diffuse");

		glUniform1i(d, 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuse);

		GLint e = glGetUniformLocation(shader, "uEnvMap");

		glUniform1i(e, 1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, environment);

	}

	void Unbind()
	{
		shader.Unbind();
	}

	Texture diffuse;
	Texture environment;

	Shader shader;
};