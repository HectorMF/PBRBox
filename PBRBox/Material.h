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
		shader.bind();

		glUniform3f(glGetUniformLocation(shader.getProgram(), "uLightPos"), 2.0, 2.0, 2.0);

		GLint d = glGetUniformLocation(shader.getProgram(), "uEnvMap");

		glUniform1i(d, 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, environment);

		glUniform1i(glGetUniformLocation(shader.getProgram(), "uShadowMap"), 1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, shadowTex);
	}

	void Unbind()
	{
		shader.unbind();
	}

	Texture environment;
	GLuint shadowTex;

	Shader shader;
};