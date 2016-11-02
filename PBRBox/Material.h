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

		glUniform3f(glGetUniformLocation(shader, "uLightPos"), 2.0, 2.0, 2.0);

		GLint d = glGetUniformLocation(shader, "material.diffuse");

		glUniform1i(d, 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, diffuse);

		GLint e = glGetUniformLocation(shader, "uEnvMap");

		glUniform1i(e, 1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, environment);

		glUniform1i(glGetUniformLocation(shader, "uShadowMap"), 2);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, shadowTex);
		glUniform1i(glGetUniformLocation(shader, "uMetallicMap"), 3);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, metallic);

		glUniform1i(glGetUniformLocation(shader, "uReflectionMap"), 4);
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, reflection);

		glUniform1i(glGetUniformLocation(shader, "uRoughnessMap"),5);
		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, roughness);

		glUniform1i(glGetUniformLocation(shader, "uNormalMap"), 6);
		glActiveTexture(GL_TEXTURE6);
		glBindTexture(GL_TEXTURE_2D, normal);

		glUniform1i(glGetUniformLocation(shader, "uUVMap"), 7);
		glActiveTexture(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_2D, uv);

	}

	void Unbind()
	{
		shader.Unbind();
	}
	GLuint uv;
	GLuint normal;
	GLuint reflection;
	GLuint metallic;
	GLuint roughness;

	GLuint diffuse;
	Texture environment;
	GLuint shadowTex;
	Shader shader;
};