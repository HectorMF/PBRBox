#pragma once
#pragma once

#include "Material.h"
#include "MathUtil.h"

class SkyboxMaterial : public Material
{
public:
	GLuint m_cubeMap;


	SkyboxMaterial()
	{
		shader = Shader("shaders\\Skybox.vert", "shaders\\Skybox.frag");
	}

	void bind()
	{
		shader.bind();

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_cubeMap);
		glUniform1i(glGetUniformLocation(shader.getProgram(), "uSkybox"), 3);

	}
};