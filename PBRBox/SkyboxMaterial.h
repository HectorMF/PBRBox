#pragma once
#pragma once

#include "Material.h"
#include "MathUtil.h"

class SkyboxMaterial : public Material
{
public:

	SkyboxMaterial(Environment* environment)
	{
		m_environment = environment;
		shader = Shader("shaders\\Skybox.vert", "shaders\\Skybox.frag");
	}

	void bind()
	{
		shader.bind();

		if (m_environment)
		{
			shader.setUniform("uRadianceMap", m_environment->radiance);
			shader.setUniform("uIrradianceMap", m_environment->irradiance);
			shader.setUniform("uSpecularMap", m_environment->specular);
		}

	}
};