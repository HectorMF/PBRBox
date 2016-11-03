#pragma once

#include "Material.h"

class PBRMaterial : public Material 
{

	bool m_dirty;

	//material information
	glm::vec4 m_albedo;
	float m_metalness;
	float m_roughness;
	float m_ior;

	Texture m_albedoMap;
	Texture m_normalMap;
	Texture m_roughnessMap;
	Texture m_metalnessMap;

	//environment information
public:
	Texture m_reflectionMap;
	Texture m_irradianceMap;
	Texture m_hammersleyPointMap;

	PBRMaterial()
	{
		shader = Shader("shaders\\Lambert.vert", "shaders\\Lambert.frag");
		m_ior = 1.4;
		m_albedo = glm::vec4(0, 0, 0, 1);
		m_roughness = .5f;
		m_metalness = 0.0;
	}

	void Bind()
	{
		shader.Bind();


		unsigned int albedoLoc = glGetUniformLocation(shader.getProgram(), "uAlbedo");
		unsigned int roughnessLoc = glGetUniformLocation(shader.getProgram(), "uRoughness");
		unsigned int metalnessLoc = glGetUniformLocation(shader.getProgram(), "uMetalness");
		unsigned int normalLoc = glGetUniformLocation(shader.getProgram(), "uNormal");


		glUniform1i(albedoLoc, 2);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, m_albedoMap);

		glUniform1i(metalnessLoc, 3);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, m_metalnessMap);

		glUniform1i(roughnessLoc, 4);
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, m_roughnessMap);

		glUniform1i(normalLoc, 5);
		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, m_normalMap);



		glUniform3fv(albedoLoc, to_linear(m_albedo));
		glUniform1f(roughnessLoc, m_roughness);
		glUniform1f(metalnessLoc, m_metalness);

	}

	void setAlbedo(const glm::vec4& albedo)
	{
		m_albedo = albedo;
	}

	void setAlbedoMap(const Texture& albedo)
	{
		m_albedoMap = albedo;
	}

	void setNormalMap(const Texture& normals)
	{
		m_normalMap = normals;
	}

	void setMetalness(const float& metalness)
	{
		m_metalness = metalness;
	}

	void setMetalnessMap(const Texture& metalness)
	{
		m_metalnessMap = metalness;
	}

	void setRoughness(const float& roughness)
	{
		m_roughness = roughness;
	}

	void setRoughnessMap(const Texture& roughness)
	{
		m_roughnessMap = roughness;
	}

	void setIOR(const float& ior)
	{
		m_ior = ior;
	}
};