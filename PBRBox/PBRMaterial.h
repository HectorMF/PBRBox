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
	Texture m_reflectionMap;
	Texture m_irradianceMap;
	Texture m_hammersleyPointMap;

	PBRMaterial()
	{
		shader = 
		m_ior = 1.4;
		m_albedo = glm::vec4(0, 0, 0, 1);
		m_roughness = .5f;
		m_metalness = 0.0;
	}

	void Bind()
	{

	}

	void setAlbedo(const glm::vec4& albedo)
	{


	}

	void setMetalness(const float& metalness)
	{

	}

	void setMetalness(const Texture& metalness)
	{

	}

	void setRoughness(const float& roughness)
	{

	}

	void setRoughness(const Texture& roughness)
	{

	}

	void setIOR(const float& ior)
	{

	}


};