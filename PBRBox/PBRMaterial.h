#pragma once

#include "Material.h"

class PBRMaterial : public Material 
{
	bool m_dirty;
	//material information
	glm::vec4 albedo;
	float metalness;
	float roughness;
	float IOR;

	Texture albedoMap;
	Texture normalMap;
	Texture roughnessMap;
	Texture metalnessMap;

	//environment information
	Texture reflectionMap;
	Texture irradianceMap;
	Texture hammersleyPointMap;


	void Bind()
	{

	}

	void setAlbedo(const glm::vec4& albedo)
	{
	}

	void setAlbedo(const Texture& albedo)
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