#pragma once

#include "Loader.h"
#include "PBRMaterial.h"

class MaterialLoader : public Loader<PBRMaterial>
{
public:
	MaterialLoader()
	{
		extensions.push_back(".gbmat");
	}


	PBRMaterial* load(ResourceManager* resourceManager, std::string filename, PBRMaterial* mat) override
	{
		std::ifstream file(filename);
		if (file)
		{
			cereal::XMLInputArchive archive(file);
			archive(*mat);
		}
		mat->m_environment = resourceManager->load<Environment>(mat->m_environment.filePath);
		if(mat->m_albedoMap.filePath != "")
			mat->m_albedoMap = resourceManager->load<Texture>(mat->m_albedoMap.filePath);
		if (mat->m_ambientOcclusion.filePath != "")
			mat->m_ambientOcclusion = resourceManager->load<Texture>(mat->m_ambientOcclusion.filePath);
		if (mat->m_metalnessMap.filePath != "")
			mat->m_metalnessMap = resourceManager->load<Texture>(mat->m_metalnessMap.filePath);
		if (mat->m_roughnessMap.filePath != "")
			mat->m_roughnessMap = resourceManager->load<Texture>(mat->m_roughnessMap.filePath);
		if (mat->m_normalMap.filePath != "")
			mat->m_normalMap = resourceManager->load<Texture>(mat->m_normalMap.filePath);


		return mat;
	}
};