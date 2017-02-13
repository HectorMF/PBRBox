#pragma once
#include "Loader.h"
#include "Texture.h"
#include "Environment.h"
#include "ResourceManager.h"

class EnvironmentLoader : public Loader<Environment>
{
public:

	EnvironmentLoader()
	{
		extensions.push_back(".gbenv");
	}

	Environment* load(ResourceManager* resourceManager, std::string filename, Environment* env) override
	{

		std::ifstream file(filename);
		if (file)
		{
			cereal::XMLInputArchive archive(file);
			archive(*env);
		}

		env->radiance = resourceManager->load<Texture>(env->radiance.filePath);
		env->irradiance = resourceManager->load<Texture>(env->irradiance.filePath);
		env->specular = resourceManager->load<Texture>(env->specular.filePath);
		//env->radiance.manager = resourceManager;
		//env->irradiance.manager = resourceManager;
		//env->specular.manager = resourceManager;
		stbi_set_flip_vertically_on_load(true);
		env->brdf = resourceManager->load<Texture>(env->brdf.filePath);
		stbi_set_flip_vertically_on_load(false);
	
		return env;
	}
};