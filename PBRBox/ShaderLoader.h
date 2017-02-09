#pragma once
#include "Loader.h"
#include "Shader.h"

class EnvironmentLoader : public Loader<Shader>
{
public:

	EnvironmentLoader()
	{
		extensions.push_back(".gbenv");
	}

	Shader* load(ResourceManager* resourceManager, std::string filename, Shader* shad) override
	{

		std::ifstream file(filename);
		if (file)
		{
			cereal::XMLInputArchive archive(file);
			archive(*shad);
		}

		std::ifstream vShaderFile;
		std::ifstream fShaderFile;
		// ensures ifstream objects can throw exceptions:
		vShaderFile.exceptions(std::ifstream::badbit);
		fShaderFile.exceptions(std::ifstream::badbit);
		try
		{
			// Open files
			vShaderFile.open(shad->vertexPath);
			fShaderFile.open(shad->fragmentPath);
			std::stringstream vShaderStream, fShaderStream;
			// Read file's buffer contents into streams
			vShaderStream << vShaderFile.rdbuf();
			fShaderStream << fShaderFile.rdbuf();
			// close file handlers
			vShaderFile.close();
			fShaderFile.close();
			// Convert stream into string
			shad->vertexCode = vShaderStream.str();
			shad->fragmentCode = fShaderStream.str();
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		}

		if (shad->autoCompile)
			shad->compile();

		return shad;
	}
};