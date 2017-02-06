#pragma once
#include "Loader.h"
#include "Texture.h"
#include "Environment.h"

class EnvironmentLoader : public Loader<Environment>
{
public:

	EnvironmentLoader()
	{
		extensions.push_back(".gbenv");
	}

	Environment* load(std::string filename) override
	{
		int width, height, bpp;
		unsigned char* image = stbi_load(filename.c_str(), &width, &height, &bpp, 4);

		Texture* tex = new Texture();
		tex->width = width;
		tex->height = height;
		tex->data = image;
		printf("LOADED JPG!!!!\n");
		return tex;
	}


};