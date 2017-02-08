#pragma once
#include "Loader.h"
#include "Texture.h"
#include "stb_image.h"

class JPGLoader : public Loader<Texture>
{
public:

	JPGLoader()
	{
		extensions.push_back(".jpg");
		extensions.push_back(".jpeg");
	}

	Texture* load(ResourceManager* resourceManager, std::string filename, Texture* tex) override
	{
		int width, height, bpp;
		unsigned char* image = stbi_load(filename.c_str(), &width, &height, &bpp, 4);

		tex->width = width;
		tex->height = height;
		tex->data = image;
		tex->upload();
		printf("LOADED JPG!!!!\n");
		return tex;
	}
};