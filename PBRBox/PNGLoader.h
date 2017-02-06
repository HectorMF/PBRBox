#pragma once

#include "Loader.h"
#include "Texture.h"
#include "stb_image.h"
#include "TextureDescriptor.h"

class PNGLoader : public Loader<Texture>
{
public:
	PNGLoader()
	{
		extensions.push_back(".png");
	}

	ResourceDescriptor<Texture>* loadDescriptor(std::string filename) override
	{
		return new TextureDescriptor(filename);
	}

	Texture* load(ResourceManager* resourceManager, std::string filename, ResourceDescriptor<Texture>* descriptor) override
	{
		int width, height, bpp;
		unsigned char* image = stbi_load(filename.c_str(), &width, &height, &bpp, 4);

		Texture* tex = new Texture();
		tex->width = width;
		tex->height = height;
		tex->data = image;
		printf("LOADED PNG!!!!\n");
		return tex;
	}
};