#pragma once
#include "Loader.h"
#include "Texture.h"
#include "stb_image.h"
#include "TextureDescriptor.h"

class JPGLoader : public Loader<Texture>
{
public:

	JPGLoader()
	{
		extensions.push_back(".jpg");
		extensions.push_back(".jpeg");
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
		printf("LOADED JPG!!!!\n");
		return tex;
	}
};