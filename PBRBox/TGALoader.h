#pragma once

#include "Loader.h"
#include "Texture.h"
#include "stb_image.h"

class TGALoader : public Loader<Texture>
{
public:
	TGALoader()
	{
		extensions.push_back(".tga");
	}

	Texture* load(ResourceManager* resourceManager, std::string filename, Texture* tex) override
	{
		int width, height, bpp;
		unsigned char* image = stbi_load(filename.c_str(), &width, &height, &bpp, 4);

		TextureData* data = new TextureData();
		data->setData(width, height, image);
		data->setFormat(GL_RGBA);
		data->setDataType(GL_UNSIGNED_BYTE);

		tex->data = data;
		tex->upload();
		printf("LOADED TGA!!!!\n");
		return tex;
	}
};
