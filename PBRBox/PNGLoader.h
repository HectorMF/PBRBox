#pragma once

#include "Loader.h"
#include "Texture.h"
#include "stb_image.h"
#include "tinyxml2.h"

class PNGLoader : public Loader<Texture>
{
public:
	PNGLoader()
	{
		extensions.push_back(".png");
	}

	std::vector<ResourceDescription> getAssetDependencies(std::string filename) override
	{
		tinyxml2::XMLDocument doc;
		doc.LoadFile((filename + ".gbr").c_str());

		// Structure of the XML file:
		// - Element "PLAY"      the root Element, which is the 
		//                       FirstChildElement of the Document
		// - - Element "TITLE"   child of the root PLAY Element
		// - - - Text            child of the TITLE Element

		// Navigate to the title, using the convenience function,
		// with a dangerous lack of error checking.
		const char* title = doc.FirstChildElement("PLAY")->FirstChildElement("TITLE")->GetText();
		printf("Name of play (1): %s\n", title);

		// Text is just another Node to TinyXML-2. The more
		// general way to get to the XMLText:
		tinyxml2::XMLText* textNode = doc.FirstChildElement("PLAY")->FirstChildElement("TITLE")->FirstChild()->ToText();
		title = textNode->Value();
		printf("Name of play (2): %s\n", title);
	}

	Texture* load(ResourceManager* resourceManager, std::string filename) override
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