#pragma once
#include "ResourceDescriptor.h"
#include "Texture.h"

class TextureDescriptor : public ResourceDescriptor<Texture>
{
protected:
	unsigned int target;
	unsigned int colorSpace;
	unsigned int minFilter;
	unsigned int magFilter;
	unsigned int uWrap;
	unsigned int vWrap;
	bool generateMM;

public:
	TextureDescriptor(){}

	TextureDescriptor(std::string filename)
	{
		save(filename);
	}

	void save(std::string filename)
	{
		tinyxml2::XMLDocument* doc = new tinyxml2::XMLDocument();
		tinyxml2::XMLNode* element = doc->InsertEndChild(doc->NewElement("Parameters"));
		
		tinyxml2::XMLElement* tar = doc->NewElement("Target");
		tar->SetText(target);

		tinyxml2::XMLElement* space = doc->NewElement("ColorSpace");
		space->SetText(colorSpace);

		tinyxml2::XMLElement* minF = doc->NewElement("MinFilter");
		minF->SetText(minFilter);

		tinyxml2::XMLElement* magF = doc->NewElement("MagFilter");
		magF->SetText(magFilter);

		tinyxml2::XMLElement* uW = doc->NewElement("UWrap");
		uW->SetText(uWrap);

		tinyxml2::XMLElement* vW = doc->NewElement("VWrap");
		vW->SetText(vWrap);

		tinyxml2::XMLElement* mip = doc->NewElement("GenerateMipMaps");
		mip->SetText(generateMM);

		element->InsertEndChild(tar);
		element->InsertEndChild(space);
		element->InsertEndChild(minF);
		element->InsertEndChild(magF);
		element->InsertEndChild(uW);
		element->InsertEndChild(vW);
		element->InsertEndChild(mip);

		doc->Print();

		doc->SaveFile((filename + ".gbr").c_str());
		delete doc;
	}

	void load(std::string filename)
	{
		tinyxml2::XMLDocument doc;
		doc.LoadFile((filename + ".gbr").c_str());

		tinyxml2::XMLNode* element = doc.FirstChildElement("Parameters");
		
		tinyxml2::XMLElement* tar = element->FirstChildElement("Target");
		target = std::stoi(tar->GetText());

		tinyxml2::XMLElement* space = element->FirstChildElement("ColorSpace");
		colorSpace = std::stoi(space->GetText());

		tinyxml2::XMLElement* minF = element->FirstChildElement("MinFilter");
		minFilter = std::stoi(minF->GetText());

		tinyxml2::XMLElement* magF = element->FirstChildElement("MagFilter");
		magFilter = std::stoi(magF->GetText());

		tinyxml2::XMLElement* uW = element->FirstChildElement("UWrap");
		uWrap = std::stoi(uW->GetText());

		tinyxml2::XMLElement* vW = element->FirstChildElement("VWrap");
		vWrap = std::stoi(vW->GetText());

		tinyxml2::XMLElement* mip = element->FirstChildElement("GenerateMipMaps");
		generateMM = mip->GetText();
	}
};