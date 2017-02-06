#pragma once
#include "ResourceDescriptor.h"
#include "Texture.h"

class TextureDescriptor : public ResourceDescriptor<Texture>
{
	void save(std::string filename, Texture* texture)
	{
		tinyxml2::XMLDocument* doc = new tinyxml2::XMLDocument();
		tinyxml2::XMLNode* element = doc->InsertEndChild(doc->NewElement("element"));

		tinyxml2::XMLElement* sub[3] = { doc->NewElement("sub"), doc->NewElement("sub"), doc->NewElement("sub") };
		for (int i = 0; i<3; ++i) {
			sub[i]->SetAttribute("attrib", i);
		}
		element->InsertEndChild(sub[2]);
		tinyxml2::XMLNode* comment = element->InsertFirstChild(doc->NewComment("comment"));
		comment->SetUserData((void*)2);
		element->InsertAfterChild(comment, sub[0]);
		element->InsertAfterChild(sub[0], sub[1]);
		sub[2]->InsertFirstChild(doc->NewText("& Text!"));
		doc->Print();

		// And now deletion:
		element->DeleteChild(sub[2]);
		doc->DeleteNode(comment);

		element->FirstChildElement()->SetAttribute("attrib", true);
		element->LastChildElement()->DeleteAttribute("attrib");

		XMLTest("Programmatic DOM", true, doc->FirstChildElement()->FirstChildElement()->BoolAttribute("attrib"));
		int value = 10;
		int result = doc->FirstChildElement()->LastChildElement()->QueryIntAttribute("attrib", &value);
		XMLTest("Programmatic DOM", result, (int)XML_NO_ATTRIBUTE);
		XMLTest("Programmatic DOM", value, 10);

		doc->Print();

		{
			XMLPrinter streamer;
			doc->Print(&streamer);
			printf("%s", streamer.CStr());
		}
		{
			XMLPrinter streamer(0, true);
			doc->Print(&streamer);
			XMLTest("Compact mode", "<element><sub attrib=\"1\"/><sub/></element>", streamer.CStr(), false);
		}
		doc->SaveFile("./resources/out/pretty.xml");
		doc->SaveFile("./resources/out/compact.xml", true);
		delete doc;
	}

	void load(std::string filename)
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
};