#pragma once
#include <map>
#include <typeindex>
#include <typeinfo>
#include "Loader.h"

#include <cereal/archives/xml.hpp>
#include <fstream>

#include "ResourceSerialization.h"
#include "ResourceBase.h"
#include "ResourceHandle.h"

class ResourceManager
{
	static unsigned int UUIDGEN;
	std::map<std::type_index, std::map<std::string, void*>> loaders;

	std::map<std::string, unsigned int> fileNameMap;

	std::map<std::string, unsigned int> aliasMap;
	std::map<unsigned int, std::shared_ptr<ResourceBase>> resources;

public:
	ResourceManager()
	{

	}
	~ResourceManager()
	{
		//cleanup loaders
		for (auto const& entry1 : loaders)
		{
			for (auto const& entry2 : entry1.second)
			{
				delete entry2.second;
			}
		}
	}

	template<typename T>
	T* get(unsigned int id)
	{
		return static_cast<T*>(resources[id].get());
	}

	template<typename T>
	ResourceHandle<T> load(std::string filename)
	{
		if (fileNameMap.find(filename) != fileNameMap.end())
		{
			ResourceHandle<T> handle;
			handle.filePath = filename;
			handle.uid = fileNameMap[filename];
			handle.manager = this;
			return handle;
		}

		std::type_index type = std::type_index(typeid(T));
		Loader<T>* loader = getLoader<T>(type, filename);
		if (loader == nullptr)
			printf("Loader not found\n");
		else
			printf("Loader found!\n");

		std::shared_ptr<T> resource;

		std::ifstream file(filename + ".xml");
		if (file)
		{
			cereal::XMLInputArchive archive(file);	
			archive(resource);
		}
		else
		{
			resource = std::make_shared<T>();
		}

		loader->load(this, filename, resource.get());

		//resource->filePath = filename;
		//resource->uniqueID = UUIDGEN;
	
		//resource->manager = this;

		resources[UUIDGEN] = resource;
		fileNameMap[filename] = UUIDGEN;

		ResourceHandle<T> handle;
		handle.filePath = filename;
		handle.uid = UUIDGEN;
		handle.manager = this;

		UUIDGEN++;
		//for (int i = 0; i < descriptor->dependencies.size(); i++)
			//load(descriptor->dependencies[i]);

		return handle;// loader->load(this, filename, descriptor);
	}

	template<typename T>
	Loader<T>* getLoader(std::type_index type, std::string filename = "") 
	{
		std::map<std::string, void*> typeLoaders = loaders[type]; 

		if (typeLoaders.empty() || loaders.size() < 1) return nullptr;
		if (filename == "") return static_cast<Loader<T>*>(typeLoaders[""]);

		Loader<T>* result = nullptr;
		int l = 0;

		for (auto const& entry : typeLoaders) {
			Loader<T>* loader = static_cast<Loader<T>*>(entry.second);

			if (entry.first.size() > l && loader->hasValidExtension(filename))
			{
				result = loader;
				l = entry.first.length();
			}
		}
		return result;
	}

	template<typename T>
	void addLoader(Loader<T>* loader)
	{
		for(std::string extension : loader->getFileExtensions())
			loaders[loader->getResourceType()][extension] = loader;
	}
};