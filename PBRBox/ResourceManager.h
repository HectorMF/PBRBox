#pragma once
#include <map>
#include <typeindex>
#include <typeinfo>
#include "Loader.h"
#include "Resource.h"

class ResourceManager
{
	std::map<std::type_index, std::map<std::string, void*>> loaders;

	std::map<std::string, std::string> aliasMap;
	std::map<std::string, Resource*> resources;


public:
	ResourceManager(){}
	~ResourceManager(){}

	template<typename T>
	T* load(std::string filename)
	{
		std::type_index type = std::type_index(typeid(T));
		Loader<T>* loader = getLoader<T>(type, filename);
		if (loader == nullptr)
			printf("Loader not found\n");
		else
			printf("Loader found!\n");

		ResourceDescriptor<T>* descriptor = loader->loadDescriptor(filename);

		for (int i = 0; i < descriptor->dependencies.size(); i++)
			load(descriptor->dependencies[i]);

		return loader->load(this, filename, descriptor);
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