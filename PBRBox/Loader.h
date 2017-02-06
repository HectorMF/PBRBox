#pragma once
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>
#include "ResourceDescriptor.h"
#include "tinyxml2.h"

class ResourceManager;

template<class T>
class Loader
{

protected:
	std::type_index type;
	std::vector<std::string> extensions;

public:
	inline bool ends_with(std::string const & value, std::string const & ending)
	{
		if (ending.size() > value.size()) return false;
		return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
	}

	Loader() : type(std::type_index(typeid(T)))
	{
	}

	virtual ResourceDescriptor<T>* loadDescriptor(std::string filename) = 0;

	virtual T* load(ResourceManager* resourceManager, std::string filename, ResourceDescriptor<T>* descriptor) = 0;

	std::type_index getResourceType()
	{
		return type;
	}

	std::vector<std::string> getFileExtensions()
	{
		return extensions;
	}

	bool hasValidExtension(std::string filename)
	{
		for (int i = 0; i < extensions.size(); i++)
		{
			if (ends_with(filename, extensions[i]))
				return true;
		}
		return false;
	}



	//virtual T Load(std::string filename) = 0;
};