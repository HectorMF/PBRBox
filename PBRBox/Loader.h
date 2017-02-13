#pragma once
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

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

	virtual T* load(ResourceManager* resourceManager, std::string filename, T* obj) = 0;

	//virtual void save(T* obj, std::string filename);
	//virtual void unload();

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