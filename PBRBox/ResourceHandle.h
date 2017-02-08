#pragma once

#include <string>
class ResourceManager;

template <typename T>
class ResourceHandle
{
public:
	unsigned int uid;
	std::string filePath;
	ResourceManager* manager;


	T* operator -> ()
	{
		return manager->get<T>(uid);
	}
};
