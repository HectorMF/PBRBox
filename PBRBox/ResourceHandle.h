#pragma once
#include <string>
class ResourceManager;

template <typename T>
class ResourceHandle
{
	//friend ResourceManager;
//protected:

public:
	unsigned int uid = 0;
	std::string filePath;
	ResourceManager* manager;

	operator T*() const { return manager->get<T>(uid); }

	T* operator -> ()
	{
		if (!uid && filePath.size() > 0)
		{
			ResourceHandle<T> handle = manager->load<T>(filePath);
			uid = handle.uid;
		}

		return manager->get<T>(uid);
	}
};
