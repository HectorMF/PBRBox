#pragma once

#include "Loader.h"
#include "Material.h"

class MaterialLoader : public Loader<Material>
{
public:
	MaterialLoader()
	{
		extensions.push_back(".gbmat");
	}

	Material* load(std::string filename) override
	{

		printf("LOADED Material!!!!\n");
		return nullptr;
	}
};