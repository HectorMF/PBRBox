#pragma once
#include "Texture.h"
#include "ResourceBase.h"
#include "ResourceHandle.h"

class Environment : public ResourceBase
{
public:

	void foo(){}

	//Blur amount;
	ResourceHandle<Texture> radiance;
	ResourceHandle<Texture> irradiance;
	ResourceHandle<Texture> specular;
	ResourceHandle<Texture> brdf;
};