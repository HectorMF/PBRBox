#pragma once
#include "Texture.h"

class Environment
{
public:
	
	//Blur amount;
	Texture* radiance;
	Texture* irradiance;
	Texture* specular;
};