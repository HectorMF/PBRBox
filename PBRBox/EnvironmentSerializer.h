#pragma once
#include <cereal/types/polymorphic.hpp>

#include "Environment.h"

template<class Archive>
void serialize(Archive& archive, Environment& env)
{
	archive(env.radiance.filePath, env.irradiance.filePath, env.specular.filePath, env.brdf.filePath);
}

CEREAL_REGISTER_TYPE(Environment);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ResourceBase, Environment);
