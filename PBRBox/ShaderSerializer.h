#pragma once
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

#include "Shader.h"

template<class Archive>
void serialize(Archive & archive, Shader& shad)
{
	archive(shad.m_version, shad.autoCompile, shad.vertexPath, shad.fragmentPath, shad.flags);
}

CEREAL_REGISTER_TYPE(Shader);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ResourceBase, Shader);
