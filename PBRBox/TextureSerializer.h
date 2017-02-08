#pragma once
#include <cereal/types/polymorphic.hpp>

#include "Texture.h"

template<class Archive>
void serialize(Archive & archive, Texture& tex)
{
	archive(tex.target, tex.colorSpace, tex.minFilter, tex.magFilter, tex.uWrap, tex.vWrap, tex.generateMipMaps);
}

CEREAL_REGISTER_TYPE(Texture);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ResourceBase, Texture);

