#pragma once
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

#include "Model.h"

template<class Archive>
void serialize(Archive & archive, Model& shad)
{

}

CEREAL_REGISTER_TYPE(Model);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ResourceBase, Model);
