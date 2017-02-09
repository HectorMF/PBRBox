#pragma once
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/bitset.hpp>

#include "PBRMaterial.h"

template<class Archive>
void serialize(Archive & archive, PBRMaterial& mat)
{
	archive(mat.m_environment.filePath, mat.m_albedoMap.filePath, mat.m_ambientOcclusion.filePath, mat.m_metalnessMap.filePath, mat.m_roughnessMap.filePath, mat.m_normalMap.filePath,
		mat.m_albedo.x, mat.m_albedo.y, mat.m_albedo.z, mat.m_albedo.w, mat.m_metalness, mat.m_roughness, mat.m_bUseVertexColors, mat.m_permutation);
}

CEREAL_REGISTER_TYPE(PBRMaterial);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ResourceBase, Material);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Material, PBRMaterial);