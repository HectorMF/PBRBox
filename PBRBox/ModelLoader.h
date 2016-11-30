
#pragma once

#include "Model.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>   
#include <assimp/postprocess.h>


class ModelLoader
{
public:
	ModelLoader();
	virtual ~ModelLoader();

	virtual bool load(Model* model, const std::string &file);

protected:
	void ModelLoader::processHierarchy(aiNode* node, ModelNode* targetParent, glm::mat4 accTransform);
	//ModelNode* ModelLoader::processHierarchy(aiNode* node, ModelNode* parent, const aiScene* scene);

	Geometry processMesh(aiMesh* mesh, const aiScene* scene);

	void ModelLoader::getMaterialOfType(std::map<aiTextureType, std::tuple<std::string, std::string>>& textures, aiMaterial* mat, aiTextureType type, std::string typeName);
};