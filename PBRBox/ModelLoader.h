#pragma once
#include "Loader.h"
#include "Model.h"
#include "stb_image.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>   
#include <assimp/postprocess.h>

#include "PBRMaterial.h"

class ModelLoader : public Loader<Model>
{
public:

	ModelLoader()
	{
		Assimp::Importer importer;
		std::string extensionList;
		importer.GetExtensionList(extensionList);
		extensionList.erase(std::remove(extensionList.begin(), extensionList.end(), '*'), extensionList.end());

		std::istringstream ss(extensionList);
		std::string token;

		while (std::getline(ss, token, ';')) {
			extensions.push_back(token);
			std::cout << token << '\n';
		}
	}

	glm::mat4 aiTransformToGLM(aiMatrix4x4t<float> &t)
	{
		return glm::transpose(glm::mat4(
			t[0][0], t[0][1], t[0][2], t[0][3],
			t[1][0], t[1][1], t[1][2], t[1][3],
			t[2][0], t[2][1], t[2][2], t[2][3],
			t[3][0], t[3][1], t[3][2], t[3][3]));
	}

	void processHierarchy(Model* model, aiNode* node, SceneNode* targetParent, glm::mat4 accTransform)
	{
		SceneNode* parent;
		glm::mat4 transform;
		// if node has meshes, create a new scene object for it
		if (node->mNumMeshes > 0)
		{
			SceneNode* mNode = new SceneNode();
			mNode->m_parent = targetParent;
			mNode->m_name = node->mName.C_Str();
			//mNode->setTran = accTransform;

			printf("NUM MESHES ON NODE %d\n", node->mNumMeshes);
			for (int i = 0; i < node->mNumMeshes; i++)
			{
				mNode->mesh = model->m_meshes[node->mMeshes[i]];
				//mNode->add();
			}

			// the new object is the parent for all child nodes
			targetParent->add(mNode);
			parent = mNode;
			transform = glm::mat4();
		}
		else
		{
			// if no meshes, skip the node, but keep its transformation
			parent = targetParent;
			transform = accTransform *aiTransformToGLM(node->mTransformation);
		}
		// continue for all child nodes
		for (int i = 0; i < node->mNumChildren; i++)
			processHierarchy(model, node->mChildren[i], parent, transform);
	}

	void ModelLoader::getMaterialOfType(std::map<aiTextureType, std::tuple<std::string, std::string>>& textures, aiMaterial* mat, aiTextureType type, std::string typeName)
	{
		if (mat->GetTextureCount(type))
		{
			aiString str;
			mat->GetTexture(type, 0, &str);

			std::tuple<std::string, std::string> texture(typeName, str.C_Str());
			textures[type] = texture;
		}
	}

	Mesh* processMesh(ResourceManager* resourceManager, aiMesh* mesh, const aiScene* scene)
	{
		Geometry geometry;

		glm::vec3 mins(FLT_MAX, FLT_MAX, FLT_MAX);
		glm::vec3 maxes(FLT_MIN, FLT_MIN, FLT_MIN);

		std::map<aiTextureType, std::tuple<std::string, std::string>> textures;

		// Walk through each of the mesh's vertices
		for (int i = 0; i < mesh->mNumVertices; i++)
		{
			Vertex vertex;
			vertex.position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
			vertex.normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
			vertex.tangent = glm::vec3(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z);
			vertex.bitangent = glm::vec3(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z);
			vertex.texCoord = glm::vec2(0);

			//a complex mesh can can have multiple tex coords per vertex, we don't handle that
			if (mesh->mTextureCoords[0]) // Does the mesh contain texture coordinates?
				vertex.texCoord = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);

			mins = glm::min(vertex.position, mins);
			maxes = glm::max(vertex.position, maxes);

			geometry.addVertex(vertex);
		}

		// Now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
		for (int i = 0; i < mesh->mNumFaces; i++)
		{
			aiFace face = mesh->mFaces[i];
			// Retrieve all indices of the face and store them in the indices vector
			// for (int j = 0; j < face.mNumIndices; j++)
			geometry.addTriangle(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
		}

		PBRMaterial* pbr = new PBRMaterial();
		pbr->m_environment.manager = resourceManager;

		aiString name;
		// Process materials
		if (mesh->mMaterialIndex >= 0)
		{
			aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

			material->Get(AI_MATKEY_NAME, name);

			aiString str;
			material->GetTexture(aiTextureType_DIFFUSE, 0, &str);
			
			if (str.length != 0)
				pbr->setAlbedoMap(resourceManager->load<Texture>(str.C_Str()));
		

			GLenum err = glGetError();
			if (err != GL_NO_ERROR)
			{
				printf("OpenGL error %08x\n", err);
				abort();
			}

			material->GetTexture(aiTextureType_SHININESS, 0, &str);

			if (str.length != 0)
				pbr->setRoughnessMap(resourceManager->load<Texture>(str.C_Str()));
			

			err = glGetError();
			if (err != GL_NO_ERROR)
			{
				printf("OpenGL error %08x\n", err);
				abort();

			}
			material->GetTexture(aiTextureType_NORMALS, 0, &str);

			if (str.length != 0)
				pbr->setNormalMap(resourceManager->load<Texture>(str.C_Str()));

			material->GetTexture(aiTextureType_AMBIENT, 0, &str);

			if (str.length != 0)
				pbr->setMetalnessMap(resourceManager->load<Texture>(str.C_Str()));


			err = glGetError();
			if (err != GL_NO_ERROR)
			{
				printf("OpenGL error %08x\n", err);
				abort();
			}
		}

		//m->m_boundingBox = gb::AABox3f(mins.x, mins.y, mins.z, maxes.x - mins.x, maxes.y - mins.y, maxes.z - mins.z);
		//m->m_material_name = name.C_Str();
		//for (auto tex : textures)
		//	m->m_textureData.push_back(tex.second);

		return new Mesh(geometry, pbr);
	}

	Model* load(ResourceManager* resourceManager, std::string filename, Model* model) override
	{
		Assimp::Importer importer;

		const aiScene* scene = importer.ReadFile(filename, aiProcess_FlipUVs | aiProcess_PreTransformVertices | aiProcessPreset_TargetRealtime_Quality);

		if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
		{
			printf("Error Loading %s : %s", filename.c_str(), importer.GetErrorString());
			return false;
		}
		// Retrieve the directory path of the filepath
		//model.path = path.substr(0, path.find_last_of('/'));
		for (int i = 0; i < scene->mNumMeshes; i++)
			model->m_meshes.push_back(processMesh(resourceManager, scene->mMeshes[i], scene));
		// Process ASSIMP's root node recursively
		//	model->m_hierarchy = scene->mRootNode->mTransformation;
		model->m_hierarchy = new SceneNode();

		processHierarchy(model, scene->mRootNode, model->m_hierarchy, glm::mat4());
		return model;
	}
};