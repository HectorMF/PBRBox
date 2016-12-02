#include "ModelLoader.h"

ModelLoader::ModelLoader() {}
ModelLoader::~ModelLoader() {}

glm::mat4 aiTransformToGLM(aiMatrix4x4t<float> &t)
{
	return glm::transpose(glm::mat4(
		t[0][0], t[0][1], t[0][2], t[0][3],
		t[1][0], t[1][1], t[1][2], t[1][3],
		t[2][0], t[2][1], t[2][2], t[2][3],
		t[3][0], t[3][1], t[3][2], t[3][3]));
}

bool ModelLoader::load(Model* model, const std::string &file)
{
	std::string fullpath = file;

	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(fullpath, aiProcess_FlipUVs | aiProcess_PreTransformVertices | aiProcessPreset_TargetRealtime_Quality);

	if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
	{
		printf("Error Loading %s : %s", file.c_str(), importer.GetErrorString());
		return false;
	}

	// Retrieve the directory path of the filepath
	//model.path = path.substr(0, path.find_last_of('/'));
	for (int i = 0; i < scene->mNumMeshes; i++)
		model->m_meshes.push_back(processMesh(scene->mMeshes[i], scene));
	// Process ASSIMP's root node recursively
//	model->m_hierarchy = scene->mRootNode->mTransformation;
	model->m_hierarchy = new ModelNode();
	model->m_hierarchy->m_transform = glm::mat4();
	processHierarchy(scene->mRootNode, model->m_hierarchy, glm::mat4());
}

void ModelLoader::processHierarchy(aiNode* node, ModelNode* targetParent, glm::mat4 accTransform)
{
	ModelNode* parent;
	glm::mat4 transform;
	// if node has meshes, create a new scene object for it
	if (node->mNumMeshes > 0)
	{
		ModelNode* mNode = new ModelNode();
		mNode->m_parent = targetParent;
		mNode->m_name = node->mName.C_Str();
		mNode->m_transform = accTransform;

		for (int i = 0; i < node->mNumMeshes; i++)
		{
			mNode->m_meshes.push_back(node->mMeshes[i]);
		}

		// the new object is the parent for all child nodes
		targetParent->addChild(mNode);
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
		processHierarchy(node->mChildren[i], parent, transform);
}

/*ModelNode* ModelLoader::processHierarchy(aiNode* node, ModelNode* parent, const aiScene* scene)
{
	ModelNode* mNode = new ModelNode();
	mNode->m_parent = parent;
	mNode->m_name = node->mName.C_Str();
	//get the transformation relative to the parent node
	aiMatrix4x4t<float> t = node->mTransformation;
	mNode->m_transform = glm::mat4(
		t.a1, t.a2, t.a3, t.a4,
		t.b1, t.b2, t.b3, t.b4,
		t.c1, t.c2, t.c3, t.c4,
		t.d1, t.d2, t.d3, t.d4);

	// Process each mesh located at the current node
	for (int i = 0; i < node->mNumMeshes; i++)
	{
		mNode->m_meshes.push_back(node->mMeshes[i]);
	}

	// After we've processed all of the meshes (if any) we then recursively process each of the children nodes
	for (int i = 0; i < node->mNumChildren; i++)
	{
		mNode->m_children.push_back(processHierarchy(node->mChildren[i], mNode, scene));
	}

	return mNode;
}*/

Geometry ModelLoader::processMesh(aiMesh* mesh, const aiScene* scene)
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
		//for (int j = 0; j < face.mNumIndices; j++)
		geometry.addTriangle(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
	}

	aiString name;
	// Process materials
	if (mesh->mMaterialIndex >= 0)
	{
		aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

		material->Get(AI_MATKEY_NAME, name);

		getMaterialOfType(textures, material, aiTextureType_DIFFUSE, "DiffuseTex");
		getMaterialOfType(textures, material, aiTextureType_SPECULAR, "SpecularTex");
		getMaterialOfType(textures, material, aiTextureType_SPECULAR, "DiffuseTex");
	}

	//m->m_boundingBox = gb::AABox3f(mins.x, mins.y, mins.z, maxes.x - mins.x, maxes.y - mins.y, maxes.z - mins.z);
	//m->m_material_name = name.C_Str();
	//for (auto tex : textures)
	//	m->m_textureData.push_back(tex.second);
	return geometry;
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