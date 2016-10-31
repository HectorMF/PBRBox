#pragma once
#include "Model.h"
#include "Camera.h"
#include "Shader.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
class ModelInstance
{
public: 
	ModelInstance(Model* data);
	void init();

	void render(Shader& shader);;
	ModelNode* m_nodes;
protected:
	Model* m_modelData;
	//ModelNode* m_nodes;

	ModelNode* parseHierarchy(const ModelNode* node, ModelNode* parent);
	void renderNode(const ModelNode* node, glm::mat4 transform, Shader& shader);
};