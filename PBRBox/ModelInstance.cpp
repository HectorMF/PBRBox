#include "ModelInstance.h"


ModelInstance::ModelInstance(Model* data)
{
	m_modelData = data;
}

void ModelInstance::init()
{
	m_nodes = m_modelData->copyHierarchy();
}

void ModelInstance::render(Shader& shader)
{
	renderNode(m_nodes, m_nodes->m_transform, shader);
}


void ModelInstance::renderNode(const ModelNode* node, glm::mat4 transform, Shader& shader)
{
	//transform *= node->M_transform;
	GLint modelLoc = glGetUniformLocation(shader.Program, "uModelMatrix");

	glm::mat4 trans = transform * node->m_transform;
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(trans));

	for (int i = 0; i < node->getNumMeshes(); i++)
	{
//		Mesh* mesh = m_modelData->getMeshes()[node->m_meshes[i]];
		//mesh->getMatieral()->enable();
		//mesh->getMatieral()->m_pShaderProgram->setUniformMatrix4fv("modelView", 1, false, (gb::Matrix4f)glGetModelView());
		//mesh->getMatieral()->m_pShaderProgram->setUniformMatrix4fv("projection", 1, false, (gb::Matrix4f)glGetProjection());
		//mesh->render();
		//mesh->getMatieral()->disable();
	}

	for (int i = 0; i < node->getNumChildren(); i++)
	{
		renderNode(node->m_children[i], trans, shader);
	}
}