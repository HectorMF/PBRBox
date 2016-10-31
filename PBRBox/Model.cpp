#include "Model.h"
#include "ModelLoader.h"

Model::Model() {}

Model::Model(const std::string &file) 
{
	load(file);
}

Model::~Model() { }

void Model::render(const Camera& camera)
{
	renderNode(camera, m_hierarchy, m_hierarchy->m_transform);
}


void Model::renderNode(const Camera& camera, const ModelNode* node, glm::mat4 transform)
{
	//transform *= node->M_transform;
	glm::mat4 trans = transform * node->m_transform;
	for (int i = 0; i < node->getNumMeshes(); i++)
	{
		Mesh* mesh = m_meshes[node->m_meshes[i]];

		
		//glUniformMatrix4fv(glGetUniformLocation(mesh->m_material.shader, "camera.mModel"), 1, GL_FALSE, glm::value_ptr(trans));
		mesh->transform = trans;
		//mesh->getMatieral()->enable();
		//mesh->getMatieral()->m_pShaderProgram->setUniformMatrix4fv("modelView", 1, false, (gb::Matrix4f)glGetModelView());
		//mesh->getMatieral()->m_pShaderProgram->setUniformMatrix4fv("projection", 1, false, (gb::Matrix4f)glGetProjection());
		mesh->render(camera);
		//mesh->getMatieral()->disable();
	}

	for (int i = 0; i < node->getNumChildren(); i++)
	{
		renderNode(camera, node->m_children[i], trans);
	}
}

void Model::load(const std::string &file)
{
	ModelLoader loader;
	loader.load(this, file);

	//m_AABB = gb::AABox3f(0, 0, 0, 0, 0, 0);
	//for (int i = 0; i < m_meshes.size(); i++)
	//	m_AABB.unionize(m_meshes[i]->getAABB());
}

void Model::save(const std::string &file)
{
	//Assertion(0, "Unsupported!");
}
