
#include "Mesh.h"
#include "glm\glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

Mesh::Mesh(Geometry& geometry, Material* material)
{
	m_material = material;
//	m_material.uploadToGPU();
	m_geometry = geometry;
	m_geometry.uploadToGPU();
}

Mesh::~Mesh()
{ }

void Mesh::render()
{
	glBindVertexArray(m_geometry.getVAO());
	glDrawElements(GL_TRIANGLES, m_geometry.getNumIndices(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}
