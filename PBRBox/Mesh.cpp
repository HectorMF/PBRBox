
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
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//GLenum drawMode = GL_TRIANGLES;
	//Wireframe mode
	/*if (displayMode == 1)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	//Point rendering mode
	if (displayMode == 2)
	{
		glPointSize(2.0f);
		drawMode = GL_POINTS;
	}*/
	
	glBindVertexArray(m_geometry.getVAO());
	glDrawElements(GL_TRIANGLES, m_geometry.getNumIndices(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}
