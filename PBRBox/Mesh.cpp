
#include "Mesh.h"
#include "glm\glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

Mesh::Mesh(Geometry& geometry, Material& material)
{
	m_material = material;
//	m_material.uploadToGPU();
	m_geometry = geometry;
	m_geometry.uploadToGPU();
}

Mesh::~Mesh()
{ }

void Mesh::render(const Camera& camera)
{
	m_material.Bind();

	glm::mat4 model = transform;
	glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.view, camera.up);
	glm::mat4 projection = glm::perspective(45.0f, (float)camera.resolution.x / (float)camera.resolution.y, 0.1f, 1000.0f);

	glm::mat4 normal = glm::transpose(glm::inverse(view * model));
	glm::mat4 invProjection = glm::inverse(projection);
	glm::mat4 transView = glm::transpose(view);

	GLint m = glGetUniformLocation(m_material.shader, "camera.mModel");
	GLint v = glGetUniformLocation(m_material.shader, "camera.mView");
	GLint p = glGetUniformLocation(m_material.shader, "camera.mProjection");
	GLint n = glGetUniformLocation(m_material.shader, "camera.mNormal");
	GLint i = glGetUniformLocation(m_material.shader, "camera.mInvView");

	glUniformMatrix4fv(m, 1, GL_FALSE, glm::value_ptr(model));
	glUniformMatrix4fv(v, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(p, 1, GL_FALSE, glm::value_ptr(projection));
	glUniformMatrix4fv(n, 1, GL_FALSE, glm::value_ptr(normal));
	glUniformMatrix4fv(i, 1, GL_FALSE, glm::value_ptr(glm::inverse(view)));
	
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	/*GLenum drawMode = GL_TRIANGLES;
	//Wireframe mode
	if (displayMode == 1)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	//Point rendering mode
	if (displayMode == 2)
	{
		glPointSize(2.0f);
		drawMode = GL_POINTS;
	}
	*/
	glBindVertexArray(m_geometry.getVAO());
	glDrawElements(GL_TRIANGLES, m_geometry.getNumIndices(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	m_material.Unbind();
}
