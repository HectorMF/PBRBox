#pragma once

#include <vector>
#include <GL/glew.h>
#include "glm\glm.hpp"

//static dynamic
//colors
class Geometry
{
public:
	Geometry();
	~Geometry();

	void uploadToGPU();

	unsigned int getVAO() const;
	unsigned int getNumIndices() const;
	unsigned int getNumVertices() const;

	const std::vector<unsigned int>& getIndices() const;
	const std::vector<glm::vec3>& getVertices() const;
	const std::vector<glm::vec3>& getNormals() const;
	const std::vector<glm::vec2>& getTexCoords() const;

	void setIndices(std::vector<unsigned int> indices);
	void setVertices(std::vector<glm::vec3> vertices);
	void setNormals(std::vector<glm::vec3> normals);
	void setUVs(std::vector<glm::vec2> uvs);

	void addTriangle(glm::uvec3 triangle);
	void addQuad(glm::uvec4 quad);
	void addVertex(glm::vec3 vertices);
	void addNormal(glm::vec3 normal);
	void addUV(glm::vec2 uv);

	//void computeNormals();
	/*
	{
		for (int i = 0; i < m_vertPositions.size(); i++)
			m_vertNormals.push_back(gb::Vec3f(0));

		for (int j = 0; j < m_numFaces; j++)
		{
			gb::Vec3ui face = m_faces[j];

			gb::Vec3f v1 = m_vertPositions[face.x];
			gb::Vec3f v2 = m_vertPositions[face.y];
			gb::Vec3f v3 = m_vertPositions[face.z];

			gb::Vec3f u = v2 - v1;
			gb::Vec3f v = v3 - v1;

			gb::Vec3f n = Cross(u, v);

			m_vertNormals[face.x] = Normalize(m_vertNormals[face.x] + n);
			m_vertNormals[face.y] = Normalize(m_vertNormals[face.y] + n);
			m_vertNormals[face.z] = Normalize(m_vertNormals[face.z] + n);
		}
	}*/

protected:
	//Bounding Box
	bool dirtyIndices;
	bool dirtyVertices;
	bool dirtyNormals;
	bool dirtyUVs;

	/* GPU Information */
	unsigned int m_VAO, m_VBO, m_IBO;

	std::vector<unsigned int> m_indices;
	std::vector<glm::vec3> m_vertices;
	std::vector<glm::vec3> m_normals;
	std::vector<glm::vec2> m_texCoords;
};