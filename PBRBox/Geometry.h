/***************************************************************
*  Copyright (C) 2013 Ohio Supercomputer Center, Ohio State University
*
* This file and its content is protected by a software license.
* You should have received a copy of this license with this file.
* If not, please contact the Ohio Supercomputer Center immediately:
* Attn: Brad Hittle Re: 1224 Kinnear Rd, Columbus, Ohio 43212
*            bhittle@osc.edu
***************************************************************/

#pragma once

#include <vector>
#include <GL/glew.h>
#include "glm\glm.hpp"

//To Do:  static vs dynamic geometry
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
	//Bounding sphere?
	bool dirtyIndices;
	bool dirtyVertices;
	bool dirtyNormals;
	bool dirtyColors;
	bool dirtyTangents;
	bool dirtyBitangents;
	bool dirtyUVs;

	/* GPU Information */
	unsigned int m_VAO, m_VBO, m_IBO;

	std::vector<unsigned int> m_indices;
	std::vector<glm::vec3> m_vertices;
	std::vector<glm::vec3> m_normals;
	std::vector<glm::vec3> m_tangents;
	std::vector<glm::vec3> m_bitangents;
	std::vector<glm::vec2> m_texCoords;
	std::vector<glm::vec4> m_colors;
};