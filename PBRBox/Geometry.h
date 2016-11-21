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
#include "glm\gtc\epsilon.hpp"
#include <cmath>

//To Do: static vs dynamic geometry on gpu 
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
	const std::vector<glm::vec3>& getTangents() const;
	const std::vector<glm::vec3>& getBitangents() const;
	const std::vector<glm::vec2>& getTexCoords() const;
	const std::vector<glm::vec4>& getColors() const;

	void setIndices(std::vector<unsigned int> indices);
	void setVertices(std::vector<glm::vec3> vertices);
	void setNormals(std::vector<glm::vec3> normals);
	void setTangents(std::vector<glm::vec3> tangents);
	void setBitangents(std::vector<glm::vec3> bitangents);
	void setTexCoords(std::vector<glm::vec2> uvs);
	void setColors(std::vector<glm::vec4> colors);

	void addTriangle(glm::uvec3 triangle);
	void addQuad(glm::uvec4 quad);
	void addVertex(glm::vec3 vertices);
	void addNormal(glm::vec3 normal);
	void addUV(glm::vec2 uv);

	void computeNormals();
	void computeTangents();

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