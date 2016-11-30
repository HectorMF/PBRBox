/***************************************************************
*  Copyright (C) 2016 Ohio Supercomputer Center, Ohio State University
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

/*! \file
\brief creating and using a GLSL shader program, consisting of a vertex shader and fragment shader
*/

//TODO: static vs dynamic geometry on gpu 

//! A vertex handles 
struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec3 bitangent;
	glm::vec2 texCoord;
	glm::vec4 color;
};

//! This class stores and manages triangular mesh data on the GPU and CPU
class Geometry
{
public:

	//! Get the current log for this object
	//! \return a pointer to a read-only buffer containing the log


	Geometry();
	~Geometry();

	Geometry(const Geometry &geom) // copy constructor
		:m_indices(geom.m_indices),
		m_positions(geom.m_positions),
		m_normals(geom.m_normals),
		m_tangents(geom.m_tangents),
		m_bitangents(geom.m_bitangents),
		m_texCoords(geom.m_texCoords),
		m_colors(geom.m_colors),
		m_VAO(geom.m_VAO),
		m_VBO(geom.m_VBO),
		m_IBO(geom.m_IBO)
	{}

	//! Uploads the Mesh data to the GPU
	void uploadToGPU();

	//! \return the OpenGL Vertex array object handle for this geometry
	unsigned int getVAO() const;
	//! This is a helper class that manages all the materials.  When we enable, we setup all the binds
	unsigned int getNumIndices() const;
	//! This is a helper class that manages all the materials.  When we enable, we setup all the binds
	unsigned int getNumVertices() const;

	//! This is a helper class that manages all the materials.  When we enable, we setup all the binds
	const std::vector<unsigned int>& getIndices() const;
	const std::vector<glm::vec3>& getPositions() const;
	const std::vector<glm::vec3>& getNormals() const;
	const std::vector<glm::vec3>& getTangents() const;
	const std::vector<glm::vec3>& getBitangents() const;
	const std::vector<glm::vec2>& getTexCoords() const;
	const std::vector<glm::vec4>& getColors() const;

	void setIndices(std::vector<unsigned int> indices);
	void setPositions(std::vector<glm::vec3> vertices);
	void setNormals(std::vector<glm::vec3> normals);
	void setTangents(std::vector<glm::vec3> tangents);
	void setBitangents(std::vector<glm::vec3> bitangents);
	void setTexCoords(std::vector<glm::vec2> uvs);
	void setColors(std::vector<glm::vec4> colors);

	//! Adds a vertex to the mesh
	//! \return the storage index of the vertex in the mesh
	unsigned int addVertex(Vertex vertex)
	{
		m_positions.push_back(vertex.position);
		m_normals.push_back(vertex.normal);
		m_tangents.push_back(vertex.tangent);
		m_bitangents.push_back(vertex.bitangent);
		m_texCoords.push_back(vertex.texCoord);
		m_colors.push_back(vertex.color);
		return m_positions.size() - 1;
	}

	//! \return the vertex information stored at the given index
	Vertex getVertex(unsigned int index)
	{
		Vertex vertex;
		vertex.position = m_positions[index];
		vertex.normal = m_normals[index];
		vertex.tangent = m_tangents[index];
		vertex.bitangent = m_bitangents[index];
		vertex.texCoord = m_texCoords[index];
		vertex.color = m_colors[index];
		return vertex;
	}

	void addTriangle(glm::uvec3 triangle);
	void addQuad(glm::uvec4 quad);

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
	std::vector<glm::vec3> m_positions;
	std::vector<glm::vec3> m_normals;
	std::vector<glm::vec3> m_tangents;
	std::vector<glm::vec3> m_bitangents;
	std::vector<glm::vec2> m_texCoords;
	std::vector<glm::vec4> m_colors;
};