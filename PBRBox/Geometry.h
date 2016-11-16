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

	//http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
	void computeTangents()
	{
		std::vector<glm::vec3> temp;
		if (m_tangents.size() > 0)
		{
			temp = m_bitangents;
		}
		m_tangents.clear();
		m_bitangents.clear();

		m_tangents.resize(getNumVertices());
		m_bitangents.resize(getNumVertices());
		int triangleOffset = 0;
		for (int i = 0; i < m_indices.size()/3; i ++)
		{
			unsigned int i1 = m_indices[triangleOffset + 0];
			unsigned int i2 = m_indices[triangleOffset + 1];
			unsigned int i3 = m_indices[triangleOffset + 2];

			// Shortcuts for vertices
			glm::vec3 & v1 = m_vertices[i1];
			glm::vec3 & v2 = m_vertices[i2];
			glm::vec3 & v3 = m_vertices[i3];

			// Shortcuts for UVs
			glm::vec2 & w1 = m_texCoords[i1];
			glm::vec2 & w2 = m_texCoords[i2];
			glm::vec2 & w3 = m_texCoords[i3];





			// position differences p1->p2 and p1->p3
			glm::vec3 v = v2 - v1;
			glm::vec3 w = v3 - v1;

			// texture offset p1->p2 and p1->p3
			float sx = w2.x - w1.x, sy = w2.y - w1.y;
			float tx = w3.x - w1.x, ty = w3.y - w1.y;
			float dirCorrection = (tx * sy - ty * sx) < 0.0f ? -1.0f : 1.0f;
			// when t1, t2, t3 in same position in UV space, just use default UV direction.
			if (0 == sx && 0 == sy && 0 == tx && 0 == ty) {
				sx = 0.0; sy = 1.0;
				tx = 1.0; ty = 0.0;
			}

			// tangent points in the direction where to positive X axis of the texture coord's would point in model space
			// bitangent's points along the positive Y axis of the texture coord's, respectively
			glm::vec3 tangent, bitangent;
			tangent.x = (w.x * sy - v.x * ty) * dirCorrection;
			tangent.y = (w.y * sy - v.y * ty) * dirCorrection;
			tangent.z = (w.z * sy - v.z * ty) * dirCorrection;

			bitangent.x = (w.x * sx - v.x * tx) * dirCorrection;
			bitangent.y = (w.y * sx - v.y * tx) * dirCorrection;
			bitangent.z = (w.z * sx - v.z * tx) * dirCorrection;

			m_tangents[i1] += tangent;
			m_tangents[i2] += tangent;
			m_tangents[i3] += tangent;

			m_bitangents[i1] += bitangent;
			m_bitangents[i2] += bitangent;
			m_bitangents[i3] += bitangent;






























			/*float x1 = v2.x - v1.x;
			float x2 = v3.x - v1.x;
			float y1 = v2.y - v1.y;
			float y2 = v3.y - v1.y;
			float z1 = v2.z - v1.z;
			float z2 = v3.z - v1.z;

			float s1 = w2.x - w1.x;
			float s2 = w3.x - w1.x;
			float t1 = w2.y - w1.y;
			float t2 = w3.y - w1.y;

			float r = 1.0F / (s1 * t2 - s2 * t1);
			glm::vec3 sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
			glm::vec3 tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

			m_tangents[i1] += sdir;
			m_tangents[i2] += sdir;
			m_tangents[i3] += sdir;

			m_bitangents[i1] += tdir;
			m_bitangents[i2] += tdir;
			m_bitangents[i3] += tdir;*/

			triangleOffset += 3;
		}

		for (int v = 0; v < m_vertices.size(); v++)
		{
			glm::vec3 n = m_normals[v];
			glm::vec3 t = m_tangents[v];
			glm::vec3 b = m_bitangents[v];

			// Gram-Schmidt orthogonalize
			t = glm::normalize(t - n * glm::dot(n, t));
			b = glm::normalize(b - n * glm::dot(n, b));

			bool invalid_tangent = std::isinf(t.x) || std::isinf(t.y) || std::isinf(t.z);
			bool invalid_bitangent = std::isinf(b.x) || std::isinf(b.y) || std::isinf(b.z);
			if (invalid_tangent != invalid_bitangent) {
				if (invalid_tangent) {
					t = glm::normalize(glm::cross(n, b));
				}
				else {
					b = glm::normalize(glm::cross(t, n));
				}
			}

			if (glm::dot(glm::cross(n, t), b) < 0.0f) {
				t = t * -1.0f;
			}

			float test = glm::dot(t, b);


			if(test > .1)
				printf("Dot %f\n",test);
			if (temp.size() > 0)
			{
			
				bool t = glm::all(glm::lessThan(glm::abs(m_bitangents[v] - temp[v]), glm::vec3(.02, .02, .02)));
				//if (!t)
				//	printf("not equal <%f, %f, %f> <%f, %f, %f>\n", m_bitangents[v].x, m_bitangents[v].y, m_bitangents[v].z, temp[v].x, temp[v].y, temp[v].z);
				//else
				//	printf("equal\n");
			}
			m_tangents[v] = t;
			m_bitangents[v] = b;
			// Calculate handedness
			//m_tangents[v]. = (Dot(Cross(n, t), tan2[a]) < 0.0F) ? -1.0F : 1.0F;
		}

	}

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