#include "Geometry.h"

Geometry::Geometry() {}

Geometry::~Geometry()
{
	glDeleteBuffers(1, &m_IBO);
	glDeleteBuffers(1, &m_VBO);
	glDeleteVertexArrays(1, &m_VAO);
}

unsigned int Geometry::getNumIndices() const
{
	return m_indices.size();
}

unsigned int Geometry::getNumVertices() const
{
	return m_vertices.size();
}

const std::vector<unsigned int>& Geometry::getIndices() const
{
	return m_indices;
}

const std::vector<glm::vec3>& Geometry::getVertices() const
{
	return m_vertices;
}

const std::vector<glm::vec3>& Geometry::getNormals() const
{
	return m_normals;
}

const std::vector<glm::vec3>& Geometry::getTangents() const
{
	return m_tangents;
}

const std::vector<glm::vec3>& Geometry::getBitangents() const
{
	return m_bitangents;
}

const std::vector<glm::vec2>& Geometry::getTexCoords() const
{
	return m_texCoords;
}

void Geometry::setIndices(std::vector<unsigned int> indices)
{
	m_indices = indices;
}

void Geometry::setVertices(std::vector<glm::vec3> vertices)
{
	m_vertices = vertices;
}

void Geometry::setNormals(std::vector<glm::vec3> normals)
{
	m_normals = normals;
}

void Geometry::setTangents(std::vector<glm::vec3> tangents)
{
	m_tangents = tangents;
}

void Geometry::setBitangents(std::vector<glm::vec3> bitangents)
{
	m_bitangents = bitangents;
}

void Geometry::setTexCoords(std::vector<glm::vec2> uvs)
{
	m_texCoords = uvs;
}

void Geometry::addTriangle(glm::uvec3 triangle)
{
	m_indices.push_back(triangle.x);
	m_indices.push_back(triangle.y);
	m_indices.push_back(triangle.z);
}

void Geometry::addQuad(glm::uvec4 quad)
{
	m_indices.push_back(quad.x);
	m_indices.push_back(quad.y);
	m_indices.push_back(quad.z);

	m_indices.push_back(quad.y);
	m_indices.push_back(quad.z);
	m_indices.push_back(quad.w);
}

void Geometry::addVertex(glm::vec3 vertex)
{
	m_vertices.push_back(vertex);
}

void Geometry::addNormal(glm::vec3 normal)
{
	m_normals.push_back(normal);
}

void Geometry::addUV(glm::vec2 uv)
{
	m_texCoords.push_back(uv);
}

unsigned int Geometry::getVAO() const
{
	return m_VAO;
}

void Geometry::uploadToGPU()
{
	//if (initialized) return;
	//initialized = true;
	//the VBO contains interleaved vertex data for better data locality? research seems to indicate that this could be pointless
	//To do: make sure we dont need to align this for better speed
	struct PackedVertex
	{
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec3 tangent;
		glm::vec3 bitangent;
		glm::vec2 texCoord;
	};

	std::vector<PackedVertex> gpuVertices; 

	if (m_normals.size() < getNumVertices())
		computeNormals();
	if (m_tangents.size() < getNumVertices())
		computeTangents();
	if(m_texCoords.size() < getNumVertices())
		m_texCoords.resize(getNumVertices());

	for (int i = 0; i < getNumVertices(); i++)
	{
		PackedVertex vertex;
		vertex.position = m_vertices[i];
		vertex.normal = m_normals[i];
		vertex.tangent = m_tangents[i];
		vertex.bitangent = m_bitangents[i];
		vertex.texCoord = m_texCoords[i];
		gpuVertices.push_back(vertex);
	}

	glGenVertexArrays(1, &m_VAO);
	glGenBuffers(1, &m_VBO);
	glGenBuffers(1, &m_IBO);

	glBindVertexArray(m_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
	glBufferData(GL_ARRAY_BUFFER, getNumVertices() * sizeof(PackedVertex), &gpuVertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, getNumIndices() * sizeof(GLuint), &getIndices()[0], GL_STATIC_DRAW);

	/* this is where we designate how to split the intereleaved data */
	//Positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PackedVertex), (GLvoid*)0);

	//Normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(PackedVertex), (GLvoid*)offsetof(PackedVertex, normal));

	//Tangent
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(PackedVertex), (GLvoid*)offsetof(PackedVertex, tangent));

	//Bitangent
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(PackedVertex), (GLvoid*)offsetof(PackedVertex, bitangent));

	//Texture Coords
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(PackedVertex), (GLvoid*)offsetof(PackedVertex, texCoord));


	glBindVertexArray(0);
}

void Geometry::computeTangents()
{
	m_tangents.clear();
	m_bitangents.clear();

	m_tangents.resize(getNumVertices());
	m_bitangents.resize(getNumVertices());

	int triangleOffset = 0;
	for (int i = 0; i < m_indices.size() / 3; i++)
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

		if (glm::dot(glm::cross(n, t), b) < 0.0f)
			t = t * -1.0f;

		m_tangents[v] = t;
		m_bitangents[v] = b;
	}
}

void Geometry::computeNormals()
{
	m_normals.clear();
	m_normals.resize(getNumVertices());

	//for (int i = 0; i < m_vertices.size(); i++)
	//	m_normals.push_back(glm::vec3(0));
	unsigned int triangleOffset = 0;
	for (int i = 0; i < m_indices.size() / 3; i++)
	{
		unsigned int i1 = m_indices[triangleOffset + 0];
		unsigned int i2 = m_indices[triangleOffset + 1];
		unsigned int i3 = m_indices[triangleOffset + 2];

		glm::vec3 v1 = m_vertices[i1];
		glm::vec3 v2 = m_vertices[i2];
		glm::vec3 v3 = m_vertices[i3];

		glm::vec3 u = v2 - v1;
		glm::vec3 v = v3 - v1;

		glm::vec3 n = glm::cross(u, v);

		m_normals[i1] = glm::normalize(m_normals[i1] + n);
		m_normals[i2] = glm::normalize(m_normals[i2] + n);
		m_normals[i3] = glm::normalize(m_normals[i3] + n);

		triangleOffset += 3;
	}
}