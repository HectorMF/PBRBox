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
		m_normals.resize(getNumVertices());

	if(m_texCoords.size() < getNumVertices())
		m_texCoords.resize(getNumVertices());
	if (m_tangents.size() < getNumVertices())
		computeTangents();

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
