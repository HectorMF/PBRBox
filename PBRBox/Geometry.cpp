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

void Geometry::setUVs(std::vector<glm::vec2> uvs)
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
	//the VBO contains interleaved vertex data for better data locality,
	struct PackedVertex
	{
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec2 texCoord;
	};

	std::vector<PackedVertex> gpuVertices;

	for (int i = 0; i < getNumVertices(); i++)
	{
		PackedVertex vertex;
		vertex.position = getVertices()[i];
		vertex.normal = getNormals()[i];
		vertex.texCoord = getTexCoords()[i];
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
	//Texture Coords
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(PackedVertex), (GLvoid*)offsetof(PackedVertex, texCoord));

	glBindVertexArray(0);
}
