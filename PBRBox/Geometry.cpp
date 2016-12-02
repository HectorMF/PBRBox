#include "Geometry.h"
#include <random>
float get_random()
{
	static std::default_random_engine e;
	static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
	return dis(e);
}
Geometry::Geometry() {}

Geometry::~Geometry()
{
	glDeleteBuffers(1, &m_IBO);
	glDeleteBuffers(1, &m_VBO);
	glDeleteVertexArrays(1, &m_VAO);
}

unsigned int Geometry::getNumTriangles() const
{
	return m_indices.size() / 3;
}

unsigned int Geometry::getNumIndices() const
{
	return m_indices.size();
}

unsigned int Geometry::getNumVertices() const
{
	return m_positions.size();
}

const std::vector<unsigned int>& Geometry::getIndices() const
{
	return m_indices;
}

const std::vector<glm::vec3>& Geometry::getPositions() const
{
	return m_positions;
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

const std::vector<glm::vec4>& Geometry::getColors() const
{
	return m_colors;
}

void Geometry::setIndices(std::vector<unsigned int> indices)
{
	m_indices = indices;
}

void Geometry::setPositions(std::vector<glm::vec3> positions)
{
	m_positions = positions;
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

void Geometry::setColors(std::vector<glm::vec4> colors)
{
	m_colors = colors;
}

void Geometry::addTriangle(unsigned int i1, unsigned int i2, unsigned int i3)
{
	m_indices.push_back(i1);
	m_indices.push_back(i2);
	m_indices.push_back(i3);
}

void Geometry::addQuad(unsigned int i1, unsigned int i2, unsigned int i3, unsigned int i4)
{
	m_indices.push_back(i1);
	m_indices.push_back(i2);
	m_indices.push_back(i3);

	m_indices.push_back(i2);
	m_indices.push_back(i3);
	m_indices.push_back(i4);
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


	std::vector<Vertex> gpuVertices;

	//if (m_normals.size() < getNumVertices())
		computeNormals();
	if (m_tangents.size() < getNumVertices())
		computeTangents();
	if(m_texCoords.size() < getNumVertices())
		m_texCoords.resize(getNumVertices());
	if (m_colors.size() < getNumVertices())
	{
		m_colors.clear();
		for (int i = 0; i < getNumVertices(); i++)
			m_colors.push_back(glm::vec4(get_random(), get_random(), get_random(),1));
		//m_colors.resize(getNumVertices());
	}
		
	for (int i = 0; i < getNumVertices(); i++)
	{
		Vertex vertex;
		vertex.position = m_positions[i];
		vertex.normal = m_normals[i];
		vertex.tangent = m_tangents[i];
		vertex.bitangent = m_bitangents[i];
		vertex.texCoord = m_texCoords[i];
		vertex.color = m_colors[i];
		gpuVertices.push_back(vertex);
	}

	glGenVertexArrays(1, &m_VAO);
	glGenBuffers(1, &m_VBO);
	glGenBuffers(1, &m_IBO);

	glBindVertexArray(m_VAO);

	glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
	glBufferData(GL_ARRAY_BUFFER, getNumVertices() * sizeof(Vertex), &gpuVertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, getNumIndices() * sizeof(GLuint), &getIndices()[0], GL_STATIC_DRAW);

	/* this is where we designate how to split the intereleaved data */
	//Positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)0);

	//Normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));

	//Tangent
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, tangent));

	//Bitangent
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, bitangent));

	//Texture Coords
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, texCoord));

	//Texture Coords
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, color));


	glBindVertexArray(0);
}

void Geometry::computeTangents()
{
	m_tangents.clear();
	m_bitangents.clear();

	m_tangents.resize(getNumVertices());
	m_bitangents.resize(getNumVertices());


	std::vector<glm::vec3> tan1;
	std::vector<glm::vec3> tan2;
	
	tan1.resize(getNumVertices());
	tan2.resize(getNumVertices());

	for (long a = 0; a <  getNumTriangles(); a+=3)
	{
		unsigned int i1 = m_indices[a + 0];
		unsigned int i2 = m_indices[a + 1];
		unsigned int i3 = m_indices[a + 2];

		// Shortcuts for vertices
		glm::vec3 & v1 = m_positions[i1];
		glm::vec3 & v2 = m_positions[i2];
		glm::vec3 & v3 = m_positions[i3];

		// Shortcuts for UVs
		glm::vec2 & w1 = m_texCoords[i1];
		glm::vec2 & w2 = m_texCoords[i2];
		glm::vec2 & w3 = m_texCoords[i3];

		float x1 = v2.x - v1.x;
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
		glm::vec3 sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
			(t2 * z1 - t1 * z2) * r);
		glm::vec3 tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
			(s1 * z2 - s2 * z1) * r);

		tan1[i1] += sdir;
		tan1[i2] += sdir;
		tan1[i3] += sdir;

		tan2[i1] += tdir;
		tan2[i2] += tdir;
		tan2[i3] += tdir;
	}

	for (long a = 0; a < getNumVertices(); a++)
	{
		glm::vec3 n = m_normals[a];
		glm::vec3 t = tan1[a];

		// Gram-Schmidt orthogonalize
		m_tangents[a] = glm::vec4(glm::normalize(t - n * glm::dot(n, t)), 1);

		// Calculate handedness
		m_tangents[a].w = (glm::dot(glm::cross(n, t), tan2[a]) < 0.0F) ? -1.0F : 1.0F;
	}

	/*int triangleOffset = 0;
	for (int i = 0; i < m_indices.size() / 3; i++)
	{
		unsigned int i1 = m_indices[triangleOffset + 0];
		unsigned int i2 = m_indices[triangleOffset + 1];
		unsigned int i3 = m_indices[triangleOffset + 2];

		// Shortcuts for vertices
		glm::vec3 & v1 = m_positions[i1];
		glm::vec3 & v2 = m_positions[i2];
		glm::vec3 & v3 = m_positions[i3];

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

	for (int v = 0; v < m_positions.size(); v++)
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
	}*/
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

		glm::vec3 v1 = m_positions[i1];
		glm::vec3 v2 = m_positions[i2];
		glm::vec3 v3 = m_positions[i3];

		glm::vec3 u = v2 - v1;
		glm::vec3 v = v3 - v1;

		glm::vec3 n = glm::cross(u, v);

		m_normals[i1] = glm::normalize(m_normals[i1] + n);
		m_normals[i2] = glm::normalize(m_normals[i2] + n);
		m_normals[i3] = glm::normalize(m_normals[i3] + n);

		triangleOffset += 3;
	}
}