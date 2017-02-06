#pragma once

#include "Geometry.h"

//#include <glm/gtc/matrix_transform.hpp>
namespace Shapes
{
	Geometry renderQuad()
	{
		Geometry g;
		g.setPositions({ { 1, 1, 0 },{ -1, 1, 0 },{ 1, -1, 0 },{ -1, -1, 0 } });
		g.setIndices({ 0, 1, 2, 1, 2, 3 });
		g.setNormals({ { 0, 0, 1 },{ 0, 0, 1 },{ 0, 0, 1 },{ 0, 0, 1 } });
		g.setTexCoords({ { 1, 1 },{ 0, 1 },{ 1, 0 },{ 0, 0 } });
		return g;
	}

	Geometry plane(float width = 1, float height = 1, int widthSegments = 1, int heightSegments = 1) 
	{
		Geometry plane;

		float width_half = width / 2;
		float height_half = height / 2;

		int gridX = widthSegments;
		int gridY = heightSegments;

		int gridX1 = gridX + 1;
		int gridY1 = gridY + 1;

		float segment_width = width / gridX;
		float segment_height = height / gridY;

		for (int iy = 0; iy < gridY1; iy++) 
		{
			float y = iy * segment_height - height_half;

			for (int ix = 0; ix < gridX1; ix++) 
			{
				float x = ix * segment_width - width_half;

				Vertex v;
				v.position = glm::vec3(x, 0, -y);
				v.normal = glm::vec3(0, -1, 0);
				v.texCoord = glm::vec2(ix / gridX, 1 - (iy / gridY));
				plane.addVertex(v);
			}
		}

		for (int iy = 0; iy < gridY; iy++) 
		{
			for (int ix = 0; ix < gridX; ix++) 
			{
				int a = ix + gridX1 * iy;
				int b = ix + gridX1 * (iy + 1);
				int c = (ix + 1) + gridX1 * (iy + 1);
				int d = (ix + 1) + gridX1 * iy;

				plane.addTriangle(a, b, d);
				plane.addTriangle(b, c, d);
			}
		}

		return plane;
	}

	Geometry sphere(float radius, int segments = 32)
	{
		Geometry sphere;

		glm::mat4 identity;

		int totalZRotationSteps = 2 + segments;
		int totalYRotationSteps = 2 * totalZRotationSteps;

		for (int zRotationStep = 0; zRotationStep <= totalZRotationSteps; zRotationStep++)
		{
			float normalizedZ = zRotationStep / (float)totalZRotationSteps;
			float angleZ = (normalizedZ * M_PI);

			for (int yRotationStep = 0; yRotationStep <= totalYRotationSteps; yRotationStep++)
			{
				float normalizedY = yRotationStep / (float)totalYRotationSteps;
				float angleY = normalizedY * M_PI * 2;

				glm::mat4 matRotZ = glm::rotate(identity, -angleZ, glm::vec3(0, 0, 1));
				glm::mat4 matRotY = glm::rotate(identity, angleY, glm::vec3(0, 1, 0));

				glm::vec4 temp = matRotZ * glm::vec4(0, 1, 0, 0);
				temp = matRotY * temp;
				temp *= -radius;

				Vertex v;
				v.position = glm::vec3(temp.x, temp.y, temp.z);
				v.normal = glm::normalize(v.position);
				v.texCoord = glm::vec2(normalizedY, 1 - normalizedZ);
				sphere.addVertex(v);
			}

			if (zRotationStep > 0)
			{
				int verticesCount = sphere.getNumVertices();
				unsigned int firstIndex = verticesCount - 2 * (totalYRotationSteps + 1);
				for (; (firstIndex + totalYRotationSteps + 2) < verticesCount; firstIndex++)
				{
					sphere.addTriangle(firstIndex, firstIndex + 1, firstIndex + totalYRotationSteps + 1);
					sphere.addTriangle(firstIndex + totalYRotationSteps + 1, firstIndex + 1, firstIndex + totalYRotationSteps + 2);
				}
			}
		}
		return sphere;
	}

	Geometry cylinder()
	{

	}

	Geometry box(float width, float height, float depth)
	{
		std::vector<glm::vec3> vertices;
		std::vector<glm::vec3> normals;
		std::vector<glm::vec2> uvs;
		std::vector<unsigned int> indices;

		float w = width * .5;
		float h = height * .5;
		float d = depth * .5;

		//up face
		vertices.insert(vertices.end(), { { w, h, d }, { -w, h, d }, { w, h, -d }, { -w, h, -d } });
		indices.insert(indices.end(), { 0, 1, 2, 1, 2, 3 });
		normals.insert(normals.end(), { { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 1, 0 } });
		uvs.insert(uvs.end(), { { 1, 1 }, { 0, 1 }, { 1, 0 }, { 0, 0 } });

		//front face
		vertices.insert(vertices.end(), { { w, -h, d }, { -w, -h, d }, { w, h, d }, { -w, h, d } });
		indices.insert(indices.end(), { 4, 5, 6, 5, 6, 7 });
		normals.insert(normals.end(), { { 0, 0, 1 }, { 0, 0, 1 }, { 0, 0, 1 }, { 0, 0, 1 } });
		uvs.insert(uvs.end(), { { 1, 1 }, { 0, 1 }, { 1, 0 }, { 0, 0 } });

		//bottom face
		vertices.insert(vertices.end(), { { w, -h, -d }, { -w, -h, -d },{ w, -h, d },{ -w, -h, d } });
		indices.insert(indices.end(), { 8, 9, 10, 9, 10, 11 });
		normals.insert(normals.end(), { { 0, -1, 0 },{ 0, -1, 0 },{ 0, -1, 0 },{ 0, -1, 0 } });
		uvs.insert(uvs.end(), { { 1, 1 },{ 0, 1 },{ 1, 0 },{ 0, 0 } });

		//back face
		vertices.insert(vertices.end(), { { -w, -h, -d },{ w, -h, -d },{ -w, h, -d },{ w, h, -d } });
		indices.insert(indices.end(), { 12, 13, 14, 13, 14, 15 });
		normals.insert(normals.end(), { { 0, 0, -1 },{ 0, 0, -1 },{ 0, 0, -1 },{ 0, 0, -1 } });
		uvs.insert(uvs.end(), { { 1, 1 },{ 0, 1 },{ 1, 0 },{ 0, 0 } });

		//left face
		vertices.insert(vertices.end(), { { -w, -h, d },{ -w, -h, -d },{ -w, h, d },{ -w, h, -d } });
		indices.insert(indices.end(), { 16, 17, 18, 17, 18, 19 });
		normals.insert(normals.end(), { { -1, 0, 0 },{ -1, 0, 0 },{ -1, 0, 0 },{ -1, 0, 0 } });
		uvs.insert(uvs.end(), { { 1, 1 },{ 0, 1 },{ 1, 0 },{ 0, 0 } });

		//right face
		vertices.insert(vertices.end(), { { w, -h, -d },{ w, -h, d },{ w, h, -d },{ w, h, d } });
		indices.insert(indices.end(), { 20, 21, 22, 21, 22, 23 });
		normals.insert(normals.end(), { { 1, 0, 0 },{ 1, 0, 0 },{ 1, 0, 0 },{ 1, 0, 0 } });
		uvs.insert(uvs.end(), { { 1, 1 },{ 0, 1 },{ 1, 0 },{ 0, 0 } });

		Geometry g;
		g.setPositions(vertices);
		g.setNormals(normals);
		g.setTexCoords(uvs);
		g.setIndices(indices);
		return g;
	}

	Geometry cube(float size)
	{
		return box(size, size, size);
	}

	Geometry cone()
	{

	}

	Geometry icosahedron(float radius, float detail)
	{

	}

	Geometry skybox()
	{
		Geometry g;
	/*	g.addVertex({ -1.0f, 1.0f, -1.0f });
		g.addVertex({ -1.0f, -1.0f, -1.0f });
		g.addVertex({ 1.0f, -1.0f, -1.0f });
		g.addVertex({ 1.0f, -1.0f, -1.0f });
		g.addVertex({ 1.0f,  1.0f, -1.0f });
		g.addVertex({ -1.0f,  1.0f, -1.0f });

		g.addVertex({ -1.0f, -1.0f,  1.0f });
		g.addVertex({ -1.0f, -1.0f, -1.0f });
		g.addVertex({ -1.0f,  1.0f, -1.0f });
		g.addVertex({ -1.0f,  1.0f, -1.0f });
		g.addVertex({ -1.0f,  1.0f,  1.0f });
		g.addVertex({ -1.0f, -1.0f,  1.0f });

		g.addVertex({ 1.0f, -1.0f, -1.0f });
		g.addVertex({ 1.0f, -1.0f,  1.0f });
		g.addVertex({ 1.0f,  1.0f,  1.0f });
		g.addVertex({ 1.0f,  1.0f,  1.0f });
		g.addVertex({ 1.0f,  1.0f, -1.0f });
		g.addVertex({ 1.0f, -1.0f, -1.0f });

		g.addVertex({ -1.0f, -1.0f,  1.0f });
		g.addVertex({ -1.0f,  1.0f,  1.0f });
		g.addVertex({ 1.0f,  1.0f,  1.0f });
		g.addVertex({ 1.0f,  1.0f,  1.0f });
		g.addVertex({ 1.0f, -1.0f,  1.0f });
		g.addVertex({ -1.0f, -1.0f,  1.0f });

		g.addVertex({ -1.0f,  1.0f, -1.0f });
		g.addVertex({ 1.0f,  1.0f, -1.0f });
		g.addVertex({ 1.0f,  1.0f,  1.0f });
		g.addVertex({ 1.0f,  1.0f,  1.0f });
		g.addVertex({ -1.0f,  1.0f,  1.0f });
		g.addVertex({ -1.0f,  1.0f, -1.0f });

		g.addVertex({ -1.0f, -1.0f, -1.0f });
		g.addVertex({ -1.0f, -1.0f,  1.0f });
		g.addVertex({ 1.0f, -1.0f, -1.0f });
		g.addVertex({ 1.0f, -1.0f, -1.0f });
		g.addVertex({ -1.0f, -1.0f,  1.0f });
		g.addVertex({ 1.0f, -1.0f,  1.0f });
		*/
		//g.setVertices(skyboxVertices);
		//g.setNormals(normals);
		//g.setUVs(uvs);
		//g.setIndices(indices);
		return g;
	}
}