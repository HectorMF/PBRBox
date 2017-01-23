#pragma once

#include <glm\glm.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//class EnvironmentTools
//{
//public:
	// http://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
	float GGX(float NdotV, float a)
	{
		float k = a / 2;
		return NdotV / (NdotV * (1.0f - k) + k);
	}

	// http://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
	float G_Smith(float a, float nDotV, float nDotL)
	{
		return GGX(nDotL, a * a) * GGX(nDotV, a * a);
	}

	float radicalInverse_VdC(unsigned int bits) {
		bits = (bits << 16u) | (bits >> 16u);
		bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
		bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
		bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
		bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
		return float(bits) * 2.3283064365386963e-10; // / 0x100000000
	}

	// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
	glm::vec2 Hammersley(unsigned int i, unsigned int N) {
		return glm::vec2(float(i) / float(N), radicalInverse_VdC(i));
	}

	glm::vec3 ImportanceSampleGGX(glm::vec2 Xi, float Roughness, glm::vec3 N)
	{
		float a = Roughness * Roughness;
		float Phi = 2 * 3.1415926535f * Xi.x;
		float CosTheta = sqrt((1 - Xi.y) / (1 + (a*a - 1) * Xi.y));
		float SinTheta = sqrt(1 - CosTheta * CosTheta);
		glm::vec3 H;
		H.x = SinTheta * cos(Phi);
		H.y = SinTheta * sin(Phi);
		H.z = CosTheta;
		glm::vec3 UpVector = abs(N.z) < 0.999 ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
		glm::vec3 TangentX = normalize(cross(UpVector, N));
		glm::vec3 TangentY = cross(N, TangentX);
		// Tangent to world space
		return TangentX * H.x + TangentY * H.y + N * H.z;
	}

	glm::vec2 IntegrateBRDF(float Roughness, float NoV)
	{
		glm::vec3 V;
		V.x = sqrt(1.0f - NoV * NoV); // sin
		V.y = 0;
		V.z = NoV; // cos
		float A = 0;
		float B = 0;

		glm::vec3 N(0, 0, 1);
		const unsigned int NumSamples = 1024;
		for (unsigned int i = 0; i < NumSamples; i++)
		{
			glm::vec2 Xi = Hammersley(i, NumSamples);
			glm::vec3 H = ImportanceSampleGGX(Xi, Roughness, N);
			glm::vec3 L = 2 * dot(V, H) * H - V;
			float NoL = glm::clamp(L.z, 0.0f, 1.0f);
			float NoH = glm::clamp(H.z, 0.0f, 1.0f);
			float VoH = glm::clamp(dot(V, H), 0.0f, 1.0f);
			if (NoL > 0)
			{
				float G = G_Smith(Roughness, NoV, NoL);
				float G_Vis = G * VoH / (NoH * NoV);
				float Fc = pow(1 - VoH, 5);
				A += (1 - Fc) * G_Vis;
				B += Fc * G_Vis;
			}
		}
		return glm::vec2(A / (float)NumSamples, B / (float)NumSamples);
	}
	GLuint computeBRDFLUT(int size)
	{
		float* imageData = new float[size * size * 2];
		float step = 1.0f / (float)size;
		int offset = 0;
		for (int y = 0; y < size; y++)
		{
			for (int x = 0; x < size; x++)
			{
				glm::vec2 d = IntegrateBRDF(step * (y + .5) , step * (x +.5));
				imageData[offset + 0] = d.x;
				imageData[offset + 1] = d.y;
				offset += 2;
			}
		}

		//stbi_write_png("TestBRDFOut.png", size, size, 4, imageData, 0);
		
		GLuint TextureName = 0;
		glGenTextures(1, &TextureName);
		glBindTexture(GL_TEXTURE_2D, TextureName);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16, size, size, 0, GL_RG, GL_FLOAT, &imageData[0]);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		delete imageData;
		return TextureName;
	}


	/*glm::vec3 PrefilterEnvMap(float Roughness, glm::vec3 R)
	{
		glm::vec3 N = R;
		glm::vec3 V = R;
		glm::vec3 prefilteredColor = glm::vec3(0);
		float totalWeight = 0;

		const unsigned int NumSamples = 1024;
		for (unsigned int i = 0; i < NumSamples; i++)
		{
			glm::vec2 Xi = Hammersley(i, NumSamples);
			glm::vec3 H = ImportanceSampleGGX(Xi, Roughness, N);
			glm::vec3 L = 2 * dot(V, H) * H - V;
			float NoL = glm::clamp(dot(N, L), 0.0f, 1.0f);
			if (NoL > 0)
			{
				prefilteredColor += EnvMap.SampleLevel(EnvMapSampler, L, 0).rgb * NoL;
				totalWeight += NoL;
			}
		}
		return prefilteredColor / totalWeight;
	}*/
//};



