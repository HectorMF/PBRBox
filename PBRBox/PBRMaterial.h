#pragma once

#include <bitset>
#include "Material.h"
#include "MathUtil.h"

class PBRMaterial : public Material 
{
private:

	void checkBit(int pos, bool val)
	{
		if (m_permutation[pos] != val)
		{
			m_permutation.set(pos, val);
			m_dirty = true;
		}
	}

	bool m_dirty;

	//material information
	glm::vec4 m_albedo;
	float m_metalness;
	float m_roughness;

	Texture m_albedoMap;
	Texture m_normalMap;
	Texture m_roughnessMap;
	Texture m_metalnessMap;

	enum TextureMap
	{
		Roughness = 0,
		Normals = 1,
		Metalness = 2,
		Albedo = 3
	};
	
	std::bitset<4> m_permutation;

	//environment information
public:
	GLuint m_sampler;
	GLuint m_radianceMap;
	GLuint m_irradianceMap;


	Texture m_hammersleyPointMap;

	PBRMaterial()
	{
		shader = Shader("shaders\\Standard.vert", "shaders\\Standard.frag", false);
		shader.setVersion(330);
		shader.compile();
		m_albedo = glm::vec4(1, 1, 1, 1);
		m_roughness = .5f;
		m_metalness = 0.0;
		m_dirty = false;
	}

	void bind()
	{
		shader.bind();
		if (m_dirty)
		{
			shader.clearFlags();
			if(m_permutation[TextureMap::Albedo])
				shader.addFlag("#define USE_ALBEDO_MAP");
			if(m_permutation[TextureMap::Roughness])
				shader.addFlag("#define USE_ROUGHNESS_MAP");
			if(m_permutation[TextureMap::Metalness])
				shader.addFlag("#define USE_METALNESS_MAP");
			if(m_permutation[TextureMap::Normals])
				shader.addFlag("#define USE_NORMAL_MAP");
			shader.compile();
			m_dirty = false;
		}

		glUniform1i(glGetUniformLocation(shader.getProgram(), "uShadowMap"), 8);
		glActiveTexture(GL_TEXTURE8);
		glBindTexture(GL_TEXTURE_2D, shadowTex);


		GLint d = glGetUniformLocation(shader.getProgram(), "uRadianceMap");
		glUniform1i(d, 7);
		glActiveTexture(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_2D, m_radianceMap);

		GLint d1 = glGetUniformLocation(shader.getProgram(), "uIrradianceMap");
		glUniform1i(d1, 9);
		glActiveTexture(GL_TEXTURE9);
		glBindTexture(GL_TEXTURE_2D, m_irradianceMap);
		

		shader.setUniform("uLightPos", 0.0, 10.0, 0.0);

		if (m_permutation[TextureMap::Albedo])
			shader.setUniform("uAlbedo", m_albedoMap);
		else
			shader.setUniform("uAlbedo", sRGBToLinear(m_albedo));

		if (m_permutation[TextureMap::Roughness])
			shader.setUniform("uRoughness", m_roughnessMap);
		else
			shader.setUniform("uRoughness", m_roughness);

		if (m_permutation[TextureMap::Metalness])
			shader.setUniform("uMetalness", m_metalnessMap);
		else
			shader.setUniform("uMetalness", m_metalness);

		if (m_permutation[TextureMap::Normals])
			shader.setUniform("uNormal", m_normalMap);
	}

	void setAlbedo(const glm::vec4& albedo)
	{
		m_albedo = albedo;
	}

	void setAlbedoMap(const Texture& albedo)
	{
		m_albedoMap = albedo;
		checkBit(TextureMap::Albedo, true);
	}

	void setNormalMap(const Texture& normals)
	{
		m_normalMap = normals;
		checkBit(TextureMap::Normals, true);
	}

	void setMetalness(const float& metalness)
	{
		m_metalness = metalness;
		checkBit(TextureMap::Metalness, false);
	}

	void setMetalnessMap(const Texture& metalness)
	{
		m_metalnessMap = metalness;
		checkBit(TextureMap::Metalness, true);
	}

	void setRoughness(const float& roughness)
	{
		m_roughness = roughness;
		checkBit(TextureMap::Roughness, false);
	}

	void setRoughnessMap(const Texture& roughness)
	{
		m_roughnessMap = roughness;
		checkBit(TextureMap::Roughness, true);
	}
};