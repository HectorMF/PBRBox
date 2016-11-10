#pragma once

#include "Material.h"
#include "MathUtil.h"

class PBRMaterial : public Material 
{
	bool m_dirty;

	//material information
	glm::vec4 m_albedo;
	float m_metalness;
	float m_roughness;
	float m_ior;

	Texture m_albedoMap;
	Texture m_normalMap;
	Texture m_roughnessMap;
	Texture m_metalnessMap;

	bool hasAlbedoMap;
	bool hasNormalMap;
	bool hasRoughnessMap;
	bool hasMetalnessMap;

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
		m_ior = 1.4;
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
			if(hasAlbedoMap)
				shader.addFlag("#define USE_ALBEDO_MAP");
			if(hasRoughnessMap)
				shader.addFlag("#define USE_ROUGHNESS_MAP");
			if(hasMetalnessMap)
				shader.addFlag("#define USE_METALNESS_MAP");
			if(hasNormalMap)
				shader.addFlag("#define USE_NORMAL_MAP");
			shader.compile();
			m_dirty = false;
		}

		glUniform1i(glGetUniformLocation(shader.getProgram(), "uShadowMap"), 1);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, shadowTex);

		GLint d = glGetUniformLocation(shader.getProgram(), "uRadianceMap");
		glUniform1i(d, 7);
		glActiveTexture(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_2D, m_radianceMap);

		GLint d1 = glGetUniformLocation(shader.getProgram(), "uIrradianceMap");
		glUniform1i(d1, 8);
		glActiveTexture(GL_TEXTURE8);
		glBindTexture(GL_TEXTURE_2D, m_irradianceMap);
		

		glUniform3f(glGetUniformLocation(shader.getProgram(), "uLightPos"), 0.0, 10.0, 0.0);


		unsigned int albedoLoc = glGetUniformLocation(shader, "uAlbedo");
		unsigned int roughnessLoc = glGetUniformLocation(shader, "uRoughness");
		unsigned int metalnessLoc = glGetUniformLocation(shader, "uMetalness");
		unsigned int normalLoc = glGetUniformLocation(shader, "uNormal");
		unsigned int iorLoc = glGetUniformLocation(shader, "uIOR");

		if (hasAlbedoMap)
		{
			glUniform1i(albedoLoc, 2);
			glActiveTexture(GL_TEXTURE2);
			glBindTexture(GL_TEXTURE_2D, m_albedoMap);
		}

		if (hasRoughnessMap)
		{
			glUniform1i(roughnessLoc, 4);
			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_2D, m_roughnessMap);
		}

		if (hasMetalnessMap)
		{
			glUniform1i(metalnessLoc, 3);
			glActiveTexture(GL_TEXTURE3);
			glBindTexture(GL_TEXTURE_2D, m_metalnessMap);
		}

		if (hasNormalMap)
		{
			glUniform1i(normalLoc, 5);
			glActiveTexture(GL_TEXTURE5);
			glBindTexture(GL_TEXTURE_2D, m_normalMap);

		}

	

	

	

	
		glUniform4fv(albedoLoc, 1, glm::value_ptr(sRGBToLinear(m_albedo)));
		glUniform1f(roughnessLoc, m_roughness);
		glUniform1f(metalnessLoc, m_metalness);
		glUniform1f(iorLoc, m_ior);
	}

	void setAlbedo(const glm::vec4& albedo)
	{
		m_albedo = albedo;
	}

	void setAlbedoMap(const Texture& albedo)
	{
		m_albedoMap = albedo;
		hasAlbedoMap = true;
		m_dirty = true;
	}

	void setNormalMap(const Texture& normals)
	{
		m_normalMap = normals;
		hasNormalMap = true;
		m_dirty = true;
	}

	void setMetalness(const float& metalness)
	{
		m_metalness = metalness;
	}

	void setMetalnessMap(const Texture& metalness)
	{
		m_metalnessMap = metalness;
		hasMetalnessMap = true;
		m_dirty = true;
	}

	void setRoughness(const float& roughness)
	{
		m_roughness = roughness;
	}

	void setRoughnessMap(const Texture& roughness)
	{
		m_roughnessMap = roughness;
		hasRoughnessMap = true;
		m_dirty = true;
	}

	void setIOR(const float& ior)
	{
		m_ior = ior;
	}
};