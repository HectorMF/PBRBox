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
	GLuint m_radianceMap;
	GLuint m_irradianceMap;


	Texture m_hammersleyPointMap;

	PBRMaterial()
	{
		shader = Shader("shaders\\Standard.vert", "shaders\\Standard.frag");
		m_ior = 1.4;
		m_albedo = glm::vec4(.5, .5, .5, 1);
		m_roughness = .5f;
		m_metalness = 0.0;
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
			//shader.compile();
			m_dirty = false;
		}


		GLint d = glGetUniformLocation(shader.getProgram(), "uRadianceMap");
		glUniform1i(d, 4);
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_radianceMap);

		GLint d1 = glGetUniformLocation(shader.getProgram(), "uIrradianceMap");
		glUniform1i(d1, 5);
		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_irradianceMap);

		glUniform3f(glGetUniformLocation(shader.getProgram(), "uLightPos"), 0.0, 10.0, 0.0);


		unsigned int albedoLoc = glGetUniformLocation(shader, "uAlbedo");
		unsigned int roughnessLoc = glGetUniformLocation(shader, "uRoughness");
		unsigned int metalnessLoc = glGetUniformLocation(shader, "uMetalness");
		unsigned int normalLoc = glGetUniformLocation(shader, "uNormal");
		unsigned int iorLoc = glGetUniformLocation(shader, "uIOR");

		/*glUniform1i(albedoLoc, 2);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, m_albedoMap);

		glUniform1i(metalnessLoc, 3);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, m_metalnessMap);

		glUniform1i(roughnessLoc, 4);
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, m_roughnessMap);

		glUniform1i(normalLoc, 5);
		glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, m_normalMap);*/

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
	}

	void setNormalMap(const Texture& normals)
	{
		m_normalMap = normals;
	}

	void setMetalness(const float& metalness)
	{
		m_metalness = metalness;
	}

	void setMetalnessMap(const Texture& metalness)
	{
		m_metalnessMap = metalness;
	}

	void setRoughness(const float& roughness)
	{
		m_roughness = roughness;
	}

	void setRoughnessMap(const Texture& roughness)
	{
		m_roughnessMap = roughness;
	}

	void setIOR(const float& ior)
	{
		m_ior = ior;
	}
};