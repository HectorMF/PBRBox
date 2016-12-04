#pragma once
#pragma once

#include <bitset>
#include "Material.h"
#include "MathUtil.h"

class UnlitMaterial : public Material
{
private:
	bool m_dirty;
	bool m_bUseVertexColors;
	bool m_bUseDiffuseMap;

	glm::vec4 m_diffuseColor;
	Texture m_diffuseMap;

public:

	UnlitMaterial()
	{
		shader = Shader("shaders\\Unlit.vert", "shaders\\Unlit.frag", false);
		shader.setVersion(330);
		m_dirty = true;
		m_bUseVertexColors = false;
		m_bUseDiffuseMap = false;
	}

	void bind()
	{
		if (m_dirty)
		{
			shader.clearFlags();
			if (m_bUseDiffuseMap)
				shader.addFlag("#define USE_DIFFUSE_MAP");
			if (m_bUseVertexColors)
				shader.addFlag("#define USE_VERTEX_COLORS");
			shader.compile();
			m_dirty = false;
		}

		shader.bind();

		if (m_bUseDiffuseMap)
			shader.setUniform("uDiffuse", m_diffuseMap);
		else
			shader.setUniform("uDiffuse", sRGBToLinear(m_diffuseColor));
	}

	void setDiffuseColor(const glm::vec4& diffuse)
	{
		m_diffuseColor = diffuse;
		m_dirty = true;
	}

	void setDiffuseMap(const Texture& diffuse)
	{
		m_diffuseMap = diffuse;
		m_bUseDiffuseMap = true;
		m_dirty = true;
	}

	void useVertexColors()
	{
		m_bUseVertexColors = true;
		m_dirty = true;
	}
};