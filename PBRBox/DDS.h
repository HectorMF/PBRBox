#pragma once

#include <gli/gli.hpp>

/*
GLuint initSampler()
{
	GLuint SamplerName = 0;
	glGenSamplers(1, &SamplerName);
	glSamplerParameteri(SamplerName, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glSamplerParameteri(SamplerName, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(SamplerName, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glSamplerParameteri(SamplerName, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glSamplerParameteri(SamplerName, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glSamplerParameterfv(SamplerName, GL_TEXTURE_BORDER_COLOR, &glm::vec4(0.0f)[0]);
	glSamplerParameterf(SamplerName, GL_TEXTURE_MIN_LOD, -1000.f);
	glSamplerParameterf(SamplerName, GL_TEXTURE_MAX_LOD, 1000.f);
	glSamplerParameterf(SamplerName, GL_TEXTURE_LOD_BIAS, 0.0f);
	glSamplerParameteri(SamplerName, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	glSamplerParameteri(SamplerName, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glSamplerParameterf(SamplerName, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f);

	return SamplerName;
}

GLuint create_texture(char const* Filename)
{
	GLuint TextureName = 0;
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &TextureName);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, TextureName);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);

	gli::texture Texture(gli::load(Filename));
	assert(!Texture.empty());

	gli::gl GL(gli::gl::PROFILE_GL33);
	gli::gl::format const Format = GL.translate(Texture.format(), Texture.swizzles());
	gli::gl::target const Target = GL.translate(Texture.target());

	glTexImage3D(
		Target, 0,
		Format.Internal,
		static_cast<GLsizei>(Texture.extent().x), static_cast<GLsizei>(Texture.extent().y), static_cast<GLsizei>(Texture.faces()),
		0,
		Format.External, Format.Type,
		Texture.data());

	return TextureName;
}


*/
GLuint load_brdf(char const* filename)
{
	//texture.setMinFilter('NEAREST');
}

GLuint create_texture(char const* Filename)
{
	gli::texture Texture = gli::load(Filename);
	if (Texture.empty())
		return 0;

	gli::gl GL(gli::gl::PROFILE_GL33);
	gli::gl::format const Format = GL.translate(Texture.format(), Texture.swizzles());
	GLenum Target = GL.translate(Texture.target());

	GLuint TextureName = 0;
	glGenTextures(1, &TextureName);
	glBindTexture(Target, TextureName);
	glTexParameteri(Target, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(Target, GL_TEXTURE_MAX_LEVEL, static_cast<GLint>(Texture.levels() - 1));
	glTexParameteri(Target, GL_TEXTURE_SWIZZLE_R, Format.Swizzles[0]);
	glTexParameteri(Target, GL_TEXTURE_SWIZZLE_G, Format.Swizzles[1]);
	glTexParameteri(Target, GL_TEXTURE_SWIZZLE_B, Format.Swizzles[2]);
	glTexParameteri(Target, GL_TEXTURE_SWIZZLE_A, Format.Swizzles[3]);

	glm::tvec3<GLsizei> const Extent(Texture.extent());
	GLsizei const FaceTotal = static_cast<GLsizei>(Texture.layers() * Texture.faces());
	int val;
	switch (Texture.target())
	{
	case gli::TARGET_1D:
		glTexStorage1D(
			Target, static_cast<GLint>(Texture.levels()), Format.Internal, Extent.x);
		break;
	case gli::TARGET_1D_ARRAY:
	case gli::TARGET_2D:
	case gli::TARGET_CUBE:

		val = (Texture.target() == gli::TARGET_2D) ? Extent.y : FaceTotal;
		glTexStorage2D(
			Target, static_cast<GLint>(Texture.levels()), GL_SRGB8,
			Extent.x, Extent.y);
		break;
	case gli::TARGET_2D_ARRAY:
	case gli::TARGET_3D:
	case gli::TARGET_CUBE_ARRAY:
		glTexStorage3D(
			Target, static_cast<GLint>(Texture.levels()), Format.Internal,
			Extent.x, Extent.y,
			Texture.target() == gli::TARGET_3D ? Extent.z : FaceTotal);
		break;
	default:
		assert(0);
		break;
	}

	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	{
		printf("OpenGL error %08x\n", err);
		abort();
	}

	for (std::size_t Layer = 0; Layer < Texture.layers(); ++Layer)
		for (std::size_t Face = 0; Face < Texture.faces(); ++Face)
			for (std::size_t Level = 0; Level < Texture.levels(); ++Level)
			{
				GLsizei const LayerGL = static_cast<GLsizei>(Layer);
				glm::tvec3<GLsizei> Extent(Texture.extent(Level));
				Target = gli::is_target_cube(Texture.target())
					? static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + Face)
					: Target;

				switch (Texture.target())
				{
				case gli::TARGET_1D:
					if (gli::is_compressed(Texture.format()))
						glCompressedTexSubImage1D(
							Target, static_cast<GLint>(Level), 0, Extent.x,
							Format.Internal, static_cast<GLsizei>(Texture.size(Level)),
							Texture.data(Layer, Face, Level));
					else
						glTexSubImage1D(
							Target, static_cast<GLint>(Level), 0, Extent.x,
							Format.External, Format.Type,
							Texture.data(Layer, Face, Level));
					break;
				case gli::TARGET_1D_ARRAY:
				case gli::TARGET_2D:
				case gli::TARGET_CUBE:
					if (gli::is_compressed(Texture.format()))
						glCompressedTexSubImage2D(
							Target, static_cast<GLint>(Level),
							0, 0,
							Extent.x,
							Texture.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y,
							Format.Internal, static_cast<GLsizei>(Texture.size(Level)),
							Texture.data(Layer, Face, Level));
					else
						glTexSubImage2D(
							Target, static_cast<GLint>(Level),
							0, 0,
							Extent.x,
							Texture.target() == gli::TARGET_1D_ARRAY ? LayerGL : Extent.y,
							Format.External, Format.Type,
							Texture.data(Layer, Face, Level));
					break;
				case gli::TARGET_2D_ARRAY:
				case gli::TARGET_3D:
				case gli::TARGET_CUBE_ARRAY:
					if (gli::is_compressed(Texture.format()))
						glCompressedTexSubImage3D(
							Target, static_cast<GLint>(Level),
							0, 0, 0,
							Extent.x, Extent.y,
							Texture.target() == gli::TARGET_3D ? Extent.z : LayerGL,
							Format.Internal, static_cast<GLsizei>(Texture.size(Level)),
							Texture.data(Layer, Face, Level));
					else
						glTexSubImage3D(
							Target, static_cast<GLint>(Level),
							0, 0, 0,
							Extent.x, Extent.y,
							Texture.target() == gli::TARGET_3D ? Extent.z : LayerGL,
							Format.External, Format.Type,
							Texture.data(Layer, Face, Level));
					break;
				default: assert(0); break;
				}
			}

	float aniso = 0.0f;
	//glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);
	//printf("Anisotropy %f\n", aniso);
	//aniso = 1.0f;
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
	//float aniso = 4.0f;
	//glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);
	//glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
	//printf("Anisotropy %f\n", aniso);
	//BGFX_TEXTURE_MIN_ANISOTROPIC | BGFX_TEXTURE_MAG_ANISOTROPIC | BGFX_TEXTURE_U_CLAMP | BGFX_TEXTURE_V_CLAMP,

	//glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_U, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);

	//glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_NEAREST);
	//glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	return TextureName;
}

/*
GLuint create_texture(char const* Filename)
{
	GLuint TextureName = 0;
	gli::texture_cube Texture(gli::FORMAT_RGBA8_UNORM_PACK8, gli::texture_cube::extent_type(2), 1);
	assert(!Texture.empty());

	Texture[0].clear<glm::u8vec4>(glm::u8vec4(255, 0, 0, 255));
	Texture[1].clear<glm::u8vec4>(glm::u8vec4(255, 128, 0, 255));
	Texture[2].clear<glm::u8vec4>(glm::u8vec4(255, 255, 0, 255));
	Texture[3].clear<glm::u8vec4>(glm::u8vec4(0, 255, 0, 255));
	Texture[4].clear<glm::u8vec4>(glm::u8vec4(0, 255, 255, 255));
	Texture[5].clear<glm::u8vec4>(glm::u8vec4(0, 0, 255, 255));

	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &TextureName);
	glBindTexture(GL_TEXTURE_CUBE_MAP, TextureName);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, GLint(Texture.levels() - 1));
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameterfv(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BORDER_COLOR, &glm::vec4(0.0f)[0]);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_LOD, -1000.f);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LOD, 1000.f);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_LOD_BIAS, 0.0f);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

	gli::gl GL(gli::gl::PROFILE_GL32);
	gli::gl::format const Format = GL.translate(Texture.format(), Texture.swizzles());

	for (gli::texture_cube::size_type Face = 0; Face < Texture.faces(); ++Face)
	{
		glTexImage2D(
			GL_TEXTURE_CUBE_MAP_POSITIVE_X + GLenum(Face),
			0,
			Format.Internal,
			static_cast<GLsizei>(Texture.extent().x), static_cast<GLsizei>(Texture.extent().y),
			0,
			Format.External, Format.Type,
			Texture[Face].data());
	}

	return TextureName;
}
*/
//glDeleteTextures(1, &TextureName);
//glDeleteSamplers(1, &SamplerName);