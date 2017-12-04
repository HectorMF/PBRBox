#pragma once
#include <string>
#include <GL/glew.h>
#include "ResourceBase.h"
#include "TextureData.h"

enum class ColorSpace { Gamma = GL_SRGB, Linear = GL_RGBA };

enum class Filter {
	Nearest = GL_NEAREST, Linear = GL_LINEAR, MipMapNearestNearest = GL_NEAREST_MIPMAP_NEAREST,
	MipMapNearestLinear = GL_NEAREST_MIPMAP_LINEAR, MipMapLinearNearest = GL_LINEAR_MIPMAP_NEAREST,
	MipMapLinearLinear = GL_LINEAR_MIPMAP_LINEAR
};

enum class Wrap
{
	Clamp = GL_CLAMP_TO_EDGE, ClampBorder = GL_CLAMP_TO_BORDER, Repeat = GL_REPEAT, MirroredRepeat = GL_MIRRORED_REPEAT
};

enum class TextureType
{
	//2D = GL_TEXTURE_2D
};

class Texture : public ResourceBase
{
public:
	void foo() {}
	/*
	GLuint getId();
	GLenum getTarget() const { }
	GLuint getInternalFormat() const { }

	void bind(uint8_t textureUnit = 0) const;
	void unbind(uint8_t textureUnit = 0) const;

	GLuint getWidth() const {}
	GLuint getHeight() const {}
	GLuint getDepth() const {}

	float getAspectRatio() const { return getWidth() / (float)getHeight(); }

	void setWrapS(Wrap wrap) { m_wrapS = wrap; }
	void setWrapT(Wrap wrap) { m_wrapT = wrap; }
	void setWrapR(Wrap wrap) { m_wrapR = wrap; }

	GLuint getWrapS() { return static_cast<unsigned>(m_wrapS); }
	GLuint getWrapT() { return static_cast<unsigned>(m_wrapT); }
	GLuint getWrapR() { return static_cast<unsigned>(m_wrapR); }

	void setMinFilter(Filter filter);
	void setMagFilter(Filter filter);

	void setMaxAnisotropy(GLfloat anisotropy);

	void setBorderColor(const glm::fvec4& color) { m_hasBorder = true; m_borderColor = color; }
	glm::fvec4 getBorderColor() { return m_borderColor; }
	bool hasBorder() { return m_hasBorder; }
	
	
	bool m_hasBorder;
	glm::fvec4 m_borderColor;
	
	Wrap m_wrapS;
	Wrap m_wrapT;
	Wrap m_wrapR;*/




	unsigned int id;
	operator unsigned int() const { return id; }

	unsigned int target = GL_TEXTURE_2D;
	ColorSpace colorSpace = ColorSpace::Gamma;
	Filter minFilter = Filter::MipMapLinearLinear;
	Filter magFilter = Filter::Linear;
	Wrap uWrap = Wrap::Repeat;
	Wrap vWrap = Wrap::Repeat;

	bool generateMipMaps = true;

	TextureData* data;

	//unsigned int width, height;
	//unsigned char* data;

	bool hasBeenUploaded = false;
	Texture()
	{	
		glGenTextures(1, &id);
	}

	~Texture()
	{
		if(hasBeenUploaded)
			glDeleteTextures(1, &id);
		delete data;
	}

	void upload() {
		hasBeenUploaded = true;

		glBindTexture(target, id); /* Binding of texture name */
		unsigned int test = static_cast<unsigned int>(colorSpace);

		unsigned int t = GL_UNSIGNED_BYTE;
		glTexImage2D(target, 0, static_cast<unsigned>(colorSpace), data->getWidth(), data->getHeight(), 0, data->getFormat(), data->getDataType(), data->getData());

		if(generateMipMaps)
			glGenerateMipmap(target);

		glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<unsigned int>(minFilter));
		glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<unsigned int>(magFilter));

		glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<unsigned int>(uWrap));
		glTexParameteri(target, GL_TEXTURE_WRAP_T, static_cast<unsigned int>(vWrap));

		//glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);

		glBindTexture(target, 0);
	}

	void bind()
	{
		glBindTexture(target, id);
	}


	void bind(unsigned int unit)
	{
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(target, id);
	}

	/*void unbind()
	{
		glBindTexture()
	}*/

	void setBorderColor()
	{
		//glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);
	}
}; 

/*
import{ UVMapping } from '../constants';
import{ MirroredRepeatWrapping, ClampToEdgeWrapping, RepeatWrapping, LinearEncoding, UnsignedByteType, RGBAFormat, LinearMipMapLinearFilter, LinearFilter } from '../constants';
import{ _Math } from '../math/Math';
import{ Vector2 } from '../math/Vector2';



function Texture(image, mapping, wrapS, wrapT, magFilter, minFilter, format, type, anisotropy, encoding) {

	Object.defineProperty(this, 'id', { value: TextureIdCount() });

	this.uuid = _Math.generateUUID();

	this.name = '';
	this.sourceFile = '';

	this.image = image != = undefined ? image : Texture.DEFAULT_IMAGE;
	this.mipmaps = [];

	this.mapping = mapping != = undefined ? mapping : Texture.DEFAULT_MAPPING;

	this.wrapS = wrapS != = undefined ? wrapS : ClampToEdgeWrapping;
	this.wrapT = wrapT != = undefined ? wrapT : ClampToEdgeWrapping;

	this.magFilter = magFilter != = undefined ? magFilter : LinearFilter;
	this.minFilter = minFilter != = undefined ? minFilter : LinearMipMapLinearFilter;

	this.anisotropy = anisotropy != = undefined ? anisotropy : 1;

	this.format = format != = undefined ? format : RGBAFormat;
	this.type = type != = undefined ? type : UnsignedByteType;

	this.offset = new Vector2(0, 0);
	this.repeat = new Vector2(1, 1);

	this.generateMipmaps = true;
	this.premultiplyAlpha = false;
	this.flipY = true;
	this.unpackAlignment = 4;	// valid values: 1, 2, 4, 8 (see http://www.khronos.org/opengles/sdk/docs/man/xhtml/glPixelStorei.xml)


								// Values of encoding !== THREE.LinearEncoding only supported on map, envMap and emissiveMap.
								//
								// Also changing the encoding after already used by a Material will not automatically make the Material
								// update.  You need to explicitly call Material.needsUpdate to trigger it to recompile.
	this.encoding = encoding != = undefined ? encoding : LinearEncoding;

	this.version = 0;
	this.onUpdate = null;

}
*/