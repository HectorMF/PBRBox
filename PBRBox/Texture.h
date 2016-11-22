#pragma once
#include "stb_image.h"
#include <string>
#include <GL/glew.h>

enum class ColorSpace { Gamma, Linear };
class Texture
{
public:
	unsigned int id;
	operator unsigned int() const { return id; }

	Texture(){}
	Texture(std::string file, ColorSpace space = ColorSpace::Gamma)
	{
		int width, height, bpp;
		unsigned char* image = stbi_load(file.c_str(), &width, &height, &bpp, 4);

		glGenTextures(1, &id); /* Texture name generation */
		glBindTexture(GL_TEXTURE_2D, id); /* Binding of texture name */
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); /* We will use linear interpolation for magnification filter */
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); /* We will use linear interpolation for minifying filter */
		if (space == ColorSpace::Gamma)
		{
					glGenerateMipmap(GL_TEXTURE_2D);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); /* We will use linear interpolation for magnification filter */
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); /* We will use linear interpolation for minifying filter */
	
			glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
			glGenerateMipmap(GL_TEXTURE_2D);
		}/* Texture specification */
		if (space == ColorSpace::Linear)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image); /* Texture specification */

		float aniso;
		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);
	//	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
		glBindTexture(GL_TEXTURE_2D, 0);



		//printf("Anisotropy %f\n", aniso);
		//aniso = 1.0f;
		//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
		//float aniso = 4.0f;
		//glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);
		//glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
		//printf("Anisotropy %f\n", aniso);
		//BGFX_TEXTURE_MIN_ANISOTROPIC | BGFX_TEXTURE_MAG_ANISOTROPIC | BGFX_TEXTURE_U_CLAMP | BGFX_TEXTURE_V_CLAMP,


		stbi_image_free(image);
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