#pragma once
#include <GL/glew.h>
#include <string>
#include <map>
#include <memory>
#include "glm\glm.hpp"
#include "Texture.h"

enum class Attachment
{
	Depth = GL_DEPTH_ATTACHMENT, 
	Stencil = GL_STENCIL_ATTACHMENT, 
	Depth_Stencil = GL_DEPTH_STENCIL_ATTACHMENT,
	Color = GL_COLOR_ATTACHMENT0,
	Color0 = GL_COLOR_ATTACHMENT0,
	Color1 = GL_COLOR_ATTACHMENT1,
	Color2 = GL_COLOR_ATTACHMENT2,
	Color3 = GL_COLOR_ATTACHMENT3,
	Color4 = GL_COLOR_ATTACHMENT4,
	Color5 = GL_COLOR_ATTACHMENT5,
	Color6 = GL_COLOR_ATTACHMENT6,
	Color7 = GL_COLOR_ATTACHMENT7,
	Color8 = GL_COLOR_ATTACHMENT8,
	Color9 = GL_COLOR_ATTACHMENT9,
	Color10 = GL_COLOR_ATTACHMENT10,
	Color11 = GL_COLOR_ATTACHMENT11,
	Color12 = GL_COLOR_ATTACHMENT12,
	Color13 = GL_COLOR_ATTACHMENT13,
	Color14 = GL_COLOR_ATTACHMENT14,
	Color15 = GL_COLOR_ATTACHMENT15
};


#include <GL/GLFrameBufferObject.h>

#include <Utilities/Assertion.h>

namespace gb
{
	GLFrameBufferObject::GLFrameBufferObject()
	{
		m_iFramebufferObject = 0;
		m_iOldFramebufferObject = 0;

		//we are reserving 4 render targets!
		m_renderTargets.reserve(4);
		m_pDepthTexture = NULL;

		m_bFBOBound = false;
	}

	GLFrameBufferObject::~GLFrameBufferObject()
	{
	}

	//! This is where we generate our frame buffer.  
	void GLFrameBufferObject::generate()
	{
		glGenFramebuffers(1, &m_iFramebufferObject);
		printOpenGLError();
	}

	//! Setup a texture to render to.  Note, we can setup multiple render targets with idx
	void GLFrameBufferObject::setRenderToTexture(GLTextureObject *pTexture, const int &idx)
	{
		Assertion(idx >= 0 && idx < 4, format("idx:%i", idx).c_str());
		Assertion(m_bFBOBound, "fbo not bound!");

		//we need to push it or replace it...
		if (m_renderTargets.size() <= idx)
			m_renderTargets.push_back(pTexture);
		else
			m_renderTargets[idx] = pTexture;

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_2D, m_renderTargets[idx]->m_iTextureID, 0);
		printOpenGLError();

		gl_CheckFramebufferStatus();
	}

	//! render to depth buffer here
	void GLFrameBufferObject::setRenderToTextureDepth(GLTextureObject *pDepth)
	{
		Assertion(pDepth != NULL, "!pDepth");
		Assertion(m_bFBOBound, "fbo not bound!");

		m_pDepthTexture = pDepth;

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_pDepthTexture->m_iTextureID, 0);

		gl_CheckFramebufferStatus();
	}

	//! Bind and unbind the FBO
	void GLFrameBufferObject::bind()
	{
		//Fetch the previous binding, so we can restore to it when we are done with our rendering
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &m_iOldFramebufferObject);
		glBindFramebuffer(GL_FRAMEBUFFER, m_iFramebufferObject);

		m_bFBOBound = true;
	}

	void GLFrameBufferObject::unbind()
	{
		m_bFBOBound = false;

		glBindFramebuffer(GL_FRAMEBUFFER, m_iOldFramebufferObject);
	}
}



class FrameBuffer
{
public:
	FrameBuffer(int width, int height, bool alpha, bool depth = true, bool stencil = false)
	{
		m_id = 0;
		depthTexture = 0;

		glGenFramebuffers(1, &m_id);




		//Depth texture. Slower than a depth buffer, but you can sample it later in your shader
		glGenTextures(1, &depthTexture);
		glBindTexture(GL_TEXTURE_2D, depthTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, size.x, size.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		GLfloat borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);


		

		glBindFramebuffer(GL_FRAMEBUFFER, m_id);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);

		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	~FrameBuffer()
	{
		glDeleteFramebuffers(1, &m_id);
	}

	void clear(const glm::vec4 color = glm::vec4(0, 0, 0, 255))
	{

	}

	glm::ivec2 getSize() const
	{

	}
	
	void bindTexture(std::shared_ptr<Texture> tex, Attachment attachment)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_id);
		glFramebufferTexture2D(GL_FRAMEBUFFER, static_cast<unsigned int>(attachment), GL_TEXTURE_2D, tex->getId(), 0);
		textures[attachment] = tex;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	std::shared_ptr<Texture> getTexture(Attachment attachment)
	{
		return textures[attachment];
	}

	GLuint getId()
	{
		return m_id;
	}

	void bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_id);
	}

	void unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

protected:


	std::string m_name;

	unsigned int m_width;
	unsigned int m_height;


	std::map<Attachment, std::shared_ptr<Texture>> textures;

	glm::vec2 size;

	GLuint m_id;
	GLuint depthTexture;
};