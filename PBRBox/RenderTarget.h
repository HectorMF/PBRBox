#pragma once
#include <GL/glew.h>
#include "glm\glm.hpp"

class RenderTarget
{
public:
	RenderTarget()
	{
		FramebufferName = 0;
		depthTexture = 0;

		glGenFramebuffers(1, &FramebufferName);
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

		//Depth texture. Slower than a depth buffer, but you can sample it later in your shader

		glGenTextures(1, &depthTexture);
		glBindTexture(GL_TEXTURE_2D, depthTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);

		glDrawBuffer(GL_NONE); // No color buffer is drawn to.

							   // Always check that our framebuffer is ok
		//if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		//	return false;
		Unbind();
	}

	~RenderTarget()
	{

	}
	operator unsigned int() const { return depthTexture; }

	void Bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
	}

	void Unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	GLuint FramebufferName;
	GLuint depthTexture;
};