#pragma once

#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <GL/glew.h>
#include "Texture.h"

#include <glm/gtc/type_ptr.hpp>
class Shader
{
	std::string m_version;
public:
	GLuint m_program;

	std::string vertexCode;
	std::string fragmentCode;

	std::vector<std::string> flags;

	unsigned int m_boundTextures;

	operator GLuint() const { return m_program; }

	void setVersion(int version)
	{
		m_version = "#version 400 core\n";
	}

	void addFlag(std::string flag)
	{
		flags.push_back(flag);
	}

	void clearFlags()
	{
		flags.clear();
	}

	unsigned int getProgram() const { return m_program; }

	inline int getUniformLocation(std::string val)
	{
		return glGetUniformLocation(m_program, val.c_str());
	}

	Shader(){
		m_program = 0;
		m_boundTextures = 0;
		m_version = "";
	}

	~Shader()
	{

	}

	void setUniform(const std::string name, float x, float y, float z)
	{
		int location = getUniformLocation(name);
		glUniform3f(location, x, y, z);
	}

	void setUniform(const std::string name, float x, float y, float z, float w)
	{
		int location = getUniformLocation(name);
		glUniform4f(location, x, y, z, w);
	}

	void setUniform(const std::string name, const glm::vec3 & v)
	{
		int location = getUniformLocation(name);
		glUniform3fv(location, 1, glm::value_ptr(v));
	}

	void setUniform(const std::string name, const glm::vec4 & v)
	{
		int location = getUniformLocation(name);
		glUniform4fv(location, 1, glm::value_ptr(v));
	}

	void setUniform(const std::string name, const glm::mat4 &m)
	{
		int location = getUniformLocation(name);
		glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(m));
	}

	void setUniform(const std::string name, const glm::mat3 & m)
	{
		int location = getUniformLocation(name);
		glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(m));
	}

	void setUniform(const std::string name, float val)
	{
		int location = getUniformLocation(name);
		glUniform1f(location, val);
	}

	void setUniform(const std::string name, int val)
	{
		int location = getUniformLocation(name);
		glUniform1i(location, val);
	}

	void setUniform(const std::string name, bool val)
	{
		int location = getUniformLocation(name);
		glUniform1i(location, val);
	}

	void setUniform(const std::string name, Texture texture)
	{
		int location = getUniformLocation(name);
		if (location >= 0) {
			glUniform1i(location, m_boundTextures);
			glActiveTexture(GL_TEXTURE0 + m_boundTextures);
			glBindTexture(texture.textureType, texture);
			m_boundTextures++;
		}
	}

	// Uses the current shader
	void bind()
	{
		glUseProgram(m_program);
		assert(m_boundTextures == 0);
	}

	void unbind()
	{
		for (int i = 0; i < m_boundTextures; i++)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			glBindTexture(GL_TEXTURE_2D, 0);
		}

		m_boundTextures = 0; 

		glUseProgram(0);
	}

	void compile()
	{
		std::string header;
		for (int i = 0; i < flags.size(); i++)
			header += flags[i] + "\n";

		std::string vertexShader = m_version + vertexCode;
		std::string fragmentShader = m_version + header + fragmentCode;

		const GLchar* vShaderCode = vertexShader.c_str();
		const GLchar * fShaderCode = fragmentShader.c_str();
		// 2. Compile shaders
		GLuint vertex, fragment;
		GLint success;
		GLchar infoLog[2048];
		// Vertex Shader
		vertex = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex, 1, &vShaderCode, NULL);
		glCompileShader(vertex);
		// Print compile errors if any
		glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(vertex, 2048, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		// Fragment Shader
		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &fShaderCode, NULL);
		glCompileShader(fragment);
		// Print compile errors if any
		glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fragment, 2048, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		// Shader Program
		m_program = glCreateProgram();
		glAttachShader(m_program, vertex);
		glAttachShader(m_program, fragment);
		glLinkProgram(m_program);
		// Print linking errors if any
		glGetProgramiv(m_program, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(m_program, 2048, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		}
		// Delete the shaders as they're linked into our program now and no longer necessery
		glDeleteShader(vertex);
		glDeleteShader(fragment);
	}

	// Constructor generates the shader on the fly
	Shader(const GLchar* vertexPath, const GLchar* fragmentPath, bool autoCompile = true)
	{
		m_boundTextures = 0;
		// 1. Retrieve the vertex/fragment source code from filePath
		std::ifstream vShaderFile;
		std::ifstream fShaderFile;
		// ensures ifstream objects can throw exceptions:
		vShaderFile.exceptions(std::ifstream::badbit);
		fShaderFile.exceptions(std::ifstream::badbit);
		try
		{
			// Open files
			vShaderFile.open(vertexPath);
			fShaderFile.open(fragmentPath);
			std::stringstream vShaderStream, fShaderStream;
			// Read file's buffer contents into streams
			vShaderStream << vShaderFile.rdbuf();
			fShaderStream << fShaderFile.rdbuf();
			// close file handlers
			vShaderFile.close();
			fShaderFile.close();
			// Convert stream into string
			vertexCode = vShaderStream.str();
			fragmentCode = fShaderStream.str();
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
		}

		if (autoCompile) compile();
	}

};

#endif