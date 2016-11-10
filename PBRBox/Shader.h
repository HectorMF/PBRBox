#pragma once

#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <GL/glew.h>

class Shader
{
	std::string m_version;
public:
	GLuint m_program;

	std::string vertexCode;
	std::string fragmentCode;

	std::vector<std::string> flags;

	operator GLuint() const { return m_program; }

	void setVersion(int version)
	{
		m_version = "#version 330 core\n";
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

	inline unsigned int getUniform(std::string val)
	{
		return glGetUniformLocation(m_program, val.c_str());
	}

	Shader(){
		m_program = 0;
		m_version = "";
	}

	~Shader()
	{

	}

	// Uses the current shader
	void bind()
	{
		glUseProgram(m_program);
	}

	void unbind()
	{
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