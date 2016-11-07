#version 330 core
layout (location = 0) in vec3 position;

out vec3 TexCoords;


struct Camera
{
	vec3 vViewPos;
	vec3 vViewDirection;
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};

uniform Camera camera;

void main()
{
    gl_Position = camera.mProjection * camera.mView * vec4(position, 1.0);  
    TexCoords = position;
} 