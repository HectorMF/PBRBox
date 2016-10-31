#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 aUV;

struct Camera
{
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};

uniform Camera camera;

//vertex position in the eye coordinates (view space)
out vec3 ecPosition;
//normal in the eye coordinates (view space)
out vec3 ecNormal;

void main() {
    //transform vertex into the eye space
    vec4 pos = camera.mView * camera.mModel * vec4(position, 1.0);
    ecPosition = pos.xyz;
    ecNormal = vec3(camera.mNormal * vec4(normal, 1.0));

    //project the vertex, the rest is handled by WebGL
    gl_Position = camera.mProjection * pos;
}