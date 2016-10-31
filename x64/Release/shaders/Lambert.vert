#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 aUV;

struct Camera
{
	vec3 mViewDirection;
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};

//current transformation matrices coming from Context
uniform mat4 uProjectionMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uModelMatrix;
uniform mat4 uNormalMatrix;
uniform mat4 uInvViewMatrix;

uniform Camera camera;
out vec2 uv;

//user supplied light position
uniform vec3 uLightPos;

//vertex position in the eye coordinates (view space)
out vec3 ecPosition;
//normal in the eye coordinates (view space)
out vec3 ecNormal;
//light position in the eye coordinates (view space)
out vec3 ecLightPos;

void main() {
	uv = aUV;
    //transform vertex into the eye space
    vec4 pos = camera.mView * camera.mModel * vec4(position, 1.0);
    ecPosition = pos.xyz;
    ecNormal = vec3(camera.mNormal * vec4(normal, 1.0));

    ecLightPos = vec3(camera.mView * camera.mModel * vec4(uLightPos, 1.0));

    //project the vertex, the rest is handled by WebGL
    gl_Position = camera.mProjection * pos;
}