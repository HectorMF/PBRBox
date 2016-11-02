#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 auv;

struct Camera
{
	vec3 mViewDirection;
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};

uniform Camera camera;

out vec2 UV;

void main() {
	UV = auv;
    gl_Position = camera.mModel * vec4(position, 1.0);
}