#version 330 core
layout (location = 0) in vec3 position;

struct Camera
{
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};

uniform Camera camera;

out vec3 wcNormal;

void main() {
	mat4 invProjMatrix = inverse(camera.mProjection);
	mat4 transViewMatrix = transpose(camera.mView);

    //transform from the normalized device coordinates back to the view space
    vec3 unprojected = vec3(invProjMatrix * vec4(position, 1.0));

    //transfrom from the view space back to the world space
    //and use it as a sampling vector
    wcNormal = vec3(transViewMatrix * vec4(unprojected,1.0));

    gl_Position = vec4(position, 1.0);
}