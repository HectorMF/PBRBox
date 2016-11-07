#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 aUV;

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

//current transformation matrices coming from Context
uniform mat4 lightSpaceMatrix;
//user supplied light position
uniform vec3 uLightPos;





out vec2 uv;


//vertex position in the eye coordinates (view space)
out vec3 ecPosition;
//normal in the eye coordinates (view space)
out vec3 ecNormal;
//light position in the eye coordinates (view space)
out vec3 ecLightPos;

out vec4 fragPosLightSpace;

out vec3 wPos;
out vec3 wNormal;
out vec3 lightPos;



void main() {

	lightPos = uLightPos;
	uv = aUV;
	
	wPos = vec3( camera.mModel * vec4(position, 1.0));
    wNormal = transpose(inverse(mat3( camera.mModel))) * normal;
	
	fragPosLightSpace = lightSpaceMatrix * camera.mModel * vec4(position, 1.0);
	

    //transform vertex into the eye space
    vec4 pos = camera.mView * camera.mModel * vec4(position, 1.0);
    ecPosition = pos.xyz;
    ecNormal = vec3(camera.mNormal * vec4(normal, 1.0));

    ecLightPos = vec3(camera.mView * camera.mModel * vec4(uLightPos, 1.0));

    //project the vertex, the rest is handled by WebGL
    gl_Position = camera.mProjection * pos;
}