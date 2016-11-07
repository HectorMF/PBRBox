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


out vec4 fragPosLightSpace;


out vec3 lightPos;


out vec3 VSPosition;
out vec3 VSNormal;
out vec3 vLightPosition;

out vec3 WSPosition;
out vec3 WSNormal;
out vec3 EyePosition;

void main() 
{
	vec4 wsPosition = camera.mModel * vec4(position, 1.0);
	vec4 vsPosition = camera.mView * wsPosition;
	vec4 vsNormal = camera.mNormal * vec4(normal, 1.0);
	VSPosition = vsPosition.xyz;
	VSNormal = vsNormal.xyz;
	
	WSPosition = wsPosition.xyz;
	WSNormal = (camera.mInvView * vsNormal).xyz;
	
	
	vec4 eyeDirViewSpace	= vsPosition - vec4( 0, 0, 0, 1 );
	EyePosition	= -vec3( camera.mInvView * eyeDirViewSpace );
	
	lightPos = uLightPos;
	uv = aUV;
	
	
	
	fragPosLightSpace = lightSpaceMatrix * camera.mModel * vec4(position, 1.0);
	

	vLightPosition = (camera.mView * camera.mModel * vec4(uLightPos, 1.0)).xyz;

    //project the vertex, the rest is handled by WebGL
    gl_Position = camera.mProjection * vsPosition;
}