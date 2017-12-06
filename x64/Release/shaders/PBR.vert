layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 tangent;
layout (location = 3) in vec3 bitangent;
layout (location = 4) in vec2 uv;
layout (location = 5) in vec4 color;

out V2F
{
	vec3 position;
	vec3 normal;
	vec3 tangent;
	vec3 bitangent;
    vec2 uv;
	vec4 color;
} vs_out;


struct Camera
{
	vec3 position;
	vec3 viewDirection;
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};

uniform Camera camera;

/*struct Light
{
	vec3
};*/

//current transformation matrices coming from Context
uniform mat4 lightSpaceMatrix;
//user supplied light position
uniform vec3 uLightPos;


out vec4 fragPosLightSpace;


out vec3 lightPos;


out vec3 VSPosition;
out vec3 VSNormal;
out vec3 vLightPosition;

out vec3 WSPosition;
out vec3 WSNormal;

centroid out vec3 WSNormalCentroid;
out vec3 EyePosition;

void main() 
{
	vec4 a_normal = vec4(normal, 0.0) * 2.0 - 1.0;

	vec4 wsPosition = camera.mModel * vec4(position, 1.0);
	vec4 vsPosition = camera.mView * wsPosition;
	vec4 vsNormal = camera.mNormal * vec4(normal, 0.0);
	VSPosition = vsPosition.xyz;
	//VSNormal = normalize(vsNormal.xyz);
	
	vec3 wcNormal = normalize(vec3(camera.mInvView * vec4(VSNormal, 0.0)));
	
	WSPosition = wsPosition.xyz;
	WSNormal = vec3(camera.mModel* vec4(normal,0));
	WSNormalCentroid = WSNormal;//normalize(camera.mInvView * vsNormal).xyz);
	VSNormal = normalize(vec3(camera.mView * vec4(WSNormal,0)));
	EyePosition	= normalize(camera.position - vec3(wsPosition));

	lightPos = uLightPos;

	mat3 normalMatrix = transpose(inverse(mat3(camera.mModel)));
	vec3 T = normalize(normalMatrix * tangent);
    vec3 B = normalize(normalMatrix * bitangent);
    vec3 N = normalize(normalMatrix * normal);

	vs_out.position = position;
	vs_out.normal = N;
	vs_out.tangent = T;
	vs_out.bitangent = B;
	vs_out.color = color;
	vs_out.uv = uv;
	
	fragPosLightSpace = lightSpaceMatrix * camera.mModel * vec4(position, 1.0);
	

	vLightPosition = (camera.mView * camera.mModel * vec4(uLightPos, 1.0)).xyz;

    //project the vertex, the rest is handled by WebGL
    gl_Position = camera.mProjection * vsPosition;
}