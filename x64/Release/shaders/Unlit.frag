
const float PI = 3.14159265;
const float TwoPI = 6.28318530718;

float uBrightness = 1;
int maxLOD = 8; 

out vec4 fragColor;

in vec4 fragPosLightSpace;


in vec3 lightPos;

in vec3 VSPosition;
in vec3 VSNormal;
in vec3 vLightPosition;

in vec3 WSPosition;
in vec3 WSNormal;
in vec3 EyePosition;


#define saturate(x) clamp(x, 0.0, 1.0)

in V2F
{
	vec3 position;
	vec3 normal;
	vec3 tangent;
	vec3 bitangent;
    vec2 uv;
	vec4 color;
} fs_in;


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

#ifdef USE_DIFFUSE_MAP
	uniform sampler2D uDiffuse;
	vec3 getDiffuse() 
	{
		return texture2D(uDiffuse, fs_in.uv).rgb;
	}
#else
	uniform vec4 uDiffuse; 
	vec3 getDiffuse() 
	{
		return uDiffuse.rgb;
	}
#endif

#ifdef USE_VERTEX_COLORS
	vec3 getVertexColor()
	{
		return fs_in.color.rgb;
	}
#else
	vec3 getVertexColor()
	{
		return vec3(1, 1, 1);
	}
#endif

void main() 
{
	vec3 diffuse = getDiffuse() * getVertexColor();
	fragColor = vec4(diffuse,1);
}