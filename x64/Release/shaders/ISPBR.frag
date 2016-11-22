
const float PI = 3.14159265;
const float TwoPI = 6.28318530718;

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
uniform samplerCube uRadianceMap;
uniform samplerCube uIrradianceMap;

#ifdef USE_ALBEDO_MAP
	uniform sampler2D uAlbedo;
	vec3 getAlbedo() 
	{
		return texture2D(uAlbedo, fs_in.uv).rgb;
	}
#else
	uniform vec4 uAlbedo; 
	vec3 getAlbedo() 
	{
		return uAlbedo.rgb;
	}
#endif

#ifdef USE_ROUGHNESS_MAP
	uniform sampler2D uRoughness;
	float getRoughness() 
	{
		return texture2D(uRoughness, fs_in.uv).r;
	}
#else
	uniform float uRoughness;
	float getRoughness() 
	{
		return uRoughness;
	}
#endif

#ifdef USE_METALNESS_MAP
	uniform sampler2D uMetalness;
	float getMetalness() {
		return texture2D(uMetalness, fs_in.uv).r;
	}
#else
	uniform float uMetalness;
	float getMetalness() {
		return uMetalness;
	}
#endif

#ifdef USE_NORMAL_MAP
	uniform sampler2D uNormal;

	//use the tangent and bitangent to calculate the world normal from tangent space
	vec3 getNormal() 
	{
		vec3 normalRGB = texture2D(uNormal, fs_in.uv).rgb;
		//convert from rgb to tangent space
		vec3 normalMap = normalRGB * 2.0 - 1.0;
		//what does this do? handedness?
		//normalMap.y *= -1.0;
		
		mat3 TBN = transpose(mat3(fs_in.tangent, fs_in.bitangent, fs_in.normal));
		vec3 normal = (camera.mModel * vec4(normalMap * TBN,0)).rgb;
		
		return normalize(normal);
	}
#else
	vec3 getNormal() {
		return normalize(WSNormal);
	}
#endif

#ifdef USE_AO_MAP
	uniform sampler2D uAmbientOcclusion;
	float getAO()
	{
		return texture2D(uAmbientOcclusion, fs_in.uv).r;
	}
#else
	float getAO()
	{
		return 1;
	}
#endif

// http://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
float GGX(float NdotV, float a)
{
	float k = a / 2;
	return NdotV / (NdotV * (1.0f - k) + k);
}

// http://graphicrants.blogspot.com.au/2013/08/specular-brdf-reference.html
float G_Smith(float a, float nDotV, float nDotL)
{
	return GGX(nDotL, a * a) * GGX(nDotV, a * a);
}

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float radicalInverse_VdC(uint bits) {
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 Hammersley(uint i, uint N) {
	return vec2(float(i)/float(N), radicalInverse_VdC(i));
}

vec3 ImportanceSampleGGX(vec2 Xi, float Roughness, vec3 N )
{
	float a = Roughness * Roughness;
	float Phi = 2 * PI * Xi.x;
	float CosTheta = sqrt( (1 - Xi.y) / ( 1 + (a*a - 1) * Xi.y ) );
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );
	vec3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;
	vec3 UpVector = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
	vec3 TangentX = normalize( cross( UpVector, N ) );
	vec3 TangentY = cross( N, TangentX );
	// Tangent to world space
	return TangentX * H.x + TangentY * H.y + N * H.z;
}

vec3 SpecularIBL(vec3 specularColor, float roughness, float occlusion, vec3 N, vec3 V )
{
	vec3 SpecularLighting = vec3(0);
	uint NumSamples = uint(512);

	for(uint i = uint(0); i < NumSamples; i++ )
	{
		vec2 Xi = Hammersley( i, NumSamples );
		vec3 H = ImportanceSampleGGX( Xi, roughness, N );
		vec3 L = 2 * dot( V, H ) * H - V;
		float NoV = saturate( dot( N, V ) );
		float NoL = saturate( dot( N, L ) );
		float NoH = saturate( dot( N, H ) );
		float VoH = saturate( dot( V, H ) );
		if( NoL > 0 )
		{
			vec3 SampleColor = textureLod(uRadianceMap, L, 0).rgb;
			float G = G_Smith( roughness, NoV, NoL );
			float Fc = pow( 1 - VoH, 5 );
			vec3 F = (1 - Fc) * specularColor + Fc;
			// Incident light = SampleColor * NoL
			// Microfacet specular = D*G*F / (4*NoL*NoV)
			// pdf = D * NoH / (4 * VoH)
			SpecularLighting += SampleColor * F * G * VoH / (NoH * NoV);
		}
	}

	SpecularLighting *= mix(occlusion, 1.0, roughness) / NumSamples;

	return SpecularLighting;
}

void main() 
{
	float roughness = getRoughness();
	float ao = getAO();
	int nbSamples = 32;
	float roughness2 = pow(roughness, 2);
	float roughness4 = pow(roughness, 4);

	float metalness = getMetalness();
	
	vec3 albedo = getAlbedo();
	vec3 normal = getNormal();
	
	
	vec3 eye_vec = normalize(camera.position - WSPosition);
	float ndv = dot(eye_vec, normal);
  
	vec3 ecEyeDir = normalize(-VSPosition);
    //direction towards the camera in the world space
    vec3 wcEyeDir = vec3(camera.mInvView * vec4(ecEyeDir, 0.0));
	
	vec3 N = normal;
	vec3 V = wcEyeDir;
	vec3 L = lightPos - WSPosition;
	vec3 H = normalize(L + V);
	vec3 R = reflect(-V, N);
	
    float NdotH = saturate(dot(N, H));
    float LdotH = saturate(dot(L, H));
	float NdotL = saturate(dot(N, L));
    float NdotV = abs(dot(N, V)) + 1e-5f;
	float VdotH = saturate(dot(V, H));

	vec3 diffuseColor = albedo * (1 - metalness);
	vec3 specularColor = mix(vec3(0.04), albedo, metalness);
	
	vec3 irradiance = texture(uIrradianceMap, N).rgb;
	
	vec3 diffuse  		= ao * diffuseColor * irradiance;
    vec3 specular 		= SpecularIBL(specularColor, roughness, ao, N, V );
	vec3 color			= diffuse + specular;

	//color = pow(color, vec3(1/2.2));
	fragColor = vec4(color ,1.0f);
}