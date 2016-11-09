#version 330 core
const float PI = 3.14159265;
const float TwoPI = 6.28318530718;

varying in vec2 uv;

out vec4 fragColor;



in vec4 fragPosLightSpace;



in vec3 VSPosition;
in vec3 VSNormal;
in vec3 vLightPosition;

in vec3 WSPosition;
in vec3 WSNormal;
in vec3 EyePosition;

uniform sampler2D uRadianceMap;
uniform sampler2D uIrradianceMap;


vec4 sample_equirectangular_map(vec3 dir, sampler2D sampler, float lod) 
{
	vec2 uv;
	uv.x = atan( dir.z, dir.x );
	uv.y = acos( dir.y );
	uv /= vec2( 2 * PI, PI );
	
 	return textureLod( sampler, uv, lod );
}


vec2 envMapEquirect(vec3 wcNormal, float flipEnvMap) {
  //I assume envMap texture has been flipped the WebGL way (pixel 0,0 is a the bottom)
  //therefore we flip wcNorma.y as acos(1) = 0
  float phi = acos(wcNormal.y);
  float theta = atan(flipEnvMap * wcNormal.x, wcNormal.z) + PI;
  return vec2(theta / TwoPI, phi / PI);
}

vec2 envMapEquirect(vec3 wcNormal) {
    //-1.0 for left handed coordinate system oriented texture (usual case)
    return envMapEquirect(wcNormal, -1.0);
}

#ifdef USE_ALBEDO_MAP
	uniform sampler2D uAlbedo;
	vec3 getAlbedo() 
	{
		return texture2D(uAlbedo, uv).rgb;
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
		return texture2D(uRoughness, uv).r;
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
		return texture2D(uMetalness, uv).r;
	}
#else
	uniform float uMetalness;
	float getMetalness() {
		return uMetalness;
	}
#endif

#ifdef USE_NORMAL_MAP
	uniform sampler2D uNormal;
	#pragma glslify: perturb = require('glsl-perturb-normal')
	vec3 getNormal() {
		vec3 normalRGB = texture2D(uNormal, uv).rgb;
		vec3 normalMap = normalRGB * 2.0 - 1.0;

		normalMap.y *= -1.0;

		vec3 N = normalize(vNormalView);
		vec3 V = normalize(vEyeDirView);

		vec3 normalView = perturb(normalMap, N, V, uv);
		vec3 normalWorld = vec3(uInverseViewMatrix * vec4(normalView, 0.0));
		return normalize(normalWorld);
	}
#else
	vec3 getNormal() {
		return normalize(WSNormal);
	}
#endif


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



// http://the-witness.net/news/2012/02/seamless-cube-map-filtering/
vec3 fix_cube_lookup( vec3 v, float cube_size, float lod ) {
	float M = max(max(abs(v.x), abs(v.y)), abs(v.z));
	float scale = 1 - exp2(lod) / cube_size;
	if (abs(v.x) != M) v.x *= scale;
	if (abs(v.y) != M) v.y *= scale;
	if (abs(v.z) != M) v.z *= scale;
	return v;
}



vec3 F_Schlick (vec3 f0 , float f90 , float u )
{
	return f0 + ( f90 - f0 ) * pow (1 - u , 5);
}

vec3 Fresnel_Schlick( vec3 specularColor, float vdotH )
{
	return specularColor + ( 1.0 - specularColor ) * pow( 1.0 - vdotH, 5.0 );
}

float SchlickFresnel(float u)
{
    float m = clamp(1.0-u, 0.0, 1.0);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

// GGX Normal distribution
float getNormalDistribution( float roughness4, float NoH )
{
	float d = ( NoH * roughness4 - NoH ) * NoH + 1;
	return roughness4 / ( d*d );
}

// Smith GGX geometric shadowing from "Physically-Based Shading at Disney"
float getGeometricShadowing( float roughness4, float NoV, float NoL, float VoH, vec3 L, vec3 V )
{	
	float gSmithV = NoV + sqrt( NoV * (NoV - NoV * roughness4) + roughness4 );
	float gSmithL = NoL + sqrt( NoL * (NoL - NoL * roughness4) + roughness4 );
	return 1.0 / ( gSmithV * gSmithL );
}

vec3 getFresnel( vec3 specularColor, float VoH )
{
	vec3 specularColorSqrt = sqrt( clamp( vec3(0, 0, 0), vec3(0.99, 0.99, 0.99), specularColor ) );
	vec3 n = ( 1 + specularColorSqrt ) / ( 1 - specularColorSqrt );
	vec3 g = sqrt( n * n + VoH * VoH - 1 );
	return 0.5 * pow( (g - VoH) / (g + VoH), vec3(2.0) ) * ( 1 + pow( ((g+VoH)*VoH - 1) / ((g-VoH)*VoH + 1), vec3(2.0) ) );
}

vec3 getDiffuse( vec3 diffuseColor, float roughness4, float NoV, float NoL, float VoH )
{
	float VoL = 2 * VoH - 1;
	float c1 = 1 - 0.5 * roughness4 / (roughness4 + 0.33);
	float cosri = VoL - NoV * NoL;
	float c2 = 0.45 * roughness4 / (roughness4 + 0.09) * cosri * ( cosri >= 0 ? min( 1, NoL / NoV ) : NoL );
	return diffuseColor / PI * ( NoL * c1 + c2 );
}

float D_GGX(float ndoth, float m)
{
	float m2 = m* m;
	float f = (ndoth * m2 - ndoth) * ndoth + 1;
	return m2 / (f * f);
}

float V_SmithGGXCorrelated ( float NdotL , float NdotV , float alphaG )
{
	// Original formulation of G_SmithGGX Correlated
	// lambda_v = ( -1 + sqrt ( alphaG2 * (1 - NdotL2 ) / NdotL2 + 1)) * 0.5 f;
	// lambda_l = ( -1 + sqrt ( alphaG2 * (1 - NdotV2 ) / NdotV2 + 1)) * 0.5 f;
	// G_SmithGGXCorrelated = 1 / (1 + lambda_v + lambda_l );
	// V_SmithGGXCorrelated = G_SmithGGXCorrelated / (4.0 f * NdotL * NdotV );

	// This is the optimize version
	float alphaG2 = alphaG * alphaG ;
	// Caution : the " NdotL *" and " NdotV *" are explicitely inversed , this is not a mistake .
	float Lambda_GGXV = NdotL * sqrt (( - NdotV * alphaG2 + NdotV ) * NdotV + alphaG2 );
	float Lambda_GGXL = NdotV * sqrt (( - NdotL * alphaG2 + NdotL ) * NdotL + alphaG2 );

	return 0.5f / ( Lambda_GGXV + Lambda_GGXL );
}

float SmithGGXVisibility(in float nDotL, in float nDotV, in float roughness)
{
	float rough2 = roughness * roughness;
	float gSmithV = nDotV + sqrt(nDotV * (nDotV - nDotV * rough2) + rough2);
	float gSmithL = nDotL + sqrt(nDotL * (nDotL - nDotL * rough2) + rough2);
	return 1.0 / ( gSmithV * gSmithL );
}

/*
float Fr_DisneyDiffuse ( float NdotV , float NdotL , float LdotH , float linearRoughness )
{
	float energyBias = lerp (0 , 0.5 , linearRoughness );
	float energyFactor = lerp (1.0 , 1.0 / 1.51 , linearRoughness );
	float fd90 = energyBias + 2.0 * LdotH * LdotH * linearRoughness ;
	vec3 f0 = vec3 (1.0 , 1.0 , 1.0);
	float lightScatter = F_Schlick ( f0 , fd90 , NdotL ).r;
	float viewScatter = F_Schlick (f0 , fd90 , NdotV ).r;

	return lightScatter * viewScatter * energyFactor ;
}*/

vec3 EnvBRDFApprox( vec3 SpecularColor, float Roughness, float NoV )
{
	const vec4 c0 = vec4( -1, -0.0275, -0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, -0.04 );
	vec4 r = Roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( -9.28 * NoV ) ) * r.x + r.y;
	vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;
	return SpecularColor * AB.x + AB.y;
}

#define saturate(x) clamp(x, 0.0, 1.0)

void main() 
{

	float roughness = getRoughness();
	float roughness4 = pow(roughness, 4);
	float metalness = getMetalness();
	vec3 albedo = getAlbedo();
	vec3 normal = getNormal();
	
	vec3 ecEyeDir = normalize(-VSPosition);
    //direction towards the camera in the world space
    vec3 wcEyeDir = vec3(camera.mInvView * vec4(ecEyeDir, 0.0));
	
	vec3 N = normalize( WSNormal );
	vec3 V = wcEyeDir;
	vec3 L = vLightPosition - WSPosition;
	vec3 H = normalize(L + V);
	vec3 R = reflect(-V, N);
        
	//vec3 N = normalize(VSNormal);
	
	//vec3 V = normalize(-VSPosition);
	//
	
    float NdotH = saturate(dot(N, H));
    float LdotH = saturate(dot(L, H));
	float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
	float VdotH = saturate(dot(V, H));

	vec3 diffuseColor	= albedo * (1 - metalness);
	vec3 specularColor = mix(vec3(0.04), albedo, metalness );
	vec3 fresnel = Fresnel_Schlick(specularColor, NdotV);
	
	int numMips			= 7;
	float lod = (1.0 - roughness)*(numMips - 1.0);
	float mip			= numMips - 1 + log2(roughness);
	vec3 lookup			= -reflect(V, N);
	//lookup			= fix_cube_lookup(lookup, 256, mip );
	vec3 radiance		= textureLod(uRadianceMap, envMapEquirect(R), mip).rgb;
	vec3 irradiance		= texture(uIrradianceMap, envMapEquirect(N)).rgb;
	
	vec3 reflectance = EnvBRDFApprox(specularColor, roughness4, NdotV);
	
	vec3 diffuse  		= diffuseColor * irradiance;
    vec3 specular 		= reflectance * radiance;
	vec3 color			= diffuse + specular;
	
	fragColor = vec4(color, 1.0);
	
	
	
	/*
	float distribution = getNormalDistribution( roughness, NdotH );
	float geom = getGeometricShadowing( roughness, NdotV, NdotL, VdotH, L, V );

	vec3 fresnel = getFresnel(specularColor, VdotH);
	vec3 diffuse = getDiffuse( diffuseColor, roughness, NdotV, NdotL, VdotH );
	
	
	vec3 specular			= NdotL * ( distribution * fresnel * geom );
	vec3 color	= vec3(1,1,1) * ( diffuse + specular );	//= uLightColor * ( diffuse + specular );
	*/
	
	
	
	/*float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5 + 2 * LdotH*LdotH * roughness;
    float Fd = mix(1, Fd90, FL) * mix(1, Fd90, FV);
	
	float Fss90 = LdotH*LdotH*roughness;
    float Fss = mix(1, Fss90, FL) * mix(1, Fss90, FV);
    float ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5);
	
	

	vec3 ecEyeDir = normalize(-ecPosition);
    //direction towards the camera in the world space
    vec3 wcEyeDir = vec3(camera.mInvView * vec4(ecEyeDir, 0.0));
    //surface normal in the world space

    //vec3 wcNormal = vec3(camera.mInvView * vec4(N, 0.0));
	vec3 wcNormal =  normal;
	
    vec3 reflectionWorld = reflect(-wcEyeDir, normalize(wcNormal));
	vec3 reflection = texture2D(uReflectionMap, envMapEquirect(reflectionWorld));
	
	
	

	
	
	
	//vec3 F = F_Schlick(f0, f90, LdotH);
	
	vec3 fresnel = Fresnel_Schlick(vec3(1,.765,.336057), LdotH);
	float D = D_GGX(NdotH, roughness);
	float vis = SmithGGXVisibility(NdotV, NdotL, roughness);
	
	float Fr = D * fresnel * vis / PI;
	float diffuseBRDF = Fr_DisneyDiffuse(NdotV, NdotL, LdotH, roughness) / PI;
	
	
	
	*/
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
