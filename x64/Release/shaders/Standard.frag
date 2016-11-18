
const float PI = 3.14159265;
const float TwoPI = 6.28318530718;



in V2F
{
	vec3 position;
	vec3 normal;
	vec3 tangent;
	vec3 bitangent;
    vec2 uv;
} fs_in;









#define maxLod 7
#define M_INV_PI 0.31830988618
#define M_PI 3.14159265359
out vec4 fragColor;



in vec4 fragPosLightSpace;


in vec3 lightPos;

in vec3 VSPosition;
in vec3 VSNormal;
in vec3 vLightPosition;

in vec3 WSPosition;
in vec3 WSNormal;
in vec3 EyePosition;

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

uniform sampler2D uShadowMap;



float ShadowCalculation(vec4 fragPosLightSpace, vec3 wcPosition, vec3 wcNormal)
{
  // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // Get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(uShadowMap, projCoords.xy).r; 
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // Calculate bias (based on depth map resolution and slope)
 
    vec3 lightDir = normalize(lightPos - wcPosition);
    float bias = max(0.01 * (1.0 - dot(wcNormal, lightDir)), 0.005);
    // Check whether current frag pos is in shadow
    // float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(uShadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(uShadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    // Keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projCoords.z > 1.0)
        shadow = 0.0;
        
    return shadow;
}

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


float Fr_DisneyDiffuse ( float NdotV , float NdotL , float LdotH , float linearRoughness )
{
	float energyBias = mix (0 , 0.5 , linearRoughness );
	float energyFactor = mix (1.0 , 1.0 / 1.51 , linearRoughness );
	float fd90 = energyBias + 2.0 * LdotH * LdotH * linearRoughness ;
	vec3 f0 = vec3 (1.0 , 1.0 , 1.0);
	float lightScatter = F_Schlick ( f0 , fd90 , NdotL ).r;
	float viewScatter = F_Schlick (f0 , fd90 , NdotV ).r;

	return lightScatter * viewScatter * energyFactor ;
}

vec3 EnvBRDFApprox( vec3 SpecularColor, float Roughness, float NoV )
{
	const vec4 c0 = vec4( -1, -0.0275, -0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, -0.04 );
	vec4 r = Roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( -9.28 * NoV ) ) * r.x + r.y;
	vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;
	return SpecularColor * AB.x + AB.y;
}






//--------------------------------------------------------------------------------------------------
vec3 pow3(vec3 a, float b)
{
return vec3(pow(a.x,b),pow(a.y,b),pow(a.z,b));
}
//--------------------------------------------------------------------------------------------------
vec3 mix3(vec3 a, vec3 b, float c)
{
return a * (1 - c) + c * b;
}
//--------------------------------------------------------------------------------------------------
 float radicalInverse_VdC(uint bits) 
 {
     bits = (bits << 16u) | (bits >> 16u);
     bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
     bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
     bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
     bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
     return float(bits); // / 0x100000000
 }
//--------------------------------------------------------------------------------------------------
 vec2 hammersley2d(uint i, uint N, uvec2 Random)
 {
	float X =  fract(float(i)/float(N) + float( Random.x & uint( 0xffff) ) / (1<<16) );
	float Y = float(uint(radicalInverse_VdC(i)) ^ Random.y) *  2.3283064365386963e-10;
    return vec2(X,Y);
 }
//--------------------------------------------------------------------------------------------------
uvec2 ScrambleTEA(uvec2 v)
{
	uint y 			= v[0];
	uint z 			= v[1];
	uint sum 		= uint(0);
	uint iCount 	= uint(4);
	 for(uint i = uint(0); i < iCount; ++i)
	{
		sum += uint(0x9e3779b9);
		y += (z << 4u) + 0xA341316Cu  ^ z + sum ^ (z >> 5u) + 0xC8013EA4u;
		z += (y << 4u) + 0xAD90777Du ^ y + sum ^ (y >> 5u) +  0x7E95761Eu;
	}
	return uvec2(y, z);
}


 float normal_distrib(
 float ndh,
   float Roughness)
 {
   // use GGX / Trowbridge-Reitz, same as Disney and Unreal 4
   // cf http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p3
   float alpha = Roughness * Roughness;
   float tmp = alpha / max(1e-8,(ndh*ndh*(alpha*alpha-1.0)+1.0));
   return tmp * tmp * M_INV_PI;
}
 
float probabilityGGX(float ndh, float vdh, float Roughness)
{
	return normal_distrib(ndh, Roughness) * ndh / (4.0*vdh);
}
 
//--------------------------------------------------------------------------------------------------
float getLOD(float roughness)
{
 // return 3;
  return max((maxLod-4) * min(roughness *3,1),3);
}
//--------------------------------------------------------------------------------------------------
float getMipFB3(float roughness,float vdh,float ndh,float SamplesNum)
{
	float width = 500 ;
	float omegaS = 1 /  (SamplesNum *probabilityGGX(ndh, vdh, max(roughness,0.01)));
	float omegaP = 4.0 * 3.141592 / (6.0 * width * width ) ;
	return clamp (0.5 * log2 ( omegaS / omegaP ) , 3, maxLod );
}
//--------------------------------------------------------------------------------------------------
// Appoximation of joint Smith term for GGX
// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
float Vis_SmithJointApprox( float Roughness, float NoV, float NoL )
{
	float a = Roughness * Roughness;
	float Vis_SmithV = NoL * ( NoV * ( 1 - a ) + a );
	float Vis_SmithL = NoV * ( NoL * ( 1 - a ) + a );
	return 0.5 / ( Vis_SmithV + Vis_SmithL );
}
//--------------------------------------------------------------------------------------------------
// [Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"]
vec3 F_Schlick( vec3 SpecularColor, float VoH )
{
	float Fc = pow( 1 - VoH, 5);					
	// Anything less than 2% is physically impossible and is instead considered to be shadowing
	return min( 50.0 * SpecularColor.g, 1) * Fc + (1 - Fc) * SpecularColor;
}

vec3 importanceSampleGGX(vec2 Xi, vec3 A, vec3 B, vec3 C, float roughness)
{
  float a = roughness*roughness;
  float cosT = sqrt((1.0-Xi.y)/(1.0+(a*a-1.0)*Xi.y));
  float sinT = sqrt(1.0-cosT*cosT);
  float phi = 2.0*M_PI*Xi.x;
  return (sinT*cos(phi)) * A + (sinT*sin(phi)) * B + cosT * C;
}

vec4 ComputeUE4BRDF(vec3 diffColor, float metallic, float roughness, float occlusion, int SamplesNum)
{
	vec3 normal_vec = getNormal();
  //- Double-sided normal if back faces are visible
/* if (isBackFace()) 
  {
    normal_vec = -normal_vec;
  }*/

  vec3 eye_vec = normalize(camera.position - WSPosition);
  float ndv =   dot(eye_vec, normal_vec);//abs(dot(eye_vec, normal_vec)) + 1e-5f;//
  //- Trick to remove black artefacts
  //- Backface ? place the eye at the opposite - removes black zones
  if (ndv < 0) 
  {
    eye_vec = reflect(eye_vec, normal_vec);
    ndv = abs(ndv);
  }


	

  //- Diffuse contribution
  vec3 diffuseIrradiance	= texture(uIrradianceMap, normal_vec).rgb;
  vec3 contribE 			= occlusion * diffuseIrradiance * diffColor * (1.0 - metallic);
  vec3 specColor 			= mix3(vec3(0.04), diffColor, metallic);
  float lerp2uniform		= max(2*(roughness  - 0.5) ,0.0001);

  //- Specular contribution
 
	
  //- Create a local basis for BRDF work
  vec3 Tp = normalize(fs_in.tangent - normal_vec*dot(fs_in.tangent, normal_vec)); // local tangent
  vec3 Bp = normalize(fs_in.bitangent - normal_vec*dot(fs_in.bitangent, normal_vec)- Tp*dot(fs_in.bitangent, Tp)); // local bitangent
  vec3 contribS = vec3(0.0);
  uvec2 Random = ScrambleTEA(uvec2(gl_FragCoord.xy ));

  //- Brute force sampling 
  if (false)
  {
  SamplesNum = 512;
  }

  for(int i = 0; i < SamplesNum; ++i)
  {
    vec2 Xi 	 = hammersley2d(uint(i), uint(SamplesNum),Random);
    vec3 Hn 	 = importanceSampleGGX(Xi,Tp,Bp,normal_vec,roughness);
    vec3 Ln 	 = -reflect(eye_vec,Hn);

    float ndl 	= max(1e-8, (dot(normal_vec, Ln)));
    float vdh 	= max(1e-8, dot(eye_vec, Hn));
    float ndh 	= max(1e-8, dot(normal_vec, Hn));
	float PDF	= (4 * vdh / ndh);
	 
	//Blend importance sampled Irradiance with Uniform distribution Irradiance when roughness --> 1 
	float mip = getMipFB3( roughness, vdh, ndh, SamplesNum);	
	vec3 irr = textureLod(uRadianceMap, Ln, 0).rgb;
	vec3 specularIrradiance = mix3(irr, diffuseIrradiance, lerp2uniform);
	contribS +=  specularIrradiance * F_Schlick(specColor, vdh ) * clamp(Vis_SmithJointApprox( roughness, ndv, ndl ) * PDF * ndl ,0,1);
  }
  // Remove occlusions on shiny reflections
  contribS *= mix(occlusion, 1.0, roughness) / float(SamplesNum);

  //- Emissive
  //vec3 contribEm = emissive_intensity * texture2D(emissive_tex, fs_in.uv).rgb;

  //- Sum diffuse + spec + emissive
  return vec4(contribS + contribE, 1.0);
 // return vec4(contribE, 1.0);
}





























#define saturate(x) clamp(x, 0.0, 1.0)

void main() 
{


	float roughness = getRoughness();
	float occlusion = 1;
	int nbSamples = 1;
	float roughness2 = pow(roughness, 2.2);
	float roughness4 = pow(roughness, 4);

	float metalness = pow(getMetalness(),2.2);
	
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
        
	//vec3 N = normalize(VSNormal);
	
	//vec3 V = normalize(-VSPosition);
	//
	
    float NdotH = saturate(dot(N, H));
    float LdotH = saturate(dot(L, H));
	float NdotL = saturate(dot(N, L));
    float NdotV = abs(dot(N, V)) + 1e-5f;
	float VdotH = saturate(dot(V, H));

	vec3 diffuseColor	= albedo * (1 - metalness);
	vec3 specularColor = mix(vec3(0.04), albedo, metalness);
	
	
	
	
	
	
	
	
	
	
	
	
	float distribution		= getNormalDistribution( roughness, NdotH );
	//vec3 fresnel			= getFresnel( specularColor, NdotV );
	float geom				= getGeometricShadowing( roughness, NdotV, NdotL, VdotH, L, V );

	// get the specular and diffuse and combine them
	//vec3 diffuse			= getDiffuse( diffuseColor, roughness, NdotV, NdotL, VdotH );
	//vec3 specular			= NdotL * ( distribution * fresnel * geom );
	
	
	
	
	vec3 fresnel = Fresnel_Schlick(specularColor, NdotV);
	float vis = V_SmithGGXCorrelated(NdotV, NdotL, roughness);
	float D = D_GGX(NdotH, roughness);
	vec3 Fr = D * fresnel *vis/PI;
	 
	float Fd = Fr_DisneyDiffuse(NdotV, NdotL, LdotH, roughness2 )/PI;
	
	
	int numMips			= 7;
	float lod 			= (1.0 - roughness)*(numMips - 1.0);
	float mip			= numMips - 1 + log2(roughness2);
	vec3 lookup			= -reflect(V, N);
	//lookup			= fix_cube_lookup(lookup, 256, mip );
	
	//float3 vReflection = 2.0f * vNormal * dot(vViewDirection, vNormal) - vViewDirection;
	//float fA = fRoughness * fRoughness;
	//vReflection = lerp(vNormal, vReflection, (1.0f - fA) * (sqrt(1.0f - fA) + fA));
	
	
	
	vec3 radiance		= textureLod(uRadianceMap, R, mip).rgb;
	vec3 irradiance		= texture(uIrradianceMap, N).rgb;
	
	
	
	
	vec3 reflectance = EnvBRDFApprox(specularColor, roughness4, NdotV);
	
	vec3 diffuse  		= diffuseColor * irradiance;
    vec3 specular 		= reflectance * radiance;
	vec3 color			= diffuse + specular;
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

	
	
	//vec3 color				= ( diffuse * irradiance + specular * radiance);
	
	float shadow = ShadowCalculation(fragPosLightSpace, WSPosition, N);   
	shadow = min(shadow, 0.15);
	//color *=  (1.0 - shadow);
	
	fragColor = vec4( ComputeUE4BRDF(albedo, metalness, roughness * roughness, occlusion, nbSamples).rgb,1.0f);
	//fragColor = vec4(eye_vec, 1.0);
	
	
	
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
