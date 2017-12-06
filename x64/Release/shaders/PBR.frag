
const float PI = 3.14159265;
const float TwoPI = 6.28318530718;

float uBrightness = 1;
int maxLOD = 9; 

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
uniform samplerCube uRadianceMap;
uniform samplerCube uIrradianceMap;
uniform samplerCube uSpecularMap;

uniform sampler2D uBRDFLUT;

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
		
		mat3 TBN = mat3(fs_in.tangent, fs_in.bitangent, fs_in.normal);
		vec3 normal = TBN * normalMap;
	
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



vec3 cubemapSeamlessFixDirection(const in vec3 direction, const in float scale )
{
    vec3 dir = direction;
    // http://seblagarde.wordpress.com/2012/06/10/amd-cubemapgen-for-physically-based-rendering/
    float M = max(max(abs(dir.x), abs(dir.y)), abs(dir.z));

    if (abs(dir.x) != M) dir.x *= scale;
    if (abs(dir.y) != M) dir.y *= scale;
    if (abs(dir.z) != M) dir.z *= scale;

    return dir;
}

/*vec3 textureCubeLodEXTFixed(const in samplerCube texture, const in vec3 direction, const in float lodInput )
{
    float lod = min( uEnvironmentLodRange[0], lodInput );

    // http://seblagarde.wordpress.com/2012/06/10/amd-cubemapgen-for-physically-based-rendering/
    float scale = 1.0 - exp2(lod) / uEnvironmentSize[0];
    vec3 dir = cubemapSeamlessFixDirection( direction, scale);

    return textureLod(texture, dir, lod ).rgb;
}*/

float occlusionHorizon( const in vec3 R, const in vec3 normal)
{
   // if ( uOcclusionHorizon == 0)
   //     return 1.0;

// http://marmosetco.tumblr.com/post/81245981087
// TODO we set a min value (10%) to avoid pure blackness (in case of pure metal)
    float factor = clamp( 1.0 + 1.3 * dot(R, normal), 0.1, 1.0 );
    return factor * factor;
}


float linRoughnessToMipmap( float roughnessLinear )
{
    return sqrt(roughnessLinear);
}


#define FLT_EPSILON     1.192092896e-07f  

const int nMipOffset = 3;
const int nMips = 11;
//const float fUserScale = .3864;

float RoughnessFromPerceptualRoughness(float fPerceptualRoughness)
{
    return fPerceptualRoughness*fPerceptualRoughness;
}
 
float PerceptualRoughnessFromRoughness(float fRoughness)
{
    return sqrt(max(0.0,fRoughness));
}
 

float SpecularPowerFromPerceptualRoughness(float fPerceptualRoughness)
{
    float fRoughness = RoughnessFromPerceptualRoughness(fPerceptualRoughness);
    return (2.0/max(FLT_EPSILON, fRoughness*fRoughness))-2.0;
}
 
float PerceptualRoughnessFromSpecularPower(float fSpecPower)
{
    float fRoughness = sqrt(2.0/(fSpecPower + 2.0));
    return PerceptualRoughnessFromRoughness(fRoughness);
}

float roughnessToMip(float roughness, float NdotR) // log2 distribution
{
	//float temp = (2.0/(roughness*roughness))-2.0;
	//return (nMips-1-nMipOffset) - log2(temp)*fUserScale;

   float fSpecPower = SpecularPowerFromPerceptualRoughness(roughness);
   fSpecPower /= (4*max(NdotR, FLT_EPSILON));		// see section "Pre-convolved Cube Maps vs Path Tracers"
   float fScale = PerceptualRoughnessFromSpecularPower(fSpecPower);

 //  float fScale = roughness*(1.7 - 0.7*roughness);    // approximate remap from LdotR based distribution to NdotH
   return fScale*(nMips- 1- nMipOffset);
  // return roughness*(nMips-1-nMipOffset);


}

float BurleyToMip(float fPerceptualRoughness, float NdotR)
{
    float fSpecPower = SpecularPowerFromPerceptualRoughness(fPerceptualRoughness);
    fSpecPower /= (4*max(NdotR, FLT_EPSILON));      // see section "Pre-convolved Cube Maps vs Path Tracers"
    float fScale = PerceptualRoughnessFromSpecularPower(fSpecPower);
    return fPerceptualRoughness*(nMips-1-nMipOffset);
}
 



vec3 prefilterEnvMap( float roughnessLinear, const in vec3 N, const in vec3 R )
{
    float lod = roughnessToMip(roughnessLinear, dot(N,R)); //( uEnvironmentMaxLod - 1.0 );
    return textureLod(uSpecularMap, R, lod).rgb;//textureCubeLodEXTFixed( uRadianceMap, R, lod );
}

vec3 getSpecularDominantDir( const in vec3 N, const in vec3 R, const in float realRoughness ) {
    float smoothness = 1.0 - realRoughness;
    return mix( N, R, smoothness * ( sqrt( smoothness ) + realRoughness ) );
}

vec2 integrateBRDF( float r, float NoV )
{
    vec4 rgba = texture2D( uBRDFLUT, vec2(NoV, r) );

    const float div = 1.0/65535.0;
    float b = (rgba[3] * 65280.0 + rgba[2] * 255.0);
    float a = (rgba[1] * 65280.0 + rgba[0] * 255.0);

    return vec2( rgba[0],rgba[1] );// * div;
}

// https://www.unrealengine.com/blog/physically-based-shading-on-mobile
vec3 integrateBRDFApprox( const in vec3 specular, float roughness, float NoV )
{
    const vec4 c0 = vec4( -1, -0.0275, -0.572, 0.022 );
    const vec4 c1 = vec4( 1, 0.0425, 1.04, -0.04 );
    vec4 r = roughness * c0 + c1;
    float a004 = min( r.x * r.x, exp2( -9.28 * NoV ) ) * r.x + r.y;
    vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;
    return specular * AB.x + AB.y;
}

vec3 approximateSpecularIBL( const in vec3 specularColor,
                             float rLinear,
                             const in vec3 N,
                             const in vec3 V )
{

    float roughnessLinear = max( rLinear, 0.0);
    float NoV = clamp( dot( N, V ), 0.0, 1.0 );
    vec3 R = normalize( (2.0 * NoV ) * N - V);


	
    vec3 vR = getSpecularDominantDir(N, R, rLinear*rLinear);
    float RdotNsat = saturate(dot(N, R));
 

    float l = BurleyToMip(rLinear, RdotNsat);

    // From Sebastien Lagarde Moving Frostbite to PBR page 69
    // so roughness = linRoughness * linRoughness
   // vec3 dominantR = getSpecularDominantDir( N, R,  roughnessLinear);

   // vec3 dir = ( vec4(dominantR, 0)).rgb;
    vec3 prefilteredColor = textureLod(uSpecularMap, vR, l).rgb; //prefilterEnvMap( roughnessLinear, N, dir );


   // marmoset tricks
    //prefilteredColor *= occlusionHorizon( dominantR, (camera.mView * vec4(N,0)).rgb );

	//return uBrightness * prefilteredColor * integrateBRDFApprox( specularColor, roughnessLinear, NoV );
    vec4 envBRDF = texture2D( uBRDFLUT, vec2(NoV, roughnessLinear));
	
	//return vec3(roughnessToMip(roughnessLinear)/7);
	return prefilteredColor * ( specularColor * envBRDF.x + envBRDF.y );
}

 
float bias(float value, float b)
{
    return (b > 0.0) ? pow(value, log(b) / log(0.5)) : 0.0;
}
 

// contrast function.
float gain(float value, float g)
{
    return 0.5 * ((value < 0.5) ? bias(2.0 * value, 1.0 - g) : (2.0 - bias(2.0 - 2.0 * value, 1.0 - g)));
}
 
 
 
float EmpiricalSpecularAO(float ao, float perceptualRoughness)
{
    // basically a ramp curve allowing ao on very diffuse specular
    // and gradually less so as the reflection hardens.
    float fSmooth = 1-perceptualRoughness;
    float fSpecAo = gain(ao,0.5+max(0.0,fSmooth*0.4));
 
    return min(1.0,fSpecAo + mix(0.0, 0.5, fSmooth*fSmooth*fSmooth*fSmooth));
}


vec3 computeIBL_UE4( const in vec3 normal,
                     const in vec3 view,
                     const in vec3 albedo,
                     const in float roughness,
                     const in vec3 specular,
					 const in float ao)
{

    vec3 color = vec3(0.0);
    if ( albedo != color ) { // skip if no diffuse
       color += uBrightness * albedo * texture(uIrradianceMap, normal).rgb * ao;//* evaluateDiffuseSphericalHarmonics(normal,view );
    }
	//color = texture(uRadianceMap, normal).rgb;
    color += approximateSpecularIBL(specular, roughness, normal, view) * EmpiricalSpecularAO(ao, roughness);
	//float mip = (roughness <.01) ? 1 : 0;
	//color = vec3(mip);
    return color;
}
float adjustRoughness ( float inputRoughness , float avgNormalLength )
{
	// Based on The Order : 1886 SIGGRAPH course notes implementation
	if ( avgNormalLength < 1.0f)
	{
	float avgNormLen2 = avgNormalLength * avgNormalLength ;
	float kappa = (3 * avgNormalLength - avgNormalLength * avgNormLen2 ) /
	(1 - avgNormLen2 ) ;
	float variance = 1.0f / (2.0 * kappa ) ;
	 return sqrt ( inputRoughness * inputRoughness + variance );
	}
	return ( inputRoughness );
 }

 float D_GGX(float ndoth, float m)
{
	float m2 = m* m;
	float f = (ndoth * m2 - ndoth) * ndoth + 1;
	return m2 / (f * f);
}










































// lights
uniform vec3 lightPositions[4] = vec3[4](
		vec3(-6.0f,  6.0f, 6.0f),
        vec3( 6.0f,  6.0f, 6.0f),
        vec3(-6.0f, -6.0f, 6.0f),
        vec3( 6.0f, -6.0f, 6.0f));

uniform vec3 lightColors[4]= vec3[4](
		vec3(1.0f, 1.0f,1.0f),
        vec3(1.0f, 1.0f,1.0f),
        vec3(1.0f,1.0f,1.0f),
        vec3(1.0f,1.0f,1.0f));
  
  



// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}   
// ----------------------------------------------------------------------------




// marmoset horizon occlusion http://marmosetco.tumblr.com/post/81245981087
float ApproximateSpecularSelfOcclusion(vec3 vR, vec3 vertNormalNormalized)
{
    const float fFadeParam = 1.3;
    float rimmask = clamp( 1 + fFadeParam * dot(vR, vertNormalNormalized), 0.0, 1.0);
    rimmask *= rimmask;
 
    return rimmask;
}
 


 
// frostbite presentation (moving frostbite to pbr)
vec3 GetSpecularDominantDir(vec3 vN, vec3 vR, float fRealRoughness)
{
    float fInvRealRough = saturate(1 - fRealRoughness);
    float lerpFactor = fInvRealRough * (sqrt(fInvRealRough)+fRealRoughness);
 
    return mix(vN, vR, lerpFactor);
}


 

float GetReductionInMicrofacets(float perceptualRoughness)
{
    // this is not needed if you separately precompute an integrated FG term such as proposed
    // by epic. Alternatively this simple analytical approximation retains the energy
    // loss associated with Integral GGX(NdotH)*NdotH * (NdotL>0) dH which
    // for GGX equals 1/(roughness^2+1) when integrated over the half sphere.
    // without the NdotL>0 indicator term the integral equals one.
    float roughness = RoughnessFromPerceptualRoughness(perceptualRoughness);
    return 1.0 / (roughness*roughness+1.0);
}





 
vec3 EvalBRDF(vec3 vN, vec3 org_normal, vec3 to_cam,vec3 V,  float perceptualRoughness, float metalness, vec3 albedo, float ao)
{
    const int numMips = 11;
    const int nrBrdfMips = numMips-3;

    float VdotN = clamp(dot(to_cam, vN), 0.0, 1.0f);    // same as NdotR
    vec3 vRorg = 2*vN*VdotN-to_cam;
 
    vec3 vR = GetSpecularDominantDir(vN, vRorg, RoughnessFromPerceptualRoughness(perceptualRoughness));
    float RdotNsat = saturate(dot(vN, vR));
 

    float l = BurleyToMip(perceptualRoughness, RdotNsat);

 
 
    // fxcomposer uses a right hand coordinate frame (unlike d3d which uses left)
    // and has Y axis up. We've exported accordingly in Lys. For conventional
    // d3d11 just set Y axis as up in Lys before export.
	
    vec3 specRad = textureLod(uSpecularMap, vR,  l).rgb;
	 
    vec3 diffRad = texture(uIrradianceMap, vN).rgb;
 
 
    vec3 spccol = mix( vec3 (0.04), albedo, metalness);
    vec3 dfcol = mix( vec3( 0.0), albedo, 1 - metalness);
 
    // fresnel
    float fT = 1.0-RdotNsat;
    float fT5 = fT*fT; fT5 = fT5*fT5*fT;
    spccol = mix(spccol, vec3( 1.0), fT5);
 

    // take reduction in brightness into account.
    float fFade = 1;//GetReductionInMicrofacets(perceptualRoughness);
    fFade *= EmpiricalSpecularAO(ao, perceptualRoughness);
	//fFade *= ApproximateSpecularSelfOcclusion(vR, org_normal);

  	vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metalness);
 	vec3 F = fresnelSchlickRoughness(max(dot(vN, V), 0.0), F0, perceptualRoughness);
    vec2 brdf  = texture(uBRDFLUT, vec2(max(dot(vN, V), 0.0), perceptualRoughness)).rg;
	vec3 specular = specRad * (F * brdf.x + brdf.y);

    // final result
    return ao* dfcol*diffRad + fFade * specular ;
}
 
centroid in vec3 WSNormalCentroid;

const float CURV_MOD = 2;
const float MAX_POW = 2000;
const float POW_MOD_MIN = 5;
const float POW_MOD_MAX = 15;
void main() 
{
	//TODO:: prefilter normal distribution
	//http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr_v2.pdf
	//pg 92

	//float vertexColor = getVertexColor();
	
	
	
	float ao = getAO();
	int nbSamples = 32;
	//float roughness2 = pow(roughness, 2);
	//float roughness4 = pow(roughness, 4);

	float metalness = getMetalness();
	
	vec3 albedo = getAlbedo() * getVertexColor();
	vec3 normal = getNormal();
		float roughness = getRoughness() ;


	

	vec3 eye_vec = normalize(camera.position - WSPosition);
	float ndv = dot(eye_vec, normal);
  
	vec3 ecEyeDir = normalize(-VSPosition);
    //direction towards the camera in the world space
    vec3 wcEyeDir = vec3(camera.mInvView * vec4(ecEyeDir, 0.0));
	if(!gl_FrontFacing)
		normal*= -1;


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


	vec3 ddxN = dFdx(normal);
	vec3 ddyN = dFdy(normal);
	float curv2 = max( dot( ddxN, ddxN ), dot( ddyN, ddyN ) );
	float gloss_max = -0.0909 - 0.0909 * log2(CURV_MOD*curv2);
	float gloss = min(1-roughness, gloss_max);
	gloss = min(MAX_POW, exp2(1 + mix(POW_MOD_MIN, POW_MOD_MAX, gloss )));
	//Compute specular this way:

	float D = (gloss  + 1) * pow(NdotH, gloss);
	//specularColor = lightColor.rgb * NdotL * D;

	vec3 vNormalWsDdx = dFdx( normal );
	vec3 vNormalWsDdy = dFdy( normal );

	float flGeometricRoughnessFactor = pow( saturate( max( dot( vNormalWsDdx, vNormalWsDdx ), dot( vNormalWsDdy, vNormalWsDdy ) ) ), 0.33);
	curv2= flGeometricRoughnessFactor;
	roughness = max(roughness, flGeometricRoughnessFactor);
	//vRoughness.xy = max( vRoughness.xy, flGeometricRoughnessFactor.xx )


	vec3 diffuseColor =  albedo * (1 - metalness);
	vec3 specularColor = mix(vec3(0.04), albedo, metalness);

	//vec3 irradiance = texture(uIrradianceMap, N).rgb;
	
	
	//lookup			= fix_cube_lookup(lookup, 256, mip );
	
	//float3 vReflection = 2.0f * vNormal * dot(vViewDirection, vNormal) - vViewDirection;
	//float fA = fRoughness * fRoughness;
	//vReflection = lerp(vNormal, vReflection, (1.0f - fA) * (sqrt(1.0f - fA) + fA));
	
	
	
	//vec3 radiance		= textureLod(uRadianceMap, R, mip).rgb;

	//vec3 diffuse  		= Fd *  irradiance;
   // vec3 specular 		= Fr * radiance;
	//vec3 color			= diffuse + specular;

	//color = pow(color, vec3(1/2.2));

	float shadow = ShadowCalculation(fragPosLightSpace, WSPosition, N);   
	shadow = min(shadow, 0.8);
	//color *=  (1.0 - shadow);
	//vec4 t = texture2D( uBRDFLUT, fs_in.uv);
	
//vec4(integrateBRDF(fs_in.uv.x, fs_in.uv.y),0,1);//


	vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metalness);
   
           
    // reflectance equation
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) 
    {

		// calculate per-light radiance
        vec3 L = normalize(lightPositions[i] - WSPosition);
        vec3 H = normalize(V + L);
        float distance = length(lightPositions[i] - WSPosition);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i];// * attenuation;

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);    
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);        
        
		vec3 nominator    = NDF * G * F;
        float denominator = 4 * max(dot(N,V), 0.0) * max(dot(N,L), 0.0) + 0.001; // 0.001 to prevent divide by zero.
        vec3 specular = nominator / denominator;

         // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals 
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metalness;	  
             
            
        // scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);        

        // add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * radiance * NdotL; // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    }   
  
    //vec3 ambient = vec3(0.03) * albedo * ao;
    //vec3 color = ambient + Lo;
	
    //color = color / (color + vec3(1.0));





	vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metalness;	  
    
    vec3 irradiance = texture(uIrradianceMap, N).rgb;
    vec3 diffuse    = irradiance * albedo;
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 7.0;
    vec3 prefilteredColor = textureLod(uSpecularMap, getSpecularDominantDir(N, R, roughness),  roughnessToMip(roughness, dot(N,R))).rgb;    
    vec2 brdf  = texture(uBRDFLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + specular) * ao;
    
    vec3 color = ambient + Lo;

    // HDR tonemapping
   // color = color / (color + vec3(1.0));
    // gamma correct
    //color = pow(color, vec3(1.0/2.2)); 

    fragColor = vec4((kD * diffuse + specular) , 1.0);



	float roughnessLinear = max( roughness, 0.0);
    float NoV = clamp( dot( N, V ), 0.0, 1.0 );
    vec4 envBRDF = clamp(texture2D( uBRDFLUT, vec2(NoV, roughnessLinear)), 0.0, 1.0 );

	//fragColor = vec4(EvalBRDF(N, fs_in.normal, eye_vec, V, roughness, metalness, albedo, ao), 1);



	// Lo + *  (1.0 - shadow)
	float haha = dot(WSNormal,WSNormal) ;
	vec3 col = vec3(haha,haha, haha);
	if(haha >= 1.01)
	 col = vec3(1,0, 0);
	//else
	//haha = 0;
	if(flGeometricRoughnessFactor> roughness)
	haha = flGeometricRoughnessFactor - roughness;
	else haha = 0;
	fragColor = vec4((computeIBL_UE4(N, V, diffuseColor, roughness, specularColor, ao)), 1.0f);//  vec4(envBRDF.xy,0,1);// vec4(normalize(((camera.mView * vec4(N, 0.0)) + 1) *.5).rgb, 1.0);//
}
