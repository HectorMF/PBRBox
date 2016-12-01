
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
uniform samplerCube uRadianceMap;
uniform samplerCube uIrradianceMap;
uniform samplerCube uSpecularMap;

uniform sampler2D uIntegrateBRDF;

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

#ifdef USE_VERTEX_COLORS
	vec3 getVertexColor()
	{
		return fs_in.color.rgb;
	}
#else
	vec3 getVertexColor()
	{
		return vec3(1,1,1);
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

vec3 prefilterEnvMap( float roughnessLinear, const in vec3 R )
{
    float lod = linRoughnessToMipmap(roughnessLinear) * maxLOD; //( uEnvironmentMaxLod - 1.0 );
    return textureLod(uSpecularMap, R, lod).rgb;//textureCubeLodEXTFixed( uRadianceMap, R, lod );
}

vec3 getSpecularDominantDir( const in vec3 N, const in vec3 R, const in float realRoughness ) {
    float smoothness = 1.0 - realRoughness;
    return mix( N, R, smoothness * ( sqrt( smoothness ) + realRoughness ) );
}

vec2 integrateBRDF( float r, float NoV )
{
    vec4 rgba = texture2D( uIntegrateBRDF, vec2(NoV, r ) );

    const float div = 1.0/65535.0;
    float b = (rgba[3] * 65280.0 + rgba[2] * 255.0);
    float a = (rgba[1] * 65280.0 + rgba[0] * 255.0);

    return vec2( a, b ) * div;
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


    // From Sebastien Lagarde Moving Frostbite to PBR page 69
    // so roughness = linRoughness * linRoughness
    vec3 dominantR = getSpecularDominantDir( N, R, roughnessLinear*roughnessLinear );

    vec3 dir = ( vec4(dominantR, 0)).rgb;
    vec3 prefilteredColor = prefilterEnvMap( roughnessLinear, dir );


   // marmoset tricks
   // prefilteredColor *= occlusionHorizon( dominantR, VSNormal );

	return uBrightness * prefilteredColor * integrateBRDFApprox( specularColor, roughnessLinear, NoV );
   // vec2 envBRDF = integrateBRDF( roughnessLinear, NoV );
   // return uBrightness * prefilteredColor * ( specularColor * envBRDF.x + envBRDF.y );
}


vec3 computeIBL_UE4( const in vec3 normal,
                     const in vec3 view,
                     const in vec3 albedo,
                     const in float roughness,
                     const in vec3 specular)
{

    vec3 color = vec3(0.0);
    if ( albedo != color ) { // skip if no diffuse
        color += uBrightness * albedo * texture(uIrradianceMap, normal).rgb;//* evaluateDiffuseSphericalHarmonics(normal,view );
    }

    color += approximateSpecularIBL(specular, roughness, normal, view);

    return color;
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

	//vec3 irradiance = texture(uIrradianceMap, N).rgb;
	
	int numMips			= 7;
	float lod 			= (1.0 - roughness)*(numMips - 1.0);
	float mip			= numMips - 1 + log2(roughness2);
	vec3 lookup			= -reflect(V, N);
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
	shadow = min(shadow, 0.35);
	//color *=  (1.0 - shadow);
	fragColor = vec4(computeIBL_UE4(N, V, diffuseColor, roughness, specularColor), 1.0f);// vec4(normalize(((camera.mView * vec4(N, 0.0)) + 1) *.5).rgb,1.0);
}