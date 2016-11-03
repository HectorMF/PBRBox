#version 330 core

float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 Uncharted2Tonemap(vec3 x) {
   return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

//Based on Filmic Tonemapping Operators http://filmicgames.com/archives/75
vec3 tonemapUncharted2(vec3 color) {
    float ExposureBias = 2.0;
    vec3 curr = Uncharted2Tonemap(ExposureBias * color);

    vec3 whiteScale = 1.0 / Uncharted2Tonemap(vec3(W));
    return curr * whiteScale;
}



vec3 tonemapFilmic(vec3 color) {
    vec3 x = max(vec3(0.0), color - 0.004);
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}














//vertex position, normal and light position in the eye/view space
in vec3 ecPosition;
in vec3 ecNormal;
in vec3 ecLightPos;
in vec2 uv1;
in vec4 shadowCoord;

uniform sampler2D uEnvMap;
uniform sampler2D uReflectionMap;

uniform sampler2D uShadowMap;


#ifdef USE_ALBEDO_MAP
	uniform sampler2D uAlbedoColor; //assumes sRGB color, not linear

	vec3 getAlbedo() 
	{
		return toLinear(texture2D(uAlbedoColor, vTexCord0).rgb);
	}
#else
	uniform vec4 uAlbedoColor; //assumes sRGB color, not linear

	vec3 getAlbedo() 
	{
		return toLinear(uAlbedoColor.rgb);
	}
#endif

#ifdef USE_ROUGHNESS_MAP
uniform sampler2D uRoughness; //assumes sRGB color, not linear
float getRoughness() {
    return texture2D(uRoughness, vTexCord0).r;
}
#else
uniform float uRoughness;
float getRoughness() {
    return uRoughness;
}
#endif


#ifdef USE_METALNESS_MAP
uniform sampler2D uMetalness; //assumes sRGB color, not linear
float getMetalness() {
    return toLinear(texture2D(uMetalness, vTexCord0).r);
}
#else
uniform float uMetalness;
float getMetalness() {
    return uMetalness;
}
#endif

#ifdef USE_NORMAL_MAP
uniform sampler2D uNormalMap;
#pragma glslify: perturb = require('glsl-perturb-normal')
vec3 getNormal() {
    vec3 normalRGB = texture2D(uNormalMap, vTexCord0).rgb;
    vec3 normalMap = normalRGB * 2.0 - 1.0;

    normalMap.y *= -1.0;

    vec3 N = normalize(vNormalView);
    vec3 V = normalize(vEyeDirView);

    vec3 normalView = perturb(normalMap, N, V, vTexCord0);
    vec3 normalWorld = vec3(uInverseViewMatrix * vec4(normalView, 0.0));
    return normalWorld;
}
#else
vec3 getNormal() {
    return normalize(vNormalWorld);
}
#endif


uniform sampler2D uAlbedoMap;
uniform sampler2D uMetallicMap;
uniform sampler2D uRoughnessMap;
uniform sampler2D uNormalMap;




in vec4 fragPosLightSpace;
in vec3 wPos;
in vec3 wNormal;
in vec3 lightPos;


struct Camera
{
	vec3 mViewDirection;
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};
uniform Camera camera;


struct PBRMaterial
{
	sampler2D albedo;
	sampler2D metallic;
	sampler2D roughness;
	sampler2D normal;
};


struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float     shininess;
};  


uniform Material material;

float ShadowCalculation(vec4 fragPosLightSpace)
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
    vec3 normal = normalize(wNormal);
    vec3 lightDir = normalize(lightPos - wPos);
    float bias = max(0.01 * (1.0 - dot(wNormal, lightDir)), 0.005);
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

out vec4 fragColor;

float beckmannDistribution(float x, float roughness) {
  float NdotH = max(x, 0.0001);
  float cos2Alpha = NdotH * NdotH;
  float tan2Alpha = (cos2Alpha - 1.0) / cos2Alpha;
  float roughness2 = roughness * roughness;
  float denom = 3.141592653589793 * roughness2 * cos2Alpha * cos2Alpha;
  return exp(tan2Alpha / roughness2) / denom;
}

float cookTorranceSpecular(
  vec3 lightDirection,
  vec3 viewDirection,
  vec3 surfaceNormal,
  float roughness,
  float fresnel) {

  float VdotN = max(dot(viewDirection, surfaceNormal), 0.0);
  float LdotN = max(dot(lightDirection, surfaceNormal), 0.0);

  //Half angle vector
  vec3 H = normalize(lightDirection + viewDirection);

  //Geometric term
  float NdotH = max(dot(surfaceNormal, H), 0.0);
  float VdotH = max(dot(viewDirection, H), 0.000001);
  float x = 2.0 * NdotH / VdotH;
  float G = min(1.0, min(x * VdotN, x * LdotN));
  
  //Distribution term
  float D = beckmannDistribution(NdotH, roughness);

  //Fresnel term
  float F = pow(1.0 - VdotN, fresnel);

  //Multiply terms and done
  return  G * F * D / max(3.14159265 * VdotN * LdotN, 0.000001);
}

vec3 Specular_Fresnel_Schlick( in vec3 SpecularColor, in vec3 PixelNormal, in vec3 LightDir )
{
    float NdotL = max( 0, dot( PixelNormal, LightDir ) );
    return SpecularColor + ( 1 - SpecularColor ) * pow( ( 1 - NdotL ), 5 );
}

vec3 Specular_Frenel_Schlick( in vec3 SpecularColor, in vec3 ViewDir, in vec3 LightDir ) 
{ 
	vec3 Half = normalize( ViewDir + LightDir ); 
	float HdotV = clamp( dot( Half, ViewDir ), 0, 1 ); 
	return SpecularColor + ( 1 - SpecularColor ) * pow( ( 1 - HdotV ), 5 ); 
}

vec3 Fresnel_Shlick(vec3 specularColor, float vdoth)
{
	return specularColor + (1.0 - specularColor) * pow(1.0 - vdoth, 5.0);
}

const float PI = 3.14159265;
const float TwoPI = 6.28318530718;

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


const float gamma = 2.2;

vec3 toLinear(vec3 v) {
  return pow(v, vec3(gamma));
}

vec4 toLinear(vec4 v) {
  return vec4(toLinear(v.rgb), v.a);
}

vec3 toGamma(vec3 v) {
  return pow(v, vec3(1.0 / gamma));
}

vec4 toGamma(vec4 v) {
  return vec4(toGamma(v.rgb), v.a);
}
float lambert(vec3 lightDirection, vec3 surfaceNormal) 
{
  return max(0.0, dot(lightDirection, surfaceNormal));
}

vec3 reflect(vec3 I, vec3 N) {
    return I - 2.0 * dot(N, I) * N;
}

vec3 EnvBRDFApprox( vec3 SpecularColor, float Roughness, float NoV ) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022 );
    const vec4 c1 = vec4( 1.0, 0.0425, 1.04, -0.04 );
    vec4 r = Roughness * c0 + c1;
    float a004 = min( r.x * r.x, exp2( -9.28 * NoV ) ) * r.x + r.y;
    vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;
    return SpecularColor * AB.x + AB.y;
}
float saturate(float f) {
    return clamp(f, 0.0, 1.0);
}


mat3 cotangentFrame(vec3 N, vec3 p, vec2 uv) {
  // get edge vectors of the pixel triangle
  vec3 dp1 = dFdx(p);
  vec3 dp2 = dFdy(p);
  vec2 duv1 = dFdx(uv);
  vec2 duv2 = dFdy(uv);

  // solve the linear system
  vec3 dp2perp = cross(dp2, N);
  vec3 dp1perp = cross(N, dp1);
  vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
  vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

  // construct a scale-invariant frame 
  float invmax = 1.0 / sqrt(max(dot(T,T), dot(B,B)));
  return mat3(T * invmax, B * invmax, N);
}

vec3 perturb(vec3 map, vec3 N, vec3 V, vec2 texcoord) {
  mat3 TBN = cotangentFrame(N, -V, texcoord);
  return normalize(TBN * map);
}

vec3 getNormal() {
    vec3 normalRGB = texture2D(uNormalMap, vec2(uv1.x, uv1.y)).rgb;
    vec3 normalMap = normalRGB * 2.0 - 1.0;

    normalMap.y *= -1.0;

    vec3 N = normalize(ecNormal);
    vec3 V = normalize(-ecPosition);

    vec3 normalView = perturb(normalMap, N, V, vec2(uv1.x, uv1.y));
    vec3 normalWorld = vec3(camera.mInvView * vec4(normalView, 0.0));
    return normalWorld;
}

void main() {
	vec2 uv = vec2(uv1.x,  uv1.y);
     //normalize the normal, we do it here instead of vertex
     //shader for smoother gradients
	vec3 nn = texture2D(uNormalMap, uv).rgb;
	
    vec3 N = normalize(getNormal());
    vec3 L = normalize(ecLightPos - ecPosition);
	vec3 H = normalize( camera.mViewDirection + (normalize(-ecLightPos)));
   
	//Geometric term
	float NdotH = max(dot(N, H), 0.0);
	float VdotH = max(dot(camera.mViewDirection, H), 0.000001);
	  
    //calculate direction towards the light

    //diffuse intensity
    float Id = lambert(L, N);
	
	//surface and light color, full white
    vec4 baseColor = toLinear(vec4(1.0)); 
    vec4 lightColor = toLinear(vec4(1.0)); 

    //lighting in the linear space
	vec3 specular = Fresnel_Shlick(vec3(.333), VdotH);
	
    vec3 finalColor = texture2D(material.diffuse, uv).rgb;

    //direction towards they eye (camera) in the view (eye) space
    vec3 ecEyeDir = normalize(-ecPosition);
    //direction towards the camera in the world space
    vec3 wcEyeDir = vec3(camera.mInvView * vec4(ecEyeDir, 0.0));
    //surface normal in the world space

    //vec3 wcNormal = vec3(camera.mInvView * vec4(N, 0.0));
	vec3 wcNormal =  N;
	
    vec3 reflectionWorld = reflect(-wcEyeDir, normalize(wcNormal));
	vec4 reflection;
	
	vec3 ambient = vec3(.4);
	bool isMetallic = (texture2D(uMetallicMap, uv).r + texture2D(uMetallicMap, uv).g + texture2D(uMetallicMap, uv).b )> 0.0;
	
	if(isMetallic)
		reflection = texture2D(uReflectionMap, envMapEquirect(reflectionWorld));
	else
		reflection = vec4(1);//texture2D(uEnvMap, envMapEquirect(reflectionWorld));
	 
	 
	 
	 
	 
	 
	 
	float uIor = 1.4;
	float metalness = texture2D(uMetallicMap, uv).r;
	float roughness = texture2D(uRoughnessMap, uv).r;
	vec3 albedo = texture2D(material.diffuse, uv).rgb;

	
	vec3 F0 = vec3(abs((1.0 - uIor) / (1.0 + uIor)));
    F0 = F0 * F0;
    //F0 = vec3(0.04); //0.04 is default for non-metals in UE4
    F0 = mix(F0, albedo, metalness);

    //float NdotV = saturate( dot( normalWorld, eyeDirWorld ) );
	float NdotV = saturate( dot( wcNormal, ecEyeDir ) );
    vec3 reflectance = EnvBRDFApprox( F0, roughness, NdotV );
	 
	 
	 
	 
	 
	vec3 diffuseColor = albedo * (1.0 - metalness);

    //TODO: No kd? so not really energy conserving
    //we could use disney brdf for irradiance map to compensate for that like in Frostbite
	vec3 reflectionColor = texture2D(uEnvMap, envMapEquirect(reflectionWorld)).rgb;
	vec3 irradianceColor = texture2D(uReflectionMap, envMapEquirect(reflectionWorld)).rgb;
    vec3 color = diffuseColor * irradianceColor + reflectionColor * reflectance;
	 
	 
	 
	 
	 
	 
	 
	 
	float shadow = ShadowCalculation(fragPosLightSpace);   
	shadow = min(shadow, 0.75);
    vec3 lighting =  (1.0 - shadow) * color;  
	
	
	color = color;
    //reflection vector in the world space. We negate wcEyeDir as the reflect function expect incident vector pointing towards the surface
    fragColor = vec4(color, 1.0);//toGamma(finalColor);    
}