#version 330 core
//vertex position, normal and light position in the eye/view space
in vec3 ecPosition;
in vec3 ecNormal;
in vec3 ecLightPos;
in vec2 uv;
in vec4 shadowCoord;

uniform sampler2D uEnvMap;
uniform sampler2D uShadowMap;

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
    float bias = max(0.05 * (1.0 - dot(wNormal, lightDir)), 0.005);
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

void main() {

     //normalize the normal, we do it here instead of vertex
     //shader for smoother gradients
    vec3 N = normalize(ecNormal);
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
    vec3 wcNormal = vec3(camera.mInvView * vec4(ecNormal, 0.0));
    vec3 reflectionWorld = reflect(-wcEyeDir, normalize(wcNormal));
	vec4 reflection = texture2D(uEnvMap, envMapEquirect(reflectionWorld));
	
	vec3 ambient = vec3(.4);
	
	float shadow = ShadowCalculation(fragPosLightSpace);   
	shadow = min(shadow, 0.75);
    vec3 lighting =  (1.0 - shadow) * finalColor ;//* reflection.rgb;  
    //reflection vector in the world space. We negate wcEyeDir as the reflect function expect incident vector pointing towards the surface
    fragColor = vec4(lighting, 1.0);//toGamma(finalColor);    
}