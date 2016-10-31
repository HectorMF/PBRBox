#version 330 core
//vertex position, normal and light position in the eye/view space
in vec3 ecPosition;
in vec3 ecNormal;
in vec3 ecLightPos;
in vec2 uv;

uniform sampler2D uEnvMap;

struct Camera
{
	vec3 mViewDirection;
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float     shininess;
};  


uniform Material material;
uniform Camera camera;

out vec4 fragColor;


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

    //calculate direction towards the light
    vec3 L = normalize(ecLightPos - ecPosition);

    //diffuse intensity
    float Id = lambert(L, N);

	//surface and light color, full white
    vec4 baseColor = toLinear(vec4(1.0)); 
    vec4 lightColor = toLinear(vec4(1.0)); 

     //lighting in the linear space
    vec4 finalColor = vec4(texture2D(material.diffuse, uv).rgb * lightColor.rgb * Id, 1.0);
   
    //direction towards they eye (camera) in the view (eye) space
    vec3 ecEyeDir = normalize(-ecPosition);
    //direction towards the camera in the world space
    vec3 wcEyeDir = vec3(camera.mInvView * vec4(ecEyeDir, 0.0));
    //surface normal in the world space
    vec3 wcNormal = vec3(camera.mInvView * vec4(ecNormal, 0.0));

    //reflection vector in the world space. We negate wcEyeDir as the reflect function expect incident vector pointing towards the surface
    vec3 reflectionWorld = reflect(-wcEyeDir, normalize(wcNormal));

    fragColor = texture2D(uEnvMap, envMapEquirect(reflectionWorld));  
}