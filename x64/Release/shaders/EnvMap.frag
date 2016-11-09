#version 330

in vec3 wcNormal;

uniform sampler2D uEnvMap;

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

void main() {
    fragColor = texture2D(uEnvMap, envMapEquirect(normalize(wcNormal)));
}