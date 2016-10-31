#version 330 core
//vertex position, normal and light position in the eye/view space
in vec3 ecPosition;
in vec3 ecNormal;

struct Camera
{
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};

uniform Camera camera;

out vec4 fragColor;

void main() {
     //normalize the normal, we do it here instead of vertex
     //shader for smoother gradients
    vec3 n = normalize(ecNormal);
   
    //direction towards they eye (camera) in the view (eye) space
    vec3 ecEyeDir = normalize(-ecPosition);
    //direction towards the camera in the world space
    vec3 wcEyeDir = vec3(camera.mInvView * vec4(ecEyeDir, 0.0));
    //surface normal in the world space
    vec3 wcNormal = vec3(camera.mInvView * vec4(ecNormal, 0.0));

    //reflection vector in the world space. We negate wcEyeDir as the reflect function expect incident vector pointing towards the surface
    vec3 reflectionWorld = reflect(-wcEyeDir, normalize(wcNormal));

    fragColor = vec4((n + 1) * .5, 1.0);   
}