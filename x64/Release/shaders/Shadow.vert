#version 330 core

layout (location = 0) in vec3 position;

// Values that stay constant for the whole mesh.
uniform mat4 depthMVP;

void main(){
 gl_Position =  depthMVP * vec4(position,1);
}