#version 330 core
in vec2 UV;

uniform sampler2D diffuse;


 out vec4 fragColor;
void main() 
{
    fragColor = vec4(texture2D(diffuse, UV).r, texture2D(diffuse, UV).r,texture2D(diffuse, UV).r, 1.0);    
}