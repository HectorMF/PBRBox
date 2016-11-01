#version 330 core
in vec2 UV;

uniform sampler2D uShadowMap;


out vec4 fragColor;
void main() 
{
	float depthValue = texture(uShadowMap, UV).z;
    fragColor = vec4(vec3(depthValue), 1.0);    
}
