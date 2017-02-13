#version 330 core
in vec3 TexCoords;
out vec4 fragColor;

uniform samplerCube uSpecularMap;

void main()
{    
    fragColor = vec4(textureLod(uSpecularMap, TexCoords, 0).rgb, 1);
}