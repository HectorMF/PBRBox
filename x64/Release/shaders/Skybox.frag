#version 330 core
in vec3 TexCoords;
out vec4 fragColor;

uniform samplerCube uRadianceMap;

void main()
{    
    fragColor = vec4(pow(textureLod(uRadianceMap, TexCoords, 0).rgb, vec3(1)), 1);
}