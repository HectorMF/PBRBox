#version 330 core
in vec3 TexCoords;
out vec4 fragColor;

uniform samplerCube skybox;

void main()
{    
    fragColor = vec4(textureLod(skybox, TexCoords, 3).rgb, 1);
}