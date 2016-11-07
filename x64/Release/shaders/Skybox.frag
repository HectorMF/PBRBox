#version 330 core

in vec3 TexCoords;

out vec4 color;

uniform samplerCube uSkybox;

void main()
{    
    color = texture(uSkybox, TexCoords);
}
  