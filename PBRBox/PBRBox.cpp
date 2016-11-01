// PBRBox.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


#include "stdafx.h"
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <sstream>
#include <iostream>
#include "Camera.h"
#include "MouseKeyboardInput.h"
#include "Model.h"
#include "ModelInstance.h"
#include "Shader.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#ifndef M_PI
#define M_PI 3.14156265
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "GeometryUtil.h"
#include "Texture.h"
#include "Scene.h"
#include "Renderer.h"
// test scenes

Camera* hostRendercam = NULL;

int screenWidth, screenHeight;

Scene scene;

RenderTarget* shadowTarget;
Renderer* renderer;

Material* shadowMat;
Material* diffuseMat;

Material* depthMat;
Mesh* depthQuad;

Material* envMat;
/* Handler for window re-size event. Called back when the window first appears and
whenever the window is re-sized with its new width and height */
void reshape(GLsizei newwidth, GLsizei newheight)
{
	// Set the viewport to cover the new window
	interactiveCamera->setResolution(newwidth, newheight);
	renderer->windowSize = glm::vec2(newwidth, newheight);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
////	glOrtho(0.0, screenWidth, screenHeight, 0.0, 0.0, 100.0);
//	glMatrixMode(GL_MODELVIEW);

	glutPostRedisplay();
}

void initializeScene()
{
	Texture diffuse = Texture("data\\BB8 New\\Body diff MAP.jpg");
	Texture environment = Texture("data\\Mono_Lake_B\\Mono_Lake_B_HiRes_TMap.jpg");

	Material normalMat;
	normalMat.shader = Shader("shaders\\NormalShader.vert", "shaders\\\NormalShader.frag");
	normalMat.environment = environment;

	Material* mirrorMat = new Material();
	mirrorMat->shader = Shader("shaders\\Mirror.vert", "shaders\\Mirror.frag");
	mirrorMat->environment = environment;

	envMat = new Material();
	envMat->shader = Shader("shaders\\EnvMap.vert", "shaders\\EnvMap.frag");
	envMat->environment = environment;

	diffuseMat = new Material();
	diffuseMat->shader = Shader("shaders\\Lambert.vert", "shaders\\Lambert.frag");
	diffuseMat->environment = environment;
	diffuseMat->diffuse = diffuse;


	depthMat = new Material();
	depthMat->shader = Shader("shaders\\Diffuse.vert", "shaders\\Diffuse.frag");

	//Model* model = new Model("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\x64\\Release\\data\\BB8 New\\bb8.fbx");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));


	Mesh* skyBoxQuad = new Mesh(Shapes::renderQuad(), envMat);
	scene.skybox = skyBoxQuad;

	Mesh* groundPlane = new Mesh(Shapes::plane(3, 3), diffuseMat);
	
	scene.add(groundPlane);


	depthQuad = new Mesh(Shapes::renderQuad(), depthMat);
	depthQuad->transform = glm::translate(depthQuad->transform, glm::vec3(0, 1, 0));
	//scene.add(depthQuad);

	Geometry sphereMesh = Shapes::sphere(.2);

	for (int x = -2; x <= 2; x++)
	{
		for (int z = -2; z <= 2; z++)
		{
			Mesh* sphere = new Mesh(sphereMesh, diffuseMat);
			sphere->transform = glm::translate(sphere->transform, glm::vec3(x * .5, .2, z * .5));
			scene.add(sphere);
		}
	}

	Geometry lightMesh = Shapes::sphere(.2);
	Mesh* light = new Mesh(lightMesh, mirrorMat);
	light->transform = glm::translate(light->transform, glm::vec3(1,1,1));
	scene.add(light);
	
	renderer = new Renderer();
	renderer->clearColor = glm::vec4(1, 0, 1, 1);
	shadowTarget = new RenderTarget();

	shadowMat = new Material();
	shadowMat->shader = Shader("shaders\\Shadow.vert", "shaders\\Shadow.frag");

	depthMat->shadowTex = shadowTarget->depthTexture;
	diffuseMat->shadowTex = shadowTarget->depthTexture;
	//Model* model = new Model("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\IrrigationTool.obj");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));
	
	//scene.add(model);
}

// display function called by glutMainLoop(), gets executed every frame 
void disp(void)
{
	// if camera has moved, reset the accumulation buffer

	// build a new camera for each frame on the CPU
	interactiveCamera->buildRenderCamera(hostRendercam);

	renderer->setRenderTarget(*shadowTarget);

	renderer->setOverrideMaterial(*shadowMat);

	renderer->renderShadow(scene);

	renderer->clearRenderTarget();

	renderer->clearOverideMaterial();

	//diffuseMat->shadowTex = shadowTarget->depthTexture;
	//glUniformMatrix4fv(t, 1, GL_FALSE, glm::value_ptr(shadowTarget->depthTexture));

	renderer->render(scene, *hostRendercam);
//	depthQuad->m_material->Bind();
//	depthQuad->render();
//	depthQuad->m_material->Unbind();
	glutSwapBuffers();
}

void deleteCudaAndCpuMemory() {

	delete hostRendercam;
	delete interactiveCamera;
}

int main(int argc, char** argv) {

	// create a CPU camera
	hostRendercam = new Camera();
	// initialise an interactive camera on the CPU side
	initCamera();
	interactiveCamera->buildRenderCamera(hostRendercam);

	//initHDR(); // initialise the HDR environment map
			   // initialise GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB); // specify the display mode to be RGB and single buffering
	glutInitWindowPosition(100, 100); // specify the initial window position
	glutInitWindowSize(scrwidth, scrheight); // specify the initial window size
	glutCreateWindow("PBRBox"); // create the window and set title
	//initGL(100, 100);
																		// initialise OpenGL:
	glClearColor(.3, .3, .3, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, scrwidth, 0.0, scrheight);
	fprintf(stderr, "OpenGL initialized \n");

	// register callback function to display graphics
	glutDisplayFunc(disp);
	glutIdleFunc(disp);
	// functions for user interaction
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialkeys);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	// initialise GLEW
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "glew initialized  \n");


	initializeScene();
	fprintf(stderr, "VBO created  \n");
	// enter the main loop and start rendering
	fprintf(stderr, "Entering glutMainLoop...  \n");
	printf("Rendering started...\n");
	glutMainLoop();

	deleteCudaAndCpuMemory();
}



