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
// test scenes

Camera* hostRendercam = NULL;

int screenWidth, screenHeight;

Scene scene;
/* Handler for window re-size event. Called back when the window first appears and
whenever the window is re-sized with its new width and height */
void reshape(GLsizei newwidth, GLsizei newheight)
{
	// Set the viewport to cover the new window
	interactiveCamera->setResolution(newwidth, newheight);
	glViewport(0, 0, screenWidth = newwidth, screenHeight = newheight);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
////	glOrtho(0.0, screenWidth, screenHeight, 0.0, 0.0, 100.0);
//	glMatrixMode(GL_MODELVIEW);

	glutPostRedisplay();
}

void initializeScene()
{
	Texture diffuse = Texture("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\x64\\Release\\data\\BB8 New\\Body diff MAP.jpg");
	Texture environment = Texture("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\x64\\Release\\data\\Mono_Lake_B\\Mono_Lake_B_HiRes_TMap.jpg");

	Material normal;
	normal.shader = Shader("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\NormalShader.vert", "C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\NormalShader.frag");
	normal.environment = environment;

	Material mirror;
	mirror.shader = Shader("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\Mirror.vert", "C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\Mirror.frag");
	mirror.environment = environment;

	Material envMat;
	envMat.shader = Shader("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\EnvMap.vert", "C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\EnvMap.frag");
	envMat.environment = environment;

	Material diffuseMat;
	diffuseMat.shader = Shader("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\Lambert.vert", "C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\Lambert.frag");
	diffuseMat.environment = environment;
	diffuseMat.diffuse = diffuse;

	//Model* model = new Model("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\x64\\Release\\data\\BB8 New\\bb8.fbx");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));

	scene.clearColor = glm::vec4(1, 0, 1, 1);
	Mesh* skyBoxQuad = new Mesh(Shapes::renderQuad(), envMat);
	scene.skybox = skyBoxQuad;

	Mesh* groundPlane = new Mesh(Shapes::plane(1, 1, 32, 32), mirror);
	scene.add(groundPlane);

	Geometry sphereMesh = Shapes::sphere(.1);
	for (int x = -1; x <= 1; x++)
	{
		for (int z = -1; z <= 1; z++)
		{
			Mesh* sphere = new Mesh(sphereMesh, normal);
			sphere->transform = glm::translate(sphere->transform, glm::vec3(x * .3, .1, z * .3));
			scene.add(sphere);
		}
	}


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
	
	scene.render(*hostRendercam);

	glutSwapBuffers();
}


void deleteCudaAndCpuMemory() {

	delete hostRendercam;
	delete interactiveCamera;
}

/* Initialize OpenGL Graphics */
void initGL(int w, int h)
{
	glViewport(0, 0, w, h); // use a screen size of WIDTH x HEIGHT
	glEnable(GL_TEXTURE_2D);     // Enable 2D texturing

	glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
	glLoadIdentity();
	glOrtho(0.0, w, h, 0.0, 0.0, 100.0);

	glMatrixMode(GL_MODELVIEW);    // Set the matrix mode to object modeling

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the window
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



