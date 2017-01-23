// PBRBox.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


#include "stdafx.h"
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
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

#include "PBRMaterial.h"
#include "UnlitMaterial.h"
#include "SkyboxMaterial.h"
#include "DDS.h"
#include "FluentMesh.h"
#include "Environment.h"
#include "Volume.h"
#include "EnvironmentTools.h"
// test scenes

Camera* hostRendercam = NULL;

int screenWidth, screenHeight;

Scene scene;

RenderTarget* shadowTarget;
Renderer* renderer;

Material* shadowMat;

Material* depthMat;
Mesh* depthQuad;
Mesh* gun1;
Mesh* hovercraft;
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
	
	glEnable(GL_FRAMEBUFFER_SRGB);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

	shadowTarget = new RenderTarget();

	GLuint brdfLUT = computeBRDFLUT(256); // load_brdf("data\\out128.raw");

	Environment* operatingRoom = new Environment();

	operatingRoom->radiance.id = create_texture("data\\neuroArm\\neuroArm_cube_radiance.dds");
	operatingRoom->irradiance.id = create_texture("data\\neuroArm\\neuroArm_cube_irradiance.dds");
	operatingRoom->specular.id = create_texture("data\\neuroArm\\neuroArm_cube_specular.dds");

	operatingRoom->radiance.textureType = GL_TEXTURE_CUBE_MAP;
	operatingRoom->irradiance.textureType = GL_TEXTURE_CUBE_MAP;
	operatingRoom->specular.textureType = GL_TEXTURE_CUBE_MAP;

	/*
	GLuint radiance = create_texture("data\\pisa\\pisa_cube_radiance.dds");
	GLuint irradiance = create_texture("data\\pisa\\pisa_cube_irradiance.dds");
	GLuint specular = create_texture("data\\pisa\\pisa_cube_specular.dds");

	GLuint radiance = create_texture("data\\winterForest\\winterForest_cube_radiance.dds");
	GLuint irradiance = create_texture("data\\winterForest\\winterForest_cube_irradiance.dds");
	GLuint specular = create_texture("data\\winterForest\\winterForest_cube_specular.dds");

	GLuint radiance = create_texture("data\\arches\\arches_cube_radiance.dds");
	GLuint irradiance = create_texture("data\\arches\\arches_cube_irradiance.dds");
	GLuint specular = create_texture("data\\arches\\arches_cube_specular.dds");*/

	Volume* vol = new Volume("data\\3L_768x768x768_type_uc_1channels.raw");

	Model* irrigationTool = new Model("data\\IrrigationTool.obj");
	
	PBRMaterial* irrigationMat = new PBRMaterial();

	irrigationMat->setEnvironement(operatingRoom);
	irrigationMat->setAlbedo(glm::vec4(.96f, .96f, .9686f, 1));
	irrigationMat->setMetalness(1);
	irrigationMat->setRoughness(.05f);
	irrigationMat->shadowTex = shadowTarget->depthTexture;
	irrigationMat->m_BRDFLUT = brdfLUT;

	Mesh* irrigation = new Mesh(irrigationTool->m_meshes[0], irrigationMat);
	irrigation->transform = glm::scale(irrigation->transform, glm::vec3(.05));
	irrigation->transform = glm::rotate(irrigation->transform, glm::radians(90.0f), glm::vec3(1,0,0));
	irrigation->transform = glm::rotate(irrigation->transform, glm::radians(90.0f), glm::vec3(0, 0, 1));
	irrigation->transform = glm::translate(irrigation->transform, glm::vec3(0, 0, 1));
	scene.add(irrigation);

	Mesh* skyBoxQuad = new Mesh(Shapes::cube(1), new SkyboxMaterial(operatingRoom));
	scene.skybox = skyBoxQuad;

	PBRMaterial* gunMat = new PBRMaterial();
	gunMat->setEnvironement(operatingRoom);
	gunMat->setAlbedoMap(Texture("data\\cerberus\\Cerberus_A.png"));
	gunMat->setMetalnessMap(Texture("data\\cerberus\\Cerberus_M.jpg"));
	gunMat->setRoughnessMap(Texture("data\\cerberus\\Cerberus_R.jpg"));
	gunMat->setNormalMap(Texture("data\\cerberus\\Cerberus_N.jpg", ColorSpace::Linear));
	gunMat->m_BRDFLUT = brdfLUT;


	//Resource<Texture>::Load("data\\cerberus\\Cerberus_A.png");












	depthMat = new Material();
	depthMat->shader = Shader("shaders\\Diffuse.vert", "shaders\\Diffuse.frag");


	Model* mandarineMesh = new Model("data\\mandarine\\Mandarine.obj");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));

	Geometry orange = mandarineMesh->m_meshes[0];

	UnlitMaterial* mandMat = new UnlitMaterial();

	mandMat->setDiffuseMap(Texture("data\\mandarine\\mandarine.jpg"));
	Mesh* mandarine = new Mesh(orange, mandMat);
	mandarine->transform = glm::translate(mandarine->transform, glm::vec3(0, -2, 0));
	scene.add(mandarine);



	Model* hovercraftModel = new Model("data\\hovercraft\\hovercraft.fbx");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));

	Geometry hovercraftGeo = hovercraftModel->m_meshes[2];

	PBRMaterial* hovercraftmat = new PBRMaterial();

	hovercraftmat->setEnvironement(operatingRoom);
	hovercraftmat->m_BRDFLUT = brdfLUT;


	hovercraftmat->setAlbedoMap(Texture("data\\hovercraft\\Base_Color_0.png"));
	hovercraftmat->setAmbientOcclusionMap(Texture("data\\hovercraft\\AO.png"));
	hovercraftmat->setNormalMap(Texture("data\\hovercraft\\NornalGL_0.png", ColorSpace::Linear));
	hovercraftmat->setMetalnessMap(Texture("data\\hovercraft\\Metalnes_0.png"));
	hovercraftmat->setRoughnessMap(Texture("data\\hovercraft\\Roughtnes_0.png"));

	hovercraft = new Mesh(hovercraftGeo, hovercraftmat);

	hovercraft->transform = glm::scale(hovercraft->transform, glm::vec3(.01, .01, .01));
	hovercraft->transform = glm::rotate(hovercraft->transform, glm::radians(90.0f),glm::vec3(1, 0, 0));
	hovercraft->transform = glm::rotate(hovercraft->transform, glm::radians(180.0f), glm::vec3(0, 1, 0));
	hovercraft->transform = glm::rotate(hovercraft->transform, glm::radians(90.0f), glm::vec3(0, 0, 1));
	scene.add(hovercraft);



	Model* skullModel = new Model("data\\skull\\skull.OBJ");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));

	Geometry skullGeo = skullModel->m_meshes[0];

	PBRMaterial* skullmat = new PBRMaterial();

	skullmat->setEnvironement(operatingRoom);
	skullmat->m_BRDFLUT = brdfLUT;
	skullmat->setAlbedo(glm::vec4(255, 219, 145, 255) / 255.0f);
	//skullmat->setAlbedoMap(Texture("data\\skull\\albedo.jpg"));
	skullmat->setNormalMap(Texture("data\\skull\\Normal.jpg", ColorSpace::Linear));
	skullmat->setMetalness(1);
	skullmat->setRoughness(.15);
	Mesh* skull = new Mesh(skullGeo, skullmat);
	skull->transform = glm::translate(skull->transform, glm::vec3(0, -5, 0));
	scene.add(skull);



	Model* model = new Model("data\\cerberus\\Cerberus.obj");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));
	
	Geometry gun = model->m_meshes[0];

	gun1 = new Mesh(gun, gunMat);

	//gun1->transform = glm::scale(gun1->transform, glm::vec3(.5, .5, .5));
	gun1->transform = glm::translate(gun1->transform, glm::vec3(0, 3, 0));
	scene.add(gun1);



	PBRMaterial* diffuseMat = new PBRMaterial();

	diffuseMat->setEnvironement(operatingRoom);
	diffuseMat->setAlbedoMap(Texture("data\\iron\\albedo.png"));
	diffuseMat->setMetalnessMap(Texture("data\\iron\\metalness.png"));

	diffuseMat->setRoughnessMap(Texture("data\\iron\\roughness.png"));
	diffuseMat->setAmbientOcclusionMap(Texture("data\\iron\\ao.png"));
	diffuseMat->setNormalMap(Texture("data\\iron\\normal.png", ColorSpace::Linear));
	diffuseMat->shadowTex = shadowTarget->depthTexture;
	diffuseMat->m_BRDFLUT = brdfLUT;
	Mesh* groundPlane = new Mesh(Shapes::plane(5,5), diffuseMat);
	//scene.add(groundPlane);

	std::vector<Geometry> fluentModel;// = loadModel("data\\test01.dat");

	PBRMaterial* fluentMat = new PBRMaterial();


	fluentMat->setEnvironement(operatingRoom);
	fluentMat->m_BRDFLUT = brdfLUT;
	fluentMat->setAlbedo(glm::vec4(1, 1, 1, 1));
	fluentMat->setMetalness(0);
	fluentMat->setRoughness(.4);
	fluentMat->shadowTex = shadowTarget->depthTexture;
	fluentMat->useVertexColors();
	for (int i = 0; i < fluentModel.size(); i++)
	{
		fluentModel[i].computeNormals();
		fluentModel[i].computeTangents();
		Mesh* fluentMesh = new Mesh(fluentModel[i], fluentMat);

		fluentMesh->transform = glm::translate(fluentMesh->transform, glm::vec3(0,-7,0));
		scene.add(fluentMesh);
	}

	depthQuad = new Mesh(Shapes::renderQuad(), depthMat);
	depthQuad->transform = glm::scale(depthQuad->transform, glm::vec3(.25, .25, 0));

	//scene.add(depthQuad);

	Geometry sphereMesh = Shapes::sphere(.2);

	for (int x = -4; x <= 4; x++)
	{
		for (int z = -4; z <= 4; z++)
		{

			PBRMaterial* diffuseMat1 = new PBRMaterial();

			diffuseMat1->setEnvironement(operatingRoom);
			diffuseMat1->setAlbedo(glm::vec4(1, 1, 1, 1));
			diffuseMat1->setMetalness((x + 4) / 8.0f);
			diffuseMat1->setRoughness((z + 4) / 8.0f);
			diffuseMat1->shadowTex = shadowTarget->depthTexture;
			diffuseMat1->m_BRDFLUT = brdfLUT;
			Mesh* sphere = new Mesh(sphereMesh, diffuseMat1);
			sphere->transform = glm::translate(sphere->transform, glm::vec3(x * .5, 6, z * .5));
			scene.add(sphere);
		}
	}
	
	renderer = new Renderer();
	renderer->clearColor = glm::vec4(1, 0, 1, 1);

	shadowMat = new Material();
	shadowMat->shader = Shader("shaders\\Shadow.vert", "shaders\\Shadow.frag");

	depthMat->shadowTex = shadowTarget->depthTexture;
	gunMat->shadowTex = shadowTarget->depthTexture;

	//Model* model = new Model("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\IrrigationTool.obj");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));
	
	//scene.add(model);
}

// display function called by glutMainLoop(), gets executed every frame 
void disp(void)
{
	gun1->transform = glm::rotate(gun1->transform, .001f, glm::vec3(0,1,0));
	hovercraft->transform = glm::rotate(hovercraft->transform, .003f, glm::vec3(0, 0, 1));
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
	glDisable(GL_DEPTH_TEST);
	depthQuad->m_material->bind();
	depthQuad->render();
	depthQuad->m_material->unbind();

	glEnable(GL_DEPTH_TEST);
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
	glutSetOption(GLUT_MULTISAMPLE, 4);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH |GLUT_MULTISAMPLE); // specify the display mode to be RGB and single buffering
	glutInitWindowPosition(100, 100); // specify the initial window position
	glutInitWindowSize(scrwidth, scrheight); // specify the initial window size
	glutCreateWindow("PBRBox"); // create the window and set title
	//initGL(100, 100);
																		// initialise OpenGL:
	glClearColor(.3, .3, .3, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, scrwidth, 0.0, scrheight);
	fprintf(stderr, "OpenGL initialized \n");



	glEnable(GL_MULTISAMPLE);
	glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);

	// detect current settings
	GLint iMultiSample = 0;
	GLint iNumSamples = 0;
	glGetIntegerv(GL_SAMPLE_BUFFERS, &iMultiSample);
	glGetIntegerv(GL_SAMPLES, &iNumSamples);
	printf("MSAA on, GL_SAMPLE_BUFFERS = %d, GL_SAMPLES = %d\n", iMultiSample, iNumSamples);


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



