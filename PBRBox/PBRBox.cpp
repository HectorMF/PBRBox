// PBRBox.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <Windows.h>
#include "ResourceManager.h"
#include "JPGLoader.h"
#include "PNGLoader.h"
#include "TGALoader.h"
#include "MaterialLoader.h"
#include "DDSLoader.h"
#include "ModelLoader.h"

#include <sstream>
#include <iostream>
#include "Camera.h"
#include "MouseKeyboardInput.h"
#include "Model.h"
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


#include "FluentMesh.h"
#include "Environment.h"
#include "EnvironmentLoader.h"
#include "Volume.h"
//#include "EnvironmentTools.h"

#include "Animation\Animation.h"
// test scenes

Camera* hostRendercam = NULL;
Camera* cudaRendercam2 = NULL;

int screenWidth, screenHeight;

Scene scene;

RenderTarget* shadowTarget;
Renderer* renderer;

Material* shadowMat;

Material* depthMat;
Mesh* depthQuad;
Mesh* gun1;
Mesh* hovercraft;
ResourceManager* rm;
SceneNode* node2;
SceneNode* node3;

TweenManager* tm;


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cuda_pathtracer.h"
unsigned int framenumber = 0;
GLuint vbo;
void *d_vbo_buffer = NULL;
// image buffer storing accumulated pixel samples
glm::vec3* accumulatebuffer;
// final output buffer storing averaged pixel samples
glm::vec3* finaloutputbuffer;

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

void createVBO(GLuint* vbo)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//Initialize VBO
	unsigned int size = screenWidth * screenHeight * sizeof(glm::vec3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Register VBO with CUDA
	cudaGLRegisterBufferObject(*vbo);
}

void initializeScene()
{

	createVBO(&vbo);
	glEnable(GL_FRAMEBUFFER_SRGB);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

	tm = new TweenManager();

	rm = new ResourceManager();

	rm->addLoader(new JPGLoader());
	rm->addLoader(new PNGLoader());
	rm->addLoader(new DDSLoader());
	rm->addLoader(new TGALoader());
	rm->addLoader(new EnvironmentLoader());
	rm->addLoader(new MaterialLoader());
	rm->addLoader(new ModelLoader());

	rm->load<Texture>("Error.png");
	ResourceHandle<Environment> operatingRoom = rm->load<Environment>("data\\Environments\\Arches.gbenv");

	rm->setDefault<Texture>("Error.png");
	rm->setDefault<Environment>("data\\Environments\\Arches.gbenv");

	scene.environment = operatingRoom;

	shadowTarget = new RenderTarget();


	//rm->load<Volume>("data\\artifix\\artifix_small.raw");

	// new Environment();
	
	//operatingRoom->radiance = rm->load<Texture>("data\\neuroArm\\neuroArm_cube_radiance.dds");
	//operatingRoom->irradiance = rm->load<Texture>("data\\neuroArm\\neuroArm_cube_irradiance.dds");
	//operatingRoom->specular = rm->load<Texture>("data\\neuroArm\\neuroArm_cube_specular.dds");

	
	//operatingRoom->brdf = brdfLUT;
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

	//Volume* vol = new Volume("data\\3L_768x768x768_type_uc_1channels.raw");

	ResourceHandle<Model> irrigationTool = rm->load<Model>("data\\IrrigationTool.obj");

	PBRMaterial* irrigationMat = new PBRMaterial();
	irrigationMat->setEnvironment(operatingRoom);
	irrigationMat->setAlbedo(glm::vec4(.96f, .96f, .9686f, 1));
	irrigationMat->setMetalness(1);
	irrigationMat->setRoughness(.05f);

	irrigationTool->m_meshes[0]->m_material = irrigationMat;
	irrigationTool->m_hierarchy->scale = { .1,.1,.1 };
	irrigationTool->m_hierarchy->position = { -3, 0, 0 };
	scene.add(irrigationTool->m_hierarchy);

	//Mesh* irrigation = new Mesh(irrigationTool->m_meshes[0]->m_geometry, irrigationMat);
	//SceneNode* node = new SceneNode();
	//node->position = glm::vec3(0, 0, 0);
	//node->scale = glm::vec3(1, 1, 1);
	//node->rotation = glm::rotate(node->rotation, glm::radians(90.0f), glm::vec3(1, 0, 0));
	//node->m_transform = glm::rotate(node->m_transform, glm::radians(90.0f), glm::vec3(1,0,0));
	//node->m_transform = glm::rotate(node->m_transform, glm::radians(90.0f), glm::vec3(0, 0, 1));
	//node->m_transform = glm::translate(node->m_transform, glm::vec3(0, 0, 1));
	//node->mesh = irrigation;
	//scene.root = node;

	Mesh* skyBoxQuad = new Mesh(Shapes::cube(1), new SkyboxMaterial(operatingRoom));
	scene.skybox = skyBoxQuad;

	depthMat = new Material();
	depthMat->shader = Shader("shaders\\Diffuse.vert", "shaders\\Diffuse.frag");

//	Model* model = new Model("data\\head\\Infinite-Level_02.OBJ");
	ResourceHandle<Model> model = rm->load<Model>("data\\sponza\\SponzaNoFlag.obj");

	model->m_hierarchy->scale = glm::vec3(.01, .01, .01);

	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));
	scene.add(model->m_hierarchy);
	
	/*
	Model* model = new Model("data\\cerberus\\Cerberus.obj");

	Geometry gun = model->m_meshes[0];

	ResourceHandle<Texture> gunA = rm->load<Texture>("data\\cerberus\\Cerberus_A.png");
	ResourceHandle<Texture> gunM = rm->load<Texture>("data\\cerberus\\Cerberus_M.jpg");
	ResourceHandle<Texture> gunR = rm->load<Texture>("data\\cerberus\\Cerberus_R.jpg");
	ResourceHandle<Texture> gunN = rm->load<Texture>("data\\cerberus\\Cerberus_N.jpg");

	PBRMaterial* gunMat = new PBRMaterial();
	gunMat->setEnvironment(operatingRoom);
	gunMat->setAlbedoMap(gunA);
	gunMat->setMetalnessMap(gunM);
	gunMat->setRoughnessMap(gunR);
	gunMat->setNormalMap(gunN);
	gunMat->shadowTex = shadowTarget->depthTexture;
	
	std::ofstream file("GunMat.gbmat");
	if (file)
	{
		cereal::XMLOutputArchive archive(file);
		archive(*gunMat);
	}
	*/

	Model* headModel = rm->load<Model>("data\\head\\Infinite-Level_02.obj");
	headModel->m_hierarchy->position = glm::vec3(3, 1, 0);
	headModel->m_hierarchy->rotation = glm::quat();
	Tween<glm::vec3> t;
	t.start({ 3, 1, 0 }).end({ 0, 1, 0 }).duration(3).ease(Easing::BounceInOut).target(headModel->m_hierarchy->position);
	
	Tween<glm::vec3> t2;
	t2.start({ 1, 1, 1 }).end({ 13, 13, 13 }).ease(Easing::CircularIn).duration(3).target(headModel->m_hierarchy->scale);

	TweenSequence ts;
	ts.begin(SequenceType::Parallel)
	.add(t)
	.add(t2)
	.end();
	ts.loop(4, LoopType::YoYo);
	tm->start(ts);

	scene.add(headModel->m_hierarchy);

	ResourceHandle<Model> model4 = rm->load<Model>("data\\cerberus\\Cerberus.obj");
	Geometry gun4 = model4->m_meshes[0]->m_geometry;
	ResourceHandle<PBRMaterial> gunMat = rm->load<PBRMaterial>("GunMat.gbmat");

	ResourceHandle<Texture> gunA = rm->load<Texture>("data\\head\\Map-COL.jpg");

	//ResourceHandle<Texture> gunR = rm->load<Texture>("data\\head\\NormalMap.dds");
	ResourceHandle<Texture> gunN = rm->load<Texture>("data\\head\\Infinite-Level_02_Tangent_SmoothUV.jpg");
	//ResourceHandle<Texture> gunAO = rm->load<Texture>("data\\head\\SpecularAOMap.dds");

	PBRMaterial* gunMat2 = new PBRMaterial();

	gunMat2->setEnvironment(operatingRoom);
	gunMat2->setAlbedoMap(gunA); //setAlbedo(glm::vec4(255, 219, 145, 255) / 255.0f); //
	gunMat2->setMetalness(0);
	gunMat2->setRoughness(.5f);
//	gunMat->setNormalMap(gunN);


	/*Mesh* gun1 = new Mesh(gun->m_geometry, gunMat2);
	node2 = new SceneNode();
	node2->mesh = gun1;
	node2->scale = glm::vec3(6, 6, 6);
	node2->position = glm::vec3(0, 0, 4);
	scene.add(node2);*/

	node3 = new SceneNode();
	node3->mesh = new Mesh(gun4, gunMat);
	node3->scale = glm::vec3(1, 1, 1);
	node3->position = glm::vec3(0, 1, 0);
	scene.add(node3);


	Geometry sphereMesh = Shapes::sphere(.2);

	PBRMaterial* diffuseMat1 = new PBRMaterial();

	diffuseMat1->setEnvironment(operatingRoom);
	diffuseMat1->setAlbedo(glm::vec4(1, 1, 1, 1));
	diffuseMat1->setMetalness(1);
	diffuseMat1->setRoughness(0);

	Mesh* sphere = new Mesh(sphereMesh, diffuseMat1);
	//sphere->transform = glm::translate(sphere->transform, glm::vec3(x * .5, 6, z * .5));
	//SceneNode* node3 = new SceneNode();
	//node3->mesh = sphere;
	//scene.root = node3;


	for (int x = -4; x <= 4; x++)
	{
		for (int z = -4; z <= 4; z++)
		{

			PBRMaterial* diffuseMat1 = new PBRMaterial();

			diffuseMat1->setEnvironment(operatingRoom);
			diffuseMat1->setAlbedo(glm::vec4(1, 1, 1, 1));
			diffuseMat1->setMetalness((x + 4) / 8.0f);
			diffuseMat1->setRoughness((z + 4) / 8.0f);

			Mesh* sphere = new Mesh(sphereMesh, diffuseMat1);
			//sphere->transform = glm::translate(sphere->transform, glm::vec3(x * .5, 6, z * .5));
			SceneNode* node4 = new SceneNode();
			node4->mesh = sphere;
			node4->position = glm::vec3(x*.5, z* .5, 0);
			//scene.add(node4);
		}
	}

	PBRMaterial* lightMat = new PBRMaterial();

	diffuseMat1->setEnvironment(operatingRoom);
	diffuseMat1->setAlbedo(glm::vec4(1, 1, 1, 1));
	diffuseMat1->setMetalness(0);
	diffuseMat1->setRoughness(0);

	//sphere->transform = glm::translate(sphere->transform, glm::vec3(x * .5, 6, z * .5));
	SceneNode* node5 = new SceneNode();
	node5->mesh = sphere;
	node5->position = glm::vec3(-6.0f, 6.0f, 6.0f);
	//scene.add(node5);

	SceneNode* node6 = new SceneNode();
	node6->mesh = sphere;
	node6->position = glm::vec3(6.0f, 6.0f, 6.0f);
	//scene.add(node6);

	SceneNode* node7 = new SceneNode();
	node7->mesh = sphere;
	node7->position = glm::vec3(-6.0f, -6.0f, 6.0f);
	//scene.add(node7);

	SceneNode* node8 = new SceneNode();
	node8->mesh = sphere;
	node8->position = glm::vec3(6.0f, -6.0f, 6.0f);
	//scene.add(node8);


	/*



	//Resource<Texture>::Load("data\\cerberus\\Cerberus_A.png");












	


	Model* mandarineMesh = new Model("data\\mandarine\\Mandarine.obj");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));

	Geometry orange = mandarineMesh->m_meshes[0];

	UnlitMaterial* mandMat = new UnlitMaterial();

	mandMat->setDiffuseMap(Texture("data\\mandarine\\mandarine.jpg"));
	Mesh* mandarine = new Mesh(orange, mandMat);
	//mandarine->transform = glm::translate(mandarine->transform, glm::vec3(0, -2, 0));
	//scene.add(mandarine);



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

	//hovercraft->transform = glm::scale(hovercraft->transform, glm::vec3(.01, .01, .01));
	//hovercraft->transform = glm::rotate(hovercraft->transform, glm::radians(90.0f),glm::vec3(1, 0, 0));
	//hovercraft->transform = glm::rotate(hovercraft->transform, glm::radians(180.0f), glm::vec3(0, 1, 0));
	//hovercraft->transform = glm::rotate(hovercraft->transform, glm::radians(90.0f), glm::vec3(0, 0, 1));
	//scene.add(hovercraft);



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
	//skull->transform = glm::translate(skull->transform, glm::vec3(0, -5, 0));
	//scene.add(skull);



	Model* model = new Model("data\\cerberus\\Cerberus.obj");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));
	
	Geometry gun = model->m_meshes[0];

	gun1 = new Mesh(gun, gunMat);

	//gun1->transform = glm::scale(gun1->transform, glm::vec3(.5, .5, .5));
	//gun1->transform = glm::translate(gun1->transform, glm::vec3(0, 3, 0));
	//scene.add(gun1);



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

		//fluentMesh->transform = glm::translate(fluentMesh->transform, glm::vec3(0,-7,0));
		//scene.add(fluentMesh);
	}

	depthQuad = new Mesh(Shapes::renderQuad(), depthMat);
	//depthQuad->transform = glm::scale(depthQuad->transform, glm::vec3(.25, .25, 0));

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
			//sphere->transform = glm::translate(sphere->transform, glm::vec3(x * .5, 6, z * .5));
			//scene.add(sphere);
		}
	}
	*/
	renderer = new Renderer();
	renderer->clearColor = glm::vec4(1, 0, 1, 1);
	renderer->shadowTarget = shadowTarget;

	shadowMat = new Material();
	shadowMat->shader = Shader("shaders\\Shadow.vert", "shaders\\Shadow.frag");

	//gunMat->shadowTex = shadowTarget->depthTexture;

	//Model* model = new Model("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\PBRBox\\IrrigationTool.obj");
	//model->m_hierarchy->m_transform = glm::scale(model->m_hierarchy->m_transform, glm::vec3(.05, .05, .05));
	//model->m_hierarchy->m_transform = glm::translate(model->m_hierarchy->m_transform, glm::vec3(2, 0, .05));
	
	//scene.add(model);
}
unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

// display function called by glutMainLoop(), gets executed every frame 
void disp(void)
{
	tm->update(.01);



	//scene.root->getChild(2)->rotation = glm::rotate(scene.root->getChild(2)->rotation, .01f, glm::vec3(0, 1, 0));
	node3->rotation = glm::rotate(node3->rotation, .01f, glm::vec3(1, -1, 0));
	scene.root->updateWorldMatrix();
	//gun1->transform = glm::rotate(gun1->transform, .001f, glm::vec3(0,1,0));
	//hovercraft->transform = glm::rotate(hovercraft->transform, .003f, glm::vec3(0, 0, 1));
	// if camera has moved, reset the accumulation buffer

	// build a new camera for each frame on the CPU
	/*interactiveCamera->buildRenderCamera(hostRendercam);

	renderer->setRenderTarget(*shadowTarget);

	renderer->setOverrideMaterial(*shadowMat);

	renderer->renderShadow(scene);

	renderer->clearRenderTarget();

	renderer->clearOverideMaterial();
	
	//diffuseMat->shadowTex = shadowTarget->depthTexture;
	//glUniformMatrix4fv(t, 1, GL_FALSE, glm::value_ptr(shadowTarget->depthTexture));

	renderer->render(scene, *hostRendercam);
	glDisable(GL_DEPTH_TEST);
//	depthQuad->m_material->bind();
//	depthQuad->render();
//	depthQuad->m_material->unbind();

	glEnable(GL_DEPTH_TEST);
	glutSwapBuffers();*/


	// if camera has moved, reset the accumulation buffer
	if (buffer_reset) { cudaMemset(accumulatebuffer, 1, screenWidth * screenHeight * sizeof(glm::vec3)); framenumber = 0; }

	buffer_reset = false;
	framenumber++;

	// build a new camera for each frame on the CPU
	interactiveCamera->buildRenderCamera(hostRendercam);

	// copy the CPU camera to a GPU camera
	cudaMemcpy(cudaRendercam2, hostRendercam, sizeof(Camera), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();

	// maps a buffer object for acces by CUDA
	cudaGLMapBufferObject((void**)&finaloutputbuffer, vbo);

	//clear all pixels:
	glClear(GL_COLOR_BUFFER_BIT);

	// calculate a new seed for the random number generator, based on the framenumber
	unsigned int hashedframes = WangHash(framenumber);

	// gateway from host to CUDA, passes all data needed to render frame (triangles, BVH tree, camera) to CUDA for execution
	cudarender(finaloutputbuffer, accumulatebuffer, framenumber, hashedframes, cudaRendercam2);

	cudaThreadSynchronize();
	cudaGLUnmapBufferObject(vbo);
	//glFlush();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, screenWidth * screenHeight);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	glutPostRedisplay();
}

void deleteCudaAndCpuMemory() {
	delete rm;
	delete hostRendercam;
	delete interactiveCamera;
}

int main(int argc, char** argv) {

	// create a CPU camera
	hostRendercam = new Camera();
	// initialise an interactive camera on the CPU side
	initCamera();
	interactiveCamera->buildRenderCamera(hostRendercam);

	// allocate GPU memory for accumulation buffer
	cudaMalloc(&accumulatebuffer, screenWidth * screenHeight * sizeof(glm::vec3));
	// allocate GPU memory for interactive camera
	cudaMalloc((void**)&cudaRendercam2, sizeof(Camera));
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

	// free CUDA memory
	cudaFree(finaloutputbuffer);
	cudaFree(accumulatebuffer);
	cudaFree(cudaRendercam2);


	deleteCudaAndCpuMemory();
}



