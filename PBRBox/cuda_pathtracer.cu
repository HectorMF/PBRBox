/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
*  Interactive camera with depth of field based on CUDA path tracer code
*  by Peter Kutz and Yining Karl Li, https://github.com/peterkutz/GPUPathTracer
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "device_launch_parameters.h"
#include "cutil_math.h"

#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

#include "cuda_pathtracer.h"

#define M_PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define NUDGE_FACTOR     1e-3f  // epsilon
#define samps  1 // samples
#define BVH_STACK_SIZE 32
#define SCREEN_DIST (screenHeight*2)

int texturewidth = 0;
int textureheight = 0;
int total_number_of_triangles;

__device__ int depth = 0;

// Textures for vertices, triangles and BVH data
// (see CudaRender() below, as well as main() to see the data setup process)

struct Ray {
	float3 orig;	// ray origin
	float3 dir;		// ray direction	
	__device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

enum Refl_t { DIFF, METAL, SPEC, REFR, COAT, CHECKER, VOLUME };  // material types

struct Sphere {

	float rad;				// radius 
	float3 pos, emi, col;	// position, emission, color 
	Refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)

	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 

													 // Ray/sphere intersection
													 // Quadratic formula required to solve ax^2 + bx + c = 0 
													 // Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
													 // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = pos - r.orig;  // 
		float t, epsilon = 0.01f;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad; // discriminant
		if (disc<0) return 0; else disc = sqrtf(disc);
		return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);
	}

};
// FORMAT: { float radius, float3 position, float3 emission, float3 colour, Refl_t material }
__device__ Sphere spheres[] = {

	// sun
	{ 1.6,{ 0.0f, 3.8, 0 },{ 6, 4, 2 },{ 0.f, 0.f, 0.f }, DIFF },  // 37, 34, 30  X: links rechts Y: op neer
																   //{ 1600, { 3000.0f, 10, 6000 }, { 17, 14, 10 }, { 0.f, 0.f, 0.f }, DIFF },

																   // horizon sun2
																   //{ 1560, { 3500.0f, 0, 7000 }, { 30, 20, 2.5 }, { 0.f, 0.f, 0.f }, DIFF },  //  150, 75, 7.5

																   // sky
																   //{ 10000, { 50.0f, 40.8f, -1060 }, { 0.1, 0.3, 0.55 }, { 0.175f, 0.175f, 0.25f }, DIFF }, // 0.0003, 0.01, 0.15, or brighter: 0.2, 0.3, 0.6
	//{ 10000,{ 50.0f, 40.8f, -1060 },{ 0.51, 0.7, 0.9 },{ 0.175f, 0.175f, 0.25f }, DIFF },

	// ground
	//{ 100000,{ 0.0f, -100001.1, 0 },{ .1, .1, .1 },{ 0.4f, 0.4f, 0.4f }, CHECKER },
	//{ 100000,{ 0.0f, -100001.2, 0 },{ 0, 0, 0 },{ 0.3f, 0.3f, 0.3f }, DIFF }, // double shell to prevent light leaking

																			  // horizon brightener
																			  //{ 110000, { 50.0f, -110048.5, 0 }, { 3.6, 2.0, 0.2 }, { 0.f, 0.f, 0.f }, DIFF },
																			  // mountains
																			  //{ 4e4, { 50.0f, -4e4 - 30, -3000 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f }, DIFF },
																			  // white Mirr
	{ 1,{ 3, 0, 0.0 },{ 0, 0.0, 0 },{ 1.0f, 1.0f, 1.0f }, REFR },
	{ 1,{ 6, 0, 0.0 },{ 0, 0.0, 0 },{ 1.0f, .5f, .5f }, REFR },
	{ 1,{ 9, 0, 0.0 },{ 0, 0.0, 0 },{ .5f, 1.0f, .5f }, REFR },

	{ 1,{ 3, 0, -4.0 },{ 0, 0.0, 0 },{ 1.0f, 1.0f, 1.0f }, SPEC },
	{ 1,{ 6, 0, -4.0 },{ 0, 0.0, 0 },{ 1.0f, .5f, .5f }, SPEC },
	{ 1,{ 9, 0, -4.0 },{ 0, 0.0, 0 },{ .5f, 1.0f, .5f }, SPEC },


	{ 1,{ 3, 0, 4.0 },{ 0, 0.0, 0 },{ 1.0f, 1.0f, 1.0f }, DIFF },
	{ 1,{ 6, 0, 4.0 },{ 0, 0.0, 0 },{ 1.0f, .5f, .5f }, DIFF },
	{ 1,{ 9, 0, 4.0 },{ 0, 0.0, 0 },{ .5f, 1.0f, .5f }, DIFF },
	// Glass
	//{ 0.3, { 0.0f, -0.4, 4 }, { .0, 0., .0 }, { 0.9f, 0.9f, 0.9f }, DIFF }
};


// Create OpenGL BGR value for assignment in OpenGL VBO buffer
__device__ int getColor(glm::vec3& p)  // converts glm::vec3 colour to int
{
	return (((unsigned)p.z) << 16) | (((unsigned)p.y) << 8) | (((unsigned)p.x));
}

// Helper function, that checks whether a ray intersects a bounding box (BVH node)
__device__ bool RayIntersectsBox(const glm::vec3& originInWorldSpace, const glm::vec3& rayInWorldSpace, int boxIdx)
{
	// set Tnear = - infinity, Tfar = infinity
	//
	// For each pair of planes P associated with X, Y, and Z do:
	//     (example using X planes)
	//     if direction Xd = 0 then the ray is parallel to the X planes, so
	//         if origin Xo is not between the slabs ( Xo < Xl or Xo > Xh) then
	//             return false
	//     else, if the ray is not parallel to the plane then
	//     begin
	//         compute the intersection distance of the planes
	//         T1 = (Xl - Xo) / Xd
	//         T2 = (Xh - Xo) / Xd
	//         If T1 > T2 swap (T1, T2) /* since T1 intersection with near plane */
	//         If T1 > Tnear set Tnear =T1 /* want largest Tnear */
	//         If T2 < Tfar set Tfar="T2" /* want smallest Tfar */
	//         If Tnear > Tfar box is missed so
	//             return false
	//         If Tfar < 0 box is behind ray
	//             return false
	//     end
	// end of for loop

	float Tnear, Tfar;
	Tnear = -FLT_MAX;
	Tfar = FLT_MAX;

	float2 limits;

	// box intersection routine
#define CHECK_NEAR_AND_FAR_INTERSECTION(c)							    \
    if (rayInWorldSpace.##c == 0.f) {						    \
	if (originInWorldSpace.##c < limits.x) return false;					    \
	if (originInWorldSpace.##c > limits.y) return false;					    \
	} else {											    \
	float T1 = (limits.x - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	float T2 = (limits.y - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	if (T1>T2) { float tmp=T1; T1=T2; T2=tmp; }						    \
	if (T1 > Tnear) Tnear = T1;								    \
	if (T2 < Tfar)  Tfar = T2;								    \
	if (Tnear > Tfar)	return false;									    \
	if (Tfar < 0.f)	return false;									    \
	}


		// If Box survived all above tests, return true with intersection point Tnear and exit point Tfar.
		return true;
}


//////////////////////////////////////////
//	BVH intersection routine	//
//	using CUDA texture memory	//
//////////////////////////////////////////

// there are 3 forms of the BVH: a "pure" BVH, a cache-friendly BVH (taking up less memory space than the pure BVH)
// and a "textured" BVH which stores its data in CUDA texture memory (which is cached). The last one is gives the 
// best performance and is used here.

//////////////////////
// PATH TRACING
//////////////////////

__device__ glm::vec3 path_trace(curandState *randstate, glm::vec3 originInWorldSpace, glm::vec3 rayInWorldSpace, int avoidSelf)
{

	// colour mask
	glm::vec3 mask = glm::vec3(1.0f, 1.0f, 1.0f);
	// accumulated colour
	glm::vec3 accucolor = glm::vec3(0.0f, 0.0f, 0.0f);

	for (int bounces = 0; bounces < 5; bounces++) {  // iteration up to 4 bounces (instead of recursion in CPU code)

		int sphere_id = -1;
		int geomtype = -1;
		float kAB = 0.f, kBC = 0.f, kCA = 0.f; // distances from the 3 edges of the triangle (from where we hit it), to be used for texturing

		float tmin = 1e20;
		float tmax = -1e20;
		float d = 1e20;
		float scene_t = 1e20;
		float inf = 1e20;
		float hitdistance = 1e20;
		glm::vec3 pointHitInWorldSpace;
		float radius = 1;
		glm::vec3 f = glm::vec3(0, 0, 0);
		glm::vec3 emit = glm::vec3(0, 0, 0);
		glm::vec3 x; // intersection point
		glm::vec3 n; // normal
		glm::vec3 nl; // oriented normal
		glm::vec3 boxnormal = glm::vec3(0, 0, 0);
		glm::vec3 dw; // ray direction of next path segment
		Refl_t refltype;

		float3 rayorig = make_float3(originInWorldSpace.x, originInWorldSpace.y, originInWorldSpace.z);
		float3 raydir = make_float3(rayInWorldSpace.x, rayInWorldSpace.y, rayInWorldSpace.z);

		// intersect all triangles in the scene stored in BVH

		// intersect all spheres in the scene
		float numspheres = sizeof(spheres) / sizeof(Sphere);
		for (int i = int(numspheres); i--;) {  // for all spheres in scene
											   // keep track of distance from origin to closest intersection point
			if ((d = spheres[i].intersect(Ray(rayorig, raydir))) && d < scene_t) { scene_t = d; sphere_id = i; geomtype = 1; }
		}
		// set avoidSelf to current triangle index to avoid intersection between this triangle and the next ray, 
		// so that we don't get self-shadow or self-reflection from this triangle...


		if (scene_t > 1e20) return glm::vec3(0.0f, 0.0f, 0.0f);

		// SPHERES:
		if (geomtype == 1) {

			Sphere &sphere = spheres[sphere_id]; // hit object with closest intersection
			x = originInWorldSpace + rayInWorldSpace * scene_t;  // intersection point on object
			n = glm::vec3(x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z);		// normal
			n = glm::normalize(n);
			nl = dot(n, rayInWorldSpace) < 0 ? n : n * -1.0f; // correctly oriented normal
			f = glm::vec3(sphere.col.x, sphere.col.y, sphere.col.z);   // object colour
			refltype = sphere.refl;
			radius = sphere.rad;
			emit = glm::vec3(sphere.emi.x, sphere.emi.y, sphere.emi.z);  // object emission
			accucolor += (mask * emit);
		}

	

		// basic material system, all parameters are hard-coded (such as phong exponent, index of refraction)

		// diffuse material, based on smallpt by Kevin Beason 
		if (refltype == DIFF) {

			// pick two random numbers
			float phi = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			glm::vec3 w = nl; w = glm::normalize(w);
			glm::vec3 u = cross((fabs(w.x) > .1 ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), w); u = glm::normalize(u);
			glm::vec3 v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			dw = u*cosf(phi)*r2s + v*sinf(phi)*r2s + w*sqrtf(1 - r2);
			dw = glm::normalize(dw);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01f;  // scene size dependent

												  // multiply mask with colour of object
			mask *= f;
		}

		if (refltype == VOLUME) {
			float distanceInside = 5;
			float density = .005;
			float hDistance = -(1 / density)*log(curand_uniform(randstate));

			if (distanceInside > hDistance)
			{


				// compute cosine weighted random ray direction on hemisphere 
				dw = glm::vec3(curand_uniform(randstate), curand_uniform(randstate), curand_uniform(randstate));
				dw = glm::normalize(dw);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x;  // scene size dependent

										   // multiply mask with colour of object
				mask *= f;
			}
		}

		// Phong metal material from "Realistic Ray Tracing", P. Shirley
		if (refltype == METAL) {

			// compute random perturbation of ideal reflection vector
			// the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
			float phi = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float phongexponent = 20;
			float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
			float sinTheta = sqrtf(1 - cosTheta * cosTheta);

			// create orthonormal basis uvw around reflection vector with hitpoint as origin 
			// w is ray direction for ideal reflection
			glm::vec3 w = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace); w = glm::normalize(w);
			glm::vec3 u = cross((fabs(w.x) > .1 ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), w); u = glm::normalize(u);
			glm::vec3 v = cross(w, u); // v is normalised by default

									   // compute cosine weighted random ray direction on hemisphere 
			dw = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
			dw = glm::normalize(dw);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01f;  // scene size dependent

												  // multiply mask with colour of object
			mask *= f;
		}

		// specular material (perfect mirror)
		if (refltype == SPEC) {

			// compute reflected ray direction according to Snell's law
			dw = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + nl * 0.01f;   // scene size dependent

													// multiply mask with colour of object
			mask *= f;
		}

		// specular material (perfect mirror)
		if (refltype == REFR) {

			glm::vec3 on;
			// compute reflected ray direction according to Snell's law
			glm::vec3 reflection = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace);
			glm::vec3 refraction;
			float ior = 1.500f;

			float t;
			float cos;

			if (dot(rayInWorldSpace, n) > 0)
			{
				t = ior;
				on = n * -1.0f;
				cos = dot(rayInWorldSpace, n) / rayInWorldSpace.length();
				cos = sqrt(1.0f - ior * ior * (1.0f - cos*cos));
			}
			else
			{
				t = 1.0 / ior;
				on = n;
				cos = -dot(rayInWorldSpace, n) / rayInWorldSpace.length();
			}
			/*

			refract(v,n,f,out)
			{
			Vec uv = v.normalize();
			float dt = dot(uv, n);
			float disc = 1.0 - f*f*(1-dt*dt)
			if(disc > 0)
			out = f * ( v - n* dt) - n*sqrt(disc);
			else

			}




			*/
			glm::vec3 uv = glm::normalize(rayInWorldSpace);
			float dt = dot(uv, on);
			float disc = 1.0 - t * t * (1 - dt * dt);
			float refProb = .5;
			if (disc > 0) {
				refraction = (rayInWorldSpace - on* dt)* t - on*sqrt(disc);

				float r0 = (1 - ior) / (1 + ior);
				r0 = r0 * r0;
				refProb = r0 + (1 - r0)*pow((1 - cos), 5);
			}


			if (curand_uniform(randstate) < refProb)
				dw = reflection;
			else
				dw = refraction;


			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x;// +nl * 0.01;   // scene size dependent

									 // multiply mask with colour of object
			mask *= f;
		}


		// COAT material based on https://github.com/peterkutz/GPUPathTracer
		// randomly select diffuse or specular reflection
		// looks okay-ish but inaccurate (no Fresnel calculation yet)
		if (refltype == COAT) {

			float rouletteRandomFloat = curand_uniform(randstate);
			float threshold = 0.05f;
			glm::vec3 specularColor = glm::vec3(1, 1, 1);  // hard-coded
			bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_glm::vec3(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

			if (reflectFromSurface) { // calculate perfectly specular reflection

									  // Ray reflected from the surface. Trace a ray in the reflection direction.
									  // TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)

				mask *= specularColor;
				dw = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01f; // scene size dependent
			}

			else {  // calculate perfectly diffuse reflection

				float r1 = 2 * M_PI * curand_uniform(randstate);
				float r2 = curand_uniform(randstate);
				float r2s = sqrtf(r2);

				// compute orthonormal coordinate frame uvw with hitpoint as origin 
				glm::vec3 w = nl; w = glm::normalize(w);
				glm::vec3 u = cross((fabs(w.x) > .1 ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), w); u = glm::normalize(u);
				glm::vec3 v = cross(w, u);

				// compute cosine weighted random ray direction on hemisphere 
				dw = u*cosf(r1)*r2s + v*sinf(r1)*r2s + w*sqrtf(1 - r2);
				dw = glm::normalize(dw);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01f;  // // scene size dependent

													   // multiply mask with colour of object
				mask *= f;
				//mask *= make_glm::vec3(0.15f, 0.15f, 0.15f);  // gold metal
			}
		} // end COAT

		if (refltype == CHECKER) {

			float sines = sinf(1 * radius * n.x) * sinf(1 * radius * n.y) *sinf(1 * radius * n.z);

			if (sines < 0)
				f = glm::vec3(0, 0, 0);
			else
				f = glm::vec3(1, 1, 1);

			float rouletteRandomFloat = curand_uniform(randstate);
			float threshold = 0.05f;
			glm::vec3 specularColor = glm::vec3(1, 1, 1);  // hard-coded
			bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_glm::vec3(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);

			if (reflectFromSurface) { // calculate perfectly specular reflection

									  // Ray reflected from the surface. Trace a ray in the reflection direction.
									  // TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)

				mask *= specularColor;
				dw = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01f; // scene size dependent
			}

			else {  // calculate perfectly diffuse reflection

				float r1 = 2 * M_PI * curand_uniform(randstate);
				float r2 = curand_uniform(randstate);
				float r2s = sqrtf(r2);

				// compute orthonormal coordinate frame uvw with hitpoint as origin 
				glm::vec3 w = nl; w = glm::normalize(w);
				glm::vec3 u = cross((fabs(w.x) > .1 ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), w); u = glm::normalize(u);
				glm::vec3 v = cross(w, u);

				// compute cosine weighted random ray direction on hemisphere 
				dw = u*cosf(r1)*r2s + v*sinf(r1)*r2s + w*sqrtf(1 - r2);
				dw = glm::normalize(dw);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01f;  // // scene size dependent

													   // multiply mask with colour of object
				mask *= f;
				//mask *= make_glm::vec3(0.15f, 0.15f, 0.15f);  // gold metal
			}
		}

		// set up origin and direction of next path segment
		originInWorldSpace = pointHitInWorldSpace;
		rayInWorldSpace = dw;
	}

	return glm::vec3(accucolor.x, accucolor.y, accucolor.z);
}

union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

__global__ void UpdateKernel(float dt)
{
	Sphere &sphere = spheres[5];
	if (sphere.pos.x > 0)
		sphere.pos += make_float3(-1, 0, 0) * dt;
}

// the core path tracing kernel, 
// running in parallel for all pixels
__global__ void CoreLoopPathTracingKernel(glm::vec3* output, glm::vec3* accumbuffer, Camera* cudaRendercam,
	 unsigned int framenumber, unsigned int hashedframenumber)
{
	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator and initialise with hashed frame number, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	glm::vec3 finalcol; // final pixel colour  
	glm::vec3 rendercampos = glm::vec3(cudaRendercam->position.x, cudaRendercam->position.y, cudaRendercam->position.z);

	int i = (screenHeight - y - 1)*screenWidth + x; // pixel index in buffer
	int pixelx = x; // pixel x-coordinate on screen
	int pixely = screenHeight - y - 1; // pixel y-coordintate on screen

	finalcol = glm::vec3(0.0f, 0.0f, 0.0f); // reset colour to zero for every pixel	

	for (int s = 0; s < samps; s++) {

		// compute primary ray direction
		// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
		glm::vec3 rendercamview = glm::vec3(cudaRendercam->view.x, cudaRendercam->view.y, cudaRendercam->view.z); 		rendercamview = glm::normalize(rendercamview); // view is already supposed to be normalized, but normalize it explicitly just in case.
		glm::vec3 rendercamup = glm::vec3(cudaRendercam->up.x, cudaRendercam->up.y, cudaRendercam->up.z); 		rendercamup = glm::normalize(rendercamup);
		glm::vec3 horizontalAxis = cross(rendercamview, rendercamup); 		horizontalAxis = glm::normalize(horizontalAxis);  // Important to normalize!
		glm::vec3 verticalAxis = cross(horizontalAxis, rendercamview); 		verticalAxis = glm::normalize(verticalAxis);  // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

		glm::vec3 middle = rendercampos + rendercamview;
		glm::vec3 horizontal = horizontalAxis * tanf(cudaRendercam->fov.x * 0.5 * (M_PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		glm::vec3 vertical = verticalAxis * tanf(-cudaRendercam->fov.y * 0.5 * (M_PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.

																							  // anti-aliasing
																							  // calculate center of current pixel and add random number in X and Y dimension
																							  // based on https://github.com/peterkutz/GPUPathTracer 
		float jitterValueX = curand_uniform(&randState) - 0.5;
		float jitterValueY = curand_uniform(&randState) - 0.5;
		float sx = (jitterValueX + pixelx) / (cudaRendercam->resolution.x - 1);
		float sy = (jitterValueY + pixely) / (cudaRendercam->resolution.y - 1);

		// compute pixel on screen
		glm::vec3 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
		glm::vec3 pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cudaRendercam->focalDistance); // Important for depth of field!		

																																	   // calculation of depth of field / camera aperture 
																																	   // based on https://github.com/peterkutz/GPUPathTracer 

		glm::vec3 aperturePoint;

		if (cudaRendercam->apertureRadius > 0.00001) { // the small number is an epsilon value.

													   // generate random numbers for sampling a point on the aperture
			float random1 = curand_uniform(&randState);
			float random2 = curand_uniform(&randState);

			// randomly pick a point on the circular aperture
			float angle = TWO_PI * random1;
			float distance = cudaRendercam->apertureRadius * sqrtf(random2);
			float apertureX = cos(angle) * distance;
			float apertureY = sin(angle) * distance;

			aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
		}
		else { // zero aperture
			aperturePoint = rendercampos;
		}

		// calculate ray direction of next ray in path
		glm::vec3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
		apertureToImagePlane = glm::normalize(apertureToImagePlane);
// ray direction, needs to be normalised
		glm::vec3 rayInWorldSpace = apertureToImagePlane;
		// in theory, this should not be required
		rayInWorldSpace = glm::normalize(rayInWorldSpace); 

		// origin of next ray in path
		glm::vec3 originInWorldSpace = aperturePoint;

		finalcol += path_trace(&randState, originInWorldSpace, rayInWorldSpace, -1) * (1.0f / samps);
	}

	// add pixel colour to accumulation buffer (accumulates all samples) 
	accumbuffer[i] += finalcol;
	// averaged colour: divide colour by the number of calculated frames so far
	glm::vec3 tempcol = accumbuffer[i] * (1.0f / framenumber);

	Colour fcolour;
	glm::vec3 colour = glm::vec3(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));
	// convert from 96-bit to 24-bit colour + perform gamma correction
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255), (unsigned char)(powf(colour.y, 1 / 2.2f) * 255), (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);
	// store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
	output[i] = glm::vec3(x, y, fcolour.c);

}

bool g_bFirstTime = true;

// the gateway to CUDA, called from C++ (in void disp() in main.cpp)
void cudarender(glm::vec3* dptr, glm::vec3* accumulatebuffer, unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam) {

	if (g_bFirstTime) {
		// if this is the first time cudarender() is called,
		// bind the scene data to CUDA textures!
		g_bFirstTime = false;

		//cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>();
		//cudaBindTexture(NULL, &g_verticesTexture, cudaPtrVertices, &channel4desc, g_verticesNo * 8 * sizeof(float));

	}

	//UpdateKernel << <1, 1 >> >(.02);
	dim3 block(16, 16, 1);   // dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 grid(screenWidth / block.x, screenHeight / block.y, 1);

	// Configure grid and block sizes:
	int threadsPerBlock = 256;
	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int fullBlocksPerGrid = ((screenWidth * screenHeight) + threadsPerBlock - 1) / threadsPerBlock;
	// <<<fullBlocksPerGrid, threadsPerBlock>>>
	CoreLoopPathTracingKernel << <grid, block >> >(dptr, accumulatebuffer, cudaRendercam, framenumber, hashedframes);

}
