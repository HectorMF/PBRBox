/*
* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>


#include "helpers.h"
#include "random.h"
#include "ColorXYZ.h"
#include "Packet.h"

using namespace optix;
/*
struct PerRayData_pathtrace
{
float3 result;
float  importance;
int    depth;
};

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<uchar4, 2>              output_buffer;
rtBuffer<float4, 2>              accum_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );
rtDeclareVariable(unsigned int,  frame, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, );


RT_PROGRAM void pinhole_camera()
{

size_t2 screen = output_buffer.size();
unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame);

// Subpixel jitter: send the ray through a different position inside the pixel each time,
// to provide antialiasing.
float2 subpixel_jitter = frame == 0 ? make_float2(0.0f, 0.0f) : make_float2(rnd( seed ) - 0.5f, rnd( seed ) - 0.5f);

float2 d = (make_float2(launch_index) + subpixel_jitter) / make_float2(screen) * 2.f - 1.f;
float3 ray_origin = eye;
float3 ray_direction = normalize(d.x*U + d.y*V + W);

optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon );

PerRayData_pathtrace prd;
prd.importance = 1.f;
prd.depth = 0;

rtTrace(top_object, ray, prd);

float4 acc_val = accum_buffer[launch_index];
if( frame > 0 ) {
acc_val += make_float4(prd.result, 0.f);////lerp( acc_val, make_float4( prd.result, 0.f), 1.0f / static_cast<float>( frame+1 ) );
} else {
acc_val = make_float4(prd.result, 0.f);
}
output_buffer[launch_index] = make_color( make_float3( acc_val )/(frame+1) );
accum_buffer[launch_index] = acc_val;
}
*/





// Scene wide variables
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(PerRayData_pathtrace, prd_path, rtPayload, );


rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(unsigned int, frame_number, , );
rtDeclareVariable(unsigned int, sqrt_num_samples, , );
rtDeclareVariable(unsigned int, rr_begin_depth, , );
rtDeclareVariable(unsigned int, pathtrace_ray_type, , );
rtDeclareVariable(unsigned int, pathtrace_shadow_ray_type, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              accum_buffer;
rtBuffer<float4, 2>				 depth_buffer;

__device__ float computeClipDepth(float eyeDist, float n, float f)
{
	float clipDepth = (f + n) / (f - n) - (1 / eyeDist)*2.0f*f*n / (f - n);
	clipDepth = clipDepth*0.5 + 0.5f;
	return clipDepth;
}

RT_PROGRAM void pathtrace_camera()
{
	size_t2 screen = output_buffer.size();

	float2 inv_screen = 1.0f / make_float2(screen) * 2.f;
	float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

	float2 jitter_scale = inv_screen / sqrt_num_samples;
	unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
	float3 result = make_float3(0.0f);

	curandState_t state;
	curand_init(tea<16>(screen.x*launch_index.y + launch_index.x, frame_number), 0, 0, &state);

	float depthHit;
	do
	{
		//
		// Sample pixel using jittering
		//
		unsigned int x = samples_per_pixel%sqrt_num_samples;
		unsigned int y = samples_per_pixel / sqrt_num_samples;
		float2 jitter = make_float2(x - curand_uniform(&state), y - curand_uniform(&state));
		float2 d = pixel + jitter*jitter_scale;
		float3 ray_origin = eye;
		float3 ray_direction = normalize(d.x*U + d.y*V + W);

		// Initialze per-ray data
		PerRayData_pathtrace prd;
		prd.result = make_float3(0.f);
		prd.attenuation = make_float3(1.f);
		prd.countEmitted = true;
		prd.done = false;
		prd.state = &state;
		// Each iteration is a segment of the ray path.  The closest hit will
		// return new segments to be traced here.
		/*if (launch_index.x == 0 && launch_index.y == 0){
		printf("%d %d \n", launch_index.x, launch_index.y);
		printf("%d \n", sizeof(PerRayData_pathtrace));
		PerRayData_pathtrace_shadow shadow_prd;
		shadow_prd.inShadow = false;
		Ray shadow_ray = make_Ray(ray_origin, ray_direction, pathtrace_shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX);
		rtTrace(top_object, shadow_ray, shadow_prd);
		}*/


		Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, 0, RT_DEFAULT_MAX);
		rtTrace(top_object, ray, prd);

		prd.result += prd.radiance;
		depthHit = prd.depth;
		/*for (;;)
		{
		Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
		rtTrace(top_object, ray, prd);

		if (prd.done)
		{
		// We have hit the background or a luminaire
		prd.result += prd.radiance * prd.attenuation;
		break;
		}

		// Russian roulette termination
		if (prd.depth >= rr_begin_depth)
		{
		float pcont = fmaxf(prd.attenuation);
		if (rnd(prd.seed) >= pcont)
		break;
		prd.attenuation /= pcont;
		}

		prd.depth++;
		prd.result += prd.radiance * prd.attenuation;

		// Update ray data for the next path segment
		ray_origin = prd.origin;
		ray_direction = prd.direction;
		}*/

		result += prd.result;
	} while (--samples_per_pixel);

	//
	// Update the output buffer
	//
	float3 pixel_color = result / (sqrt_num_samples*sqrt_num_samples);

	float4 acc_val = accum_buffer[launch_index];

	if (frame_number > 0) {
		acc_val += make_float4(pixel_color, 0.f);
	}
	else
	{
		acc_val = make_float4(pixel_color, 0.f);
	}

	output_buffer[launch_index] = acc_val / (frame_number + 1);
	accum_buffer[launch_index] = acc_val;

	depth_buffer[launch_index] = make_float4(computeClipDepth(depthHit, .1f, 750.0f));
	/*if (frame_number > 1)
	{
	float a = 1.0f / (float)frame_number;
	float3 old_color = make_float3(output_buffer[launch_index]);
	output_buffer[launch_index] = make_float4(lerp(old_color, pixel_color, 1.0f), 1.0f);
	}
	else
	{
	output_buffer[launch_index] = make_float4(pixel_color, 1.0f);
	accum_buffer[launch_index] = acc_val;
	}*/
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf("Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y);
	output_buffer[launch_index] = make_float4(bad_color);
}
