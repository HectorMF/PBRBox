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
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>


#include "random.h"
#include "utilities.h"
#include "Packet.h"
#include "MonteCarlo.h"

using namespace optix;
rtTextureSampler<uchar4, 1> tf_texture;
rtTextureSampler<unsigned char, 3> volume_texture;
//rtTextureSampler<float, 3> sdf_texture;

rtDeclareVariable(float3, half_voxel, , );
rtDeclareVariable(float3, volume_size, , );

rtDeclareVariable(float3, boxmin, , );
rtDeclareVariable(float3, boxmax, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, hitcolor, attribute hitcolor, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(int, inshadow, attribute inshadow, );
rtDeclareVariable(int, hasReflection, attribute hasReflection, );
rtDeclareVariable(unsigned char, hitIntensity, attribute hitIntensity, );
rtDeclareVariable(float3, reflectionDir, attribute reflectionDir, );
rtDeclareVariable(float3, hitPosition, attribute hitPosition, );
rtDeclareVariable(unsigned int, frame_number, , );

rtDeclareVariable(float, gradientDelta, , );
rtDeclareVariable(float, invGradientDelta, , );
rtDeclareVariable(float3, gradientDeltaX, , );
rtDeclareVariable(float3, gradientDeltaY, , );
rtDeclareVariable(float3, gradientDeltaZ, , );

rtDeclareVariable(float, stepSize, , );
rtDeclareVariable(float, stepSizeShadow, , );
rtDeclareVariable(float, densityScale, , );
rtDeclareVariable(float, invDensityScale, , );

rtDeclareVariable(PerRayData_pathtrace, prd_path, rtPayload, );
rtDeclareVariable(PerRayData_pathtrace_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(float3, hitAlbedo, attribute hitAlbedo, );

//General Data
//rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
//rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

//Packet Data
//rtDeclareVariable(PerRayData, pathtrace_packet, rtPayload, );
//rtDeclareVariable(PerRayData_Shadow, shadow_packet, rtPayload, );

//Texture Information
//rtTextureSampler<uchar4, 1> transfer_function_texture;
//rtTextureSampler<unsigned char, 3> volume_texture;
//rtTextureSampler<float, 3> signed_distance_texture;

//Bone transformation matrix
//rtDeclareVariable(optix::Matrix4x4, boneTransform, ,);

//Hit point attributes
//rtDeclareVariable(float3, albedo, attribute albedo, );
//rtDeclareVariable(float3, normal, attribute normal, );
//rtDeclareVariable(float, roughness, attribute roughness, );
//rtDeclareVariable(float, metalness, attribute metalness, );


inline __device__ unsigned char sampleVolume(float3 pos)
{
	return tex3D(volume_texture, pos.x, pos.y, pos.z);
}

inline __device__ optix::float3 getAlbedo(unsigned char intensity)
{
	uchar4 tf = tex1D(tf_texture, intensity);
	return optix::make_float3(tf.x / 255.0f, tf.y / 255.0f, tf.z / 255.0f);
}

inline __device__ float getOpacity(unsigned char intensity)
{
	return  tex1D(tf_texture, intensity).w / 255.0f;
}

inline __device__ float getMetalness(unsigned char intensity)
{
	return 0;
}

inline __device__ float getRoughness(unsigned char intensity)
{
	return 1;
}

inline __device__ float3 getEmission(unsigned char intensity)
{
	return optix::make_float3(1, 0, 0);
}

inline __device__ float getNormalizedIntensity(unsigned char intensity)
{
	return intensity / 255.0f;
}

inline __device__ bool SampleDistanceRM(float tMin, float tMax, float3& pos, float& tHit)
{
	unsigned int rseed = tea<2>(launch_index.y, launch_index.x) + frame_number;
	const float s = -log(1.0 - rnd(rseed)) * invDensityScale;
	float sum = 0.0f;
	float sigmaT = 0.0f;

	float t = tMin + rnd(rseed) * stepSize;

	while (sum < s)
	{
		pos = ray.origin + t * ray.direction;
		tHit = t;
		if (t > tMax)
			return false;
		unsigned char sample = sampleVolume(pos);
		//float normalizedSample = getNormalizedIntensity(sample);

		sigmaT = densityScale * getOpacity(sample);
		/*float skip = stepSize;

		if (sample <= .01f)
		{
		//we hit empty voxel, sample sdf for space leaping
		float leap = tex3D(sdf_texture, pos.x, pos.y, pos.z);
		//if (leap > .05f)
		skip =  fmaxf(leap - .00112763724f, skip);
		}
		*/
		sum += sigmaT * stepSize;
		t += stepSize;
	}

	return true;
}

//https://github.com/mmp/pbrt-v3/blob/master/src/media/grid.cpp
inline __device__ bool SampleDistanceRT(float tMin, float tMax, float3& pos, float& tHit)
{
	unsigned int rseed = tea<2>(launch_index.y, launch_index.x) + frame_number;
	float sigma_t = 10;
	float Tr = 1, t = tMin;
	while (true)
	{
		t -= log(1 - rnd(rseed)) * invDensityScale / sigma_t;
		if (t >= tMax)
			return false;
		pos = ray.origin + t * ray.direction;
		tHit = t;
		Tr *= 1 - fmaxf(0, getOpacity(sampleVolume(pos)));
		const float rrThreshold = .1;
		if (Tr < rrThreshold)
		{
			float q = max(.05f, 1 - Tr);
			if (rnd(rseed) < q) break;
			Tr /= 1 - q;
		}
	}

	return true;
}


inline __device__ bool FreePathRM(float3 origin, float3 direction, float tMin, float tMax)
{
	//unsigned int rseed = (512 * launch_index.y + launch_index.x) * frame_number;
	const float s = -log(curand_uniform(prd_path.state)) / densityScale;
	float sum = 0.0f;
	float sigmaT = 0.0f;

	float3 pos;
	float t = tMin + curand_uniform(prd_path.state) * stepSizeShadow;

	while (sum < s)
	{
		pos = origin + t * direction;

		if (t > tMax)
			return false;

		sigmaT = densityScale * getOpacity(sampleVolume(pos));

		sum += sigmaT * stepSizeShadow;
		t += stepSizeShadow;
	}

	return true;
}




inline __device__ bool getNearestIntersection(float tMin, float tMax, float3& pos, float& t)
{
	t = tMin;

	while (t < tMax)
	{
		pos = ray.origin + t * ray.direction;

		if (getOpacity(sampleVolume(pos)) > 0.0f)
			return true;

		t += stepSize;
	}

	return false;
}





static __device__ float3 boxnormal(float t)
{
	float3 t0 = (boxmin - ray.origin) / ray.direction;
	float3 t1 = (boxmax - ray.origin) / ray.direction;
	float3 neg = make_float3(t == t0.x ? 1 : 0, t == t0.y ? 1 : 0, t == t0.z ? 1 : 0);
	float3 pos = make_float3(t == t1.x ? 1 : 0, t == t1.y ? 1 : 0, t == t1.z ? 1 : 0);
	return pos - neg;
}


inline __device__ float3 getGradient(float3 p)
{
	float3 g;
	// note: must use +/- 0.5 since apron may only be 1 voxel wide (cannot go beyond brick)
	g.x = (tex3D(volume_texture, p.x - .5f, p.y, p.z) - tex3D(volume_texture, p.x + .5f, p.y, p.z)) / (2 * .25f);
	g.y = (tex3D(volume_texture, p.x, p.y - .5f, p.z) - tex3D(volume_texture, p.x, p.y + .5f, p.z)) / (2 * .25f);
	g.z = (tex3D(volume_texture, p.x, p.y, p.z - .5f) - tex3D(volume_texture, p.x, p.y, p.z + .5f)) / (2 * .25f);
	g = normalize(g);
	return g;
}


inline __device__  float3 NormalizedGradient2(const float3& P)
{
	float3 Gradient;

	Gradient.x = (getOpacity(sampleVolume(P + gradientDeltaX)) - getOpacity(sampleVolume(P - gradientDeltaX)));
	Gradient.y = (getOpacity(sampleVolume(P + gradientDeltaY)) - getOpacity(sampleVolume(P - gradientDeltaY)));
	Gradient.z = (getOpacity(sampleVolume(P + gradientDeltaZ)) - getOpacity(sampleVolume(P - gradientDeltaZ)));

	return normalize(Gradient);
}

inline __device__  float3 NormalizedGradient(const float3& P)
{
	float sobelX[27] = { -1, 0, 1, -3, 0, 3, -1, 0, 1,
		-3, 0, 3, -6, 0, 6, -3, 0, 3,
		-1, 0, 1, -3, 0, 3, -1, 0, 1 };

	float sobelY[27] = { -1, -3, -1, 0, 0, 0, 1, 3, 1,
		-3, -6, -3, 0, 0, 0, 3, 6, 3,
		-1, -3, -1, 0, 0, 0, 1, 3, 1 };

	float sobelZ[27] = { -1, -3, -1, -3, -6, -3, -1, -3, -1,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 3, 1, 3, 6, 3, 1, 3, 1 };

	float xSum = 0.0, ySum = 0.0, zSum = 0.0;

	float3 Gradient;

	for (int z = 0; z < 3; z++)
	{
		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				float intensity = getOpacity(sampleVolume(P + x * gradientDeltaX + y * gradientDeltaY + z * gradientDeltaZ));

				xSum += intensity * sobelX[z * 3 * 3 + y * 3 + x];
				ySum += intensity * sobelY[z * 3 * 3 + y * 3 + x];
				zSum += intensity * sobelZ[z * 3 * 3 + y * 3 + x];
			}
		}
	}


	Gradient.x = xSum;
	Gradient.y = ySum;
	Gradient.z = zSum;

	return normalize(Gradient);
}


#define LO		0
#define	MID		1.0
#define	HI		2.0



inline __device__ float getTricubic(float3 p)
{
	float tv[9];

	// find bottom-left corner of local 3x3x3 group		
	float3  q = floor3(p) - MID;				// move to bottom-left corner	

												// evaluate tri-cubic
	float3 tb = (fracf(p)) * 0.5 + 0.25;
	float3 ta = (1.0 - tb);
	float3 ta2 = ta*ta;
	float3 tb2 = tb*tb;
	float3 tab = ta*tb*2.0;

	// lookup 3x3x3 local neighborhood
	tv[0] = tex3D(volume_texture, q.x, q.y, q.z);
	tv[1] = tex3D(volume_texture, q.x + MID, q.y, q.z);
	tv[2] = tex3D(volume_texture, q.x + HI, q.y, q.z);
	tv[3] = tex3D(volume_texture, q.x, q.y + MID, q.z);
	tv[4] = tex3D(volume_texture, q.x + MID, q.y + MID, q.z);
	tv[5] = tex3D(volume_texture, q.x + HI, q.y + MID, q.z);
	tv[6] = tex3D(volume_texture, q.x, q.y + HI, q.z);
	tv[7] = tex3D(volume_texture, q.x + MID, q.y + HI, q.z);
	tv[8] = tex3D(volume_texture, q.x + HI, q.y + HI, q.z);

	float3 abc = make_float3(tv[0] * ta2.x + tv[1] * tab.x + tv[2] * tb2.x,
		tv[3] * ta2.x + tv[4] * tab.x + tv[5] * tb2.x,
		tv[6] * ta2.x + tv[7] * tab.x + tv[8] * tb2.x);

	tv[0] = tex3D(volume_texture, q.x, q.y, q.z + MID);
	tv[1] = tex3D(volume_texture, q.x + MID, q.y, q.z + MID);
	tv[2] = tex3D(volume_texture, q.x + HI, q.y, q.z + MID);
	tv[3] = tex3D(volume_texture, q.x, q.y + MID, q.z + MID);
	tv[4] = tex3D(volume_texture, q.x + MID, q.y + MID, q.z + MID);
	tv[5] = tex3D(volume_texture, q.x + HI, q.y + MID, q.z + MID);
	tv[6] = tex3D(volume_texture, q.x, q.y + HI, q.z + MID);
	tv[7] = tex3D(volume_texture, q.x + MID, q.y + HI, q.z + MID);
	tv[8] = tex3D(volume_texture, q.x + HI, q.y + HI, q.z + MID);

	float3 def = make_float3(tv[0] * ta2.x + tv[1] * tab.x + tv[2] * tb2.x,
		tv[3] * ta2.x + tv[4] * tab.x + tv[5] * tb2.x,
		tv[6] * ta2.x + tv[7] * tab.x + tv[8] * tb2.x);

	tv[0] = tex3D(volume_texture, q.x, q.y, q.z + HI);
	tv[1] = tex3D(volume_texture, q.x + MID, q.y, q.z + HI);
	tv[2] = tex3D(volume_texture, q.x + HI, q.y, q.z + HI);
	tv[3] = tex3D(volume_texture, q.x, q.y + MID, q.z + HI);
	tv[4] = tex3D(volume_texture, q.x + MID, q.y + MID, q.z + HI);
	tv[5] = tex3D(volume_texture, q.x + HI, q.y + MID, q.z + HI);
	tv[6] = tex3D(volume_texture, q.x, q.y + HI, q.z + HI);
	tv[7] = tex3D(volume_texture, q.x + MID, q.y + HI, q.z + HI);
	tv[8] = tex3D(volume_texture, q.x + HI, q.y + HI, q.z + HI);

	float3 ghi = make_float3(tv[0] * ta2.x + tv[1] * tab.x + tv[2] * tb2.x,
		tv[3] * ta2.x + tv[4] * tab.x + tv[5] * tb2.x,
		tv[6] * ta2.x + tv[7] * tab.x + tv[8] * tb2.x);

	float3 jkl = make_float3(abc.x*ta2.y + abc.y*tab.y + abc.z*tb2.y,
		def.x*ta2.y + def.y*tab.y + def.z*tb2.y,
		ghi.x*ta2.y + ghi.y*tab.y + ghi.z*tb2.y);

	return jkl.x*ta2.z + jkl.y*tab.z + jkl.z*tb2.z;
}

inline __device__ float3 getGradientTricubic(float3 p)
{
	// tri-cubic filtered gradient
	const float vs = 0.5;
	float3 g;
	g.x = (getTricubic(p + make_float3(-vs, 0, 0)) - getTricubic(p + make_float3(vs, 0, 0))) / (vs);
	g.y = (getTricubic(p + make_float3(0, -vs, 0)) - getTricubic(p + make_float3(0, vs, 0))) / (vs);
	g.z = (getTricubic(p + make_float3(0, 0, -vs)) - getTricubic(p + make_float3(0, 0, vs))) / (vs);
	g = normalize(g);
	return g;
}

inline __device__ float3 getGradientTest(float3 p)
{
	// tri-cubic filtered gradient
	float3 total = make_float3(0);
	total += getGradient(p + make_float3(-1, 0, 0));
	total += getGradient(p + make_float3(+1, 0, 0));
	total += getGradient(p + make_float3(0, -1, 0));
	total += getGradient(p + make_float3(0, +1, 0));
	total += getGradient(p + make_float3(0, 0, -1));
	total += getGradient(p + make_float3(0, 0, +1));

	total /= 6;
	return total;
}


inline __device__ float3 GradientCD(float3 pos)
{
	const float intensity[3][2] =
	{
		{ sampleVolume(pos + gradientDeltaX), sampleVolume(pos - gradientDeltaX) },
		{ sampleVolume(pos + gradientDeltaY), sampleVolume(pos - gradientDeltaY) },
		{ sampleVolume(pos + gradientDeltaZ), sampleVolume(pos - gradientDeltaZ) }
	};

	return make_float3(intensity[0][1] - intensity[0][0], intensity[1][1] - intensity[1][0], intensity[2][1] - intensity[2][0]);
}

inline __device__ float3 GradientFD(float3 pos)
{
	const float intensity[4] =
	{
		sampleVolume(pos),
		sampleVolume(pos + gradientDeltaX),
		sampleVolume(pos + gradientDeltaY),
		sampleVolume(pos + gradientDeltaZ)
	};

	return make_float3(intensity[0] - intensity[1], intensity[0] - intensity[2], intensity[0] - intensity[3]);
}

inline __device__ float3 GradientFiltered(float3 pos)
{
	float3 Offset = make_float3(gradientDeltaX.x, gradientDeltaY.y, gradientDeltaZ.z);

	float3 G0 = GradientCD(pos);
	float3 G1 = GradientCD(pos + make_float3(-Offset.x, -Offset.y, -Offset.z));
	float3 G2 = GradientCD(pos + make_float3(Offset.x, Offset.y, Offset.z));
	float3 G3 = GradientCD(pos + make_float3(-Offset.x, Offset.y, -Offset.z));
	float3 G4 = GradientCD(pos + make_float3(Offset.x, -Offset.y, Offset.z));
	float3 G5 = GradientCD(pos + make_float3(-Offset.x, -Offset.y, Offset.z));
	float3 G6 = GradientCD(pos + make_float3(Offset.x, Offset.y, -Offset.z));
	float3 G7 = GradientCD(pos + make_float3(-Offset.x, Offset.y, Offset.z));
	float3 G8 = GradientCD(pos + make_float3(Offset.x, -Offset.y, -Offset.z));

	float3 L0 = lerp(lerp(G1, G2, 0.5), lerp(G3, G4, 0.5), 0.5);
	float3 L1 = lerp(lerp(G5, G6, 0.5), lerp(G7, G8, 0.5), 0.5);

	return lerp(G0, lerp(L0, L1, 0.5), 0.75);
}

RT_PROGRAM void intersect(int s)
{
	float3 t0 = (boxmin - ray.origin) / ray.direction;
	float3 t1 = (boxmax - ray.origin) / ray.direction;
	float3 near = fminf(t0, t1);
	float3 far = fmaxf(t0, t1);
	float tmin = fmaxf(near);
	float tmax = fminf(far);

	//c_tStep *= 0.5;

	float t = tmin;
	float3 p = ray.origin + ray.direction * t;
	if (rtPotentialIntersection(t))
	{
		hitPosition = rtTransformVector(RT_OBJECT_TO_WORLD, p);
		geometric_normal = shading_normal = make_float3(0,1,0);// NormalizedGradient(p);

		inshadow = 0;
		hasReflection = 1;
		//if (FreePathRM(p, make_float3(0, 1, 0), 0, 1))
		//	inshadow = 1;
		//hitPosition = p;
		float3 tn = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

		hitIntensity = 1;// sampleVolume(p);
		hitAlbedo = make_float3(1, 0, 0);// getAlbedo(hitIntensity);
		prd_path.depth = t;
		//float3 temp = UniformSampleHemisphere(random_float2(prd_path.state), make_float3(0,1,0));
		//float3 reflection = reflect(ray.direction, tn);

		//if (FreePathRM(p, UniformSampleCone(random_float2(prd_path.state),1.0f, make_float3(0, 1, 0)), 0, 1))
		//{
		//	inshadow = 1;
		//}

		//	float3 reflection = reflect(ray.direction, tn);

		//	reflectionDir = reflection;// UniformSampleHemisphere(random_float2(prd_path.state), reflection);
		//
		//	if (FreePathRM(p, reflectionDir, 0, 1))
		//{
		//	hasReflection = 0;
		//}

		hitcolor = make_float3(.6f);

		if (rtReportIntersection(0))
		{
			return;
		}
	}
	/*if (SampleDistanceRT(tmin, tmax, p, t))
	{
		if (rtPotentialIntersection(t))
		{
			hitPosition = rtTransformVector(RT_OBJECT_TO_WORLD, p);
			geometric_normal = shading_normal = NormalizedGradient(p);

			inshadow = 0;
			hasReflection = 1;
			//if (FreePathRM(p, make_float3(0, 1, 0), 0, 1))
			//	inshadow = 1;
			//hitPosition = p;
			float3 tn = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

			hitIntensity = sampleVolume(p);
			hitAlbedo = getAlbedo(hitIntensity);
			prd_path.depth = t;
			//float3 temp = UniformSampleHemisphere(random_float2(prd_path.state), make_float3(0,1,0));
			//float3 reflection = reflect(ray.direction, tn);

			//if (FreePathRM(p, UniformSampleCone(random_float2(prd_path.state),1.0f, make_float3(0, 1, 0)), 0, 1))
			//{
			//	inshadow = 1;
			//}

			//	float3 reflection = reflect(ray.direction, tn);

			//	reflectionDir = reflection;// UniformSampleHemisphere(random_float2(prd_path.state), reflection);
			//
			//	if (FreePathRM(p, reflectionDir, 0, 1))
			//{
			//	hasReflection = 0;
			//}

			hitcolor = make_float3(.6f);

			if (rtReportIntersection(0))
			{
				return;
			}
		}
	}
	*/
	/*
	if (tmin <= tmax)
	{

	for (int i = 0; i < maxSteps; ++i)
	{
	//Done if already outside the bounding box or if the ray has passed through enough volume to create an opaque value
	if (t > tmax || sum > 0.95f)
	break;

	float3 pos = ray.origin + ray.direction * t;

	//precalculate the texture lookup
	//gb::Vec3f texLookup = gb::Vec3f ((pos.x + c_BBoxHalfSize.x) * c_invVolumeScale.x,
	//	(pos.y + c_BBoxHalfSize.y) * c_invVolumeScale.y,
	//	(pos.z + c_BBoxHalfSize.z) * c_invVolumeScale.z);
	//float3 texLookup = (pos + c_BBoxHalfSize)*c_invVolumeScale;

	pos = pos * volume_size;

	float sample = tex3D(volume_texture, pos.x, pos.y, pos.z);
	float skip = c_tStep  + c_tStep * 0.9*(rnd(rseed) - 0.5);
	if (sample <= .01)
	{
	//we hit empty voxel, sample sdf for space leaping
	//float leap = tex3D(sdf_texture, pos.x, pos.y, pos.z);
	//if (leap > .05f)
	//	skip = fmaxf(leap - .012763724f, skip);
	}
	else
	{
	if (rtPotentialIntersection(t))
	{
	float tricubicSample =  getTricubic(pos);

	geometric_normal = shading_normal = GradientFiltered(pos);
	//geometric_normal = getGradient(pos);
	hitcolor = make_float3(.6f);
	if (rtReportIntersection(0))
	{
	return;
	}
	}
	}

	t += skip;
	}
	}
	/*
	int nsamp = 1; //2;
	float count = 0;
	float inv_sigma_max = 1 / 255.0f;
	float inv_density_max = 1 / 255.0f;
	float3 normal;
	for (int n = 0; n < nsamp; n++)
	{
	/// woodcock tracking
	float dist = tmin;
	float density;

	while (1)
	{

	dist += -logf(1 - rnd(rseed)) * inv_sigma_max;
	if (dist >= tmax){
	count += 1;
	break;
	}

	float3 pos = ray.origin + ray.direction * dist;
	density = tex3D(volume_texture, pos.x, pos.y, pos.z);

	if (rnd(rseed) < density)
	{
	if (rtPotentialIntersection(dist))
	{
	float tricubicSample = getTricubic(pos);

	geometric_normal = shading_normal = getGradient(pos);
	//geometric_normal = getGradient(pos);
	hitcolor = make_float3(.6f);
	if (rtReportIntersection(0))
	{
	return;
	}
	}
	break;
	}

	}
	}

	//return count / nsamp;

	//float max_t = f_min(tmax, dist(a, b));






	/*float sample = getTricubic(pos);

	if (sample > 0.1) {

	}
	*/
	//



}




/*
HOST_DEVICE_NI void SampleVolume(Ray R, CRNG& RNG, ScatterEvent& SE)
{
float MinT;
float MaxT;

Intersection Int;

IntersectBox(R, gpVolumes[gpTracer->VolumeID].BoundingBox.MinP, gpVolumes[gpTracer->VolumeID].BoundingBox.MaxP, Int);

if (!Int.Valid)
return;

MinT = max(Int.NearT, R.MinT);
MaxT = min(Int.FarT, R.MaxT);

const float S = -log(RNG.Get1()) / gpTracer->RenderSettings.Shading.DensityScale;
float Sum = 0.0f;
float SigmaT = 0.0f;

Vec3f Ps;

const float StepSize = gpTracer->RenderSettings.Traversal.StepFactorPrimary * gpVolumes[gpTracer->VolumeID].MinStep;

MinT += RNG.Get1() * StepSize;

while (Sum < S)
{
Ps = R.O + MinT * R.D;

if (MinT >= MaxT)
return;

float Intensity = GetIntensity(gpTracer->VolumeID, Ps);

SigmaT = gpTracer->RenderSettings.Shading.DensityScale * gpTracer->Opacity1D.Evaluate(Intensity);

Sum += SigmaT * StepSize;
MinT += StepSize;
}

SE.SetValid(MinT, Ps, NormalizedGradient(gpTracer->VolumeID, Ps), -R.D, ColorXYZf());
}

HOST_DEVICE_NI bool ScatterEventInVolume(Ray R, CRNG& RNG)
{
float MinT;
float MaxT;
Vec3f Ps;

Intersection Int;

IntersectBox(R, gpVolumes[gpTracer->VolumeID].BoundingBox.MinP, gpVolumes[gpTracer->VolumeID].BoundingBox.MaxP, Int);

if (!Int.Valid)
return false;

MinT = max(Int.NearT, R.MinT);
MaxT = min(Int.FarT, R.MaxT);

const float S = -log(RNG.Get1()) / gpTracer->RenderSettings.Shading.DensityScale;
float Sum = 0.0f;
float SigmaT = 0.0f;

const float StepSize = gpTracer->RenderSettings.Traversal.StepFactorShadow * gpVolumes[gpTracer->VolumeID].MinStep;

MinT += RNG.Get1() * StepSize;

while (Sum < S)
{
Ps = R.O + MinT * R.D;

if (MinT > MaxT)
return false;

float Intensity = GetIntensity(gpTracer->VolumeID, Ps);

SigmaT = gpTracer->RenderSettings.Shading.DensityScale * gpTracer->Opacity1D.Evaluate(Intensity);

Sum += SigmaT * StepSize;
MinT += StepSize;
}

return true;
}

*/


RT_PROGRAM void bounds(int, float result[6])
{
	optix::Aabb* aabb = (optix::Aabb*)result;
	aabb->set(boxmin, boxmax);
}

