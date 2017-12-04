/*
Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

//#include "Geometry.h"

//#include "RNG.cuh"

#include <optixu/optixu_math_namespace.h>
#include "exposureMath.h"

__host__ __device__ __inline__ void CreateCS(const optix::float3& N, optix::float3& u, optix::float3& v)
{
	if ((N.x == 0) && (N.y == 0))
	{
		if (N.z < 0.0f)
			u = make_float3(-1.0f, 0.0f, 0.0f);
		else
			u = make_float3(1.0f, 0.0f, 0.0f);

		v = make_float3(0.0f, 1.0f, 0.0f);
	}
	else
	{
		// Note: The root cannot become zero if
		// N.x == 0 && N.y == 0.
		const float d = 1.0f / sqrtf(N.y*N.y + N.x*N.x);

		u = make_float3(N.y * d, -N.x * d, 0);
		v = optix::cross(N, u);
	}
}

__device__ __inline__ float CumulativeMovingAverage(const float i, const float Ai, const float Xi)
{
	return Ai + (Xi - Ai) / (i + 1);
}

/**
@brief Generate a 2D stratified sample
@param[in] Pass Pass ID
@param[in] U Random input
@param[in] NumX Kernel size X
@param[in] NumY Kernel size Y
@return Stratified sample
*/
__host__ __device__ __inline__ optix::float2 StratifiedSample2D(const int& Pass, const optix::float2& U, const int& NumX = 4, const int& NumY = 4)
{
	const float Dx = 1.0f / (float)NumX;
	const float Dy = 1.0f / (float)NumY;

	const int Y = (int)((float)Pass / (float)NumX);
	const int X = Pass - (Y * NumX);

	return make_float2((float)(X + U.x) * Dx, (float)(Y + U.y) * Dy);
}

/**
@brief Generate a 2D stratified sample
@param[in] StratumX Stratum X
@param[in] StratumY Stratum Y
@param[in] U Random input
@param[in] NumX Kernel size X
@param[in] NumY Kernel size Y
@return Stratified sample
*/
__host__ __device__ __inline__ optix::float2 StratifiedSample2D(const int& StratumX, const int& StratumY, const optix::float2& U, const int& NumX = 4, const int& NumY = 4)
{
	const float Dx = 1.0f / ((float)NumX);
	const float Dy = 1.0f / ((float)NumY);

	return make_float2((float)(StratumX + U.x) * Dx, (float)(StratumY + U.y) * Dy);
}

/**
@brief Convert a given vector from world coordinates to local coordinates
@param[in] W Vector in world coordinates
@param[in] N Normal vector in world coordinates
@return Vector in world coordinates
*/
__host__ __device__ __inline__ optix::float3 WorldToLocal(const optix::float3& W, const optix::float3& N)
{
	const optix::float3 U = optix::normalize(optix::cross(N, make_float3(0.0072f, 0.0034f, 1.0f)));
	const optix::float3 V = optix::normalize(optix::cross(N, U));

	return make_float3(optix::dot(W, U), optix::dot(W, V), optix::dot(W, N));
}

/**
@brief Convert a given vector from local coordinates to world coordinates
@param[in] W Vector in local coordinates
@param[in] N Normal vector in world coordinates
@return Vector in world coordinates
*/
__host__ __device__ __inline__ optix::float3 LocalToWorld(const optix::float3& W, const optix::float3& N)
{
	const optix::float3 U = optix::normalize(optix::cross(N, make_float3(0.0072f, 0.0034f, 1.0f)));
	const optix::float3 V = optix::normalize(optix::cross(N, U));

	return make_float3(U.x * W.x + V.x * W.y + N.x * W.z,
		U.y * W.x + V.y * W.y + N.y * W.z,
		U.z * W.x + V.z * W.y + N.z * W.z);
}

/**
@brief Convert a given vector from world coordinates to local coordinates
@param[in] W Vector in world coordinates
@param[in] N Normal vector in world coordinates
@return Vector in world coordinates
*/
__host__ __device__ __inline__ optix::float3 WorldToLocal(const optix::float3& U, const optix::float3& V, const optix::float3& N, const optix::float3& W)
{
	return make_float3(optix::dot(W, U), optix::dot(W, V), optix::dot(W, N));
}

/**
@brief Convert a given vector from local coordinates to world coordinates
@param[in] W Vector in local coordinates
@param[in] N Normal vector in world coordinates
@return Vector in world coordinates
*/
__host__ __device__ __inline__ optix::float3 LocalToWorld(const optix::float3& U, const optix::float3& V, const optix::float3& N, const optix::float3& W)
{
	return make_float3(U.x * W.x + V.x * W.y + N.x * W.z, U.y * W.x + V.y * W.y + N.y * W.z, U.z * W.x + V.z * W.y + N.z * W.z);
}

/**
@brief Computes the spherical theta
@param[in] Wl Vector in local coordinates
@return Spherical theta
*/
__host__ __device__ __inline__ float SphericalTheta(const optix::float3& Wl)
{
	return acosf(optix::clamp(Wl.y, -1.f, 1.f));
}

/**
@brief Computes the spherical phi
@param[in] Wl Vector in local coordinates
@return Spherical phi
*/
__host__ __device__ __inline__ float SphericalPhi(const optix::float3& Wl)
{
	float p = atan2f(Wl.z, Wl.x);
	return (p < 0.f) ? p + 2.f * PI_F : p;
}

/**
@brief Computes the cosine of theta (latitude), given a spherical coordinate
@param[in] Ws Spherical coordinate
@return Cosine of theta
*/
__host__ __device__ __inline__ float CosTheta(const optix::float3& Ws)
{
	return Ws.z;
}

/**
@brief Computes the absolute cosine of theta (latitude), given a spherical coordinate
@param[in] Ws Spherical coordinate
@return Absolute cosine of theta
*/
__host__ __device__ __inline__ float AbsCosTheta(const optix::float3 &Ws)
{
	return fabsf(CosTheta(Ws));
}

/**
@brief Computes the sine of phi (latitude), given a spherical coordinate
@param[in] Ws Spherical coordinate
@return Sine of theta
*/
__host__ __device__ __inline__ float SinTheta(const optix::float3& Ws)
{
	return sqrtf(max(0.f, 1.f - Ws.z * Ws.z));
}

/**
@brief Computes the squared cosine of theta (latitude), given a spherical coordinate
@param[in] Ws Spherical coordinate
@return Squared cosine of theta
*/
__host__ __device__ __inline__ float SinTheta2(const optix::float3& Ws)
{
	return 1.f - CosTheta(Ws) * CosTheta(Ws);
}

/**
@brief Computes the cosine of phi (longitude), given a spherical coordinate
@param[in] Ws Spherical coordinate
@return Cosine of phi
*/
__host__ __device__ __inline__ float CosPhi(const optix::float3& Ws)
{
	return Ws.x / SinTheta(Ws);
}

/**
@brief Computes the sine of phi (longitude), given a spherical coordinate
@param[in] Ws Spherical coordinate
@return Sine of phi
*/
__host__ __device__ __inline__ float SinPhi(const optix::float3& Ws)
{
	return Ws.y / SinTheta(Ws);
}

/**
@brief Determines whether two vectors reside in the same hemisphere
@param[in] Ww1 First vector in world coordinates
@param[in] Ww2 First vector in world coordinates
@return Whether two given vectors reside in the same hemisphere
*/
__host__ __device__ __inline__ bool SameHemisphere(const optix::float3& Ww1, const optix::float3& Ww2)
{
	return Ww1.z * Ww2.z > 0.0f;
}

/**
@brief Determines whether two vectors reside in the same hemisphere
@param[in] W1 First vector in world coordinates
@param[in] W2 First vector in world coordinates
@param[in] N First vector in world coordinates
@return Whether two given vectors reside in the same hemisphere
*/
__host__ __device__ __inline__ bool SameHemisphere(const optix::float3& W1, const optix::float3& W2, const optix::float3& N)
{
	return (optix::dot(W1, N) * optix::dot(W2, N)) >= 0.0f;
}

/**
@brief Determines whether two vectors reside in the same shading hemisphere
@param[in] W1 First vector in world coordinates
@param[in] W2 First vector in world coordinates
@param[in] N Normal vector in world coordinates
@return Whether two given vectors reside in the same shading hemisphere
*/
__host__ __device__ __inline__ bool InShadingHemisphere(const optix::float3& W1, const optix::float3& W2, const optix::float3& N)
{
	return optix::dot(W1, N) >= 0.0f && optix::dot(W2, N) >= 0.0f;
}

/**
@brief Generates a uniform sample in a disk
@param[in] U Random input
@return Uniform sample in a disk
*/
__host__ __device__ __inline__ optix::float2 UniformSampleDisk(const optix::float2& U)
{
	float r = sqrtf(U.x);
	float theta = 2.0f * PI_F * U.y;
	return make_float2(r * cosf(theta), r * sinf(theta));
}

/**
@brief Generates a uniform sample in a disk
@param[in] U Random input
@return Uniform sample in a disk
*/
__host__ __device__ __inline__ optix::float3 UniformSampleDisk(const optix::float2& U, const optix::float3& N)
{
	const optix::float2 UV = UniformSampleDisk(U);

	optix::float3 Ucs, Vcs;

	CreateCS(N, Ucs, Vcs);

	return (UV.x * Ucs) + (UV.y * Vcs);
}

/**
@brief Generates a concentric sample in a disk
@param[in] U Random input
@return Concentric sample in a disk
*/
__host__ __device__ __inline__ optix::float2 ConcentricSampleDisk(const optix::float2& U)
{
	float r, theta;
	// Map uniform random numbers to $[-1,1]^2$
	float sx = 2 * U.x - 1;
	float sy = 2 * U.y - 1;
	// Map square to $(r,\theta)$
	// Handle degeneracy at the origin

	if (sx == 0.0 && sy == 0.0)
	{
		return make_float2(0.0f);
	}

	if (sx >= -sy)
	{
		if (sx > sy)
		{
			// Handle first region of disk
			r = sx;
			if (sy > 0.0)
				theta = sy / r;
			else
				theta = 8.0f + sy / r;
		}
		else
		{
			// Handle second region of disk
			r = sy;
			theta = 2.0f - sx / r;
		}
	}
	else
	{
		if (sx <= sy)
		{
			// Handle third region of disk
			r = -sx;
			theta = 4.0f - sy / r;
		}
		else
		{
			// Handle fourth region of disk
			r = -sy;
			theta = 6.0f + sx / r;
		}
	}

	theta *= PI_F / 4.f;

	return make_float2(r*cosf(theta), r*sinf(theta));
}

/**
@brief Generates a cosine weighted hemispherical sample
@param[in] U Random input
@return Cosine weighted hemispherical sample
*/
__host__ __device__ __inline__ optix::float3 CosineWeightedHemisphere(const optix::float2& U)
{
	const optix::float2 ret = ConcentricSampleDisk(U);
	return make_float3(ret.x, ret.y, sqrtf(max(0.f, 1.f - ret.x * ret.x - ret.y * ret.y)));
}

/**
@brief Generates a cosine weighted hemispherical sample in world coordinates
@param[in] U Random input
@param[in] Wow Vector in world coordinates
@param[in] Wow Normal in world coordinates
@return Cosine weighted hemispherical sample in world coordinates
*/
__host__ __device__ __inline__ optix::float3 CosineWeightedHemisphere(const optix::float2& U, const optix::float3& N)
{
	const optix::float3 Wl = CosineWeightedHemisphere(U);

	const optix::float3 u = optix::normalize(optix::cross(N, N));
	const optix::float3 v = optix::normalize(optix::cross(N, u));

	return make_float3(u.x * Wl.x + v.x * Wl.y + N.x * Wl.z,
		u.y * Wl.x + v.y * Wl.y + N.y * Wl.z,
		u.z * Wl.x + v.z * Wl.y + N.z * Wl.z);
}

/**
@brief Computes the probability from a cosine weighted hemispherical sample
@param[in] CosTheta Cosine of theta (Latitude)
@param[in] Phi Phi (Longitude)
@return Probability from a cosine weighted hemispherical sample
*/
__host__ __device__ __inline__ float CosineWeightedHemispherePdf(const float& CosTheta, const float& Phi)
{
	return CosTheta * INV_PI_F;
}

/**
@brief Generates a spherical sample
@param[in] SinTheta Sine of theta (Latitude)
@param[in] CosTheta Cosine of theta (Latitude)
@param[in] Phi Phi (Longitude)
@return Spherical sample
*/
__host__ __device__ __inline__ optix::float3 SphericalDirection(const float& SinTheta, const float& CosTheta, const float& Phi)
{
	return make_float3(SinTheta * cosf(Phi), SinTheta * sinf(Phi), CosTheta);
}

__host__ __device__ __inline__ optix::float3 SphericalDirection(float sintheta, float costheta, float phi, const optix::float3& x, const optix::float3& y, const optix::float3& z)
{
	return sintheta * cosf(phi) * x + sintheta * sinf(phi) * y + costheta * z;
}

/**
@brief Generates a cosine weighted hemispherical sample in world coordinates
@param[in] U Random input
@param[in] Wow Vector in world coordinates
@param[in] Wow Normal in world coordinates
@return Cosine weighted hemispherical sample in world coordinates
*/
__host__ __device__ __inline__ optix::float3 SphericalDirection(const float& SinTheta, const float& CosTheta, const float& Phi, const optix::float3& N)
{
	const optix::float3 Wl = SphericalDirection(SinTheta, CosTheta, Phi);

	const optix::float3 u = optix::normalize(optix::cross(N, make_float3(0.0072f, 1.0f, 0.0034f)));
	const optix::float3 v = optix::normalize(optix::cross(N, u));

	return make_float3(u.x * Wl.x + v.x * Wl.y + N.x * Wl.z,
		u.y * Wl.x + v.y * Wl.y + N.y * Wl.z,
		u.z * Wl.x + v.z * Wl.y + N.z * Wl.z);
}

/**
@brief Generates a sample in a triangle
@param[in] U Random input
@return Sample in a triangle
*/
__host__ __device__ __inline__ optix::float2 UniformSampleTriangle(const optix::float2& U)
{
	float su1 = sqrtf(U.x);

	return make_float2(1.0f - su1, U.y * su1);
}

/**
@brief Generates a sample in a sphere
@param[in] U Random input
@return Sample in a sphere
*/
__host__ __device__ __inline__ optix::float3 UniformSampleSphere(const optix::float2& U)
{
	float z = 1.f - 2.f * U.x;
	float r = sqrtf(max(0.f, 1.f - z*z));
	float phi = 2.f * PI_F * U.y;
	float x = r * cosf(phi);
	float y = r * sinf(phi);
	return make_float3(x, y, z);
}

/**
@brief Generates a hemispherical sample
@param[in] U Random input
@return Sample in a hemisphere
*/
__host__ __device__ __inline__ optix::float3 UniformSampleHemisphere(const optix::float2& U)
{
	float z = U.x;
	float r = sqrtf(max(0.f, 1.f - z*z));
	float phi = 2 * PI_F * U.y;
	float x = r * cosf(phi);
	float y = r * sinf(phi);
	return make_float3(x, y, z);
}

/**
@brief Generates a hemispherical sample in world coordinates
@param[in] U Random input
@param[in] N Normal in world coordinates
@return Hemispherical sample in world coordinates
*/
__host__ __device__ __inline__ optix::float3 UniformSampleHemisphere(const optix::float2& U, const optix::float3& N)
{
	const optix::float3 Wl = UniformSampleHemisphere(U);

	const optix::float3 u = optix::normalize(optix::cross(N, make_float3(0.0072f, 1.0f, 0.0034f)));
	const optix::float3 v = optix::normalize(optix::cross(N, u));

	return make_float3(u.x * Wl.x + v.x * Wl.y + N.x * Wl.z,
		u.y * Wl.x + v.y * Wl.y + N.y * Wl.z,
		u.z * Wl.x + v.z * Wl.y + N.z * Wl.z);
}

/**
@brief Generates a sample in a cone
@param[in] U Random input
@param[in] CosThetaMax Maximum cone angle
@return Sample in a cone
*/
__host__ __device__ __inline__ optix::float3 UniformSampleCone(const optix::float2& U, const float& CosThetaMax)
{
	float costheta = optix::lerp(U.x, CosThetaMax, 1.f);
	float sintheta = sqrtf(1.f - costheta*costheta);
	float phi = U.y * 2.f * PI_F;
	return make_float3(cosf(phi) * sintheta, sinf(phi) * sintheta, costheta);
}

/**
@brief Generates a sample in a cone
@param[in] U Random input
@param[in] CosThetaMax Maximum cone angle
@param[in] N Normal
@return Sample in a cone
*/
__host__ __device__ __inline__ optix::float3 UniformSampleCone(const optix::float2& U, const float& CosThetaMax, const optix::float3& N)
{
	const optix::float3 Wl = UniformSampleCone(U, CosThetaMax);

	const optix::float3 u = optix::normalize(optix::cross(N, make_float3(0.0072f, 1.0f, 0.0034f)));
	const optix::float3 v = optix::normalize(optix::cross(N, u));

	return make_float3(u.x * Wl.x + v.x * Wl.y + N.x * Wl.z,
		u.y * Wl.x + v.y * Wl.y + N.y * Wl.z,
		u.z * Wl.x + v.z * Wl.y + N.z * Wl.z);
}

/**
@brief Computes the PDF of a sample in a cone
@param[in] CosThetaMax Maximum cone angle
@return PDF of a sample in a cone
*/
__host__ __device__ __inline__ float UniformConePdf(float CosThetaMax)
{
	return 1.f / (2.f * PI_F * (1.f - CosThetaMax));
}

/**
@brief Computes the probability of a spherical sample
@return Probability of a spherical sample
*/
__host__ __device__ __inline__ float UniformSpherePdf(void)
{
	return 1.0f / (4.0f * PI_F);
}

/**
@brief Uniformly samples a point on a triangle
@param[in] pIndicesV Vertex indices
@param[in] pVertices Vertices
@param[in] pIndicesVN Vertex normal indices
@param[in] pVertexNormals Vertex normals
@param[in] SampleTriangleIndex Index of the triangle to sample
@param[in] U Random input
@param[in][out] N Sampled normal
@param[in] UV Sampled texture coordinates
@return Probability of a spherical sample
*/
__host__ __device__ __inline__ optix::float3 UniformSampleTriangle(int4* pIndicesV, optix::float3* pVertices, int4* pIndicesVN, optix::float3* pVertexNormals, int SampleTriangleIndex, optix::float2 U, optix::float3& N, optix::float2& UV)
{
	const int4 Face = pIndicesV[SampleTriangleIndex];

	const optix::float3 P[3] = { pVertices[Face.x], pVertices[Face.y], pVertices[Face.z] };

	UV = UniformSampleTriangle(U);

	const float B0 = 1.0f - UV.x - UV.y;

	const optix::float3 VN[3] =
	{
		pVertexNormals[pIndicesVN[SampleTriangleIndex].x],
		pVertexNormals[pIndicesVN[SampleTriangleIndex].y],
		pVertexNormals[pIndicesVN[SampleTriangleIndex].z]
	};

	N = optix::normalize(B0 * VN[0] + UV.x * VN[1] + UV.y * VN[2]);

	return B0 * P[0] + UV.x * P[1] + UV.y * P[2];
}


/**
@brief P. Shirley's concentric disk algorithm, maps square to disk
@param[in] U Random input
@param[out] u Output u coordinate
@param[out] v Output v coordinate
*/
__host__ __device__ __inline__ void ShirleyDisk(const optix::float2& U, float& u, float& v)
{
	float phi = 0, r = 0, a = 2 * U.x - 1, b = 2 * U.y - 1;

	if (a >-b)
	{
		if (a > b)
		{
			// Reg.1
			r = a;
			phi = QUARTER_PI_F * (b / a);
		}
		else
		{
			// Reg.2
			r = b;
			phi = QUARTER_PI_F * (2 - a / b);
		}
	}
	else
	{
		if (a < b)
		{
			// Reg.3
			r = -a;
			phi = QUARTER_PI_F * (4 + b / a);
		}
		else
		{
			// Reg.4
			r = -b;

			if (b != 0)
				phi = QUARTER_PI_F * (6 - a / b);
			else
				phi = 0;
		}
	}

	u = r * cos(phi);
	v = r * sin(phi);
}

/**
@brief P. Shirley's concentric disk algorithm, maps square to disk
@param[in] N Normal
@param[in] U Random input
*/
__host__ __device__ __inline__ optix::float3 ShirleyDisk(const optix::float3& N, const optix::float2& U)
{
	float u, v;
	float phi = 0, r = 0, a = 2 * U.x - 1, b = 2 * U.y - 1;

	if (a >-b)
	{
		if (a > b)
		{
			// Reg.1
			r = a;
			phi = QUARTER_PI_F * (b / a);
		}
		else
		{
			// Reg.2
			r = b;
			phi = QUARTER_PI_F * (2 - a / b);
		}
	}
	else
	{
		if (a < b)
		{
			// Reg.3
			r = -a;
			phi = QUARTER_PI_F * (4 + b / a);
		}
		else
		{
			// Reg.4
			r = -b;

			if (b != 0)
				phi = QUARTER_PI_F * (6 - a / b);
			else
				phi = 0;
		}
	}

	u = r * cos(phi);
	v = r * sin(phi);

	optix::float3 Ucs, Vcs;

	CreateCS(N, Ucs, Vcs);

	return (u * Ucs) + (v * Vcs);
}

__host__ __device__ __inline__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
	float f = nf * fPdf, g = ng * gPdf;
	return (f * f) / (f * f + g * g);
}

