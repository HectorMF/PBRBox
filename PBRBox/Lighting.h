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

#include "ColorXYZ.h"
#include "Sample.h"
#include "MonteCarlo.h"
#include "exposureMath.h"
#include <optixu/optixu_math_namespace.h>

class Light
{
public:
	float			m_Theta;
	float			m_Phi;
	float			m_Width;
	float			m_InvWidth;
	float			m_HalfWidth;
	float			m_InvHalfWidth;
	float			m_Height;
	float			m_InvHeight;
	float			m_HalfHeight;
	float			m_InvHalfHeight;
	float			m_Distance;
	float			m_SkyRadius;
	optix::float3			m_P;
	optix::float3			m_Target;
	optix::float3			m_N;
	optix::float3			m_U;
	optix::float3			m_V;
	float			m_Area;
	float			m_AreaPdf;
	ColorRGB		m_Color;
	ColorRGB		m_ColorTop;
	ColorRGB		m_ColorMiddle;
	ColorRGB		m_ColorBottom;
	int				m_T;

	__device__ Light(void) :
		m_Theta(0.0f),
		m_Phi(0.0f),
		m_Width(1.0f),
		m_InvWidth(1.0f / m_Width),
		m_HalfWidth(0.5f * m_Width),
		m_InvHalfWidth(1.0f / m_HalfWidth),
		m_Height(1.0f),
		m_InvHeight(1.0f / m_Height),
		m_HalfHeight(0.5f * m_Height),
		m_InvHalfHeight(1.0f / m_HalfHeight),
		m_Distance(1.0f),
		m_SkyRadius(100.0f),
		m_P(optix::make_float3(1.0f, 1.0f, 1.0f)),
		m_Target(optix::make_float3(0.0f, 0.0f, 0.0f)),
		m_N(optix::make_float3(1.0f, 0.0f, 0.0f)),
		m_U(optix::make_float3(1.0f, 0.0f, 0.0f)),
		m_V(optix::make_float3(1.0f, 0.0f, 0.0f)),
		m_Area(m_Width * m_Height),
		m_AreaPdf(1.0f / m_Area),
		m_Color(10.0f),
		m_ColorTop(10.0f),
		m_ColorMiddle(10.0f),
		m_ColorBottom(10.0f),
		m_T(0)
	{
	}

	__device__  Light& operator=(const Light& Other)
	{
		m_Theta = Other.m_Theta;
		m_Phi = Other.m_Phi;
		m_Width = Other.m_Width;
		m_InvWidth = Other.m_InvWidth;
		m_HalfWidth = Other.m_HalfWidth;
		m_InvHalfWidth = Other.m_InvHalfWidth;
		m_Height = Other.m_Height;
		m_InvHeight = Other.m_InvHeight;
		m_HalfHeight = Other.m_HalfHeight;
		m_InvHalfHeight = Other.m_InvHalfHeight;
		m_Distance = Other.m_Distance;
		m_SkyRadius = Other.m_SkyRadius;
		m_P = Other.m_P;
		m_Target = Other.m_Target;
		m_N = Other.m_N;
		m_U = Other.m_U;
		m_V = Other.m_V;
		m_Area = Other.m_Area;
		m_AreaPdf = Other.m_AreaPdf;
		m_Color = Other.m_Color;
		m_ColorTop = Other.m_ColorTop;
		m_ColorMiddle = Other.m_ColorMiddle;
		m_ColorBottom = Other.m_ColorBottom;
		m_T = Other.m_T;

		return *this;
	}

	__device__  void Update(const optix::Aabb& BoundingBox)
	{
		m_InvWidth = 1.0f / m_Width;
		m_HalfWidth = 0.5f * m_Width;
		m_InvHalfWidth = 1.0f / m_HalfWidth;
		m_InvHeight = 1.0f / m_Height;
		m_HalfHeight = 0.5f * m_Height;
		m_InvHalfHeight = 1.0f / m_HalfHeight;
		m_Target = BoundingBox.center();

		// Determine light position
		m_P.x = m_Distance * cosf(m_Phi) * sinf(m_Theta);
		m_P.z = m_Distance * cosf(m_Phi) * cosf(m_Theta);
		m_P.y = m_Distance * sinf(m_Phi);

		m_P += m_Target;

		// Determine area
		if (m_T == 0)
		{
			m_Area = m_Width * m_Height;
			m_AreaPdf = 1.0f / m_Area;
		}

		if (m_T == 1)
		{
			m_P = BoundingBox.center();
			m_SkyRadius = 1000.0f * optix::length(BoundingBox.m_max - BoundingBox.m_min);
			m_Area = 4.0f * PI_F * powf(m_SkyRadius, 2.0f);
			m_AreaPdf = 1.0f / m_Area;
		}

		// Compute orthogonal basis frame
		m_N = optix::normalize(m_Target - m_P);
		m_U = optix::normalize(optix::cross(m_N, optix::make_float3(0.0f, 1.0f, 0.0f)));
		m_V = optix::normalize(optix::cross(m_N, m_U));
	}

	// Samples the light
	__device__  ColorXYZ SampleL(const optix::float3& P, optix::Ray& Rl, float& Pdf, LightingSample& LS)
	{
		ColorXYZ L = SPEC_BLACK;
		optix::float3 rOrigin;
		optix::float3 rDirection;
		
		if (m_T == 0)
		{
			rOrigin = m_P + ((-0.5f + LS.m_LightSample.m_Pos.x) * m_Width * m_U) + ((-0.5f + LS.m_LightSample.m_Pos.y) * m_Height * m_V);
			rDirection = optix::normalize(P - rOrigin);
			L = optix::dot(rDirection, m_N) > 0.0f ? Le(optix::make_float2(0.0f)) : SPEC_BLACK;
			Pdf = fabsf(optix::dot(rDirection, m_N)) > 0.0f ? distanceSquared(P, rOrigin) / (fabsf(optix::dot(rDirection, m_N)) * m_Area) : 0.0f;
		}

		if (m_T == 1)
		{
			rOrigin = m_P + m_SkyRadius * UniformSampleSphere(LS.m_LightSample.m_Pos);
			rDirection = optix::normalize(P - rOrigin);
			L = ColorXYZ(rDirection.x, rDirection.y, rDirection.z);//Le(optix::make_float2(1.0f) - 2.0f * LS.m_LightSample.m_Pos);
			Pdf = powf(m_SkyRadius, 2.0f) / m_Area;
		}

		Rl = optix::make_Ray(rOrigin, rDirection, 1, 0.0f, optix::length(P - rOrigin));

		return L;
	}

	// Intersect ray with light
	__device__  bool Intersect(optix::Ray& R, float& T, ColorXYZ& L, optix::float2* pUV = NULL, float* pPdf = NULL)
	{
		if (m_T == 0)
		{
			// Compute projection
			const float DotN = optix::dot(R.direction, m_N);

			// Rays is co-planar with light surface
			if (DotN >= 0.0f)
				return false;

			// Compute hit distance
			T = (-m_Distance - optix::dot(R.origin, m_N)) / DotN;

			// Intersection is in ray's negative direction
			if (T < R.tmin || T > R.tmax)
				return false;

			// Determine position on light
			const optix::float3 Pl = R.origin + R.direction * T;

			// Vector from point on area light to center of area light
			const optix::float3 Wl = Pl - m_P;

			// Compute texture coordinates
			const optix::float2 UV = optix::make_float2(optix::dot(Wl, m_U), optix::dot(Wl, m_V));

			// Check if within bounds of light surface
			if (UV.x > m_HalfWidth || UV.x < -m_HalfWidth || UV.y > m_HalfHeight || UV.y < -m_HalfHeight)
				return false;

			//R.m_MaxT = T;

			if (pUV)
				*pUV = UV;

			if (DotN < 0.0f)
				L = m_Color.ToXYZ() / m_Area;
			else
				L = SPEC_BLACK;

			if (pPdf)
				*pPdf = distanceSquared(R.origin, Pl) / (DotN * m_Area);

			return true;
		}

		if (m_T == 1)
		{
			T = m_SkyRadius;

			// Intersection is in ray's negative direction
			if (T < R.tmin || T > R.tmax)
				return false;

			R.tmax = T;

			optix::float2 UV = optix::make_float2(SphericalPhi(R.direction) * INV_TWO_PI_F, SphericalTheta(R.direction) * INV_PI_F);

			L = Le(optix::make_float2(1.0f) - 2.0f * UV);

			if (pPdf)
				*pPdf = powf(m_SkyRadius, 2.0f) / m_Area;

			return true;
		}

		return false;
	}

	__device__  float Pdf(const float3& P, const float3& Wi)
	{
		ColorXYZ L;
		optix::float2 UV;
		float Pdf = 1.0f;

		optix::Ray Rl = optix::make_Ray(P, Wi, 0, 0.0f, FLT_MAX);

		if (m_T == 0)
		{
			float T = 0.0f;

			if (!Intersect(Rl, T, L, NULL, &Pdf))
				return 0.0f;

			return powf(T, 2.0f) / (fabsf(optix::dot(m_N, -Wi)) * m_Area);
		}

		if (m_T == 1)
		{
			return powf(m_SkyRadius, 2.0f) / m_Area;
		}

		return 0.0f;
	}

	__device__ ColorXYZ Le(const float2& UV)
	{
		if (m_T == 0)
			return ColorXYZ::FromRGB(m_Color.r, m_Color.g, m_Color.b) / m_Area;

		if (m_T == 1)
		{
			//optix::float3 envColor = make_float3(tex2D(m_envmap, UV.x, UV.y));

			return ColorXYZ(UV.x, UV.y,0);
			//if (UV.y > 0.0f)
			//	return m_ColorTop.ToXYZ();// lerp(fabs(UV.y), m_ColorMiddle, m_ColorTop).ToXYZ();
			//else
			//	return m_ColorBottom.ToXYZ();// lerp(fabs(UV.y), m_ColorMiddle, m_ColorBottom).ToXYZ();
		}

		return SPEC_BLACK;
	}
};