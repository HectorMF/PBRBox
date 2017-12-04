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
#include <optixu/optixu_math_namespace.h>
#include "random.h"

class LightSample
{
public:
	optix::float2 m_Pos;
	float m_Component;

	__host__ __device__ LightSample(void)
	{
		m_Pos = make_float2(0.0f);
		m_Component = 0.0f;
	}

	__host__ __device__ LightSample& LightSample::operator=(const LightSample& Other)
	{
		m_Pos = Other.m_Pos;
		m_Component = Other.m_Component;

		return *this;
	}

	__host__ __device__ void LargeStep(curandState_t* state)
	{
		m_Pos = random_float2(state);
		m_Component = random_float(state);
	}
};

class BRDFSample
{
public:
	float	m_Component;
	optix::float2	m_Dir;

	__host__ __device__ BRDFSample(void)
	{
		m_Component = 0.0f;
		m_Dir = make_float2(0.0f);
	}

	__host__ __device__ BRDFSample(const float& Component, const optix::float2& Dir)
	{
		m_Component = Component;
		m_Dir = Dir;
	}

	__host__ __device__ BRDFSample& BRDFSample::operator=(const BRDFSample& Other)
	{
		m_Component = Other.m_Component;
		m_Dir = Other.m_Dir;

		return *this;
	}

	__host__ __device__ void LargeStep(curandState_t* state)
	{
		m_Component = random_float(state);
		m_Dir = random_float2(state);
	}
};

class LightingSample
{
public:
	BRDFSample		m_BsdfSample;
	LightSample 	m_LightSample;
	float			m_LightNum;

	__host__ __device__ LightingSample(void)
	{
		m_LightNum = 0.0f;
	}

	__host__ __device__ LightingSample& LightingSample::operator=(const LightingSample& Other)
	{
		m_BsdfSample = Other.m_BsdfSample;
		m_LightNum = Other.m_LightNum;
		m_LightSample = Other.m_LightSample;

		return *this;
	}

	__host__ __device__ void LargeStep(curandState_t* state)
	{
		m_BsdfSample.LargeStep(state);
		m_LightSample.LargeStep(state);

		m_LightNum = random_float(state);
	}
};

class CameraSample
{
public:
	optix::float2	m_ImageXY;
	optix::float2	m_LensUV;

	__device__ CameraSample(void)
	{
		m_ImageXY = make_float2(0.0f);
		m_LensUV = make_float2(0.0f);
	}

	__device__ CameraSample& CameraSample::operator=(const CameraSample& Other)
	{
		m_ImageXY = Other.m_ImageXY;
		m_LensUV = Other.m_LensUV;

		return *this;
	}

	__device__ void LargeStep(optix::float2& ImageUV, optix::float2& LensUV, const int& X, const int& Y, const int& KernelSize)
	{
		m_ImageXY = make_float2(X + ImageUV.x, Y + ImageUV.y);
		m_LensUV = LensUV;
	}
};