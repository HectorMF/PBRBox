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

#include "Sample.h"
#include "MonteCarlo.h"
#include "ColorXYZ.h"

class Lambertian
{
public:
	__host__ __device__  Lambertian(const ColorXYZ& Kd)
	{
		m_Kd = Kd;
	}

	__host__ __device__  ~Lambertian(void)
	{
	}

	__host__ __device__  ColorXYZ F(const optix::float3& Wo, const optix::float3& Wi)
	{
		return m_Kd * INV_PI_F;
	}

	__host__ __device__  ColorXYZ SampleF(const optix::float3& Wo, optix::float3& Wi, float& Pdf, const optix::float2& U)
	{
		Wi = CosineWeightedHemisphere(U);

		if (Wo.z < 0.0f)
			Wi.z *= -1.0f;

		Pdf = this->Pdf(Wo, Wi);

		return this->F(Wo, Wi);
	}

	__host__ __device__  float Pdf(const optix::float3& Wo, const optix::float3& Wi)
	{
		return SameHemisphere(Wo, Wi) ? AbsCosTheta(Wi) * INV_PI_F : 0.0f;
	}

	ColorXYZ	m_Kd;
};

__host__ __device__ __inline__ ColorXYZ FrDiel(float cosi, float cost, const ColorXYZ &etai, const ColorXYZ &etat)
{
	ColorXYZ Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	ColorXYZ Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
	return (Rparl*Rparl + Rperp*Rperp) / 2.f;
}

class Fresnel
{
public:
	__host__ __device__  Fresnel(float ei, float et) :
		eta_i(ei),
		eta_t(et)
	{
	}

	__host__ __device__   ~Fresnel(void)
	{
	}

	__host__ __device__  ColorXYZ Evaluate(float cosi)
	{
		// Compute Fresnel reflectance for dielectric
		cosi = optix::clamp(cosi, -1.0f, 1.0f);

		// Compute indices of refraction for dielectric
		bool entering = cosi > 0.0f;
		float ei = eta_i, et = eta_t;

		if (!entering)
			swap(ei, et);

		// Compute _sint_ using Snell's law
		float sint = ei / et * sqrtf(max(0.f, 1.f - cosi*cosi));

		if (sint >= 1.0f)
		{
			// Handle total internal reflection
			return 1.0f;
		}
		else
		{
			float cost = sqrtf(max(0.f, 1.0f - sint * sint));
			return FrDiel(fabsf(cosi), cost, ei, et);
		}
	}

	float eta_i, eta_t;
};

class Blinn
{
public:
	__host__ __device__  Blinn(const float& Exponent) :
		m_Exponent(Exponent)
	{
	}

	__host__ __device__  ~Blinn(void)
	{
	}

	__host__ __device__  void SampleF(const optix::float3& Wo, optix::float3& Wi, float& Pdf, const optix::float2& U)
	{
		// Compute sampled half-angle vector $\wh$ for Blinn distribution
		float costheta = powf(U.x, 1.f / (m_Exponent + 1));
		float sintheta = sqrtf(max(0.f, 1.f - costheta*costheta));
		float phi = U.y * 2.f * PI_F;

		optix::float3 wh = SphericalDirection(sintheta, costheta, phi);

		if (!SameHemisphere(Wo, wh))
			wh = -wh;

		// Compute incident direction by reflecting about $\wh$
		Wi = -Wo + 2.f * optix::dot(Wo, wh) * wh;

		// Compute PDF for $\wi$ from Blinn distribution
		float blinn_pdf = ((m_Exponent + 1.f) * powf(costheta, m_Exponent)) / (2.f * PI_F * 4.f * optix::dot(Wo, wh));

		if (optix::dot(Wo, wh) <= 0.f)
			blinn_pdf = 0.f;

		Pdf = blinn_pdf;
	}

	__host__ __device__  float Pdf(const optix::float3& Wo, const optix::float3& Wi)
	{
		optix::float3 wh = optix::normalize(Wo + Wi);

		float costheta = AbsCosTheta(wh);
		// Compute PDF for $\wi$ from Blinn distribution
		float blinn_pdf = ((m_Exponent + 1.f) * powf(costheta, m_Exponent)) / (2.f * PI_F * 4.f * optix::dot(Wo, wh));

		if (optix::dot(Wo, wh) <= 0.0f)
			blinn_pdf = 0.0f;

		return blinn_pdf;
	}

	__host__ __device__  float D(const optix::float3& wh)
	{
		float costhetah = AbsCosTheta(wh);
		return (m_Exponent + 2) * INV_TWO_PI_F * powf(costhetah, m_Exponent);
	}

	float	m_Exponent;
};

class Microfacet
{
public:
	__host__ __device__  Microfacet(const ColorXYZ& Reflectance, const float& Ior, const float& Exponent) :
		m_R(Reflectance),
		m_Fresnel(Ior, 1.0f),
		m_Blinn(Exponent)
	{
	}

	__host__ __device__  ~Microfacet(void)
	{
	}

	__host__ __device__  ColorXYZ F(const optix::float3& wo, const optix::float3& wi)
	{
		float cosThetaO = AbsCosTheta(wo);
		float cosThetaI = AbsCosTheta(wi);

		if (cosThetaI == 0.f || cosThetaO == 0.f)
			return SPEC_BLACK;

		optix::float3 wh = wi + wo;

		if (wh.x == 0. && wh.y == 0. && wh.z == 0.)
			return SPEC_BLACK;

		wh = optix::normalize(wh);
		float cosThetaH = optix::dot(wi, wh);

		ColorXYZ F = SPEC_WHITE;//m_Fresnel.Evaluate(cosThetaH);

		return m_R * m_Blinn.D(wh) * G(wo, wi, wh) * F / (4.f * cosThetaI * cosThetaO);
	}

	__host__ __device__  ColorXYZ SampleF(const optix::float3& wo, optix::float3& wi, float& Pdf, const optix::float2& U)
	{
		m_Blinn.SampleF(wo, wi, Pdf, U);

		if (!SameHemisphere(wo, wi))
			return SPEC_BLACK;

		return this->F(wo, wi);
	}

	__host__ __device__  float Pdf(const optix::float3& wo, const optix::float3& wi)
	{
		if (!SameHemisphere(wo, wi))
			return 0.0f;

		return m_Blinn.Pdf(wo, wi);
	}

	__host__ __device__  float G(const optix::float3& wo, const optix::float3& wi, const optix::float3& wh)
	{
		float NdotWh = AbsCosTheta(wh);
		float NdotWo = AbsCosTheta(wo);
		float NdotWi = AbsCosTheta(wi);
		float WOdotWh = fabsf(optix::dot(wo, wh));

		return min(1.f, min((2.f * NdotWh * NdotWo / WOdotWh), (2.f * NdotWh * NdotWi / WOdotWh)));
	}

	ColorXYZ		m_R;
	Fresnel		m_Fresnel;
	Blinn		m_Blinn;

};

class IsotropicPhase
{
public:
	__host__ __device__  IsotropicPhase(const ColorXYZ& Kd) :
		m_Kd(Kd)
	{
	}

	__host__ __device__  ~IsotropicPhase(void)
	{
	}

	__host__ __device__  ColorXYZ F(const optix::float3& Wo, const optix::float3& Wi)
	{
		return m_Kd * INV_PI_F;
	}

	__host__ __device__  ColorXYZ SampleF(const optix::float3& Wo, optix::float3& Wi, float& Pdf, const optix::float2& U)
	{
		Wi = UniformSampleSphere(U);
		Pdf = this->Pdf(Wo, Wi);

		return F(Wo, Wi);
	}

	__host__ __device__  float Pdf(const optix::float3& Wo, const optix::float3& Wi)
	{
		return INV_4_PI_F;
	}

	ColorXYZ	m_Kd;
};

class BRDF
{
public:
	__host__ __device__  BRDF(const optix::float3& N, const optix::float3& Wo, const ColorXYZ& Kd, const ColorXYZ& Ks, const float& Ior, const float& Exponent) :
		m_Lambertian(Kd),
		m_Microfacet(Ks, Ior, Exponent),
		m_Nn(N),
		m_Nu(optix::normalize(optix::cross(N, Wo))),
		m_Nv(optix::normalize(optix::cross(N, m_Nu)))
	{
	}

	__host__ __device__  ~BRDF(void)
	{
	}

	__host__ __device__  optix::float3 WorldToLocal(const optix::float3& W)
	{
		return make_float3(optix::dot(W, m_Nu), optix::dot(W, m_Nv), optix::dot(W, m_Nn));
	}

	__host__ __device__  optix::float3 LocalToWorld(const optix::float3& W)
	{
		return make_float3(m_Nu.x * W.x + m_Nv.x * W.y + m_Nn.x * W.z,
			m_Nu.y * W.x + m_Nv.y * W.y + m_Nn.y * W.z,
			m_Nu.z * W.x + m_Nv.z * W.y + m_Nn.z * W.z);
	}

	__host__ __device__  ColorXYZ F(const optix::float3& Wo, const optix::float3& Wi)
	{
		const optix::float3 Wol = WorldToLocal(Wo);
		const optix::float3 Wil = WorldToLocal(Wi);

		ColorXYZ R;

		R += m_Lambertian.F(Wol, Wil);
		R += m_Microfacet.F(Wol, Wil);

		return R;
	}

	__host__ __device__  ColorXYZ SampleF(const optix::float3& Wo, optix::float3& Wi, float& Pdf, const BRDFSample& S)
	{
		const optix::float3 Wol = WorldToLocal(Wo);
		optix::float3 Wil;

		ColorXYZ R;

		if (S.m_Component <= 0.5f)
		{
			m_Lambertian.SampleF(Wol, Wil, Pdf, S.m_Dir);
		}
		else
		{
			m_Microfacet.SampleF(Wol, Wil, Pdf, S.m_Dir);
		}

		Pdf += m_Lambertian.Pdf(Wol, Wil);
		Pdf += m_Microfacet.Pdf(Wol, Wil);

		R += m_Lambertian.F(Wol, Wil);
		R += m_Microfacet.F(Wol, Wil);

		Wi = LocalToWorld(Wil);

		return R;
	}

	__host__ __device__  float Pdf(const optix::float3& Wo, const optix::float3& Wi)
	{
		const optix::float3 Wol = WorldToLocal(Wo);
		const optix::float3 Wil = WorldToLocal(Wi);

		float Pdf = 0.0f;

		Pdf += m_Lambertian.Pdf(Wol, Wil);
		Pdf += m_Microfacet.Pdf(Wol, Wil);

		return Pdf;
	}

	optix::float3			m_Nn;
	optix::float3			m_Nu;
	optix::float3			m_Nv;
	Lambertian		m_Lambertian;
	Microfacet		m_Microfacet;
};

class VolumeShader
{
public:
	enum Type
	{
		Brdf,
		Phase
	};

	__host__ __device__  VolumeShader(const Type& Type, const optix::float3& N, const optix::float3& Wo, const ColorXYZ& Kd, const ColorXYZ& Ks, const float& Ior, const float& Exponent) :
		m_Type(Type),
		m_Brdf(N, Wo, Kd, Ks, Ior, Exponent),
		m_IsotropicPhase(Kd)
	{
	}

	__host__ __device__  ~VolumeShader(void)
	{
	}

	__host__ __device__  ColorXYZ F(const optix::float3& Wo, const optix::float3& Wi)
	{
		switch (m_Type)
		{
		case Brdf:
			return m_Brdf.F(Wo, Wi);

		case Phase:
			return m_IsotropicPhase.F(Wo, Wi);
		}

		return 1.0f;
	}

	__host__ __device__  ColorXYZ SampleF(const optix::float3& Wo, optix::float3& Wi, float& Pdf, const BRDFSample& S)
	{
		switch (m_Type)
		{
		case Brdf:
			return m_Brdf.SampleF(Wo, Wi, Pdf, S);

		case Phase:
			return m_IsotropicPhase.SampleF(Wo, Wi, Pdf, S.m_Dir);
		}
	}

	__host__ __device__ float Pdf(const optix::float3& Wo, const optix::float3& Wi)
	{
		switch (m_Type)
		{
		case Brdf:
			return m_Brdf.Pdf(Wo, Wi);

		case Phase:
			return m_IsotropicPhase.Pdf(Wo, Wi);
		}

		return 1.0f;
	}

	Type				m_Type;
	BRDF				m_Brdf;
	IsotropicPhase		m_IsotropicPhase;
};