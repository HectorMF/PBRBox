#pragma once

#include <optixu/optixu_math_namespace.h>
using namespace optix;

class BoundingBox
{
public:
	float3	m_MinP;
	float3	m_MaxP;

	BoundingBox(void) :
		m_MinP(FLT_MAX, FLT_MAX, FLT_MAX),
		m_MaxP(-FLT_MAX, -FLT_MAX, -FLT_MAX)
	{
	};

	__host__ __device__ BoundingBox(const float3 &v1, const float3 &v2) :
		m_MinP(v1),
		m_MaxP(v2)
	{
		optix::
	}

	__host__ __device__ BoundingBox& operator = (const BoundingBox& B)
	{
		m_MinP = B.m_MinP;
		m_MaxP = B.m_MaxP;

		return *this;
	}

	// Adds a point to this bounding box
	BoundingBox& operator += (const float3& P)
	{
		if (!Contains(P))
		{
			for (int i = 0; i < 3; i++)
			{
				if (P[i] < m_MinP[i])
					m_MinP[i] = P[i];

				if (P[i] > m_MaxP[i])
					m_MaxP[i] = P[i];
			}
		}

		return *this;
	}

	// Adds a bounding box to this bounding box
	BoundingBox& operator += (const BoundingBox& B)
	{
		*this += B.m_MinP;
		*this += B.m_MaxP;

		return *this;
	}

	float3 &operator[](int i)
	{
		return (&m_MinP)[i];
	}

	const float3 &operator[](int i) const
	{
		return (&m_MinP)[i];
	}

	__host__ __device__ float LengthX(void) const	{ return fabs(m_MaxP.x - m_MinP.x); };
	__host__ __device__ float LengthY(void) const	{ return fabs(m_MaxP.y - m_MinP.y); };
	__host__ __device__ float LengthZ(void) const	{ return fabs(m_MaxP.z - m_MinP.z); };

	__host__ __device__ float3 GetCenter(void) const
	{
		return make_float3(0.5f * (m_MinP.x + m_MaxP.x), 0.5f * (m_MinP.y + m_MaxP.y), 0.5f * (m_MinP.z + m_MaxP.z));
	}

	__host__ __device__ EContainment Contains(const float3& P) const
	{
		for (int i = 0; i < 3; i++)
		{
			if (P[i] < m_MinP[i] || P[i] > m_MaxP[i])
				return ContainmentNone;
		}

		return ContainmentFull;
	};

	__host__ __device__ EContainment Contains(const float3* pPoints, long PointCount) const
	{
		long Contain = 0;

		for (int i = 0; i < PointCount; i++)
		{
			if (Contains(pPoints[i]) == ContainmentFull)
				Contain++;
		}

		if (Contain == 0)
			return ContainmentNone;
		else
		{
			if (Contain == PointCount)
				return ContainmentFull;
			else
				return ContainmentPartial;
		}
	}

	__host__ __device__ EContainment Contains(const BoundingBox& B) const
	{
		bool ContainsMin = false, ContainsMax = false;

		if (Contains(B.m_MinP) == ContainmentFull)
			ContainsMin = true;

		if (Contains(B.m_MaxP) == ContainmentFull)
			ContainsMax = true;

		if (!ContainsMin && !ContainsMax)
			return ContainmentNone;
		else
		{
			if (ContainsMin && ContainsMax)
				return ContainmentFull;
			else
				return ContainmentPartial;
		}
	}

	__host__ __device__ EAxis GetDominantAxis(void) const
	{
		return (LengthX() > LengthY() && LengthX() > LengthZ()) ? AxisX : ((LengthY() > LengthZ()) ? AxisY : AxisZ);
	}

	__host__ __device__	float3				GetMinP(void) const		{ return m_MinP; }
	__host__ __device__	float3				GetInvMinP(void) const	{ return make_float3(1.0f) / m_MinP; }
	__host__ __device__	void				SetMinP(float3 MinP)		{ m_MinP = MinP; }
	__host__ __device__	float3				GetMaxP(void) const		{ return m_MaxP; }
	__host__ __device__	float3				GetInvMaxP(void) const	{ return make_float3(1.0f) / m_MaxP; }
	__host__ __device__	void				SetMaxP(float3 MaxP)		{ m_MaxP = MaxP; }

	HO float GetMaxLength(EAxis* pAxis = NULL) const
	{
		if (pAxis)
			*pAxis = GetDominantAxis();

		const float3& MinMax = GetExtent();

		return MinMax[GetDominantAxis()];
	}

	HO float HalfSurfaceArea(void) const
	{
		const float3 e(GetExtent());
		return e.x * e.y + e.y * e.z + e.x * e.z;
	}

	HO float GetArea(void) const
	{
		const float3 ext(m_MaxP - m_MinP);
		return float(ext.x)*float(ext.y) + float(ext.y)*float(ext.z) + float(ext.x)*float(ext.z);
	}

	HO float3 GetExtent(void) const
	{
		return m_MaxP - m_MinP;
	}

	HO float GetEquivalentRadius(void) const
	{
		return 0.5f * GetExtent().Length();
	}

	__host__ __device__ bool Inside(const float3& pt)
	{
		return (pt.x >= m_MinP.x && pt.x <= m_MaxP.x &&
			pt.y >= m_MinP.y && pt.y <= m_MaxP.y &&
			pt.z >= m_MinP.z && pt.z <= m_MaxP.z);
	}

	// Performs a line box intersection
	__host__ __device__ bool Intersect(CRay& R, float* pMinT = NULL, float* pMaxT = NULL)
	{
		// Compute intersection of line with all six bounding box planes
		const float3 InvR = make_float3(1.0f / R.m_D.x, 1.0f / R.m_D.y, 1.0f / R.m_D.z);
		const float3 BotT = InvR * (m_MinP - R.m_O);
		const float3 TopT = InvR * (m_MaxP - R.m_O);

		// re-order intersections to find smallest and largest on each axis
		const float3 MinT = make_float3(min(TopT.x, BotT.x), min(TopT.y, BotT.y), min(TopT.z, BotT.z));
		const float3 MaxT = make_float3(max(TopT.x, BotT.x), max(TopT.y, BotT.y), max(TopT.z, BotT.z));

		// find the largest tmin and the smallest tmax
		const float LargestMinT = max(max(MinT.x, MinT.y), max(MinT.x, MinT.z));
		const float SmallestMaxT = min(min(MaxT.x, MaxT.y), min(MaxT.x, MaxT.z));

		if (pMinT)
			*pMinT = LargestMinT;

		if (pMaxT)
			*pMaxT = SmallestMaxT;

		return SmallestMaxT > LargestMinT;
	}

	__host__ __device__ bool IntersectP(const CRay& ray, float* hitt0 = NULL, float* hitt1 = NULL)
	{
		float t0 = ray.m_MinT, t1 = ray.m_MaxT;

		for (int i = 0; i < 3; ++i)
		{
			// Update interval for _i_th bounding box slab
			float invRayDir = 1.f / ray.m_D[i];
			float tNear = (m_MinP[i] - ray.m_O[i]) * invRayDir;
			float tFar = (m_MaxP[i] - ray.m_O[i]) * invRayDir;

			// Update parametric interval from slab intersection $t$s
			if (tNear > tFar)
				swap(tNear, tFar);

			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  < t1 ? tFar : t1;

			if (t0 > t1)
				return false;
		}

		if (hitt0)
			*hitt0 = t0;

		if (hitt1)
			*hitt1 = t1;

		return true;
	}

	void PrintSelf(void)
	{
		printf("Min: ");
		m_MinP.PrintSelf();

		printf("Max: ");
		m_MaxP.PrintSelf();
	}
};