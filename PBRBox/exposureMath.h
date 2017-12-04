#pragma once
#include <optixu/optixu_math_namespace.h>

#define PI_F												3.141592654f	
#define HALF_PI_F											0.5f * PI_F
#define QUARTER_PI_F										0.25f * PI_F
#define TWO_PI_F											2.0f * PI_F
#define INV_PI_F											0.31830988618379067154f
#define INV_TWO_PI_F										0.15915494309189533577f
#define FOUR_PI_F											4.0f * PI_F
#define INV_4_PI_F											1.0f / FOUR_PI_F
#define	EULER_F												2.718281828f
#define RAD_F												57.29577951308232f
#define TWO_RAD_F											2.0f * RAD_F
#define DEG_TO_RAD											1.0f / RAD_F

__device__ __host__ __inline__ float  distanceSquared(optix::float3 p1, optix::float3 p2)
{
	float3 temp = (p1 - p2);
	return temp.x * temp.x + temp.y * temp.y + temp.z * temp.z;
}