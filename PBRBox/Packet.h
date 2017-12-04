#ifndef __optix_prd_h__
#define __optix_prd_h__

#include <optixu/optixu_math_namespace.h>
#include "random.h"


struct __align__(128) PerRayData_pathtrace
{
	optix::float3 result;
	optix::float3 radiance;
	optix::float3 attenuation;
	optix::float3 origin;
	optix::float3 direction;
	unsigned int seed;
	float depth;
	int countEmitted;
	int done;
	curandState_t* state;
};

struct PerRayData_pathtrace_shadow
{
	int inShadow;
	curandState_t* state;
};

struct PerRayData_volumeTrace
{

};

#endif