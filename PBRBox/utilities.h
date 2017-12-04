
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
#include <optixu/optixu_matrix_namespace.h>

inline __host__ __device__ float fracf(float v)
{
	return v - floorf(v);
}

inline __host__ __device__ float2 fracf(float2 v)
{
	return make_float2(fracf(v.x), fracf(v.y));
}

inline __host__ __device__ float3 fracf(float3 v)
{
	return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}

inline __host__ __device__ float3 floor3(float3 a)
{
	return make_float3(floorf(a.x), floorf(a.y), floorf(a.z));
}
