//
//  volumetric.cu
//  optixVolumetric
//
//  Created by Tim Tavlintsev (TVL)
//
//

#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

#include "helpers.h"
#include "random.h"
#include "Packet.h"
#include "ColorXYZ.h"

#include "Sample.h"
#include "Lighting.h"
#include "MonteCarlo.h"
#include "VolumeShader.h"

using namespace optix;


rtTextureSampler<unsigned char, 3> volume_texture;

rtTextureSampler<uchar4, 1> tf_texture;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, hitcolor, attribute hitcolor, );
rtDeclareVariable(int, inshadow, attribute inshadow, );
rtDeclareVariable(int, hasReflection, attribute hasReflection, );
rtDeclareVariable(unsigned char, hitIntensity, attribute hitIntensity, );
rtDeclareVariable(float3, hitPosition, attribute hitPosition, );
rtDeclareVariable(float3, reflectionDir, attribute reflectionDir, );
rtDeclareVariable(float3, hitAlbedo, attribute hitAlbedo, );

rtDeclareVariable(float3, diffuse_color, , );

rtDeclareVariable(PerRayData_pathtrace, prd_path, rtPayload, );
rtDeclareVariable(PerRayData_pathtrace_shadow, prd_shadow, rtPayload, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(unsigned int, pathtrace_ray_type, , );
rtDeclareVariable(unsigned int, pathtrace_shadow_ray_type, , );

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );


rtDeclareVariable(int, num_lights, , );
rtBuffer<Light>        lights;


rtDeclareVariable(rtObject, top_shadower, , );

rtDeclareVariable(float, gradientDelta, , );
rtDeclareVariable(float, invGradientDelta, , );
rtDeclareVariable(float3, gradientDeltaX, , );
rtDeclareVariable(float3, gradientDeltaY, , );
rtDeclareVariable(float3, gradientDeltaZ, , );

rtDeclareVariable(float, stepSize, , );
rtDeclareVariable(float, stepSizeShadow, , );
rtDeclareVariable(float, densityScale, , );
rtDeclareVariable(float, invDensityScale, , );
/*
RT_PROGRAM void any_hit_shadow()
{

// #define TRANSPARENT 1
#ifndef TRANSPARENT
// this material is opaque, so it fully attenuates all shadow rays
prd_shadow.attenuation = optix::make_float3(0);
rtTerminateRay();

#else
// Attenuates shadow rays for shadowing transparent objects
float3 world_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
float nDi = fabs(dot(world_normal, ray.direction));

prd_shadow.attenuation *= 1 - fresnel_schlick(nDi, 5, 1 - shadow_attenuation, make_float3(1));
if (optix::luminance(prd_shadow.attenuation))// < importance_cutoff)
rtTerminateRay();
else
rtIgnoreIntersection();
#endif

}
*/




RT_PROGRAM void debug_normal()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);

	prd_path.radiance = ((world_geo_normal + make_float3(1.0f)) *.5f);
	prd_path.done = true;
}

RT_PROGRAM void ambient_shader()
{
	prd_path.result = hitcolor;
}

void EstimateDirectLight()
{

}

__device__ ColorXYZ sampleOneLight(curandState_t* state)
{
	if (num_lights == 0)
	{
		return SPEC_BLACK;
	}

	/*LightingSample LS;
	//LS.LargeStep(state);

	int lightIndex = (int)floorf(LS.m_lightNum * float(num_lights));

	Light& light = lights[lightIndex];
	return (float)NumLights * EstimatedDirectLight(pScene, Type, Density, Light, LS, Wo, Pe, N, RNG);*/
}

rtTextureSampler<float4, 2> envmap;



__device__ float3 sampleEnvironment(float3 sample)
{
	float theta = atan2f(sample.x, sample.z);
	float phi = M_PIf * 0.5f - acosf(sample.y);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
	return make_float3(tex2D(envmap, u, v));
	//here we sample the environment
}


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
	return 1 - intensity;
}

inline __device__ float3 getEmission(unsigned char intensity)
{
	return optix::make_float3(1, 0, 0);
}

inline __device__ float getNormalizedIntensity(unsigned char intensity)
{
	return intensity / 255.0f;
}


inline __device__ bool FreePathRM(optix::Ray ray, curandState_t* rndState)
{
	//unsigned int rseed = (512 * launch_index.y + launch_index.x) * frame_number;
	const float s = -log(curand_uniform(rndState)) / densityScale;
	float sum = 0.0f;
	float sigmaT = 0.0f;

	float3 pos;
	float t = ray.tmin + curand_uniform(rndState) * stepSizeShadow;

	while (sum < s)
	{
		pos = ray.origin + t * ray.direction;

		if (t > ray.tmax)
			return false;

		sigmaT = densityScale * getOpacity(sampleVolume(pos));

		sum += sigmaT * stepSizeShadow;
		t += stepSizeShadow;
	}

	return true;
}

inline __device__ bool SampleDistanceRM(float tMin, float tMax, float3& pos, float& tHit)
{
	const float s = -log(1.0 - curand_uniform(prd_path.state)) * invDensityScale;
	float sum = 0.0f;
	float sigmaT = 0.0f;

	float t = tMin + curand_uniform(prd_path.state) * stepSize;

	while (sum < s)
	{
		pos = ray.origin + t * ray.direction;
		tHit = t;
		if (t > tMax)
			return false;

		sigmaT = densityScale * getOpacity(sampleVolume(pos));
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


RT_PROGRAM void any_hit_shadow()
{
	// this material is opaque, so it fully attenuates all shadow rays

	//if(!FreePathRM(ray, prd_shadow.state))
	prd_shadow.inShadow = true;
	rtTerminateRay();
}

/*
DEV bool NearestLight(CScene* pScene, CRay R, CColorXyz& LightColor, Vec3f& Pl, CLight*& pLight, float* pPdf = NULL)
{
bool Hit = false;

float T = 0.0f;

CRay RayCopy = R;

float Pdf = 0.0f;

for (int i = 0; i < pScene->m_Lighting.m_NoLights; i++)
{
if (pScene->m_Lighting.m_Lights[i].Intersect(RayCopy, T, LightColor, NULL, &Pdf))
{
Pl = R(T);
pLight = &pScene->m_Lighting.m_Lights[i];
Hit = true;
}
}

if (pPdf)
*pPdf = Pdf;

return Hit;
}*/

// here we sample one random light and the environment to get an estimated lighting contribution
RT_PROGRAM void singleScattering()
{
	VolumeShader::Type type = VolumeShader::Brdf;
	ColorXYZ Ld = SPEC_BLACK;
	ColorXYZ Li = SPEC_BLACK;
	ColorXYZ F = SPEC_BLACK;


	Light AreaLight;
	float intensity = 10;

	AreaLight.m_T = 0;
	AreaLight.m_Theta = 0 / RAD_F;
	AreaLight.m_Phi = 0 / RAD_F;
	AreaLight.m_Width = 5;
	AreaLight.m_Height = 5;
	AreaLight.m_Distance = 144;
	AreaLight.m_Color = ColorRGB(1, 1, 1) * intensity;

	optix::Aabb bb(make_float3(-144, -144, -144), make_float3(144, 144, 144));
	AreaLight.Update(bb);

	Light AreaLight2;

	AreaLight2.m_T = 0;
	AreaLight2.m_Theta = 0 / RAD_F;
	AreaLight2.m_Phi = 40 / RAD_F;
	AreaLight2.m_Width = 5;
	AreaLight2.m_Height = 5;
	AreaLight2.m_Distance = 144;
	AreaLight2.m_Color = ColorRGB(0, 0, 1) * intensity;

	AreaLight2.Update(bb);

	Light AreaLight3;
	AreaLight3.m_T = 1;
	AreaLight3.Update(bb);


	//lights[0].m_Height = 3;
	//lights[0].m_Distance = 4;
	//lights[0].m_Color = ColorRGB(1, 0, 0) * intensity;
	//lights[0].Update(bb);

	float lightPdf = 1.0f;
	float shaderPdf = 1.0f;

	float3 Wi;
	float3 P;
	float3 Pl;

	LightingSample LS;
	LS.LargeStep(prd_path.state);

	ColorXYZ albedo = ColorXYZ::FromRGB(0, 0, 0);
	ColorXYZ specular = ColorXYZ::FromRGB(hitAlbedo.x, hitAlbedo.y, hitAlbedo.z);
	float metalness = 1.0f;
	float roughness = 0.0f;

	float3 Wo = normalize(-ray.direction);
	float3 N = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 V = ray.direction;
	float3 R = reflect(ray.direction, N);

	VolumeShader shader(type, N, Wo, specular, albedo, 1.5f, roughness);

	optix::Ray Rl;

	float LightPdf = 1.0f, ShaderPdf = 1.0f;
	float rnd = random_float(prd_path.state);

	if (rnd < 1.2f)
		Li = AreaLight.SampleL(hitPosition, Rl, LightPdf, LS);
	else
	{
		Li = AreaLight3.SampleL(hitPosition, Rl, LightPdf, LS);
		//float3 envColor = sampleEnvironment(-make_float3(Li[0], Li[1], Li[2]));
		Li = ColorXYZ::FromRGB(1, 0, 1);// (envColor.x, envColor.y, envColor.z);
	}
	//else
	{


		/*float3 tempo = reflect(Wo, N);// 1 * UniformSampleSphere(LS.m_LightSample.m_Pos);
		float3 tempd = normalize(hitPosition - tempo);
		float m_Area = 4.0f * PI_F * powf(1, 2.0f);
		LightPdf = powf(1, 2.0f) / m_Area;

		Rl = make_Ray(hitPosition, tempo, 1, 0.0f, 100);
		float3 ce = sampleEnvironment(-tempd);
		Li =  Li.FromRGB(ce.x, ce.y, ce.z);*/
	}

	//Light pLight = AreaLight2;

	Wi = -Rl.direction;

	F = shader.F(Wo, Wi);

	ShaderPdf = shader.Pdf(Wo, Wi);


	if (!Li.IsBlack() && ShaderPdf > 0.0f && LightPdf > 0.0f)
	{
		PerRayData_pathtrace_shadow prd;
		prd.state = prd_path.state;
		rtTrace(top_object, Rl, prd);
		if (!prd.inShadow)
		{

			const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, ShaderPdf);
			if (type == VolumeShader::Brdf)
				Ld += F * Li * fabsf(dot(Wi, N)) * WeightMIS / LightPdf;
			if (type == VolumeShader::Phase)
				Ld += F * Li * WeightMIS / LightPdf;
		}
	}

	F = shader.SampleF(Wo, Wi, ShaderPdf, LS.m_BsdfSample);

	if (!F.IsBlack() && ShaderPdf > 0.0f)
	{
		//if (NearestLight(pScene, CRay(hitPosition, Wi, 0.0f), Li, Pl, pLight, &LightPdf))
		{

			LightPdf = AreaLight3.Pdf(hitPosition, Wi);

			//LightPdf = AreaLight.Pdf(hitPosition, Wi);

			if (LightPdf > 0.0f && !Li.IsBlack())
			{
				//&& !FreePathRM(CRay(Pl, Normalize(hitPosition - Pl), 0.0f, (hitPosition - Pl).Length()), RNG)
				PerRayData_pathtrace_shadow prd;
				prd.state = prd_path.state;
				optix::Ray ray = make_Ray(Pl, normalize(hitPosition - Pl), 1, 0, length(hitPosition - Pl));
				rtTrace(top_object, ray, prd);

				if (!prd.inShadow)
				{
					const float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, LightPdf);

					if (type == VolumeShader::Brdf)
						Ld += F * Li * fabsf(dot(Wi, N)) * WeightMIS / ShaderPdf;

					if (type == VolumeShader::Phase)
						Ld += F * Li * WeightMIS / ShaderPdf;
				}
			}
		}
	}




	//float3 diffuseColor = albedo * (1 - metalness);
	//float3 specularColor = lerp(make_float3(0.04f), albedo, metalness);
	ColorRGB colorr; colorr.FromXYZ(Ld[0], Ld[1], Ld[2]);
	float3 color = make_float3(1,0,0);
	//color = (N + make_float3(1, 1, 1)) *.5f;
	//float depth = computeClipDepth(t_hit, -1, 10);
	//color = make_float3(depth, depth, depth);
	//color += ColorXYZ::ToRGB(albedo);// *texture(uIrradianceMap, normal).rgb;
	//color += 

	//if (albedo != color) { // skip if no diffuse
	//	color += uBrightness * albedo * texture(uIrradianceMap, normal).rgb * getAO();//* evaluateDiffuseSphericalHarmonics(normal,view );
	//}
	//color = texture(uRadianceMap, normal).rgb;
	/*float roughnessLinear = max(rLinear, 0.0);
	float NoV = clamp(dot(N, V), 0.0, 1.0);
	vec3 R = normalize((2.0 * NoV) * N - V);


	// From Sebastien Lagarde Moving Frostbite to PBR page 69
	// so roughness = linRoughness * linRoughness
	vec3 dominantR = getSpecularDominantDir(N, R, roughnessLinear*roughnessLinear);

	vec3 dir = (vec4(dominantR, 0)).rgb;
	vec3 prefilteredColor = prefilterEnvMap(roughnessLinear, dir);


	// marmoset tricks
	//prefilteredColor *= occlusionHorizon( dominantR, (camera.mView * vec4(N,0)).rgb );

	//return uBrightness * prefilteredColor * integrateBRDFApprox( specularColor, roughnessLinear, NoV );
	vec4 envBRDF = texture2D(uBRDFLUT, vec2(NoV, roughnessLinear));
	//return vec3(roughnessToMip(roughnessLinear)/7);
	return uBrightness * prefilteredColor * (specularColor * envBRDF.x + envBRDF.y);

	color += approximateSpecularIBL(specular, roughness, normal, view);*/
	//float mip = (roughness <.01) ? 1 : 0;
	//color = vec3(mip);





	//Lv += GetEmission(intensity).ToXYZ();

	//if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li, Pl, pLight))
	//{
	//	pView->m_FrameEstimateXyza.Set(CColorXyza(Lv.c[0], Lv.c[1], Lv.c[2]), X, Y);
	//	return;
	//}
	//Lv += UniformSampleOneLight(pScene, CVolumeShader::Brdf, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe), RNG, true);
	//PerRayData_pathtrace_shadow shadow_prd;
	//shadow_prd.inShadow = false;
	//Ray shadow_ray = make_Ray(hitpoint, make_float3(0,1,0), pathtrace_shadow_ray_type, 0, RT_DEFAULT_MAX);

	//rtTrace(top_object, shadow_ray, shadow_prd);
	prd_path.radiance = make_float3(0);
	if (hasReflection)
	{
		//prd_path.radiance += sampleEnvironment(reflectionDir);
	}

	prd_path.radiance += color;// sampleEnvironment(reflectionDir);
							   // sampleEnvironment(reflection);
	if (inshadow)
	{
		//prd_path.radiance *=0.0f;
	}

	// make_float3(1.0f);
	//pView->m_FrameEstimateXyza.Set(CColorXyza(Lv.c[0], Lv.c[1], Lv.c[2]), X, Y);

}


/*
RT_PROGRAM void diffuse()
{
float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

float3 hitpoint = ray.origin + t_hit * ray.direction;

//
// Generate a reflection ray.  This will be traced back in ray-gen.
//
prd_path.origin = hitpoint;

float z1 = 1;
float z2 = 1;
float3 p;
cosine_sample_hemisphere(z1, z2, p);
optix::Onb onb(ffnormal);
onb.inverse_transform(p);
prd_path.direction = p;

// NOTE: f/pdf = 1 since we are perfectly importance sampling lambertian
// with cosine density.
//prd_path.attenuation = prd_path.attenuation * diffuse_color;
prd_path.countEmitted = false;

//
// Next event estimation (compute direct lighting).
//
unsigned int num_lights = lights.size();
float3 result = make_float3(0.0f);

for (int i = 0; i < num_lights; ++i)
{
// Choose random point on light
ParallelogramLight light = lights[i];
const float z1 = rnd(prd_path.seed);
const float z2 = rnd(prd_path.seed);
const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

// Calculate properties of light sample (for area based pdf)
const float  Ldist = length(light_pos - hitpoint);
const float3 L = normalize(light_pos - hitpoint);
const float  nDl = dot(ffnormal, L);
const float  LnDl = dot(light.normal, L);

// cast shadow ray
if (nDl > 0.0f && LnDl > 0.0f)
{
PerRayData_pathtrace_shadow shadow_prd;
shadow_prd.inShadow = false;
// Note: bias both ends of the shadow ray, in case the light is also present as geometry in the scene.
Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
rtTrace(top_object, shadow_ray, shadow_prd);

if (!shadow_prd.inShadow)
{
const float A = length(cross(light.v1, light.v2));
// convert area based pdf to solid angle
const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
result += light.emission * weight;
}
}
}

prd_path.radiance = result;
}
*/