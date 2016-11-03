
#ifdef USE_ALBEDO_MAP
	uniform sampler2D uAlbedo;
	vec3 getAlbedo() 
	{
		return texture2D(uAlbedo, vTexCord0).rgb;
	}
#else
	uniform vec4 uAlbedo; 
	vec3 getAlbedo() 
	{
		return uAlbedo.rgb;
	}
#endif

#ifdef USE_ROUGHNESS_MAP
	uniform sampler2D uRoughness;
	float getRoughness() 
	{
		return texture2D(uRoughness, vTexCord0).r;
	}
#else
	uniform float uRoughness;
	float getRoughness() 
	{
		return uRoughness;
	}
#endif

#ifdef USE_METALNESS_MAP
	uniform sampler2D uMetalness;
	float getMetalness() {
		return texture2D(uMetalness, vTexCord0).r;
	}
#else
	uniform float uMetalness;
	float getMetalness() {
		return uMetalness;
	}
#endif

#ifdef USE_NORMAL_MAP
	uniform sampler2D uNormal;
	#pragma glslify: perturb = require('glsl-perturb-normal')
	vec3 getNormal() {
		vec3 normalRGB = texture2D(uNormal, vTexCord0).rgb;
		vec3 normalMap = normalRGB * 2.0 - 1.0;

		normalMap.y *= -1.0;

		vec3 N = normalize(vNormalView);
		vec3 V = normalize(vEyeDirView);

		vec3 normalView = perturb(normalMap, N, V, vTexCord0);
		vec3 normalWorld = vec3(uInverseViewMatrix * vec4(normalView, 0.0));
		return normalWorld;
	}
#else
	vec3 getNormal() {
		return normalize(vNormalWorld);
	}
#endif
