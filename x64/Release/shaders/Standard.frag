

in vec2 uv;

out vec4 fragColor;



//vertex position in the eye coordinates (view space)
in vec3 ecPosition;
//normal in the eye coordinates (view space)
in vec3 ecNormal;
//light position in the eye coordinates (view space)
in vec3 ecLightPos;

in vec4 fragPosLightSpace;

in vec3 wPos;
in vec3 wNormal;
in vec3 lightPos;


















#ifdef USE_ALBEDO_MAP
	uniform sampler2D uAlbedo;
	vec3 getAlbedo() 
	{
		return texture2D(uAlbedo, uv).rgb;
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
		return texture2D(uRoughness, uv).r;
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
		return texture2D(uMetalness, uv).r;
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
		vec3 normalRGB = texture2D(uNormal, uv).rgb;
		vec3 normalMap = normalRGB * 2.0 - 1.0;

		normalMap.y *= -1.0;

		vec3 N = normalize(vNormalView);
		vec3 V = normalize(vEyeDirView);

		vec3 normalView = perturb(normalMap, N, V, uv);
		vec3 normalWorld = vec3(uInverseViewMatrix * vec4(normalView, 0.0));
		return normalize(normalWorld);
	}
#else
	vec3 getNormal() {
		return normalize(wNormal);
	}
#endif


struct Camera
{
	vec3 vViewPos;
	vec3 vViewDirection;
	mat4 mProjection;
	mat4 mView;
	mat4 mInvView;
	mat4 mNormal;
	mat4 mModel;
};

uniform Camera camera;

float SchlickFresnel(float u)
{
    float m = clamp(1.0-u, 0.0, 1.0);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

void main() 
{
	float roughness = getRoughness();
	float metalness = getMetalness();
	vec3 albedo = getAlbedo();
	vec3 normal = getNormal();
	
	vec3 L = normalize(ecLightPos - ecPosition);
	vec3 V = normalize(camera.vViewPos - ecPosition);
	vec3 H = normalize(L + V);
	
    float NdotH = dot(ecNormal,H);
    float LdotH = dot(L,H);
	float NdotL = dot(ecNormal,L);
    float NdotV = dot(ecNormal,V);
	
	float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5 + 2 * LdotH*LdotH * roughness;
    float Fd = mix(1, Fd90, FL) * mix(1, Fd90, FV);
	
	float Fss90 = LdotH*LdotH*roughness;
    float Fss = mix(1, Fss90, FL) * mix(1, Fss90, FV);
    float ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5);
	
	
	
	fragColor = vec4(vec3(ss), 1.0);
	/*
    vec3 L = normalize(ecLightPos - ecPosition);
	vec3 H = normalize( camera.vViewDirection + (normalize(-ecLightPos)));
   
	//Geometric term
	float NdotH = max(dot(N, H), 0.0);
	float VdotH = max(dot(camera.vViewDirection, H), 0.000001);
	  
    //calculate direction towards the light

    //diffuse intensity
    float Id = lambert(L, N);
	
	//surface and light color, full white
    vec4 baseColor = toLinear(vec4(1.0)); 
    vec4 lightColor = toLinear(vec4(1.0)); 

    //lighting in the linear space
	vec3 specular = Fresnel_Shlick(vec3(.333), VdotH);
	
    vec3 finalColor = texture2D(material.diffuse, uv).rgb;

    //direction towards they eye (camera) in the view (eye) space
    vec3 ecEyeDir = normalize(-ecPosition);
    //direction towards the camera in the world space
    vec3 wcEyeDir = vec3(camera.mInvView * vec4(ecEyeDir, 0.0));
    //surface normal in the world space

    //vec3 wcNormal = vec3(camera.mInvView * vec4(N, 0.0));
	vec3 wcNormal =  N;
	
    vec3 reflectionWorld = reflect(-wcEyeDir, normalize(wcNormal));
	vec4 reflection;
	
	vec3 ambient = vec3(.4);
	bool isMetallic = (texture2D(uMetallicMap, uv).r + texture2D(uMetallicMap, uv).g + texture2D(uMetallicMap, uv).b )> 0.0;
	
	if(isMetallic)
		reflection = texture2D(uReflectionMap, envMapEquirect(reflectionWorld));
	else
		reflection = vec4(1);//texture2D(uEnvMap, envMapEquirect(reflectionWorld));
	 
	 
	 
	 
	 
	 
	 
	float uIor = 1.4;
	float metalness = texture2D(uMetallicMap, uv).r;
	float roughness = texture2D(uRoughnessMap, uv).r;
	vec3 albedo = texture2D(material.diffuse, uv).rgb;

	
	vec3 F0 = vec3(abs((1.0 - uIor) / (1.0 + uIor)));
    F0 = F0 * F0;
    //F0 = vec3(0.04); //0.04 is default for non-metals in UE4
    F0 = mix(F0, albedo, metalness);

    //float NdotV = saturate( dot( normalWorld, eyeDirWorld ) );
	float NdotV = saturate( dot( wcNormal, ecEyeDir ) );
    vec3 reflectance = EnvBRDFApprox( F0, roughness, NdotV );
	 
	 
	 
	 
	 
	vec3 diffuseColor = albedo * (1.0 - metalness);

    //TODO: No kd? so not really energy conserving
    //we could use disney brdf for irradiance map to compensate for that like in Frostbite
	vec3 reflectionColor = texture2D(uEnvMap, envMapEquirect(reflectionWorld)).rgb;
	vec3 irradianceColor = texture2D(uReflectionMap, envMapEquirect(reflectionWorld)).rgb;
    vec3 color = diffuseColor * irradianceColor + reflectionColor * reflectance;
	 
	 
	 
	 
	 
	 
	 
	 
	float shadow = ShadowCalculation(fragPosLightSpace);   
	shadow = min(shadow, 0.75);
    vec3 lighting =  (1.0 - shadow) * color;  
	
	
	color = color;
    //reflection vector in the world space. We negate wcEyeDir as the reflect function expect incident vector pointing towards the surface
    fragColor = vec4(color, 1.0);//toGamma(finalColor);   */ 
}
