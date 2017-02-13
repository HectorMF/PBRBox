// code for depth-of-field, mouse + keyboard user interaction based on https://github.com/peterkutz/GPUPathTracer

#pragma once

#include "glm\glm.hpp"
#define M_PI_2 1.57079632679
#define M_PI 3.14156265
#define PI_OVER_TWO 1.5707963267948966192313216916397514420985
#define scrwidth 1280
#define scrheight 720












// Camera struct, used to store interactive camera data, copied to the GPU and used by CUDA for each frame
struct Camera {
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
	float apertureRadius;
	float focalDistance;
};

// class for interactive camera object, updated on the CPU for each frame and copied into Camera struct
class InteractiveCamera
{
private:

	glm::vec3 centerPosition;
	glm::vec3 viewDirection;
	float yaw;
	float pitch;
	float radius;
	float apertureRadius;
	float focalDistance;

	void fixYaw();
	void fixPitch();
	void fixRadius();
	void fixApertureRadius();
	void fixFocalDistance();

public:
	InteractiveCamera();
	virtual ~InteractiveCamera();
	void changeYaw(float m);
	void changePitch(float m);
	void changeRadius(float m);
	void changeAltitude(float m);
	void changeFocalDistance(float m);
	void strafe(float m);
	void goForward(float m);
	void rotateRight(float m);
	void changeApertureDiameter(float m);
	void setResolution(float x, float y);
	void setFOVX(float fovx);

	void buildRenderCamera(Camera* renderCamera);

	glm::vec2 resolution;
	glm::vec2 fov;
};
