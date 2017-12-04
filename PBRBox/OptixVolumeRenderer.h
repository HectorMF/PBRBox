#pragma once

#define NOMINMAX
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <iostream>
#include <fstream>

class OptixVolumeRenderer
{
private:
	optix::Context m_context;

	optix::Buffer tfBuffer;
	optix::TextureSampler tfTexSampler;

	optix::Buffer		volumeBuffer;
	optix::TextureSampler volumeTexSampler;
	optix::float3 volumeSize;

	optix::Buffer		sdfBuffer;
	optix::TextureSampler sdfTexSampler;

	float m_buffer_width;
	float m_buffer_height;
	int accumulation_frame;

	optix::float3       camera_up;
	optix::float3       camera_lookat;
	optix::float3       camera_eye;
	optix::Matrix4x4    camera_rotate;
	optix::Transform transform;
	bool         camera_dirty = true;
	bool use_pbo = true;

	optix::GeometryGroup volumeGroup;

public:

	OptixVolumeRenderer();
	virtual void createContext();
	virtual void destroyContext();
	virtual void addGeometry();
	virtual void draw();
	virtual void setupCamera();
	virtual void setSpacing(const float &space);

	virtual void resize(int width, int height);

	void DisplayBuffer(optix::Buffer buffer);
	void resizeBuffer(optix::Buffer buffer, unsigned width, unsigned height);

	void calculateCameraVariables(optix::float3 eye, optix::float3 lookat, optix::float3 up,
		float  fov, float  aspect_ratio,
		optix::float3& U, optix::float3& V, optix::float3& W, bool fov_is_vertical);
	void updateCamera();

	optix::Buffer createOutputBuffer(
		optix::Context context,
		RTformat format,
		unsigned width,
		unsigned height,
		bool use_pbo);

	void loadRawVolume(const std::string& filename, optix::float3 vol_size);

	void loadSDFVolume(const std::string& filename, optix::float3 vol_size);

	void setupCamera(optix::float3 eye, optix::float3 lookat, optix::float3 up, float vfov);

	void clearFrame();

	void setTransform(optix::Matrix4x4 transform);

};
