#include "OptixVolumeRenderer.h"
#include "GL\glew.h"

OptixVolumeRenderer::OptixVolumeRenderer() 
{
	createContext();
	setupCamera();
	updateCamera();
	addGeometry();

	//loadRawVolume(std::string("C:\\Users\\Hector\\Documents\\Gearbox\\bin\\3L-drilled_768x768x768_type_uc_1channels.raw"), { 768, 768, 768 });
	//loadSDFVolume(std::string("C:\\Users\\Hector\\Documents\\Gearbox\\bin\\Sphere_SDF.raw"), { 512, 512, 512 });

	//m_context->validate();

	accumulation_frame = 0;
	m_buffer_width = 512;
	m_buffer_height = 512;
}

void OptixVolumeRenderer::setSpacing(const float &space)
{

}

void OptixVolumeRenderer::calculateCameraVariables(optix::float3 eye, optix::float3 lookat, optix::float3 up,
	float  fov, float  aspect_ratio,
	optix::float3& U, optix::float3& V, optix::float3& W, bool fov_is_vertical)
{
	float ulen, vlen, wlen;
	W = lookat - eye; // Do not normalize W -- it implies focal length

	wlen = length(W);
	U = normalize(cross(W, up));
	V = normalize(cross(U, W));

	if (fov_is_vertical) {
		vlen = wlen * tanf(0.5f * fov * M_PIf / 180.0f);
		V *= vlen;
		ulen = vlen * aspect_ratio;
		U *= ulen;
	}
	else {
		ulen = wlen * tanf(0.5f * fov * M_PIf / 180.0f);
		U *= ulen;
		vlen = ulen / aspect_ratio;
		V *= vlen;
	}
}


void OptixVolumeRenderer::setupCamera()
{
	camera_eye = optix::make_float3(0, 141.421356f, 141.421356f);
	camera_lookat = optix::make_float3(0.0f, 0.0f, 0.0f);
	camera_up = optix::make_float3(0.0f, 1.0f, 0.0f);

	camera_rotate = optix::Matrix4x4::identity();
	camera_dirty = true;
}

void OptixVolumeRenderer::updateCamera()
{
	const float vfov = 17.285f;
	const float aspect_ratio = static_cast<float>(m_buffer_width) /
		static_cast<float>(m_buffer_height);

	optix::float3 camera_u, camera_v, camera_w;
	calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);

	const optix::Matrix4x4 frame = optix::Matrix4x4::fromBasis(
		optix::normalize(camera_u),
		optix::normalize(camera_v),
		optix::normalize(-camera_w),
		camera_lookat);
	const optix::Matrix4x4 frame_inv = frame.inverse();
	// Apply camera rotation twice to match old SDK behavior
	const optix::Matrix4x4 trans = frame*camera_rotate*camera_rotate*frame_inv;

	camera_eye = optix::make_float3(trans*make_float4(camera_eye, 1.0f));
	camera_lookat = optix::make_float3(trans*make_float4(camera_lookat, 1.0f));
	camera_up = optix::make_float3(trans*make_float4(camera_up, 0.0f));

	calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	camera_rotate = optix::Matrix4x4::identity();

	m_context["eye"]->setFloat(camera_eye);
	m_context["U"]->setFloat(camera_u);
	m_context["V"]->setFloat(camera_v);
	m_context["W"]->setFloat(camera_w);

	camera_dirty = false;
}

void OptixVolumeRenderer::createContext()
{
	m_context = optix::Context::create();
	m_context->setRayTypeCount(2);
	m_context->setEntryPointCount(1);
	m_context->setStackSize(4800);

	m_context["num_lights"]->setInt(0);
	m_context["max_depth"]->setInt(100);
	m_context["radiance_ray_type"]->setUint(0);
	m_context["shadow_ray_type"]->setUint(1);
	m_context["frame_number"]->setUint(0u);
	m_context["scene_epsilon"]->setFloat(1.e-4f);
	m_context["ambient_light_color"]->setFloat(0.6f, 0.6f, 0.6f);
	m_context["diffuse_color"]->setFloat(.6f, 0.0f, .6f);
	m_context["scene_epsilon"]->setFloat(1.e-3f);
	m_context["pathtrace_ray_type"]->setUint(0u);
	m_context["pathtrace_shadow_ray_type"]->setUint(1u);
	m_context["rr_begin_depth"]->setUint(1);
	m_context["sqrt_num_samples"]->setUint(1);

	optix::Buffer buffer = createOutputBuffer(m_context, RT_FORMAT_FLOAT4, m_buffer_width, m_buffer_height, use_pbo);
	m_context["output_buffer"]->set(buffer);

	optix::Buffer accum_buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
		RT_FORMAT_FLOAT4, m_buffer_width, m_buffer_height);
	m_context["accum_buffer"]->set(accum_buffer);

	optix::Buffer depth_buffer = createOutputBuffer(m_context, RT_FORMAT_FLOAT4, m_buffer_width, m_buffer_height, use_pbo);
	m_context["depth_buffer"]->set(depth_buffer);

	std::string ptx_path("C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\x64\\Release\\Optix\\accum_camera.ptx");
	optix::Program ray_gen_program = m_context->createProgramFromPTXFile(ptx_path, "pathtrace_camera");
	m_context->setRayGenerationProgram(0, ray_gen_program);

	// Exception program
	optix::Program exception_program = m_context->createProgramFromPTXFile(ptx_path, "exception");
	m_context->setExceptionProgram(0, exception_program);
	m_context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);
}

void OptixVolumeRenderer::destroyContext()
{
	if (m_context)
	{
		m_context->destroy();
		m_context = 0;
	}
}

void OptixVolumeRenderer::DisplayBuffer(optix::Buffer buffer)
{
	// Query buffer information
	RTsize buffer_width_rts, buffer_height_rts;
	buffer->getSize(buffer_width_rts, buffer_height_rts);
	int width = static_cast<int>(buffer_width_rts);
	int height = static_cast<int>(buffer_height_rts);
	RTformat buffer_format = buffer->getFormat();

	/*	GLboolean use_SRGB = GL_FALSE;
	if (buffer_format == RT_FORMAT_FLOAT4 || buffer_format == RT_FORMAT_FLOAT3)
	{
	glGetBooleanv(GL_FRAMEBUFFER_SRGB_CAPABLE_EXT, &use_SRGB);
	if (use_SRGB)
	glEnable(GL_FRAMEBUFFER_SRGB_EXT);
	}*/
	//glEnable(GL_FRAMEBUFFER_SRGB_EXT);
	// Check if we have a GL interop display buffer
	const unsigned pboId = buffer->getGLBOId();
	if (pboId)
	{
		static unsigned int gl_tex_id = 0;
		if (!gl_tex_id)
		{
			glGenTextures(1, &gl_tex_id);
			glBindTexture(GL_TEXTURE_2D, gl_tex_id);

			// Change these to GL_LINEAR for super- or sub-sampling
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

			// GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}



		glBindTexture(GL_TEXTURE_2D, gl_tex_id);

		// send PBO to texture
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);

		RTsize elmt_size = buffer->getElementSize();
		if (elmt_size % 8 == 0)      glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
		else if (elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		else if (elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
		else                         glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		if (buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
		else if (buffer_format == RT_FORMAT_FLOAT4)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0);
		else if (buffer_format == RT_FORMAT_FLOAT3)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, GL_RGB, GL_FLOAT, 0);
		else if (buffer_format == RT_FORMAT_FLOAT)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, width, height, 0, GL_LUMINANCE, GL_FLOAT, 0);
		//else
		//	throw Exception("Unknown buffer format");

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// 1:1 texel to pixel mapping with glOrtho(0, 1, 0, 1, -1, 1) setup:
		// The quad coordinates go from lower left corner of the lower left pixel 
		// to the upper right corner of the upper right pixel. 
		// Same for the texel coordinates.
		/*glUseProgram(13);
		GLuint sampler = 0;
		glGenSamplers(1, &sampler);
		//glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		const int texUnit = 4;
		GLuint samplerUniform = glGetUniformLocation(13, "Depthbuffer");
		glUniform1i(samplerUniform, texUnit);
		glActiveTexture(GL_TEXTURE0 + texUnit);
		glBindTexture(GL_TEXTURE_2D, gl_tex_id);
		glBindSampler(texUnit, sampler);*/

		//glDisable (GL_DEPTH_TEST);
		glUseProgram(0);

		glEnable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glDisable(GL_ALPHA_TEST);
		glDisable(GL_BLEND);

		//glClearColor(1.f, 0.f, 1.f, 0.f);
		//glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//glClear(GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-1, 1, -1, 1, -1, 1);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glActiveTexture(GL_TEXTURE0);
		glEnable(GL_TEXTURE_2D);


		//draw picture
		glColor3f(1, 0, 1);
		glBegin(GL_QUADS);
		glTexCoord2f(1.f, 1.f); glVertex3f(1.f, 1.f, 0.f);
		glTexCoord2f(1.f, 0.f); glVertex3f(1.f, -1.f, 0.f);
		glTexCoord2f(0.f, 0.f); glVertex3f(-1.f, -1.f, 0.f);
		glTexCoord2f(0.f, 1.f); glVertex3f(-1.f, 1.f, 0.f);
		glEnd();

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();

		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

		glEnable(GL_BLEND);
		//glEnable (GL_DEPTH_TEST);
		glEnable(GL_LIGHTING);
		glEnable(GL_ALPHA_TEST);

		//blendProgram.disable();
		//glUseProgram(0);

	}
	else
	{
		GLvoid* imageData = buffer->map(0, RT_BUFFER_MAP_READ);
		GLenum gl_data_type = GL_FALSE;
		GLenum gl_format = GL_FALSE;

		switch (buffer_format)
		{
		case RT_FORMAT_UNSIGNED_BYTE4:
			gl_data_type = GL_UNSIGNED_BYTE;
			gl_format = GL_BGRA;
			break;

		case RT_FORMAT_FLOAT:
			gl_data_type = GL_FLOAT;
			gl_format = GL_LUMINANCE;
			break;

		case RT_FORMAT_FLOAT3:
			gl_data_type = GL_FLOAT;
			gl_format = GL_RGB;
			break;

		case RT_FORMAT_FLOAT4:
			gl_data_type = GL_FLOAT;
			gl_format = GL_RGBA;
			break;

		default:
			fprintf(stderr, "Unrecognized buffer data type or format.\n");
			exit(2);
			break;
		}

		RTsize elmt_size = buffer->getElementSize();
		int align = 1;
		if ((elmt_size % 8) == 0) align = 8;
		else if ((elmt_size % 4) == 0) align = 4;
		else if ((elmt_size % 2) == 0) align = 2;
		glPixelStorei(GL_UNPACK_ALIGNMENT, align);

		glDrawPixels(
			static_cast<GLsizei>(width),
			static_cast<GLsizei>(height),
			gl_format,
			gl_data_type,
			imageData
			);
		buffer->unmap();
	}

	//if (use_SRGB)
	//	glDisable(GL_FRAMEBUFFER_SRGB_EXT);
}


void OptixVolumeRenderer::setupCamera(optix::float3 eye, optix::float3 lookat, optix::float3 up, float vfov)
{
	camera_eye = optix::make_float3(eye.x, eye.y, eye.z);
	camera_lookat = optix::make_float3(lookat.x, lookat.y, lookat.z);
	camera_up = optix::make_float3(up.x, up.y, up.z);

	const float aspect_ratio = static_cast<float>(m_buffer_width) /
		static_cast<float>(m_buffer_height);

	optix::float3 camera_u, camera_v, camera_w;
	calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);

	const optix::Matrix4x4 frame = optix::Matrix4x4::fromBasis(
		optix::normalize(camera_u),
		optix::normalize(camera_v),
		optix::normalize(-camera_w),
		camera_lookat);
	const optix::Matrix4x4 frame_inv = frame.inverse();
	// Apply camera rotation twice to match old SDK behavior
	const optix::Matrix4x4 trans = frame*camera_rotate*camera_rotate*frame_inv;

	camera_eye = optix::make_float3(trans*make_float4(camera_eye, 1.0f));
	camera_lookat = optix::make_float3(trans*make_float4(camera_lookat, 1.0f));
	camera_up = optix::make_float3(trans*make_float4(camera_up, 0.0f));

	calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	camera_rotate = optix::Matrix4x4::identity();

	//camera_eye = optix::make_float3(m[0][1]);
	//camera_lookat = optix::make_float3(m);
	//camera_up = optix::make_float3(trans*make_float4(camera_up, 0.0f));

	m_context["eye"]->setFloat(camera_eye);
	m_context["U"]->setFloat(camera_u);
	m_context["V"]->setFloat(camera_v);
	m_context["W"]->setFloat(camera_w);

}

void OptixVolumeRenderer::clearFrame()
{
	accumulation_frame = 0;
}

void OptixVolumeRenderer::draw()
{
	m_context["frame_number"]->setUint(accumulation_frame++);
	try {
		m_context->launch(0, m_buffer_width, m_buffer_height);
	}
	catch (optix::Exception e)
	{
		printf(("Optix Error: " + e.getErrorString() + " \n").c_str());

	}


	DisplayBuffer(m_context["output_buffer"]->getBuffer());

}

void OptixVolumeRenderer::resizeBuffer(optix::Buffer buffer, unsigned width, unsigned height)
{
	buffer->setSize(width, height);

	// Check if we have a GL interop display buffer
	const unsigned pboId = buffer->getGLBOId();
	if (pboId)
	{
		buffer->unregisterGLBuffer();
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer->getElementSize() * width * height, 0, GL_STREAM_DRAW);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		buffer->registerGLBuffer();
	}
}


void OptixVolumeRenderer::resize(int width, int height)
{
	m_buffer_width = width;
	m_buffer_height = height;

	resizeBuffer(m_context["output_buffer"]->getBuffer(), width, height);
	resizeBuffer(m_context["accum_buffer"]->getBuffer(), width, height);
	resizeBuffer(m_context["depth_buffer"]->getBuffer(), width, height);

	accumulation_frame = 0;
}

void OptixVolumeRenderer::addGeometry()
{
	const std::string volume_intersection_ptx = "C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\x64\\Release\\Optix\\volume.ptx";
	optix::Geometry volume_geometry = m_context->createGeometry();
	volume_geometry->setPrimitiveCount(1u);
	volume_geometry->setBoundingBoxProgram(m_context->createProgramFromPTXFile(volume_intersection_ptx, "bounds"));
	volume_geometry->setIntersectionProgram(m_context->createProgramFromPTXFile(volume_intersection_ptx, "intersect"));
	volume_geometry["boxmin"]->setFloat(-0.0f, -0.0f, -0.0f);
	volume_geometry["boxmax"]->setFloat(1.0f, 1.0f, 1.0f);



	// Metal material
	const std::string volume_material_ptx = "C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\x64\\Release\\Optix\\phong.ptx";
	optix::Program volume_ss = m_context->createProgramFromPTXFile(volume_material_ptx, "singleScattering");
	optix::Program volume_shadow = m_context->createProgramFromPTXFile(volume_material_ptx, "any_hit_shadow");

	optix::Material volume_material = m_context->createMaterial();
	volume_material->setClosestHitProgram(0, volume_ss);
	volume_material->setAnyHitProgram(1, volume_shadow);


	// Miss program
	{
		const std::string bg_ptx = "C:\\Users\\Javi\\Documents\\GitHub\\PBRBox\\x64\\Release\\Optix\\constantbg.ptx";
		m_context->setMissProgram(0, m_context->createProgramFromPTXFile(bg_ptx, "miss"));
		m_context->setMissProgram(1, m_context->createProgramFromPTXFile(bg_ptx, "missShadow"));
		const optix::float3 default_color = optix::make_float3(1.0f, 1.0f, 1.0f);
		const std::string texpath = "C:\\Users\\Hector\\Documents\\Gearbox\\bin\\Resources\\Environments\\neuroArm-panorama.hdr";
		//m_context["envmap"]->setTextureSampler(loadHDRTexture(m_context, texpath, default_color));
		m_context["bg_color"]->setFloat(optix::make_float3(0.34f, 0.55f, 0.85f));
	}



	// Create GIs for each piece of geometry
	std::vector<optix::GeometryInstance> gis;
	//gis.push_back( context->createGeometryInstance( glass_sphere, &glass_matl, &glass_matl+1 ) );
	//gis.push_back(context->createGeometryInstance(parallelogram, &test_matl, &test_matl + 1));


	optix::GeometryInstance volume = m_context->createGeometryInstance(volume_geometry, &volume_material, &volume_material + 1);
	// Place all in group

	volumeGroup = m_context->createGeometryGroup();
	volumeGroup->setAcceleration(m_context->createAcceleration("NoAccel"));
	volumeGroup->addChild(volume);

	transform = m_context->createTransform();
	//transformMatrix.postTrans(-36, -36, -36);
	//transform->setMatrix(false, &optix::Matrix4x4::identity()[0], NULL);
	transform->setChild(volumeGroup);

	m_context["top_object"]->set(transform);
	m_context["top_shadower"]->set(transform);
}

void  OptixVolumeRenderer::setTransform(optix::Matrix4x4 t)
{
	transform->setMatrix(false, &t[0], NULL);
	volumeGroup->getAcceleration()->markDirty();
}


optix::Buffer OptixVolumeRenderer::createOutputBuffer(
	optix::Context context,
	RTformat format,
	unsigned width,
	unsigned height,
	bool use_pbo)
{

	optix::Buffer buffer;
	if (use_pbo)
	{
		// First allocate the memory for the GL buffer, then attach it to OptiX.

		// Assume ubyte4 or float4 for now
		unsigned int elmt_size = format == RT_FORMAT_UNSIGNED_BYTE4 ? 4 : 16;

		GLuint vbo = 0;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, elmt_size * width * height, 0, GL_STREAM_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		buffer = context->createBufferFromGLBO(RT_BUFFER_OUTPUT, vbo);
		buffer->setFormat(format);
		buffer->setSize(width, height);
	}
	else
	{
		buffer = context->createBuffer(RT_BUFFER_OUTPUT, format, width, height);
	}

	return buffer;
}


void OptixVolumeRenderer::loadRawVolume(const std::string& filename, optix::float3 vol_size)
{


	std::string data;
	// Open the file for reading, binary mode 
	std::ifstream ifFile(filename, std::ios_base::binary);
	ifFile.seekg(0, std::ios::end);

	//if (ifFile.tellg() < vol_size.x * vol_size.y * vol_size.z)
	//	throw init_exception("[Error] Not enough values in Dataset for defined volume");

	data.resize(ifFile.tellg());
	ifFile.seekg(0);
	ifFile.read(const_cast<char*>(data.c_str()), data.size());

	volumeBuffer = m_context->createBuffer(RT_BUFFER_INPUT);

	volumeBuffer->setSize(vol_size.x, vol_size.y, vol_size.z);
	volumeBuffer->setFormat(RT_FORMAT_UNSIGNED_BYTE);

	unsigned char* buf_data = (unsigned char*)volumeBuffer->map();
	memcpy(buf_data, reinterpret_cast<const unsigned char *>(data.c_str()), sizeof(unsigned char)*data.size());
	volumeBuffer->unmap();

	volumeTexSampler = m_context->createTextureSampler();
	volumeTexSampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
	volumeTexSampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
	volumeTexSampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);

	volumeTexSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	volumeTexSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	volumeTexSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);

	volumeTexSampler->setBuffer(volumeBuffer);

	m_context["volume_texture"]->setTextureSampler(volumeTexSampler);

	volumeSize = vol_size;
}

void OptixVolumeRenderer::loadSDFVolume(const std::string& filename, optix::float3 vol_size)
{

	if (filename.length() == 0)
	{
		//throw init_exception("[Error] Volume filename empty.");
	}

	std::string data;
	// Open the file for reading, binary mode 
	std::ifstream ifFile(filename, std::ios_base::binary);
	ifFile.seekg(0, std::ios::end);

	data.resize(ifFile.tellg());
	ifFile.seekg(0);
	ifFile.read(const_cast<char*>(data.c_str()), data.size());

	//if (data.size() < vol_size.x*vol_size.y*vol_size.z)
	//	throw init_exception("[Error] Not enough values in Dataset for defined volume");


	sdfBuffer = m_context->createBuffer(RT_BUFFER_INPUT);

	sdfBuffer->setSize(vol_size.x, vol_size.y, vol_size.z);
	sdfBuffer->setFormat(RT_FORMAT_FLOAT);

	float* buf_data = (float*)sdfBuffer->map();
	memcpy(buf_data, reinterpret_cast<const float *>(data.c_str()), data.size());
	sdfBuffer->unmap();

	sdfTexSampler = m_context->createTextureSampler();
	sdfTexSampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
	sdfTexSampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
	sdfTexSampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);

	sdfTexSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	sdfTexSampler->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
	sdfTexSampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);

	sdfTexSampler->setBuffer(sdfBuffer);

	m_context["sdf_texture"]->setTextureSampler(sdfTexSampler);
}




/*

optix::TextureSampler loadHDRTexture(optix::Context context,
const std::string& filename,
const optix::float3& default_color)
{
// Create tex sampler and populate with default values
optix::TextureSampler sampler = context->createTextureSampler();
sampler->setWrapMode(0, RT_WRAP_REPEAT);
sampler->setWrapMode(1, RT_WRAP_REPEAT);
sampler->setWrapMode(2, RT_WRAP_REPEAT);
sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
sampler->setMaxAnisotropy(1.0f);
sampler->setMipLevelCount(1u);
sampler->setArraySize(1u);

// Read in HDR, set texture buffer to empty buffer if fails
HDRLoader hdr(filename);
if (hdr.failed()) {

// Create buffer with single texel set to default_color
optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1u, 1u);
float* buffer_data = static_cast<float*>(buffer->map());
buffer_data[0] = default_color.x;
buffer_data[1] = default_color.y;
buffer_data[2] = default_color.z;
buffer_data[3] = 1.0f;
buffer->unmap();

sampler->setBuffer(0u, 0u, buffer);
// Although it would be possible to use nearest filtering here, we chose linear
// to be consistent with the textures that have been loaded from a file. This
// allows OptiX to perform some optimizations.
sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

return sampler;
}

const unsigned int nx = hdr.width();
const unsigned int ny = hdr.height();

// Create buffer and populate with HDR data
optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny);
float* buffer_data = static_cast<float*>(buffer->map());

for (unsigned int i = 0; i < nx; ++i) {
for (unsigned int j = 0; j < ny; ++j) {

unsigned int hdr_index = ((ny - j - 1)*nx + i) * 4;
unsigned int buf_index = ((j)*nx + i) * 4;

buffer_data[buf_index + 0] = hdr.raster()[hdr_index + 0];
buffer_data[buf_index + 1] = hdr.raster()[hdr_index + 1];
buffer_data[buf_index + 2] = hdr.raster()[hdr_index + 2];
buffer_data[buf_index + 3] = hdr.raster()[hdr_index + 3];
}
}

buffer->unmap();

sampler->setBuffer(0u, 0u, buffer);
sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

return sampler;
}*/