/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
*  Interactive camera with depth of field based on CUDA path tracer code
*  by Peter Kutz and Yining Karl Li, https://github.com/peterkutz/GPUPathTracer
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef __CUDA_PATHTRACER_H_
#define __CUDA_PATHTRACER_H_

#include "glm/vec3.hpp"
#include "camera.h"
#include <ctime>

#define BVH_STACK_SIZE 32
#define screenWidth 960	// screenwidth
#define screenHeight 640 // screenheight
//#define width 1920	// screenwidth
//#define height 1072 // screenheight


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		//if (abort) exit(code);
	}
}




#define DBG_PUTS(level, msg) \
    do { if (level <= 1) { puts(msg); fflush(stdout); }} while (0)


// The gateway to CUDA, called from C++ (src/main.cpp)

void cudarender(cudaArray* tex, glm::vec3* dptr, glm::vec3* accumulatebuffer, unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam);


struct Clock {
	unsigned firstValue;
	Clock() { reset(); }
	void reset() { firstValue = clock(); }
	unsigned readMS() { return (clock() - firstValue) / (CLOCKS_PER_SEC / 1000); }
};


#endif
