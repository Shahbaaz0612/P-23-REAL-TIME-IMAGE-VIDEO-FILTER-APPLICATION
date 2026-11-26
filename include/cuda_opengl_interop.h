#ifndef CUDA_GL_INTEROP_H
#define CUDA_GL_INTEROP_H

#include <cstddef>

#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


class CudaGLInterop
{
public:

    // Create PBO and allocate memory
    bool createPBO(size_t size);

    // Register PBO with CUDA
    bool registerPBOWithCUDA();

    // Map PBO and get GPU pointer
    bool mapPBO(unsigned char** d_output);

    // Unmap PBO
    bool unmapPBO();

    // Return OpenGL PBO ID
    GLuint getPBO() const;

    // Release resources
    void cleanup();


private:

    // OpenGL PBO ID
    GLuint pbo = 0;

    // CUDA graphics resource
    cudaGraphicsResource* cudaResource = nullptr;

    // PBO memory size
    size_t bufferSize = 0;

    // Mapping status
    bool isMapped = false;
};

#endif
