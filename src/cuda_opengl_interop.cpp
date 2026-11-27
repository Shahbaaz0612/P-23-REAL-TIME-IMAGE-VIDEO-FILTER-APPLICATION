#include "cuda_opengl_interop.h"
#include <iostream>
using namespace std;

//=========================================creating PBO and allocating memory===========================================

bool CudaGLInterop::createPBO(size_t size)
{
    if (size == 0)
    {
        cout << "Invalid PBO size!" << endl;
        return false;
    }

    // Generate one PBO
    glGenBuffers(1, &pbo);

    if (pbo == 0)
    {
        cout << "PBO creation failed!" << endl;
        return false;
    }

    // Binding the PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocating memory for PBO.
    glBufferData(
        GL_PIXEL_UNPACK_BUFFER,
        size,
        nullptr,
        GL_STREAM_DRAW
    );

    // Unbind PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Store allocated size
    bufferSize = size;
    cout << "PBO created successfully!" << endl;
    cout << "PBO size: " << bufferSize << " bytes" << endl;
    return true;
}

// =========================================registering PBO with cuda=====================================================

bool CudaGLInterop::registerPBOWithCUDA()
{
    if (pbo == 0)
    {
        cout << "PBO is not created!" << endl;
        return false;
    }

    // Register PBO with CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &cudaResource,
        pbo,
        cudaGraphicsRegisterFlagsWriteDiscard
    );

    if (err != cudaSuccess)
    {
        cout << "PBO registration failed: "
             << cudaGetErrorString(err) << endl;
        return false;
    }
    cout << "PBO registered with CUDA successfully!" << endl;
    return true;
}

//=================================Mapping PBO with cuda and getting mapped GPU pointer====================================

bool CudaGLInterop::mapPBO(unsigned char** d_output)
{
    // Check PBO state
    if (cudaResource == nullptr || d_output == nullptr || isMapped)
    {
        cout << "Invalid PBO state!" << endl;
        return false;
    }

    *d_output = nullptr;

    // Map PBO for CUDA access
    cudaError_t err = cudaGraphicsMapResources(
        1,
        &cudaResource,
        0
    );

    if (err != cudaSuccess)
    {
        cout << "PBO mapping failed: "
             << cudaGetErrorString(err) << endl;
        return false;
    }

    isMapped = true;

    // Get mapped device pointer
    void* devicePointer = nullptr;
    size_t mappedSize = 0;

    err = cudaGraphicsResourceGetMappedPointer(
        &devicePointer,
        &mappedSize,
        cudaResource
    );

    if (err != cudaSuccess)
    {
        cout << "Failed to get PBO device pointer: "
             << cudaGetErrorString(err) << endl;
        cudaGraphicsUnmapResources(1, &cudaResource, 0);
        isMapped = false;
        return false;
    }

    // Verify available PBO memory
    if (mappedSize < bufferSize)
    {
        cout << "PBO memory size mismatch!" << endl;
        cudaGraphicsUnmapResources(1, &cudaResource, 0);
        isMapped = false;
        return false;
    }

    // Return GPU pointer to main.cpp
    *d_output = static_cast<unsigned char*>(devicePointer);
    return true;
}

//============================================Unmap the PBO======================================================

bool CudaGLInterop::unmapPBO()
{
    if (!isMapped)
    {
        cout << "PBO is not mapped!" << endl;
        return false;
    }

    cudaError_t err = cudaGraphicsUnmapResources(
	1,
        &cudaResource,
        0
    );

    if (err != cudaSuccess)
    {
        cout << "PBO unmapping failed: "
             << cudaGetErrorString(err) << endl;
        return false;
    }

    isMapped = false;
    return true;
}

//===========================================================================

GLuint CudaGLInterop::getPBO() const
{
    return pbo;
}

//==================================================cleaning the resources===============================================

void CudaGLInterop::cleanup()
{

    // Unregister PBO from CUDA
    if (cudaResource != nullptr && !isMapped)
    {
        cudaError_t err = cudaGraphicsUnregisterResource(cudaResource);
        if (err == cudaSuccess)
        {
            cudaResource = nullptr;
        }
        else
        {
            cout << "CUDA resource unregister failed: " << cudaGetErrorString(err) << endl;
        }
    }

    // Delete OpenGL PBO
    if (pbo != 0 && cudaResource == nullptr)
    {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    bufferSize = 0;
    cout << "CUDA-OpenGL interop cleanup completed!" << endl;
}


