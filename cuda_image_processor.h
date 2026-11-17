#ifndef CUDA_IMAGE_PROCESSOR_H
#define CUDA_IMAGE_PROCESSOR_H

#include <cuda_runtime.h>

void launch_inversion_kernel(const unsigned char* d_input,
                             unsigned char* d_invert,
                             int width,
                             int height,
                             dim3 grid,
                             dim3 block);

void launch_gray_kernel(const unsigned char* d_input,
                        unsigned char* d_gray,
                        int width,
                        int height,
                        dim3 grid,
                        dim3 block);

#endif
