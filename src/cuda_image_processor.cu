#include "cuda_image_processor.h"

__global__
void inversion_kernel(const unsigned char* d_input, unsigned char* d_invert, int width, int height){

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if(x < width && y < height){
                int index = (y * width + x) * 3;

                d_invert[index] = 255 - d_input[index];
                d_invert[index + 1] = 255 - d_input[index + 1];
                d_invert[index + 2] = 255 - d_input[index + 2];
        }
}

__global__
void gray_kernel(const unsigned char* d_input, unsigned char* d_gray, int width, int height){

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if(x < width && y < height){

                int index = (y * width + x) * 3;
                int i = (y * width + x);

                d_gray[i] =  (0.11 * d_input[index]) + (0.587 * d_input[index+1]) + (0.299 * d_input[index+2]);
        }
}

__global__
void blur_kernel(const unsigned char* d_input, unsigned char* d_blur, int width, int height){

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if(x<width && y<height){

                int sumB = 0, sumG = 0, sumR = 0;

                for(int dy=-1; dy<=1; dy++){

                        for(int dx = -1; dx <= 1; dx++){

                                int yy = y + dy;
                                int xx = x + dx;

                                if(xx>=0 && xx<width && yy>=0 && yy<height){

                                        int pixel = (xx + yy * width) * 3;

                                        sumB += d_input[pixel];
                                        sumG += d_input[pixel+1];
                                        sumR += d_input[pixel+2];
                                }
                        }
                }

                int index = (x + width * y) * 3;
                d_blur[index] = sumB/49;
                d_blur[index+1] = sumG/49;
                d_blur[index+2] = sumR/49;
        }
}
	
cudaError_t launch_inversion_kernel(const unsigned char* d_input,
                             unsigned char* d_invert,
                             int width,
                             int height,
                             dim3 grid,
                             dim3 block) {

        inversion_kernel<<<grid, block>>>(d_input, d_invert, width, height);

	return cudaGetLastError();
}

cudaError_t launch_gray_kernel(const unsigned char* d_input,
                        unsigned char* d_gray,
                        int width,
                        int height,
                        dim3 grid,
                        dim3 block) {
        
	gray_kernel<<<grid, block>>>(d_input, d_gray, width, height);

	return cudaGetLastError();
}


cudaError_t launch_blur_kernel(const unsigned char* d_input,
                        unsigned char* d_blur,
                        int width,
                        int height,
                        dim3 grid,
                        dim3 block) {
        
	blur_kernel<<<grid, block>>>(d_input, d_blur, width, height);

	return cudaGetLastError();
}


