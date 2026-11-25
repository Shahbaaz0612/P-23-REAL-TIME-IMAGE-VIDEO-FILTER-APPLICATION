#include<opencv2/opencv.hpp>
#include<iostream>
#include<cuda_runtime.h>
#include "cuda_image_processor.h"
using namespace std;

int main(){
        cv::VideoCapture cap(0,cv::CAP_V4L2);
        if(!cap.isOpened()){
                cout<<"failed to open camera!!"<<endl;
                return -1;
        }
        cout<< "======Camera Opened Successfully======" << endl;

        cout<< "Frames/Sec: " << cap.get(cv::CAP_PROP_FPS) << endl;
        cout<< "Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << endl;
        cout<< "Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
        cout<< "Backend Name: " << cap.getBackendName() << endl;

/*      cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_CONTRAST, 0);
        cap.set(cv::CAP_PROP_FPS, 60);
*/
        cv::Mat frame;
        cv::Mat invert;
        cv::Mat gray;
        cv::Mat blur;
  while(1){
                if( !cap.read(frame) ){
                        cout << "Frame capture failed!!" << endl;
                        break;
                }

                if(frame.empty()) continue;

                invert.create(frame.rows, frame.cols, CV_8UC3);
                gray.create(frame.rows, frame.cols, CV_8UC1);
                blur.create(frame.rows, frame.cols, CV_8UC3);
                cv::imshow("WebCam Frames", frame);

                size_t total_bytes = frame.cols * frame.rows * frame.channels();

                unsigned char* h_input = frame.data;
                unsigned char* d_input = nullptr;
                unsigned char* d_invert = nullptr;
                unsigned char* d_gray = nullptr;
                unsigned char* d_blur = nullptr;

                cudaError_t err = cudaMalloc((void**)&d_input, total_bytes);
                if(err != cudaSuccess){
                        cout<< " cudaMalloc0 failed: " << cudaGetErrorString(err) << endl;
                        cap.release();
                        return -1;
                }

                err = cudaMemcpy(d_input, h_input, total_bytes, cudaMemcpyHostToDevice);
                if(err != cudaSuccess){
                        cout<< " cudaMemcpy0 host failed: " << cudaGetErrorString(err) << endl;
                        cap.release();
                        return -1;
                }

                err = cudaMalloc((void**)&d_invert, total_bytes);
                if(err != cudaSuccess){
                        cout<< " cudaMalloc1 failed: " << cudaGetErrorString(err) << endl;
                        cap.release();
                        return -1;
                }

                err = cudaMalloc((void**)&d_gray, frame.rows * frame.cols);
                if(err != cudaSuccess){
                        cout<< " cudaMalloc2 failed: " << cudaGetErrorString(err) << endl;
                        cap.release();
                        return -1;
                }

                err = cudaMalloc((void**)&d_blur, total_bytes);
                if(err != cudaSuccess){
                        cout<< " cudaMalloc3 failed: " << cudaGetErrorString(err) << endl;
                        cap.release();
                        return -1;
                }

                dim3 block(16,16);
                dim3 grid( ((frame.cols + block.x - 1) / block.x), ( (frame.rows + block.y - 1) / block.y) );
                launch_inversion_kernel(d_input, d_invert, frame.cols, frame.rows, grid, block);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                        cout << "Kernel launch failed: "<< cudaGetErrorString(err) << endl;
                        cap.release();
                        return -1;
                }
                cudaDeviceSynchronize();
                err = cudaMemcpy(invert.data, d_invert, total_bytes, cudaMemcpyDeviceToHost);
                if(err != cudaSuccess){
                        cout<< " cudaMemcpy 1 failed: " << cudaGetErrorString(err) << endl;
                        cap.release();
                        return -1;
                }
                cv::imshow("Inversion Frames", invert);


                launch_gray_kernel(d_input, d_gray, frame.cols, frame.rows, grid, block);
                cudaDeviceSynchronize();

                err = cudaMemcpy(gray.data, d_gray, frame.rows * frame.cols, cudaMemcpyDeviceToHost);
                if(err != cudaSuccess){
                        cout<< " cudaMemcpy2 failed: " << cudaGetErrorString(err) << endl;
                        cap.release();
                        return -1;
                }
                cv::imshow("Gray Frames", gray);

		launch_blur_kernel(d_input, d_blur, frame.cols, frame.rows, grid, block);
              	//blur_kernel<<<grid, block>>>(d_input, d_blur, frame.cols, frame.rows);
                cudaDeviceSynchronize();
                err = cudaMemcpy(blur.data, d_blur, total_bytes, cudaMemcpyDeviceToHost);
                if(err != cudaSuccess){
                        cout<< " cudaMalloc failed: " << cudaGetErrorString(err) << endl;
                        cap.release();
                        return -1;
                }
                cv::imshow("Blur Frames", blur);
        

                cout<<"======Frame Captured Successfully======" << endl;
                cout<< "width: " << frame.cols << endl;
                cout<< "height: " << frame.rows << endl;
                cout<< "pixel size: " << frame.elemSize() <<endl;
                cout<< "channels/pixels: " << frame.channels() << endl;
                cout<< "step size: " << frame.step << endl;
                cout<< "frame type: " << frame.type() << endl;

                if(cv::waitKey(1) == 27){
                        break;
                }

                h_input = nullptr;
                cudaFree(d_gray);
                cudaFree(d_invert);
                cudaFree(d_blur);
                cudaFree(d_input);
        }
        cap.release();
        cv::destroyAllWindows();
}
