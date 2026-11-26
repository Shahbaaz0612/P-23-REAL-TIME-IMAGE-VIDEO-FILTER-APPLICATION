#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "camera_capture.h"
#include "cuda_image_processor.h"
#include "cuda_opengl_interop.h"
#include "opengl_renderer.h"

using namespace std;


// Available filter
enum class Filter
{
    INVERSION = 1,
    GRAYSCALE = 2,
    BLUR = 3
};

int main()
{
    // =========================================
    // 1. SELECT FILTER
    // =========================================

    int choice;

    cout << "\n======================================" << endl;
    cout << "   CUDA IMAGE VIDEO FILTER APPLICATION" << endl;
    cout << "======================================" << endl;

    cout << "1. Image Inversion" << endl;
    cout << "2. Grayscale" << endl;
    cout << "3. Box Blur" << endl;

    cout << "\nSelect Filter: ";

    if (!(cin >> choice) || choice < 1 || choice > 3)
    {
        cout << "Invalid filter selection!" << endl;
        return -1;
    }

    Filter selectedFilter = static_cast<Filter>(choice);


    // Determine output channels

    int channels = (selectedFilter == Filter::GRAYSCALE) ? 1 : 3;


    cout << "\nSelected Filter: ";

    if (selectedFilter == Filter::INVERSION)
        cout << "Image Inversion" << endl;

    else if (selectedFilter == Filter::GRAYSCALE)
        cout << "Grayscale" << endl;

    else
        cout << "Box Blur" << endl;


    // =========================================
    // 2. INITIALIZE CAMERA
    // =========================================

    CameraCapture camera;

    if (!camera.openCamera(0))
    {
        return -1;
    }


    // Capture first frame

    cv::Mat frame;

    if (!camera.captureFrame(frame))
    {
        cout << "Initial frame capture failed!" << endl;

        camera.releaseCamera();

        return -1;
    }


    // Verify input format

    if (frame.type() != CV_8UC3)
    {
        cout << "Camera must provide BGR CV_8UC3 frames!" << endl;

        camera.releaseCamera();

        return -1;
    }


    // Get image dimensions

    int width = frame.cols;

    int height = frame.rows;


    // =========================================
    // 3. CALCULATE MEMORY REQUIREMENTS
    // =========================================

    size_t totalPixels = static_cast<size_t>(width) * height;


    // Camera input is always BGR (3 channels)

    size_t inputBytes = totalPixels * 3;


    // Output memory depends on selected filter

    size_t pboSize = totalPixels * channels;


    cout << "\nImage Width: " << width << endl;
    cout << "Image Height: " << height << endl;

    cout << "Input Memory: " << inputBytes << " bytes" << endl;

    cout << "PBO Memory: " << pboSize << " bytes" << endl;


    // =========================================
    // 4. INITIALIZE OPENGL
    // =========================================

    OpenGLRenderer renderer;

    if (!renderer.initialize(width, height, channels))
    {
        cout << "OpenGL initialization failed!" << endl;

        camera.releaseCamera();

        return -1;
    }


    // =========================================
    // 5. CREATE AND REGISTER PBO
    // =========================================

    CudaGLInterop interop;

    // Make OpenGL context current

    renderer.makeContextCurrent();


    // Create PBO and allocate memory

    if (!interop.createPBO(pboSize))
    {
        cout << "PBO creation failed!" << endl;

        renderer.cleanup();

        camera.releaseCamera();

        return -1;
    }


    // Register PBO with CUDA

    if (!interop.registerPBOWithCUDA())
    {
        cout << "PBO registration failed!" << endl;

        interop.cleanup();

        renderer.cleanup();

        camera.releaseCamera();

        return -1;
    }


    // =========================================
    // 6. CUDA INPUT MEMORY ALLOCATION
    // =========================================

    unsigned char* d_input = nullptr;

    cudaError_t err = cudaMalloc(
        (void**)&d_input,
        inputBytes
    );

    if (err != cudaSuccess)
    {
        cout << "CUDA memory allocation failed: "
             << cudaGetErrorString(err) << endl;

        renderer.makeContextCurrent();

        interop.cleanup();

        renderer.cleanup();

        camera.releaseCamera();

        return -1;
    }


    // =========================================
    // 7. CONFIGURE CUDA GRID AND BLOCK
    // =========================================

    dim3 block(16, 16);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );


    cout << "\nCUDA initialization completed!" << endl;
    cout << "Starting frame processing..." << endl;


    // =========================================
    // 8. FRAME PROCESSING LOOP
    // =========================================

    unsigned char* d_output = nullptr;

    bool pboMapped = false;

    bool firstFrame = true;

    int exitCode = 0;


    while (!renderer.shouldClose())
    {
        // -------------------------------------
        // Step 1: Capture camera frame
        // -------------------------------------

        if (!firstFrame)
        {
            if (!camera.captureFrame(frame))
            {
                cout << "Frame capture failed!" << endl;

                exitCode = -1;
                break;
            }
        }

        firstFrame = false;


        // Ensure continuous memory

        if (!frame.isContinuous())
        {
            frame = frame.clone();
        }


        // Verify frame dimensions and format

        if (frame.cols != width ||
            frame.rows != height ||
            frame.type() != CV_8UC3)
        {
            cout << "Unexpected camera frame format!" << endl;

            exitCode = -1;
            break;
        }


        // -------------------------------------
        // Step 2: Copy frame from Host to Device
        // -------------------------------------

        err = cudaMemcpy(
            d_input,
            frame.data,
            inputBytes,
            cudaMemcpyHostToDevice
        );

        if (err != cudaSuccess)
        {
            cout << "Host to Device copy failed: "
                 << cudaGetErrorString(err) << endl;

            exitCode = -1;
            break;
        }


        // -------------------------------------
        // Step 3: Map PBO
        // -------------------------------------

        if (!interop.mapPBO(&d_output))
        {
            cout << "PBO mapping failed!" << endl;

            exitCode = -1;
            break;
        }

        pboMapped = true;


        // -------------------------------------
        // Step 4: Launch selected CUDA kernel
        // -------------------------------------

        cudaError_t kernelError = cudaSuccess;


        switch (selectedFilter)
        {
            case Filter::INVERSION:

                kernelError = launch_inversion_kernel(
                    d_input,
                    d_output,
                    width,
                    height,
                    grid,
                    block
                );

                break;


            case Filter::GRAYSCALE:

                kernelError = launch_gray_kernel(
                    d_input,
                    d_output,
                    width,
                    height,
                    grid,
                    block
                );

                break;


            case Filter::BLUR:

                kernelError = launch_blur_kernel(
                    d_input,
                    d_output,
                    width,
                    height,
                    grid,
                    block
                );

                break;
        }


        if (kernelError != cudaSuccess)
        {
            cout << "Kernel launch failed: "
                 << cudaGetErrorString(kernelError) << endl;

            exitCode = -1;
            break;
        }


        // -------------------------------------
        // Step 5: Synchronize CUDA
        // -------------------------------------

        err = cudaDeviceSynchronize();

        if (err != cudaSuccess)
        {
            cout << "CUDA kernel execution failed: "
                 << cudaGetErrorString(err) << endl;

            exitCode = -1;
            break;
        }


        // -------------------------------------
        // Step 6: Unmap PBO
        // -------------------------------------

        if (!interop.unmapPBO())
        {
            cout << "PBO unmapping failed!" << endl;

            exitCode = -1;
            break;
        }

        pboMapped = false;


        // -------------------------------------
        // Step 7: Update OpenGL texture from PBO
        // -------------------------------------

        if (!renderer.updateTextureFromPBO(interop.getPBO()))
        {
            cout << "Texture update failed!" << endl;

            exitCode = -1;
            break;
        }


        // -------------------------------------
        // Step 8: Process OpenGL window events
        // -------------------------------------

        renderer.pollEvents();
    }


    // =========================================
    // 9. CLEANUP
    // =========================================

    cout << "\nReleasing resources..." << endl;


    // Unmap PBO if it is still mapped

    if (pboMapped)
    {
        cudaDeviceSynchronize();

        if (interop.unmapPBO())
        {
            pboMapped = false;
        }
    }


    // Free CUDA input memory

    if (d_input != nullptr)
    {
        cudaFree(d_input);

        d_input = nullptr;
    }


    // Cleanup CUDA-OpenGL interoperability

    renderer.makeContextCurrent();

    interop.cleanup();


    // Release camera

    camera.releaseCamera();


    // Cleanup OpenGL resources

    renderer.cleanup();


    cout << "Application terminated!" << endl;

    return exitCode;
}
