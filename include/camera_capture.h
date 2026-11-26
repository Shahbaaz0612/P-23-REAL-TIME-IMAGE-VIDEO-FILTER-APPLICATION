#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <opencv2/opencv.hpp>

class CameraCapture
{
public:

    // Open camera
    bool openCamera(int cameraID);

    // Capture one frame
    bool captureFrame(cv::Mat& frame);

    // Release camera
    void releaseCamera();

    // Destructor
    ~CameraCapture();

private:

    cv::VideoCapture cap;
};

#endif
