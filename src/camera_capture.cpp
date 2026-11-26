#include "camera_capture.h"
#include <iostream>
using namespace std;

// Open camera
bool CameraCapture::openCamera(int cameraID)
{
    cap.open(cameraID, cv::CAP_V4L2);

    if (!cap.isOpened())
    {
        cout << "Failed to open camera!" << endl;
        return false;
    }

    cout << "====== Camera Opened Successfully ======" << endl;
    cout << "Frames/Sec: " << cap.get(cv::CAP_PROP_FPS) << endl;
    cout << "Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << endl;
    cout << "Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "Backend Name: " << cap.getBackendName() << endl;

    return true;
}


// Capture one frame
bool CameraCapture::captureFrame(cv::Mat& frame)
{
    if (!cap.read(frame))
    {
        return false;
    }
    if (frame.empty())
    {
        return false;
    }
    return true;
}


// Release camera
void CameraCapture::releaseCamera()
{
    if (cap.isOpened())
    {
        cap.release();
    }
}


// Destructor
CameraCapture::~CameraCapture()
{
    releaseCamera();
}



