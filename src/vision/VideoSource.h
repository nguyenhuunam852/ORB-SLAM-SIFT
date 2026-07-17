#pragma once

#include <opencv2/videoio.hpp>
#include <string>

// Thin wrapper around cv::VideoCapture for a file or a live camera.
class VideoSource
{
public:
    bool openFile(const std::string &path);
    bool openCamera(int index);
    bool isOpened() const;
    bool readFrame(cv::Mat &frame);
    void release();

    // Seeks a file source back to its first frame. No-op (returns false)
    // for a live camera, which has no meaningful "start" to seek to.
    bool rewindToStart();

    int frameWidth() const;
    int frameHeight() const;

private:
    cv::VideoCapture m_capture;
};
