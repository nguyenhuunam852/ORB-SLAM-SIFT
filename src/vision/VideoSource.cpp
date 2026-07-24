#include "VideoSource.h"

#include <opencv2/imgproc.hpp>

bool VideoSource::openFile(const std::string &path)
{
    release();
    return m_capture.open(path);
}

bool VideoSource::openCamera(int index)
{
    release();
    return m_capture.open(index);
}

bool VideoSource::isOpened() const
{
    return m_capture.isOpened();
}

bool VideoSource::readFrame(cv::Mat &frame)
{
    if (!m_capture.isOpened())
        return false;
    return m_capture.read(frame) && !frame.empty();
}

bool VideoSource::rewindToStart()
{
    if (!m_capture.isOpened())
        return false;
    return m_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
}

void VideoSource::release()
{
    if (m_capture.isOpened())
        m_capture.release();
}

int VideoSource::frameWidth() const
{
    return static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_WIDTH));
}

int VideoSource::frameHeight() const
{
    return static_cast<int>(m_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
}

bool VideoSource::readFrameAt(const std::string &path, int frameIndex, cv::Mat &frame)
{
    cv::VideoCapture cap(path);
    if (!cap.isOpened())
        return false;
    if (!cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex))
        return false;
    return cap.read(frame) && !frame.empty();
}
