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

    // Re-reads a single arbitrary frame from a file source by index (0-based,
    // matching cv::CAP_PROP_POS_FRAMES convention) without disturbing this
    // object's own sequential read position -- opens its own temporary
    // capture on `path`. For an image-sequence pattern (KITTI's %06d.png),
    // this is just a filename format + imread, so the seek is cheap and
    // exact. Not meant for hot per-frame use -- see its caller's own doc
    // comment (SlamWorker's ASIFT loop-closure fallback).
    static bool readFrameAt(const std::string &path, int frameIndex, cv::Mat &frame);

private:
    cv::VideoCapture m_capture;
};
