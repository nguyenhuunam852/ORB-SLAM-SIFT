#pragma once

#include <opencv2/core.hpp>

// Pinhole camera intrinsic matrix K = [fx 0 cx; 0 fy cy; 0 0 1].
// Defaults are the KITTI odometry P0 (grayscale left camera) calibration,
// e.g. dataset/sequences/00/calib.txt, so the app is ready to go on KITTI
// sequences out of the box; override from the left panel for other sources.
struct CameraIntrinsics
{
    double fx = 718.856;
    double fy = 718.856;
    double cx = 607.1928;
    double cy = 185.2157;

    cv::Mat toMat() const
    {
        return (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                           0.0, fy, cy,
                                           0.0, 0.0, 1.0);
    }
};
