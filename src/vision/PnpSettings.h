#pragma once

#include <opencv2/calib3d.hpp>

// Sentinel for PnpSettings::method selecting SlamWorker's own custom
// linear DLT (Direct Linear Transform) pose solver instead of one of
// OpenCV's cv::SOLVEPNP_* algorithms -- negative so it never collides with
// a real OpenCV enum value (all of which are >= 0).
inline constexpr int kPnpMethodDlt = -1;

// Parameters forwarded to cv::solvePnPRansac(), or to SlamWorker's own
// DLT-RANSAC implementation when method == kPnpMethodDlt.
struct PnpSettings
{
    int method = cv::SOLVEPNP_ITERATIVE;
    double reprojectionError = 8.0;
    double confidence = 0.99;
    int iterationsCount = 100;
};
