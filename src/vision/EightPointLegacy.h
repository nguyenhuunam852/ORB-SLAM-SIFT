#pragma once

// Standalone copy of the original 8-point-then-convert fundamental-matrix
// estimator (normalized 8-point + Gold Standard Sampson refinement), kept
// here for reference/benchmarking after SlamWorker::estimateTwoViewPose()
// was switched over to the direct calibrated 5-point solver
// (SlamWorker::estimateEssentialRansac()) -- see DEBUGGING.md, "Full F/E
// option menu" item 1. Not wired into SlamWorker or any live pipeline;
// deliberately dependency-free from SlamWorker so it can be benchmarked in
// isolation.

#include <opencv2/core.hpp>

#include <vector>

namespace eight_point_legacy {

// Normalized 8-point algorithm (Hartley & Zisserman Alg. 11.1) wrapped in
// RANSAC, with a final Gold Standard (Sampson-distance-minimizing
// Levenberg-Marquardt) nonlinear refinement pass over the best inlier set.
// Samples 8 correspondences per iteration, fits F linearly, scores all
// correspondences by Sampson distance, refits + refines over the largest
// inlier set. Returns an empty Mat if fewer than 8 correspondences are
// available. Byte-for-byte identical logic to what SlamWorker.cpp used to
// ship (RANSAC seed 42, 1000 iterations, 1.0 squared-pixel Sampson
// threshold) -- see SlamWorker.cpp's own kRansacSeed/kFRansacSampsonThreshold
// doc comments for why those specific values were chosen.
cv::Mat estimateFundamentalRansac(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2,
                                   cv::Mat &mask);

} // namespace eight_point_legacy
