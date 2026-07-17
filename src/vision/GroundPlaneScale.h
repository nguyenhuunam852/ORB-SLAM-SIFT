#pragma once

// Standalone module: VISO2-M-style (Geiger et al.) ground-plane-based
// monocular scale correction. Given a fixed, pre-calibrated camera height
// and ground-plane normal, and a set of already-triangulated 3D points
// expressed in the *current* camera's coordinate frame, estimates the
// implied camera height via a robust consensus vote over all points, then
// returns the scale factor (known height / estimated height) needed to
// correct that frame's translation.
//
// This is the real VISO2-M algorithm -- confirmed against the authoritative
// source itself (KIT-MRT/viso2, a maintained mirror of Geiger's original
// libviso2, src/viso_mono.cpp's estimateMotion()/smallerThanMedian()), not
// just a secondary description: restricts to the nearer half of points (by
// L1 distance) before scoring, and uses an adaptive Gaussian-kernel
// consensus vote (sigma = median L1 distance / 50, not a fixed constant --
// an earlier version of this file used a fixed mu=50 borrowed from a later
// paper's citation of the formula, Song/Chandraker/Guest PAMI 2015 eq. 4,
// which is NOT what the original source actually does; corrected after
// fetching and reading viso_mono.cpp directly). Simpler than the ON-HOLD
// homography-decomposition idea earlier in DEBUGGING.md: it needs no
// per-frame homography/H-branch machinery at all, just the already-
// triangulated map points every two-view estimator here already produces,
// plus a fixed (not per-frame-estimated) ground-plane geometry.
//
// Not wired into SlamWorker; a standalone, independently-testable module,
// matching this project's EightPointLegacy.h/ImuRotation.h pattern.

#include <opencv2/core.hpp>

#include <vector>

namespace ground_plane_scale {

struct GroundPlaneConfig
{
    // Unit normal of the ground plane, in the *current* camera's
    // coordinate frame, pointing from the ground up towards the camera
    // (i.e. opposite the camera's "down" axis for a level, forward-facing
    // rig). KITTI's rectified camera convention is X=right, Y=down,
    // Z=forward, so a perfectly level camera has groundNormalCam =
    // (0, -1, 0); a real rig has some small downward pitch, which tilts
    // this slightly towards +Z.
    cv::Mat groundNormalCam; // 3x1, CV_64F, unit length

    // Real-world camera height above the ground plane, meters. ~1.65m is
    // commonly cited for KITTI's rig in the literature -- per this
    // project's own earlier note (DEBUGGING.md's ON-HOLD section), verify
    // against the specific sequence's calibration/sensor documentation
    // rather than trusting that number blindly if this is used for real.
    double knownCameraHeight = 1.65;
};

// Estimates the implied camera height from a set of 3D points already
// expressed in the current camera's coordinate frame (caller's
// responsibility: transform world-frame map points via X_cam = R*X_world +
// t before calling this). Returns -1.0 (a sentinel meaning "not available")
// if there are fewer than 3 points or the estimate is otherwise degenerate,
// rather than silently guessing.
double estimateCameraHeight(const std::vector<cv::Point3f> &pointsInCameraFrame, const GroundPlaneConfig &config);

// Convenience wrapper: returns the scale correction factor
// (config.knownCameraHeight / estimatedHeight) to multiply the current
// frame's translation by, or -1.0 (sentinel, not available) if
// estimateCameraHeight() itself returns unavailable/degenerate.
double estimateScaleCorrection(const std::vector<cv::Point3f> &pointsInCameraFrame, const GroundPlaneConfig &config);

} // namespace ground_plane_scale
