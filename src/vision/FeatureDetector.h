#pragma once

// Feature-detector abstraction: this codebase originally hardcoded SIFT
// (float descriptors, cv::NORM_L2 matching) everywhere. ORB produces binary
// descriptors and needs cv::NORM_HAMMING instead -- pulled out into its own
// file, alongside SIFT's own construction, so SlamWorker just asks for
// "the detector" and "the right matcher for whatever detector is active"
// without caring which one it is.

#include "SiftSettings.h"

#include <opencv2/features2d.hpp>

#include <vector>

namespace feature_detector {

enum class DetectorType
{
    Sift,
    Orb
};

// Mirrors cv::ORB::create()'s own parameters, same spirit as SiftSettings.h.
struct OrbSettings
{
    int nFeatures = 2000; // matches SiftSettings' own default, for comparable feature counts
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int wtaK = 2;
    int scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;
};

// Sift -> cv::SIFT::create(...) (exactly this codebase's original call);
// Orb -> cv::ORB::create(...). Both cv::SIFT/cv::ORB derive from
// cv::Feature2D, so callers only need the base-class detectAndCompute().
cv::Ptr<cv::Feature2D> createDetector(DetectorType type, const SiftSettings &sift, const OrbSettings &orb);

// cv::NORM_L2 for Sift's float descriptors, cv::NORM_HAMMING for Orb's
// binary ones -- whichever matchDescriptors() below must be called with.
int normTypeFor(DetectorType type);

// RootSIFT (Arandjelovic & Zisserman, CVPR 2012): L1-normalize, sqrt,
// L2-normalize each descriptor row -- turns SIFT's Euclidean-distance
// matching into the Hellinger-kernel-equivalent comparison the paper shows
// is a strictly better match-quality metric for the same underlying
// histogram descriptor, at zero extra extraction cost. Only meaningful for
// SIFT's float descriptors (CV_32F) -- callers must not apply this to ORB's
// binary ones. Same transform third_party/ORB_SLAM3_SIFT/src/ORBextractor.cc
// applies (both branches, CudaSIFT and CPU cv::SIFT), kept identical here so
// this codebase's own SIFT path (SlamWorker) matches that fork's descriptor
// space instead of using raw SIFT.
cv::Mat toRootSift(const cv::Mat &descriptors);

// Lowe's ratio test was calibrated (0.75) against SIFT's continuous L2
// distances. ORB's Hamming distances are coarse integers over a small range
// (typically 0-256 for a 32-byte descriptor), so the same ratio threshold
// is far stricter in practice -- ties and near-ties between the best and
// second-best candidate are common even for a correct match, and 0.75
// rejects many of them. ORB-SLAM2's own matcher uses looser
// ratios/thresholds for exactly this reason. 0.85 for Orb, unchanged 0.75
// for Sift.
float defaultRatioFor(DetectorType type);

// Same BFMatcher + knnMatch(k=2) + Lowe's-ratio-test body this codebase
// always used, just with the norm parameterized instead of hardcoded to
// cv::NORM_L2 (which only ever made sense for SIFT's float descriptors).
//
// mutualCheck (default off, behavior-preserving): additionally requires each
// surviving A->B match's B-side point to independently pick the same A-side
// point as ITS own nearest neighbor (a B->A pass, k=1, no ratio test --
// symmetry is the filter here, not a second ratio threshold). One-directional
// ratio-test matching lets through a lot of ambiguous ties, especially for
// ORB: its Hamming distances are coarse integers over a small range (0-256
// for a 32-byte descriptor), so near-ties between the best and second-best
// candidate are common even where SIFT's continuous L2 distances would
// clearly separate them. Cross-checking is a standard cheap fix for exactly
// this (see DEBUGGING.md's "Full F/E option menu" item 4) -- garbage-in
// reduction that helps whichever RANSAC estimator (E, H, or PnP) consumes
// the resulting correspondences.
bool matchDescriptors(int normType, const cv::Mat &descA, const cv::Mat &descB,
                       std::vector<cv::DMatch> &goodMatches, float ratio = 0.75f, bool mutualCheck = false);

} // namespace feature_detector
