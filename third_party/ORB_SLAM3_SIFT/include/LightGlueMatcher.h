#ifndef LIGHTGLUEMATCHER_H
#define LIGHTGLUEMATCHER_H

// SIFT+LightGlue matcher wrapper (ONNX Runtime). See DEBUGGING.md part 58:
// ASIFT was closed as a NEGATIVE RESULT after confirming via the actual
// paper that it's structurally mismatched to small-baseline per-frame VO.
// LightGlue is a learned matcher explicitly designed for practical feature
// matching and ships official pretrained weights for plain SIFT
// descriptors -- this keeps the existing SIFT extractor untouched and only
// replaces the MATCHING step, scoped to genuinely pairwise frame-vs-frame
// call sites (TrackWithMotionModel's SearchByProjection(Frame&,Frame&,...)
// and SearchForInitialization), not the frame-vs-local-map-point-cloud
// path (SearchByProjection against mvpLocalMapPoints), which isn't a
// natural fit for a pairwise image matcher.
//
// No official ONNX export of SIFT+LightGlue exists anywhere -- the model
// this loads (weights/lightglue_sift.onnx) was exported by
// analyze/export_lightglue_sift.py, a from-scratch static-graph
// reimplementation verified numerically equivalent to the original
// PyTorch cvg/LightGlue module. See that script's header comment and
// DEBUGGING.md part 58 for the full derivation (RootSIFT preprocessing,
// keypoint normalization convention, add_scale_ori 4D positional input).

#include <opencv2/core.hpp>
#include <vector>
#include <memory>

namespace Ort {
    struct Env;
    struct Session;
}

namespace ORB_SLAM3
{

class LightGlueMatcher
{
public:
    // modelPath: path to the exported ONNX model (keypoints,descriptors ->
    // matches,mscores). Loads once; safe to reuse across many Match() calls.
    explicit LightGlueMatcher(const std::string& modelPath, int intraOpThreads = 4);
    ~LightGlueMatcher();

    // Matches kps0/desc0 (frame/keyframe 0) against kps1/desc1 (frame 1).
    // imgSize: the image dimensions both keypoint sets were detected in
    // (used for LightGlue's own keypoint-normalization convention -- see
    // export script). desc0/desc1 must be CV_32F SIFT descriptors (raw,
    // NOT RootSIFT -- this function performs that conversion internally on
    // a local copy, so callers keep using the original descriptors
    // everywhere else in the pipeline, e.g. BoW/VLAD conversion, MapPoint
    // descriptor storage).
    //
    // Returns matches01: matches01[i] = j means kps0[i] <-> kps1[j]
    // (mutual, above the model's internal 0.1 filter_threshold), -1 if kp
    // i in frame 0 has no accepted match in frame 1. Returns match count.
    int Match(const std::vector<cv::KeyPoint>& kps0, const cv::Mat& desc0,
              const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
              cv::Size imgSize, std::vector<int>& matches01) const;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

// Global instance, constructed lazily on first use (guarded, thread-unsafe
// by design -- ORB-SLAM3's tracking thread is the only caller of the two
// call sites this is scoped to). Path is fixed relative to the process's
// working directory, matching this project's other fixed-relative-path
// assets (vocabulary_sift/, settings_sift/).
LightGlueMatcher* GetLightGlueMatcher();

} // namespace ORB_SLAM3

#endif // LIGHTGLUEMATCHER_H
