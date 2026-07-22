#pragma once

// VLAD (Vector of Locally Aggregated Descriptors, Jegou et al. 2010) place
// recognition, ported from third_party/ORB_SLAM3_SIFT/include/VladVocabulary.h
// (same class, same math, stripped of that fork's DBoW2/DUtils transitive-
// include compatibility shim -- this codebase has no such dependency to
// preserve). Used by SlamWorker::tryLoopClosure() as a SIFT-compatible
// alternative to setDbowLoopClosureEnabled() (which only works with ORB's
// binary descriptors): a small k-means codebook (trained offline, see
// third_party/ORB_SLAM3_SIFT/analyze/orbslam3_vlad_train.cpp) plus a
// residual-aggregation encoding, compared via cosine similarity.

#include <opencv2/core.hpp>

#include <string>

namespace vlad {

class VladVocabulary
{
public:
    // Loads a codebook trained by orbslam3_vlad_train.cpp -- a plain-text
    // cv::FileStorage YAML file with a "centroids" (k x descriptorDim,
    // CV_32F) matrix. Returns false (leaving any previously loaded codebook
    // in place) if the file can't be read or has no valid centroids matrix.
    bool loadFromTextFile(const std::string &path);

    // Encodes a keyframe/frame's full descriptor set (CV_32F, one row per
    // keypoint, column count must match the codebook's own descriptor
    // dimension) into a single L2-normalized VLAD vector (1 x k*dim,
    // CV_32F): assign each descriptor to its nearest centroid, accumulate
    // per-centroid residuals, intra-normalize each centroid's residual
    // block, then L2-normalize the flattened whole. Empty input returns an
    // all-zero vector, not an error -- score() against an all-zero vector
    // is always 0, which callers already treat as "no match".
    cv::Mat computeVlad(const cv::Mat &descriptors) const;

    // Cosine similarity between two VLAD vectors (both already
    // L2-normalized by computeVlad(), so this is a plain dot product) --
    // returns a similarity in [-1,1] (in practice [0,1] for VLAD vectors)
    // where higher is a better match.
    float score(const cv::Mat &v1, const cv::Mat &v2) const;

    // Codebook size (k) -- number of centroids/visual words.
    unsigned int size() const;

private:
    int nearestCentroid(const cv::Mat &descriptorRow) const;

    cv::Mat mCentroids; // k x descriptorDim, CV_32F
};

} // namespace vlad
