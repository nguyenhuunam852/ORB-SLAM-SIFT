#ifndef VLADVOCABULARY_H
#define VLADVOCABULARY_H

#include <opencv2/core.hpp>

#include <cassert>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <algorithm>
#include <limits>

#include "Thirdparty/DBoW2/DUtils/Random.h"

// Replicates the same transitive include/using-directive surface the
// original DBoW2::TemplatedVocabulary.h provided (it pulls in <fstream>,
// <vector>, <string>, DUtils/Random.h, etc. and declares
// `using namespace std;` at global scope) -- every file in this codebase
// that includes ORBVocabulary.h (Frame.h, KeyFrame.h, KeyFrameDatabase.h,
// Map.h, System.h, Tracking.h, LocalMapping.h, MLPnPsolver.cc, ...) relies
// on std:: names (vector, map, set, string, ofstream, ...) being
// unqualified and fully-defined (not just forward-declared via <iosfwd>),
// and on DUtils::Random being available, via that now-removed transitive
// include. Not great practice, but matching it here avoids touching every
// one of those files just to add their own includes/using-directives.
using namespace std;

namespace ORB_SLAM3
{

// Replaces the original ORBVocabulary (a DBoW2::TemplatedVocabulary<FORB::
// TDescriptor,FORB> -- a hierarchical bag-of-words tree over ORB's binary
// descriptors) for the SIFT-based fork: DBoW2 has no descriptor class for
// SIFT's CV_32F/128-dim descriptors vendored in this project (only FORB),
// and no pretrained SIFT vocabulary exists publicly (confirmed via web
// search -- see DEBUGGING.md's ORB->SIFT swap session). VLAD (Vector of
// Locally Aggregated Descriptors, Jegou et al. 2010) is used instead: a
// small k-means codebook (trained offline, see analyze/orbslam3_vlad_train.cpp)
// plus a residual-aggregation encoding, compared via cosine similarity --
// much cheaper to train than a DBoW2 vocabulary tree and needs no new
// DBoW2 descriptor class.
//
// Kept as the literal type ORBVocabulary refers to (see ORBVocabulary.h's
// now-repointed typedef) rather than introducing a new type threaded
// through every Frame/KeyFrame/Tracking/System call site that takes an
// `ORBVocabulary*` -- zero signature churn outside this class's own
// implementation and KeyFrameDatabase.cc's internals (which already needed
// rewriting regardless, see its own doc comments).
class VladVocabulary
{
public:
    // Method name matches the original ORBVocabulary::loadFromTextFile()
    // exactly (not renamed) so System.cc's existing call site needs no
    // edit -- and it remains literally accurate: the new codebook format
    // (see analyze/orbslam3_vlad_train.cpp) is a plain-text cv::FileStorage
    // YAML file, just a different schema than DBoW2's own text vocabulary
    // format.
    bool loadFromTextFile(const std::string &path);

    // Encodes a keyframe/frame's full descriptor set (CV_32F, 128 cols,
    // one row per keypoint) into a single L2-normalized VLAD vector (1 x
    // k*128, CV_32F): assign each descriptor to its nearest centroid,
    // accumulate per-centroid residuals, intra-normalize each centroid's
    // residual block, then L2-normalize the flattened whole. Empty input
    // (no descriptors) returns an all-zero vector, not an error -- score()
    // against an all-zero vector is always 0, which every caller already
    // treats as "no match" via its own minScore/threshold check.
    cv::Mat computeVlad(const cv::Mat &descriptors) const;

    // Cosine similarity between two VLAD vectors (both already
    // L2-normalized by computeVlad(), so this is a plain dot product) --
    // a direct drop-in analog of DBoW2::TemplatedVocabulary::score(),
    // returning a similarity in [-1,1] (in practice [0,1] for VLAD
    // vectors, which have no negative-correlation structure worth
    // distinguishing) that every KeyFrameDatabase caller already treats as
    // "higher is a better match" via its own minScore/ratio thresholds.
    float score(const cv::Mat &v1, const cv::Mat &v2) const;

    // Codebook size (k) -- number of centroids/visual words.
    unsigned int size() const;

private:
    int nearestCentroid(const cv::Mat &descriptorRow) const;

    cv::Mat mCentroids; // k x 128, CV_32F
};

} // namespace ORB_SLAM3

#endif // VLADVOCABULARY_H
