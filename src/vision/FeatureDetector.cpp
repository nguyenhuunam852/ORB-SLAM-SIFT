#include "vision/FeatureDetector.h"

#include <algorithm>
#include <cmath>

namespace feature_detector {

cv::Ptr<cv::Feature2D> createDetector(DetectorType type, const SiftSettings &sift, const OrbSettings &orb)
{
    if (type == DetectorType::Orb) {
        return cv::ORB::create(orb.nFeatures, orb.scaleFactor, orb.nLevels, orb.edgeThreshold, orb.firstLevel,
                                orb.wtaK, static_cast<cv::ORB::ScoreType>(orb.scoreType), orb.patchSize,
                                orb.fastThreshold);
    }
    return cv::SIFT::create(sift.nFeatures, sift.nOctaveLayers, sift.contrastThreshold, sift.edgeThreshold,
                             sift.sigma);
}

int normTypeFor(DetectorType type)
{
    return type == DetectorType::Orb ? cv::NORM_HAMMING : cv::NORM_L2;
}

cv::Mat toRootSift(const cv::Mat &descriptors)
{
    cv::Mat out = descriptors.clone();
    CV_Assert(out.empty() || out.type() == CV_32F);
    const float eps = 1e-6f;
    for (int r = 0; r < out.rows; ++r) {
        cv::Mat row = out.row(r);
        double l1 = cv::norm(row, cv::NORM_L1) + eps;
        row /= l1;
        for (int c = 0; c < row.cols; ++c)
            row.at<float>(0, c) = std::sqrt(std::max(row.at<float>(0, c), eps));
        double l2 = cv::norm(row, cv::NORM_L2) + eps;
        row /= l2;
    }
    return out;
}

float defaultRatioFor(DetectorType type)
{
    return type == DetectorType::Orb ? 0.85f : 0.75f;
}

bool matchDescriptors(int normType, const cv::Mat &descA, const cv::Mat &descB, std::vector<cv::DMatch> &goodMatches,
                       float ratio, bool mutualCheck)
{
    goodMatches.clear();
    // descB needs >= 2 rows for a meaningful k=2 ratio test (comparing the
    // best match against a real 2nd-best) -- descA has no such requirement:
    // knnMatch() ranks each descA ROW independently against descB, so even
    // a single-row descA (e.g. fuseWindowLandmarks()'s per-landmark query,
    // see DEBUGGING.md item 19) works correctly. Previously required
    // descA.rows >= 2 too, for no algorithmic reason found -- this silently
    // made every single-row query return false before matching was even
    // attempted (confirmed: fuseWindowLandmarks()'s v4/Phase A measured 0
    // matches across an 866-frame smoke test before this fix).
    if (descA.empty() || descB.empty() || descB.rows < 2)
        return false;

    cv::BFMatcher matcher(normType);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(descA, descB, knn, 2);

    std::vector<cv::DMatch> abMatches;
    for (const auto &m : knn) {
        if (m.size() == 2 && m[0].distance < ratio * m[1].distance)
            abMatches.push_back(m[0]);
    }

    if (!mutualCheck || abMatches.empty()) {
        goodMatches = std::move(abMatches);
        return !goodMatches.empty();
    }

    // B->A pass, nearest neighbor only (k=1): no ratio test here, symmetry
    // with the A->B pass above is itself the filter. Keep only A->B matches
    // whose B-side point's own nearest A-side neighbor is the same point the
    // A->B pass picked.
    std::vector<cv::DMatch> baNearest;
    matcher.match(descB, descA, baNearest);

    goodMatches.reserve(abMatches.size());
    for (const auto &m : abMatches) {
        if (m.trainIdx >= 0 && m.trainIdx < static_cast<int>(baNearest.size())
            && baNearest[m.trainIdx].trainIdx == m.queryIdx)
            goodMatches.push_back(m);
    }
    return !goodMatches.empty();
}

} // namespace feature_detector
