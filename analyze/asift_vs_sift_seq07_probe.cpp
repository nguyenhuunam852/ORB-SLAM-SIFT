// Standalone probe: does ASIFT (cv::AffineFeature wrapping cv::SIFT) recover
// meaningfully more correspondences than plain SIFT on the exact wide-baseline
// keyframe pairs that seq07's loop-closure re-match failed on (see
// [loop][vlad-topk-diag] evidence gathered this session)? Uses the same
// matching rule as SlamWorker::matchDescriptors(): BFMatcher(NORM_L2),
// ratio=0.75, no mutual check.
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <utility>

namespace {

cv::Mat toRootSift(const cv::Mat &desc)
{
    if (desc.empty())
        return desc;
    cv::Mat out = desc.clone();
    const double eps = 1e-7;
    for (int r = 0; r < out.rows; ++r) {
        cv::Mat row = out.row(r);
        double l1 = cv::norm(row, cv::NORM_L1) + eps;
        row /= l1;
        for (int c = 0; c < row.cols; ++c)
            row.at<float>(0, c) = std::sqrt(std::max(row.at<float>(0, c), (float)eps));
        double l2 = cv::norm(row, cv::NORM_L2) + eps;
        row /= l2;
    }
    return out;
}

int countGoodMatches(const cv::Mat &descA, const cv::Mat &descB, float ratio = 0.75f)
{
    if (descA.empty() || descB.empty() || descB.rows < 2)
        return 0;
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(descA, descB, knn, 2);
    int good = 0;
    for (const auto &m : knn) {
        if (m.size() == 2 && m[0].distance < ratio * m[1].distance)
            ++good;
    }
    return good;
}

struct Pair {
    const char *label;
    const char *fileA;
    const char *fileB;
};

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <image_dir>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];

    const std::vector<Pair> pairs = {
        {"kf31(f272)_vs_kf0(f10)  [SIFT fail, matches=3]", "000271.png", "000009.png"},
        {"kf100(f879)_vs_kf0(f10) [SIFT fail, matches=3]", "000878.png", "000009.png"},
        {"kf119(f1067)_vs_kf1(f18) [SIFT ok, matches=146]", "001066.png", "000017.png"},
        {"kf120(f1075)_vs_kf0(f10) [SIFT ok, matches=100]", "001074.png", "000009.png"},
    };

    // Production default (SiftSettings::nFeatures=2000) as the fair baseline.
    const int kProdNFeatures = 2000;
    const int kMinMatchesNeeded = 20; // kLoopMinPnpInliers in SlamWorker.cpp
    auto sift = cv::SIFT::create(kProdNFeatures);

    // Escalation ladder per user's explicit ask: only pay for ASIFT when
    // plain SIFT's match count is already too low, and escalate maxTilt
    // (2 -> 4 -> 8 -> 16) one step at a time, stopping at the first tilt
    // that clears the threshold -- capped at maxTilt=16, give up if even
    // that doesn't clear it.
    const std::vector<int> kTiltLadder = {2, 4, 8, 16};

    for (const auto &p : pairs) {
        cv::Mat imgA = cv::imread(dir + "/" + p.fileA, cv::IMREAD_GRAYSCALE);
        cv::Mat imgB = cv::imread(dir + "/" + p.fileB, cv::IMREAD_GRAYSCALE);
        if (imgA.empty() || imgB.empty()) {
            std::fprintf(stderr, "failed to load %s or %s\n", p.fileA, p.fileB);
            continue;
        }

        std::vector<cv::KeyPoint> kpA, kpB;
        cv::Mat descA, descB;
        const auto t0 = cv::getTickCount();
        sift->detectAndCompute(imgA, cv::noArray(), kpA, descA);
        sift->detectAndCompute(imgB, cv::noArray(), kpB, descB);
        const double siftMs = (cv::getTickCount() - t0) * 1000.0 / cv::getTickFrequency();
        const cv::Mat descA_rs = toRootSift(descA);
        const cv::Mat descB_rs = toRootSift(descB);
        const int siftMatches = countGoodMatches(descA_rs, descB_rs);

        std::printf("=== %s ===\n", p.label);
        std::printf("  SIFT : kpA=%zu kpB=%zu matches=%d  (%.0f ms)%s\n",
                     kpA.size(), kpB.size(), siftMatches, siftMs,
                     siftMatches >= kMinMatchesNeeded ? "  -- already clears threshold, no escalation needed" : "");

        if (siftMatches >= kMinMatchesNeeded) {
            std::printf("\n");
            continue;
        }

        double cumulativeMs = siftMs;
        bool cleared = false;
        for (int tilt : kTiltLadder) {
            auto asift = cv::AffineFeature::create(cv::SIFT::create(kProdNFeatures), tilt);
            std::vector<cv::KeyPoint> akpA, akpB;
            cv::Mat adescA, adescB;
            const auto t1 = cv::getTickCount();
            asift->detectAndCompute(imgA, cv::noArray(), akpA, adescA);
            asift->detectAndCompute(imgB, cv::noArray(), akpB, adescB);
            const double asiftMs = (cv::getTickCount() - t1) * 1000.0 / cv::getTickFrequency();
            cumulativeMs += asiftMs;
            const cv::Mat adescA_rs = toRootSift(adescA);
            const cv::Mat adescB_rs = toRootSift(adescB);
            const int asiftMatches = countGoodMatches(adescA_rs, adescB_rs);

            std::printf("  ASIFT tilt=%d: kpA=%zu kpB=%zu matches=%d  (+%.0f ms, cumulative %.0f ms, %.1fx SIFT)%s\n",
                         tilt, akpA.size(), akpB.size(), asiftMatches, asiftMs, cumulativeMs,
                         cumulativeMs / std::max(1.0, siftMs),
                         asiftMatches >= kMinMatchesNeeded ? "  -- CLEARS THRESHOLD, stop escalating" : "");
            if (asiftMatches >= kMinMatchesNeeded) {
                cleared = true;
                break;
            }
        }
        if (!cleared)
            std::printf("  -- never cleared threshold even at tilt=%d, ladder exhausted\n", kTiltLadder.back());
        std::printf("\n");
    }

    return 0;
}
