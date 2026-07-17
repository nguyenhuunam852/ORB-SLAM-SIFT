// Stage 4 calibration tool for the SIFT-based ORB_SLAM3 fork
// (third_party/ORB_SLAM3_SIFT): measures real squared-L2 distance
// distributions for true vs. false SIFT descriptor matches on actual KITTI
// data, to replace ORBmatcher's placeholder TH_HIGH/TH_LOW (see
// DEBUGGING.md's ORB->SIFT swap session) with values backed by measurement
// instead of a guess -- mirrors this project's own established practice
// (see src/vision/FeatureDetector.cpp's defaultRatioFor()).
//
// Methodology: for frame pairs at small baselines (i vs i+1, i+3, i+5),
// get candidate correspondences via the real ORBextractor + ratio-test
// matching (same shape as feature_detector::matchDescriptors()), then
// geometrically verify with cv::findFundamentalMat RANSAC. RANSAC inliers
// are labeled "true matches"; RANSAC outliers among the same ratio-tested
// candidate pool are labeled "false matches". Squared-L2 descriptor
// distance is logged into both distributions and percentiles are reported.
//
// Usage: orbslam3_sift_calibrate <settings.yaml> <seq-dir> [nframes]

#include <ORBextractor.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

void LoadImages(const std::string &sequencePath, std::vector<std::string> &imageFiles)
{
    std::ifstream fTimes(sequencePath + "/times.txt");
    std::string line;
    size_t count = 0;
    while (std::getline(fTimes, line)) {
        if (!line.empty())
            ++count;
    }
    const std::string prefix = sequencePath + "/image_0/";
    imageFiles.resize(count);
    char buf[16];
    for (size_t i = 0; i < count; ++i) {
        std::snprintf(buf, sizeof(buf), "%06zu.png", i);
        imageFiles[i] = prefix + buf;
    }
}

float squaredL2(const cv::Mat &a, const cv::Mat &b)
{
    const float *pa = a.ptr<float>();
    const float *pb = b.ptr<float>();
    float dist = 0.f;
    for (int i = 0; i < a.cols; ++i) {
        const float d = pa[i] - pb[i];
        dist += d * d;
    }
    return dist;
}

float percentile(std::vector<float> v, double p)
{
    if (v.empty())
        return -1.f;
    std::sort(v.begin(), v.end());
    size_t idx = std::min(v.size() - 1, static_cast<size_t>(p * v.size()));
    return v[idx];
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <settings.yaml> <seq-dir> [nframes]\n", argv[0]);
        return 1;
    }
    const std::string settingsPath = argv[1];
    const std::string seqDir = argv[2];
    const int nframes = argc > 3 ? std::atoi(argv[3]) : 500;

    int nFeatures = 2000, nLevels = 3;
    {
        cv::FileStorage fs(settingsPath, cv::FileStorage::READ);
        if (fs.isOpened()) {
            if (!fs["ORBextractor.nFeatures"].empty())
                nFeatures = static_cast<int>(fs["ORBextractor.nFeatures"]);
            if (!fs["ORBextractor.nLevels"].empty())
                nLevels = static_cast<int>(fs["ORBextractor.nLevels"]);
        }
    }
    std::fprintf(stderr, "[config] nFeatures=%d nLevels=%d\n", nFeatures, nLevels);
    ORB_SLAM3::ORBextractor extractor(nFeatures, 1.2f, nLevels, 20, 7);

    std::vector<std::string> imageFiles;
    LoadImages(seqDir, imageFiles);
    if (imageFiles.empty()) {
        std::fprintf(stderr, "no images found in %s\n", seqDir.c_str());
        return 1;
    }
    const int limit = std::min(static_cast<int>(imageFiles.size()) - 6, nframes);

    std::vector<float> trueDist, falseDist;
    const int baselines[] = {1, 3, 5};

    for (int i = 0; i < limit; i += 5) {
        cv::Mat img1 = cv::imread(imageFiles[i], cv::IMREAD_GRAYSCALE);
        if (img1.empty())
            continue;
        std::vector<cv::KeyPoint> kp1;
        cv::Mat desc1;
        std::vector<int> lap = {0, 0};
        extractor(img1, cv::Mat(), kp1, desc1, lap);
        if (desc1.rows < 8)
            continue;

        for (int b : baselines) {
            const size_t j = static_cast<size_t>(i + b);
            if (j >= imageFiles.size())
                continue;
            cv::Mat img2 = cv::imread(imageFiles[j], cv::IMREAD_GRAYSCALE);
            if (img2.empty())
                continue;
            std::vector<cv::KeyPoint> kp2;
            cv::Mat desc2;
            extractor(img2, cv::Mat(), kp2, desc2, lap);
            if (desc2.rows < 8)
                continue;

            cv::BFMatcher matcher(cv::NORM_L2);
            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(desc1, desc2, knn, 2);

            std::vector<cv::DMatch> candidates;
            for (auto &m : knn) {
                if (m.size() == 2 && m[0].distance < 0.8f * m[1].distance)
                    candidates.push_back(m[0]);
            }
            if (candidates.size() < 8)
                continue;

            std::vector<cv::Point2f> pts1, pts2;
            pts1.reserve(candidates.size());
            pts2.reserve(candidates.size());
            for (auto &m : candidates) {
                pts1.push_back(kp1[m.queryIdx].pt);
                pts2.push_back(kp2[m.trainIdx].pt);
            }

            std::vector<uchar> inlierMask;
            cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 1.0, 0.99, inlierMask);
            if (inlierMask.empty())
                continue;

            for (size_t k = 0; k < candidates.size(); ++k) {
                const float sq = squaredL2(desc1.row(candidates[k].queryIdx), desc2.row(candidates[k].trainIdx));
                if (inlierMask[k])
                    trueDist.push_back(sq);
                else
                    falseDist.push_back(sq);
            }
        }

        if (i % 200 == 0) {
            std::fprintf(stderr, "[progress] frame %d/%d, true=%zu false=%zu\n", i, limit, trueDist.size(),
                         falseDist.size());
        }
    }

    std::fprintf(stderr, "\n=== RESULTS ===\n");
    std::fprintf(stderr, "true-match pairs (RANSAC inliers):  %zu\n", trueDist.size());
    std::fprintf(stderr, "false-match pairs (RANSAC outliers): %zu\n", falseDist.size());
    std::fprintf(stderr, "true-match squared-L2:  p50=%.1f p90=%.1f p95=%.1f p99=%.1f max=%.1f\n",
                 percentile(trueDist, 0.50), percentile(trueDist, 0.90), percentile(trueDist, 0.95),
                 percentile(trueDist, 0.99), trueDist.empty() ? -1.f : *std::max_element(trueDist.begin(), trueDist.end()));
    std::fprintf(stderr, "false-match squared-L2: min=%.1f p1=%.1f p5=%.1f p10=%.1f p50=%.1f\n",
                 falseDist.empty() ? -1.f : *std::min_element(falseDist.begin(), falseDist.end()),
                 percentile(falseDist, 0.01), percentile(falseDist, 0.05), percentile(falseDist, 0.10),
                 percentile(falseDist, 0.50));

    const float thLow = percentile(trueDist, 0.95);
    const float thHigh = percentile(trueDist, 0.99) * 1.5f;
    std::fprintf(stderr,
                  "\n[suggestion] TH_LOW ~= %.1f (95th pct of true matches), TH_HIGH ~= %.1f "
                  "(1.5x 99th pct of true matches) -- inspect the false-match percentiles above and adjust by hand "
                  "if there's little separation.\n",
                  thLow, thHigh);

    return 0;
}
