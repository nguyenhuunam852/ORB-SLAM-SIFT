// Standalone SIFT+LightGlue feasibility test (see DEBUGGING.md part 58 /
// ASIFT-investigation follow-up: ASIFT was closed as a NEGATIVE RESULT
// after confirming, via the actual paper, that it is structurally
// mismatched to small-baseline per-frame VO. LightGlue is a learned
// matcher explicitly designed for practical, general-purpose feature
// matching (not just wide-baseline) and ships official pretrained weights
// for plain SIFT descriptors -- this keeps the project's existing SIFT
// extractor untouched and only replaces the MATCHING step.
//
// No official ONNX export of SIFT+LightGlue exists (checked every release
// of fabio-sim/LightGlue-ONNX: only superpoint/disk/raco_aliked are
// supported) -- the model this test loads was exported by
// analyze/export_lightglue_sift.py, a from-scratch static-graph
// reimplementation adapted from that project's SuperPoint/DISK exporter to
// support SIFT's add_scale_ori=True config (4D keypoint encoding:
// normalized x,y + raw scale,orientation, vs the other extractors' plain
// x,y). Verified bit-for-bit-equivalent (within fp32 tolerance) against
// the original PyTorch cvg/LightGlue module on random inputs before use.
//
// CRITICAL preprocessing requirements matched from cvg/LightGlue/sift.py:
//  - Descriptors must be converted to RootSIFT (L1-normalize, sqrt,
//    L2-normalize) -- the SIFT+LightGlue weights were trained on RootSIFT,
//    not raw SIFT descriptors. Skipping this silently produces low-quality
//    matches, not a crash.
//  - Keypoint (x,y) must be normalized via LightGlue's own convention:
//    shift=(W/2,H/2), scale=max(W,H)/2, normalized=(xy-shift)/scale.
//  - Orientation must be in radians (OpenCV's KeyPoint.angle is degrees).
//  - Both images' keypoint sets are stacked as a single (2,N,...) batch,
//    so they must be zero-padded to equal length N before inference;
//    matches referencing a padded index must be discarded afterward.
//
// Usage: <this-binary> <onnx-model-path> <image0-path> <image1-path>
// Prints raw SIFT counts, LightGlue match count/score stats, and (for a
// same-scale sanity comparison) the existing project's ratio-test-based
// brute-force match count on the identical descriptor pair.

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// cvg/LightGlue/sift.py: sift_to_rootsift -- L1 normalize, sqrt, L2 normalize.
void toRootSift(cv::Mat& desc)
{
    CV_Assert(desc.type() == CV_32F);
    const float eps = 1e-6f;
    for (int r = 0; r < desc.rows; ++r) {
        cv::Mat row = desc.row(r);
        double l1 = cv::norm(row, cv::NORM_L1) + eps;
        row /= l1;
        for (int c = 0; c < row.cols; ++c)
            row.at<float>(0, c) = std::sqrt(std::max(row.at<float>(0, c), eps));
        double l2 = cv::norm(row, cv::NORM_L2) + eps;
        row /= l2;
    }
}

struct FrameFeatures {
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc; // CV_32F, RootSIFT
};

FrameFeatures extract(const cv::Mat& image, const cv::Ptr<cv::SIFT>& sift)
{
    FrameFeatures f;
    sift->detectAndCompute(image, cv::noArray(), f.kps, f.desc);
    toRootSift(f.desc);
    return f;
}

// Pack (2,N,4) keypoints (normalized x,y + raw scale,ori-in-radians) and
// (2,N,128) RootSIFT descriptors for a single image pair, zero-padded to
// equal length N = max(N0,N1).
struct PackedInput {
    std::vector<float> keypoints;   // 2*N*4
    std::vector<float> descriptors; // 2*N*128
    int N0 = 0, N1 = 0, N = 0;
};

PackedInput packPair(const FrameFeatures& f0, const FrameFeatures& f1, cv::Size imgSize)
{
    PackedInput p;
    p.N0 = static_cast<int>(f0.kps.size());
    p.N1 = static_cast<int>(f1.kps.size());
    p.N = std::max(p.N0, p.N1);
    p.keypoints.assign(static_cast<size_t>(2) * p.N * 4, 0.0f);
    p.descriptors.assign(static_cast<size_t>(2) * p.N * 128, 0.0f);

    const float shiftX = imgSize.width / 2.0f;
    const float shiftY = imgSize.height / 2.0f;
    const float scale = std::max(imgSize.width, imgSize.height) / 2.0f;

    auto fill = [&](int half, const FrameFeatures& f) {
        for (int i = 0; i < static_cast<int>(f.kps.size()); ++i) {
            const cv::KeyPoint& kp = f.kps[i];
            size_t kBase = (static_cast<size_t>(half) * p.N + i) * 4;
            p.keypoints[kBase + 0] = (kp.pt.x - shiftX) / scale;
            p.keypoints[kBase + 1] = (kp.pt.y - shiftY) / scale;
            p.keypoints[kBase + 2] = kp.size;
            p.keypoints[kBase + 3] = kp.angle * static_cast<float>(CV_PI) / 180.0f;
            size_t dBase = (static_cast<size_t>(half) * p.N + i) * 128;
            const float* row = f.desc.ptr<float>(i);
            std::copy(row, row + 128, p.descriptors.begin() + dBase);
        }
    };
    fill(0, f0);
    fill(1, f1);
    return p;
}

// Baseline: brute-force + Lowe ratio test, same spirit as this project's
// existing ORBmatcher logic, for a same-scale sanity comparison.
int ratioTestMatchCount(const cv::Mat& desc0, const cv::Mat& desc1, float ratio = 0.8f)
{
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc0, desc1, knn, 2);
    int count = 0;
    for (auto& m : knn)
        if (m.size() == 2 && m[0].distance < ratio * m[1].distance)
            ++count;
    return count;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <onnx-model-path> <image0-path> <image1-path> [nfeatures=5000]\n", argv[0]);
        return 1;
    }
    const std::string modelPath = argv[1];
    const std::string image0Path = argv[2];
    const std::string image1Path = argv[3];
    const int nFeatures = argc > 4 ? std::atoi(argv[4]) : 5000;

    cv::Mat img0 = cv::imread(image0Path, cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread(image1Path, cv::IMREAD_GRAYSCALE);
    if (img0.empty() || img1.empty()) {
        std::fprintf(stderr, "failed to load images\n");
        return 1;
    }

    // Matches this project's ORBextractor.cc SIFT config exactly (nfeatures
    // overridable via CLI arg for speed-vs-quality scaling tests -- LightGlue's
    // transformer attention is O(N^2) in keypoint count, unlike the project's
    // traditional matcher, so this matters a lot more here than it did for ASIFT).
    auto sift = cv::SIFT::create(nFeatures, /*nOctaveLayers=*/8,
                                  /*contrastThreshold=*/0.04, /*edgeThreshold=*/10.0,
                                  /*sigma=*/1.6);

    auto t0 = std::chrono::steady_clock::now();
    FrameFeatures f0 = extract(img0, sift);
    FrameFeatures f1 = extract(img1, sift);
    auto t1 = std::chrono::steady_clock::now();
    double extractMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "[lightglue-test] N0=%zu N1=%zu extract_ms=%.1f\n",
                 f0.kps.size(), f1.kps.size(), extractMs);

    int baselineMatches = ratioTestMatchCount(f0.desc, f1.desc);
    std::fprintf(stderr, "[lightglue-test] baseline ratio-test matches=%d\n", baselineMatches);

    PackedInput packed = packPair(f0, f1, img0.size());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "lightglue_test");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    Ort::Session session(env, modelPath.c_str(), opts);

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 3> kShape{2, packed.N, 4};
    std::array<int64_t, 3> dShape{2, packed.N, 128};
    Ort::Value kTensor = Ort::Value::CreateTensor<float>(memInfo, packed.keypoints.data(),
                                                          packed.keypoints.size(), kShape.data(), kShape.size());
    Ort::Value dTensor = Ort::Value::CreateTensor<float>(memInfo, packed.descriptors.data(),
                                                          packed.descriptors.size(), dShape.data(), dShape.size());

    const char* inputNames[] = {"keypoints", "descriptors"};
    const char* outputNames[] = {"matches", "mscores"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(kTensor));
    inputs.push_back(std::move(dTensor));

    auto t2 = std::chrono::steady_clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames, inputs.data(), inputs.size(),
                                outputNames, 2);
    auto t3 = std::chrono::steady_clock::now();
    double matchMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    int64_t* matchesData = outputs[0].GetTensorMutableData<int64_t>();
    float* mscoresData = outputs[1].GetTensorMutableData<float>();
    auto matchesShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t numRawMatches = matchesShape.empty() ? 0 : matchesShape[0];

    int validMatches = 0;
    double scoreSum = 0.0, scoreMin = 1e9, scoreMax = -1e9;
    for (int64_t i = 0; i < numRawMatches; ++i) {
        int64_t idx0 = matchesData[i * 3 + 1];
        int64_t idx1 = matchesData[i * 3 + 2];
        if (idx0 >= packed.N0 || idx1 >= packed.N1)
            continue; // padded slot, discard
        ++validMatches;
        double s = mscoresData[i];
        scoreSum += s;
        scoreMin = std::min(scoreMin, s);
        scoreMax = std::max(scoreMax, s);
    }

    std::fprintf(stderr,
                 "[lightglue-test] match_ms=%.1f raw_matches=%lld valid_matches=%d "
                 "(padded_discarded=%lld) score_mean=%.3f score_min=%.3f score_max=%.3f\n",
                 matchMs, static_cast<long long>(numRawMatches), validMatches,
                 static_cast<long long>(numRawMatches - validMatches),
                 validMatches ? scoreSum / validMatches : 0.0, scoreMin, scoreMax);
    std::fprintf(stderr, "[lightglue-test] SUMMARY: ratio-test=%d matches, LightGlue=%d matches\n",
                 baselineMatches, validMatches);

    return 0;
}
