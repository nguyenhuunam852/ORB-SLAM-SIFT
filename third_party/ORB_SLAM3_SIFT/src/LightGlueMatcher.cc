#include "LightGlueMatcher.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace ORB_SLAM3
{

struct LightGlueMatcher::Impl
{
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "lightglue"};
    std::unique_ptr<Ort::Session> session;

    explicit Impl(const std::string& modelPath)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);

        // Try CUDA first (LightGlue's 9-layer attention is O(N^2) in
        // keypoint count -- measured ~9s/pair on CPU at nfeatures=5000,
        // vs the paper's own ~69ms/pair GPU benchmark. This dev machine
        // has no NVIDIA GPU, so this always falls back to CPU here, but
        // the exact same binary picks up CUDA automatically wherever one
        // is available, e.g. Kaggle's free-tier GPU runtimes -- see
        // DEBUGGING.md part 58). AppendExecutionProvider_CUDA throws
        // Ort::Exception (not a link/compile error) if CUDA isn't usable,
        // since it dispatches through the generic C API at runtime.
        bool usingCuda = false;
        try {
            OrtCUDAProviderOptions cudaOpts{};
            opts.AppendExecutionProvider_CUDA(cudaOpts);
            usingCuda = true;
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[lightglue-matcher] CUDA provider unavailable (%s) -- using CPU\n", e.what());
        }

        session = std::make_unique<Ort::Session>(env, modelPath.c_str(), opts);
        std::fprintf(stderr, "[lightglue-matcher] execution provider: %s\n", usingCuda ? "CUDA" : "CPU");
    }
};

LightGlueMatcher::LightGlueMatcher(const std::string& modelPath, int /*intraOpThreads*/)
    : mImpl(std::make_unique<Impl>(modelPath))
{
    std::fprintf(stderr, "[lightglue-matcher] loaded model: %s\n", modelPath.c_str());
}

LightGlueMatcher::~LightGlueMatcher() = default;

namespace {

// cvg/LightGlue/sift.py: sift_to_rootsift -- L1 normalize, sqrt, L2 normalize.
// Required: the sift_lightglue weights were trained on RootSIFT, not raw
// SIFT descriptors. Operates on a local copy, never mutates the caller's
// descriptor Mat (other pipeline consumers -- BoW/VLAD conversion,
// MapPoint::GetDescriptor -- need the original raw descriptors).
cv::Mat toRootSift(const cv::Mat& desc)
{
    cv::Mat out = desc.clone();
    CV_Assert(out.type() == CV_32F);
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

} // namespace

int LightGlueMatcher::Match(const std::vector<cv::KeyPoint>& kps0, const cv::Mat& desc0,
                             const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
                             cv::Size imgSize, std::vector<int>& matches01) const
{
    matches01.assign(kps0.size(), -1);
    if (kps0.empty() || kps1.empty())
        return 0;

    cv::Mat rd0 = toRootSift(desc0);
    cv::Mat rd1 = toRootSift(desc1);

    const int N0 = static_cast<int>(kps0.size());
    const int N1 = static_cast<int>(kps1.size());
    const int N = std::max(N0, N1);

    std::vector<float> keypoints(static_cast<size_t>(2) * N * 4, 0.0f);
    std::vector<float> descriptors(static_cast<size_t>(2) * N * 128, 0.0f);

    const float shiftX = imgSize.width / 2.0f;
    const float shiftY = imgSize.height / 2.0f;
    const float scale = std::max(imgSize.width, imgSize.height) / 2.0f;

    auto fill = [&](int half, const std::vector<cv::KeyPoint>& kps, const cv::Mat& rd) {
        for (int i = 0; i < static_cast<int>(kps.size()); ++i) {
            const cv::KeyPoint& kp = kps[i];
            size_t kBase = (static_cast<size_t>(half) * N + i) * 4;
            keypoints[kBase + 0] = (kp.pt.x - shiftX) / scale;
            keypoints[kBase + 1] = (kp.pt.y - shiftY) / scale;
            keypoints[kBase + 2] = kp.size;
            keypoints[kBase + 3] = kp.angle * static_cast<float>(CV_PI) / 180.0f;
            size_t dBase = (static_cast<size_t>(half) * N + i) * 128;
            const float* row = rd.ptr<float>(i);
            std::copy(row, row + 128, descriptors.begin() + dBase);
        }
    };
    fill(0, kps0, rd0);
    fill(1, kps1, rd1);

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 3> kShape{2, N, 4};
    std::array<int64_t, 3> dShape{2, N, 128};
    Ort::Value kTensor = Ort::Value::CreateTensor<float>(memInfo, keypoints.data(), keypoints.size(),
                                                          kShape.data(), kShape.size());
    Ort::Value dTensor = Ort::Value::CreateTensor<float>(memInfo, descriptors.data(), descriptors.size(),
                                                          dShape.data(), dShape.size());

    const char* inputNames[] = {"keypoints", "descriptors"};
    const char* outputNames[] = {"matches", "mscores"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(kTensor));
    inputs.push_back(std::move(dTensor));

    auto outputs = mImpl->session->Run(Ort::RunOptions{nullptr}, inputNames, inputs.data(), inputs.size(),
                                        outputNames, 2);

    int64_t* matchesData = outputs[0].GetTensorMutableData<int64_t>();
    auto matchesShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t numRawMatches = matchesShape.empty() ? 0 : matchesShape[0];

    int nmatches = 0;
    for (int64_t i = 0; i < numRawMatches; ++i) {
        int64_t idx0 = matchesData[i * 3 + 1];
        int64_t idx1 = matchesData[i * 3 + 2];
        if (idx0 >= N0 || idx1 >= N1)
            continue; // padded slot, discard
        matches01[idx0] = static_cast<int>(idx1);
        ++nmatches;
    }
    return nmatches;
}

namespace {
    std::unique_ptr<LightGlueMatcher> g_lightGlueMatcher;
}

LightGlueMatcher* GetLightGlueMatcher()
{
    if (!g_lightGlueMatcher) {
        g_lightGlueMatcher = std::make_unique<LightGlueMatcher>("weights/lightglue_sift.onnx");
    }
    return g_lightGlueMatcher.get();
}

} // namespace ORB_SLAM3
