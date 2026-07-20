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

// Hard cap on keypoints fed into a single LightGlue Match() call, independent
// of whatever the extractor produced. LightGlue's attention is O(N^2) in
// keypoint count (9 self+cross attention layers) -- the paper/official
// benchmarks (and cvg/LightGlue's own default max_num_keypoints) target
// ~1-2k keypoints, well below this project's nfeatures=5000. Originally set
// to 3000 purely to bound memory after a real OOM (ASIFT + the
// translation-velocity density boost together fed it N=7000-7500+ before
// both were removed -- see DEBUGGING.md part 58 continued). Lowered to 2048
// (round number matching common max_num_keypoints configs) as a first test
// of whether matching LightGlue's training/eval keypoint density -- not
// just avoiding OOM -- improves match/tracking quality on KITTI, after
// verifying (analyze/verify_lightglue_onnx.py, real-data check) that the
// ONNX export itself is faithful and not the source of the coverage gap.
constexpr int kMaxLightGlueKeypoints = 2048;

// Indices of the top-k keypoints by response (all indices, in order, if
// kps.size()<=k).
std::vector<int> topResponseIndices(const std::vector<cv::KeyPoint>& kps, int k)
{
    std::vector<int> idx(kps.size());
    for (size_t i = 0; i < idx.size(); ++i)
        idx[i] = static_cast<int>(i);
    if (static_cast<int>(idx.size()) <= k)
        return idx;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                       [&](int a, int b) { return kps[a].response > kps[b].response; });
    idx.resize(k);
    return idx;
}

} // namespace

int LightGlueMatcher::Match(const std::vector<cv::KeyPoint>& kps0, const cv::Mat& desc0,
                             const std::vector<cv::KeyPoint>& kps1, const cv::Mat& desc1,
                             cv::Size imgSize, std::vector<int>& matches01) const
{
    matches01.assign(kps0.size(), -1);
    if (kps0.empty() || kps1.empty())
        return 0;

    // idx0[i]/idx1[j] map subsampled position -> original kps0/kps1 index.
    const std::vector<int> idx0 = topResponseIndices(kps0, kMaxLightGlueKeypoints);
    const std::vector<int> idx1 = topResponseIndices(kps1, kMaxLightGlueKeypoints);

    cv::Mat rd0 = toRootSift(desc0);
    cv::Mat rd1 = toRootSift(desc1);

    const int N0 = static_cast<int>(idx0.size());
    const int N1 = static_cast<int>(idx1.size());
    const int N = std::max(N0, N1);

    std::vector<float> keypoints(static_cast<size_t>(2) * N * 4, 0.0f);
    std::vector<float> descriptors(static_cast<size_t>(2) * N * 128, 0.0f);

    const float shiftX = imgSize.width / 2.0f;
    const float shiftY = imgSize.height / 2.0f;
    const float scale = std::max(imgSize.width, imgSize.height) / 2.0f;

    auto fill = [&](int half, const std::vector<cv::KeyPoint>& kps, const cv::Mat& rd, const std::vector<int>& idx) {
        for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
            const cv::KeyPoint& kp = kps[idx[i]];
            size_t kBase = (static_cast<size_t>(half) * N + i) * 4;
            keypoints[kBase + 0] = (kp.pt.x - shiftX) / scale;
            keypoints[kBase + 1] = (kp.pt.y - shiftY) / scale;
            keypoints[kBase + 2] = kp.size;
            keypoints[kBase + 3] = kp.angle * static_cast<float>(CV_PI) / 180.0f;
            size_t dBase = (static_cast<size_t>(half) * N + i) * 128;
            const float* row = rd.ptr<float>(idx[i]);
            std::copy(row, row + 128, descriptors.begin() + dBase);
        }
    };
    fill(0, kps0, rd0, idx0);
    fill(1, kps1, rd1, idx1);

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
    // Deliberately never destroyed (raw pointer, no delete) -- ONNX Runtime's
    // CUDA execution provider frees GPU memory (cudaFreeHost etc.) in its own
    // destructor, and at normal process-exit time the CUDA driver can already
    // be mid-teardown, turning that free into an uncaught
    // onnxruntime::OnnxRuntimeException -> std::terminate -> SIGABRT that
    // kills the whole process, including any in-flight file write (this is
    // exactly what truncated a completed KeyFrameTrajectory.txt to 0 bytes
    // on a Kaggle GPU run despite tracking/ATE having already finished
    // successfully -- see DEBUGGING.md). The OS reclaims all GPU/process
    // memory on exit regardless, so skipping this destructor is safe.
    LightGlueMatcher* g_lightGlueMatcher = nullptr;
}

LightGlueMatcher* GetLightGlueMatcher()
{
    if (!g_lightGlueMatcher) {
        g_lightGlueMatcher = new LightGlueMatcher("weights/lightglue_sift.onnx");
    }
    return g_lightGlueMatcher;
}

} // namespace ORB_SLAM3
