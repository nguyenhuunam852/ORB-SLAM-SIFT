// Stage-0 probe for CudaSift's scale/octave encoding, run against real KITTI
// frames BEFORE trusting any conversion formula in the live ORBextractor
// path. Mirrors Session 14's original Stage 0 for cv::SIFT (which caught a
// real 1-indexed-vs-0-indexed layer bug this same way) -- this project's
// own repeated, explicit lesson is that a new detector's scale/octave
// encoding must be measured against real data, not assumed, before it's
// wired into anything that indexes mvScaleFactors[]/mvLevelSigma2[] by it.
//
// CudaSift's SiftPoint (per Celebrandil/CudaSift's cudaSiftD.h) has:
//   xpos, ypos      -- sub-pixel position
//   scale           -- continuous sigma at detection (NOT an integer level)
//   orientation     -- degrees
//   subsampling     -- octave downsample factor (1.0, 2.0, 4.0, ... -- NOT
//                       a small integer octave index the way OpenCV's
//                       packed KeyPoint::octave is)
//   data[128]       -- descriptor
// None of this matches cv::SIFT's octave/layer bit-packing that this
// project's ORBextractor.cc::flatLevel() decodes -- this probe measures the
// REAL relationship between (scale, subsampling) and image structure so a
// correct flatLevel()-compatible mapping can be derived, instead of guessed.
//
// Usage: <this-binary> <kitti-sequence-image_0-dir> [num-frames=200]
// Prints, across all sampled frames: the observed range of `subsampling`
// (distinct values seen -- these are the "octaves"), and for each distinct
// subsampling value, the observed range of `scale` (continuous sigma) --
// this reveals the log-scale-space bucketing needed to derive a sub-layer
// index within each octave, analogous to cv::SIFT's `layer`.
//
// NOT YET BUILT/RUN LOCALLY -- this dev machine has no NVIDIA GPU, so this
// probe can only be compiled and executed on Kaggle. Run it FIRST and
// report the printed ranges before trusting ORBextractor.cc's
// USE_CUDASIFT-gated octave/layer conversion (currently a best-effort
// placeholder, explicitly marked, pending this probe's real measurement).

#include <opencv2/opencv.hpp>

#include <cudaImage.h>
#include <cudaSift.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <kitti-sequence-image_0-dir> [num-frames=200]\n", argv[0]);
        return 1;
    }
    const std::string seqDir = argv[1];
    const int numFrames = argc > 2 ? std::atoi(argv[2]) : 200;

    InitCuda(0);

    std::set<float> subsamplingValues;
    // subsampling -> (min scale, max scale)
    std::map<float, std::pair<float, float>> scaleRangeBySubsampling;
    long totalPts = 0;
    float globalMinScale = 1e9f, globalMaxScale = -1e9f;
    float globalMinOrient = 1e9f, globalMaxOrient = -1e9f;

    for (int i = 0; i < numFrames; ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%06d.png", i);
        const std::string path = seqDir + "/" + buf;
        cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            std::fprintf(stderr, "frame %d: could not read %s, stopping\n", i, path.c_str());
            break;
        }
        cv::Mat imgF;
        gray.convertTo(imgF, CV_32FC1);

        CudaImage cimg;
        cimg.Allocate(imgF.cols, imgF.rows, iAlignUp(imgF.cols, 128), false, nullptr, (float*)imgF.data);
        cimg.Download();

        SiftData siftData;
        // Generous cap -- this project targets nfeatures=5000; CudaSift's
        // own raw yield before any capping is unknown until measured, so
        // this leaves real headroom rather than silently truncating.
        InitSiftData(siftData, 32768, true, true);

        // initBlur/thresh are CudaSift's own mainSift.cpp demo defaults
        // (1.0f / 3.5f-ish per the reference example) -- placeholders, NOT
        // yet tuned against this project's SiftSettings.h equivalents
        // (contrastThreshold=0.04/edgeThreshold=10.0/sigma=1.6). Tuning
        // those is a separate, later step once the octave/scale mapping
        // below is confirmed correct.
        const float initBlur = 1.0f;
        const float thresh = 3.5f;
        ExtractSift(siftData, cimg, 5 /*numOctaves*/, initBlur, thresh, 0.0f /*lowestScale*/);

        SiftPoint* pts = siftData.h_data;
        for (int p = 0; p < siftData.numPts; ++p) {
            const float ss = pts[p].subsampling;
            const float sc = pts[p].scale;
            const float orient = pts[p].orientation;
            subsamplingValues.insert(ss);
            auto& range = scaleRangeBySubsampling[ss];
            if (totalPts == 0 || range.first == 0.0f) range = {sc, sc};
            range.first = std::min(range.first, sc);
            range.second = std::max(range.second, sc);
            globalMinScale = std::min(globalMinScale, sc);
            globalMaxScale = std::max(globalMaxScale, sc);
            globalMinOrient = std::min(globalMinOrient, orient);
            globalMaxOrient = std::max(globalMaxOrient, orient);
            ++totalPts;
        }

        if (i % 20 == 0)
            std::fprintf(stderr, "[cudasift-probe] frame %d: numPts=%d (running total=%ld)\n", i, siftData.numPts, totalPts);

        FreeSiftData(siftData);
    }

    std::printf("\n=== CudaSift Stage-0 probe results (%ld total keypoints) ===\n", totalPts);
    std::printf("distinct subsampling (octave-equivalent) values seen: %zu\n", subsamplingValues.size());
    for (float ss : subsamplingValues) {
        auto& range = scaleRangeBySubsampling[ss];
        std::printf("  subsampling=%.4f -> scale range [%.4f, %.4f]\n", ss, range.first, range.second);
    }
    std::printf("global scale range: [%.4f, %.4f]\n", globalMinScale, globalMaxScale);
    std::printf("global orientation range: [%.4f, %.4f] (expect [0,360) if degrees, confirms/refutes the radian-conversion assumption)\n",
                globalMinOrient, globalMaxOrient);
    std::printf("\nNext step: derive flatLevel()-compatible (octave,layer) from (subsampling,scale) using the\n"
                "ranges above -- e.g. octave = log2(subsampling), layer = bucketed position of scale within\n"
                "its subsampling group's [min,max] range -- then update ORBextractor.cc's USE_CUDASIFT path\n"
                "(currently a placeholder) and re-run this probe's sanity checks before trusting a live run.\n");

    return 0;
}
