// Trains a real DBoW2 vocabulary (hierarchical k-means tree, TF-IDF
// scoring) for the CUSTOM SlamWorker/kitti_ate pipeline's own RootSIFT
// descriptor space -- see FRootSift.h/.cpp (this session, DEBUGGING.md
// 2026-07-22) for the descriptor-class adapter this needed, since the
// vendored DBoW2 only shipped FORB.h (ORB, Hamming distance) before now.
// Distinct from analyze/orbslam3_vlad_train.cpp, which trains a VLAD
// codebook for the SEPARATE third_party/ORB_SLAM3_SIFT fork -- this tool
// targets THIS project's own SlamWorker.cpp/kitti_ate.cpp, currently using
// VLAD (see SlamWorker::setVladLoopClosureEnabled()) as its only
// SIFT-compatible loop-closure candidate search. A trained DBoW2
// vocabulary is meant to become an ALTERNATIVE, not a replacement -- VLAD
// stays wired exactly as-is.
//
// Uses feature_detector::createDetector()+toRootSift() (SlamWorker's own
// extraction path, FeatureDetector.h) rather than reimplementing SIFT
// extraction here, so training-time descriptors are byte-identical in
// distribution to what tryLoopClosure() will query the vocabulary with at
// runtime -- same rationale orbslam3_vlad_train.cpp's own header comment
// already documents for its own (different) codebase.
//
// Usage: train_sift_dbow_vocabulary <out.txt> <k> <L> <frame-stride>
//   <seq-dir> [<seq-dir> ...]
//   out.txt: output vocabulary, DBoW2::TemplatedVocabulary<cv::Mat,
//     DBoW2::FRootSift>'s own saveToTextFile()/loadFromTextFile() format.
//   k: branching factor (children per tree node), e.g. 10. Must be <= 20
//     (TemplatedVocabulary::loadFromTextFile()'s own sanity bound).
//   L: tree depth levels, e.g. 5 (k^L words -- 10^5 = 100k words at k=10).
//     Must be <= 10 (same sanity bound). Training cost/memory grows fast
//     with L -- start small (e.g. L=4, 10k words) and only go bigger once
//     a run completes in reasonable time.
//   frame-stride: only extract from every Nth frame (1 = every frame).
//     DBoW2's hierarchical k-means is far more expensive than VLAD's flat
//     single-pass k-means, so keeping the training pool small enough to
//     finish is the real lever here, not a quality one.
//   seq-dir: one or more KITTI sequence dirs (each must contain
//     times.txt and image_0/), e.g. .../sequences/00.

#include "vision/FeatureDetector.h"

#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/FRootSift.h>

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

using SiftVocabulary = DBoW2::TemplatedVocabulary<cv::Mat, DBoW2::FRootSift>;

void loadImageList(const std::string &sequencePath, std::vector<std::string> &imageFiles)
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

} // namespace

int main(int argc, char **argv)
{
    if (argc < 6) {
        std::fprintf(stderr,
                      "usage: %s <out.txt> <k> <L> <frame-stride> <seq-dir> [<seq-dir> ...]\n"
                      "  out.txt: output vocabulary (DBoW2 text format)\n"
                      "  k: branching factor, e.g. 10 (<=20)\n"
                      "  L: tree depth, e.g. 4-5 (<=10, k^L words)\n"
                      "  frame-stride: use every Nth frame (1 = all)\n"
                      "  seq-dir: one or more KITTI sequence dirs (times.txt + image_0/)\n",
                      argv[0]);
        return 1;
    }
    const std::string outPath = argv[1];
    const int k = std::atoi(argv[2]);
    const int L = std::atoi(argv[3]);
    const int stride = std::max(1, std::atoi(argv[4]));
    std::vector<std::string> seqDirs;
    for (int i = 5; i < argc; ++i)
        seqDirs.push_back(argv[i]);

    if (k <= 0 || k > 20 || L <= 0 || L > 10) {
        std::fprintf(stderr, "k must be in (0,20], L must be in (0,10] (TemplatedVocabulary's own load bounds)\n");
        return 1;
    }

    // Same detector construction SlamWorker::rebuildDetector() uses for
    // SIFT, default SiftSettings (nFeatures=2000, see item 12's own
    // DEBUGGING.md writeup for why this cap essentially never binds on
    // KITTI seq00 anyway -- irrelevant here since training just wants a
    // representative descriptor sample, not the runtime cap's exact
    // behavior).
    SiftSettings siftSettings;
    feature_detector::OrbSettings unusedOrbSettings;
    cv::Ptr<cv::Feature2D> detector =
        feature_detector::createDetector(feature_detector::DetectorType::Sift, siftSettings, unusedOrbSettings);

    std::vector<std::vector<cv::Mat>> trainingFeatures;

    for (const std::string &seqDir : seqDirs) {
        std::vector<std::string> imageFiles;
        loadImageList(seqDir, imageFiles);
        if (imageFiles.empty()) {
            std::fprintf(stderr, "[warn] no images found in %s, skipping\n", seqDir.c_str());
            continue;
        }
        std::fprintf(stderr, "[config] %zu images from %s (stride=%d)\n", imageFiles.size(), seqDir.c_str(), stride);

        for (size_t i = 0; i < imageFiles.size(); i += static_cast<size_t>(stride)) {
            cv::Mat img = cv::imread(imageFiles[i], cv::IMREAD_GRAYSCALE);
            if (img.empty())
                continue;

            std::vector<cv::KeyPoint> kps;
            cv::Mat descriptors;
            detector->detectAndCompute(img, cv::noArray(), kps, descriptors);
            if (descriptors.empty())
                continue;
            descriptors = feature_detector::toRootSift(descriptors);

            std::vector<cv::Mat> rows;
            rows.reserve(static_cast<size_t>(descriptors.rows));
            for (int r = 0; r < descriptors.rows; ++r)
                rows.push_back(descriptors.row(r).clone());
            trainingFeatures.push_back(std::move(rows));

            if (i % (static_cast<size_t>(stride) * 200) < static_cast<size_t>(stride)) {
                std::fprintf(stderr, "[progress] %s frame %zu/%zu, %zu training images collected so far\n",
                              seqDir.c_str(), i, imageFiles.size(), trainingFeatures.size());
                std::fflush(stderr);
            }
        }
    }

    if (trainingFeatures.empty()) {
        std::fprintf(stderr, "No training images collected\n");
        return 1;
    }
    long long totalDescriptors = 0;
    for (const auto &img : trainingFeatures)
        totalDescriptors += static_cast<long long>(img.size());
    std::fprintf(stderr,
                  "[config] training DBoW2 vocabulary: k=%d L=%d, %zu training images, %lld total descriptors "
                  "(this is the slow step -- hierarchical k-means, expect minutes to hours depending on k^L)\n",
                  k, L, trainingFeatures.size(), totalDescriptors);

    SiftVocabulary voc(k, L, DBoW2::TF_IDF, DBoW2::L1_NORM);
    voc.create(trainingFeatures);
    std::fprintf(stderr, "[config] vocabulary created: %u words\n", voc.size());

    std::fprintf(stderr, "[config] writing vocabulary to %s\n", outPath.c_str());
    voc.saveToTextFile(outPath);

    // Sanity check (reported, not asserted): reload through a fresh
    // vocabulary instance and compare BoW score for temporally-adjacent
    // vs temporally-distant frames from the first sequence -- mirrors
    // orbslam3_vlad_train.cpp's own end-of-run check. Expect adjacent >>
    // distant if the vocabulary captures anything meaningful.
    if (!seqDirs.empty()) {
        SiftVocabulary reloaded;
        if (!reloaded.loadFromTextFile(outPath)) {
            std::fprintf(stderr, "[warn] sanity check: failed to reload vocabulary from %s\n", outPath.c_str());
            return 0;
        }

        std::vector<std::string> imageFiles;
        loadImageList(seqDirs[0], imageFiles);
        const size_t distantIdx = imageFiles.size() > 200 ? 200 : imageFiles.size() - 1;
        if (imageFiles.size() > 1) {
            auto bowFor = [&](size_t idx) {
                cv::Mat img = cv::imread(imageFiles[idx], cv::IMREAD_GRAYSCALE);
                std::vector<cv::KeyPoint> kps;
                cv::Mat descriptors;
                detector->detectAndCompute(img, cv::noArray(), kps, descriptors);
                descriptors = feature_detector::toRootSift(descriptors);
                std::vector<cv::Mat> rows;
                rows.reserve(static_cast<size_t>(descriptors.rows));
                for (int r = 0; r < descriptors.rows; ++r)
                    rows.push_back(descriptors.row(r));
                DBoW2::BowVector bow;
                reloaded.transform(rows, bow);
                return bow;
            };
            const DBoW2::BowVector bow0 = bowFor(0);
            const DBoW2::BowVector bow1 = bowFor(1);
            const DBoW2::BowVector bowDistant = bowFor(distantIdx);
            const double scoreAdjacent = reloaded.score(bow0, bow1);
            const double scoreDistant = reloaded.score(bow0, bowDistant);
            std::fprintf(stderr,
                          "[sanity] frame0<->frame1 (adjacent) score=%.4f, "
                          "frame0<->frame%zu (distant) score=%.4f (expect adjacent notably higher)\n",
                          scoreAdjacent, distantIdx, scoreDistant);
        }
    }

    std::fprintf(stderr, "[done] vocabulary written to %s\n", outPath.c_str());
    return 0;
}
