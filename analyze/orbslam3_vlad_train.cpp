// Trains a VLAD codebook for the SIFT-based ORB_SLAM3 fork
// (third_party/ORB_SLAM3_SIFT) -- replaces DBoW2's ORBvoc.txt, which is
// trained on ORB's binary descriptors and has no SIFT-compatible
// equivalent (see VladVocabulary.h's doc comment and DEBUGGING.md's
// ORB->SIFT swap session). Links against orbslam3_sift_ext so the
// training-time ORBextractor (now SIFT-based) is byte-identical to the
// runtime one -- the strongest available guard against train/runtime
// descriptor-distribution skew.
//
// Usage: orbslam3_vlad_train <settings.yaml> <k> <out.yml> <seq-dir> [<seq-dir> ...]
//   settings.yaml: an ORB_SLAM3 settings file -- only ORBextractor.nFeatures/
//     nLevels are read from it (real Camera/Viewer keys aren't needed since
//     this tool never constructs a full ORB_SLAM3::System).
//   k: codebook size (number of VLAD centroids), e.g. 64.
//   out.yml: output codebook path, loadable via VladVocabulary::loadFromTextFile().
//   seq-dir: one or more KITTI sequence directories (each must contain
//     times.txt and image_0/), e.g. .../sequences/00.

#include <ORBextractor.h>
#include <VladVocabulary.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>
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

} // namespace

int main(int argc, char **argv)
{
    if (argc < 5) {
        std::fprintf(stderr,
                      "usage: %s <settings.yaml> <k> <out.yml> <seq-dir> [<seq-dir> ...]\n"
                      "  settings.yaml: reads ORBextractor.nFeatures/nLevels only\n"
                      "  k: VLAD codebook size (number of centroids), e.g. 64\n"
                      "  out.yml: output codebook, loadable via VladVocabulary::loadFromTextFile()\n"
                      "  seq-dir: one or more KITTI sequence dirs (times.txt + image_0/)\n",
                      argv[0]);
        return 1;
    }
    const std::string settingsPath = argv[1];
    const int k = std::atoi(argv[2]);
    const std::string outPath = argv[3];
    std::vector<std::string> seqDirs;
    for (int i = 4; i < argc; ++i)
        seqDirs.push_back(argv[i]);

    if (k <= 0) {
        std::fprintf(stderr, "k must be > 0\n");
        return 1;
    }

    int nFeatures = 2000, nLevels = 3;
    {
        cv::FileStorage fs(settingsPath, cv::FileStorage::READ);
        if (fs.isOpened()) {
            if (!fs["ORBextractor.nFeatures"].empty())
                nFeatures = static_cast<int>(fs["ORBextractor.nFeatures"]);
            if (!fs["ORBextractor.nLevels"].empty())
                nLevels = static_cast<int>(fs["ORBextractor.nLevels"]);
            fs.release();
        } else {
            std::fprintf(stderr, "[warn] could not open %s, using defaults nFeatures=%d nLevels=%d\n",
                          settingsPath.c_str(), nFeatures, nLevels);
        }
    }
    std::fprintf(stderr, "[config] nFeatures=%d nLevels(nOctaveLayers)=%d k=%d\n", nFeatures, nLevels, k);

    ORB_SLAM3::ORBextractor extractor(nFeatures, 1.2f, nLevels, 20, 7);

    constexpr int kRowCap = 400000;
    cv::Mat pool(0, 128, CV_32F);
    pool.reserve(kRowCap);

    std::mt19937 rng(12345);
    long long totalSeen = 0;

    for (const std::string &seqDir : seqDirs) {
        std::vector<std::string> imageFiles;
        LoadImages(seqDir, imageFiles);
        if (imageFiles.empty()) {
            std::fprintf(stderr, "[warn] no images found in %s, skipping\n", seqDir.c_str());
            continue;
        }
        std::fprintf(stderr, "[config] %zu images from %s\n", imageFiles.size(), seqDir.c_str());

        for (size_t i = 0; i < imageFiles.size(); ++i) {
            cv::Mat img = cv::imread(imageFiles[i], cv::IMREAD_GRAYSCALE);
            if (img.empty())
                continue;

            std::vector<cv::KeyPoint> kps;
            cv::Mat descriptors;
            std::vector<int> lap = {0, 0};
            extractor(img, cv::Mat(), kps, descriptors, lap);

            for (int r = 0; r < descriptors.rows; ++r) {
                ++totalSeen;
                if (pool.rows < kRowCap) {
                    pool.push_back(descriptors.row(r));
                } else {
                    // Reservoir sampling: keep a uniform random subset once past the cap.
                    std::uniform_int_distribution<long long> dist(0, totalSeen - 1);
                    const long long j = dist(rng);
                    if (j < kRowCap)
                        descriptors.row(r).copyTo(pool.row(static_cast<int>(j)));
                }
            }

            if (i % 200 == 0) {
                std::fprintf(stderr, "[progress] %s frame %zu/%zu, pool=%d/%d rows (seen %lld)\n",
                              seqDir.c_str(), i, imageFiles.size(), pool.rows, kRowCap, totalSeen);
                std::fflush(stderr);
            }
        }
    }

    if (pool.rows < k) {
        std::fprintf(stderr, "Only %d descriptor rows collected, need at least k=%d\n", pool.rows, k);
        return 1;
    }
    std::fprintf(stderr, "[config] training k-means on %d rows (%lld descriptors seen total)...\n", pool.rows,
                 totalSeen);

    cv::Mat labels, centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 1e-4);
    // cv::kmeans's return value IS the k-means "loss" -- compactness, the
    // sum over all points of squared L2 distance to their assigned
    // centroid (the objective k-means actually minimizes). Not a neural-
    // net loss curve (OpenCV's kmeans is a single black-box call, no
    // per-iteration callback), but this before/after-normalized number is
    // the real, directly comparable signal: lower per-descriptor
    // compactness means tighter, more separated clusters. Reported here
    // instead of silently discarded (the previous version of this file
    // never captured cv::kmeans's return value at all).
    const double compactness = cv::kmeans(pool, k, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
    std::fprintf(stderr, "[kmeans-loss] compactness=%.2f (raw sum of squared dist) "
                          "avgPerDescriptor=%.6f (compactness/%d rows) -- lower is better\n",
                 compactness, compactness / pool.rows, pool.rows);
    centers.convertTo(centers, CV_32F);

    std::fprintf(stderr, "[config] writing codebook to %s\n", outPath.c_str());
    {
        // Flat, non-dotted top-level keys -- cv::FileStorage::WRITE's node-
        // construction API rejects dotted key names (see DEBUGGING.md's
        // Session 12 crash); this format isn't parsed by Settings.cc so
        // there's no need to match its dotted-key convention anyway.
        cv::FileStorage fs(outPath, cv::FileStorage::WRITE);
        fs << "k" << k;
        fs << "descriptorDim" << 128;
        fs << "centroids" << centers;
        std::string trainedFrom;
        for (size_t i = 0; i < seqDirs.size(); ++i)
            trainedFrom += (i ? "," : "") + seqDirs[i];
        fs << "trainedFrom" << trainedFrom;
        char siftParams[128];
        std::snprintf(siftParams, sizeof(siftParams), "nFeatures=%d,nOctaveLayers=%d,contrastThreshold=0.04,"
                                                        "edgeThreshold=10.0,sigma=1.6",
                      nFeatures, nLevels);
        fs << "siftParams" << std::string(siftParams);
        fs.release();
    }

    // Sanity check (reported, not asserted -- see the plan's "measure and
    // report honestly" note): reload through the real VladVocabulary class
    // and compare VLAD cosine similarity for temporally-adjacent vs
    // temporally-distant frames from the first sequence. Expect adjacent
    // >> distant if the codebook captures anything meaningful.
    if (!seqDirs.empty()) {
        ORB_SLAM3::VladVocabulary voc;
        if (!voc.loadFromTextFile(outPath)) {
            std::fprintf(stderr, "[warn] sanity check: failed to reload codebook from %s\n", outPath.c_str());
            return 0;
        }

        std::vector<std::string> imageFiles;
        LoadImages(seqDirs[0], imageFiles);
        const size_t distantIdx = imageFiles.size() > 200 ? 200 : imageFiles.size() - 1;
        if (imageFiles.size() > 1) {
            auto vladFor = [&](size_t idx) {
                cv::Mat img = cv::imread(imageFiles[idx], cv::IMREAD_GRAYSCALE);
                std::vector<cv::KeyPoint> kps;
                cv::Mat descriptors;
                std::vector<int> lap = {0, 0};
                extractor(img, cv::Mat(), kps, descriptors, lap);
                return voc.computeVlad(descriptors);
            };
            const cv::Mat v0 = vladFor(0);
            const cv::Mat v1 = vladFor(1);
            const cv::Mat vDistant = vladFor(distantIdx);
            const float simAdjacent = voc.score(v0, v1);
            const float simDistant = voc.score(v0, vDistant);
            std::fprintf(stderr,
                          "[sanity] frame0<->frame1 (adjacent) similarity=%.4f, "
                          "frame0<->frame%zu (distant) similarity=%.4f (expect adjacent notably higher)\n",
                          simAdjacent, distantIdx, simDistant);
        }
    }

    std::fprintf(stderr, "[done] codebook written to %s\n", outPath.c_str());
    return 0;
}
