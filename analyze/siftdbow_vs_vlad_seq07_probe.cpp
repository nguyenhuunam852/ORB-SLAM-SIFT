// Standalone probe: does the real, trained SIFT-DBoW2 vocabulary
// ("sift_dbow_vocab (2).txt") score the exact seq07 wide-baseline
// loop-closure keyframe pairs any more discriminatively than VLAD did?
// (VLAD scores measured live in production: kf31->kf0=0.5224,
// kf32->kf1=0.4915, kf119->kf1=0.6232, kf120->kf0=0.6193 -- see
// [loop][vlad-topk-diag] evidence gathered this session). Same RootSIFT
// preprocessing and transform() convention as SlamWorker::insertKeyframe().
#include <opencv2/opencv.hpp>
#include "DBoW2/BowVector.h"
#include "DBoW2/FRootSift.h"
#include "DBoW2/TemplatedVocabulary.h"

#include <cstdio>
#include <vector>

using SiftVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FRootSift::TDescriptor, DBoW2::FRootSift>;

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

DBoW2::BowVector computeBow(const SiftVocabulary &vocab, const cv::Mat &descriptors)
{
    std::vector<cv::Mat> rows;
    rows.reserve(static_cast<size_t>(descriptors.rows));
    for (int r = 0; r < descriptors.rows; ++r)
        rows.push_back(descriptors.row(r));
    DBoW2::BowVector bow;
    DBoW2::FeatureVector unusedFeatVec;
    vocab.transform(rows, bow, unusedFeatVec, 4);
    return bow;
}

struct Pair {
    const char *label;
    const char *fileA;
    const char *fileB;
    double vladScore;
};

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <image_dir> <vocab_path>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    const std::string vocabPath = argv[2];

    SiftVocabulary vocab;
    if (!vocab.loadFromTextFile(vocabPath)) {
        std::fprintf(stderr, "failed to load vocab from %s\n", vocabPath.c_str());
        return 1;
    }
    std::fprintf(stderr, "loaded vocab: %u words\n", static_cast<unsigned>(vocab.size()));

    const std::vector<Pair> pairs = {
        {"kf31(f272)_vs_kf0(f10)   [VLAD score=0.5224, SIFT-match fail]", "000271.png", "000009.png", 0.5224},
        {"kf32(f280)_vs_kf1(f18)   [VLAD score=0.4915, SIFT-match fail]", "000279.png", "000017.png", 0.4915},
        {"kf100(f879)_vs_kf0(f10)  [VLAD score=0.4986, SIFT-match fail]", "000878.png", "000009.png", 0.4986},
        {"kf119(f1067)_vs_kf1(f18) [VLAD score=0.6232, SIFT-match OK]", "001066.png", "000017.png", 0.6232},
        {"kf120(f1075)_vs_kf0(f10) [VLAD score=0.6193, SIFT-match OK]", "001074.png", "000009.png", 0.6193},
    };

    auto sift = cv::SIFT::create(2000);

    for (const auto &p : pairs) {
        cv::Mat imgA = cv::imread(dir + "/" + p.fileA, cv::IMREAD_GRAYSCALE);
        cv::Mat imgB = cv::imread(dir + "/" + p.fileB, cv::IMREAD_GRAYSCALE);
        if (imgA.empty() || imgB.empty()) {
            std::fprintf(stderr, "failed to load %s or %s\n", p.fileA, p.fileB);
            continue;
        }
        std::vector<cv::KeyPoint> kpA, kpB;
        cv::Mat descA, descB;
        sift->detectAndCompute(imgA, cv::noArray(), kpA, descA);
        sift->detectAndCompute(imgB, cv::noArray(), kpB, descB);
        descA = toRootSift(descA);
        descB = toRootSift(descB);

        DBoW2::BowVector bowA = computeBow(vocab, descA);
        DBoW2::BowVector bowB = computeBow(vocab, descB);
        const double dbowScore = vocab.score(bowA, bowB);

        std::printf("=== %s ===\n", p.label);
        std::printf("  kpA=%zu kpB=%zu bowA_words=%zu bowB_words=%zu\n",
                     kpA.size(), kpB.size(), bowA.size(), bowB.size());
        std::printf("  SIFT-DBoW2 score=%.4f   VLAD score=%.4f   ratio(dbow/vlad)=%.2fx\n\n",
                     dbowScore, p.vladScore, p.vladScore > 0 ? dbowScore / p.vladScore : 0.0);
    }

    return 0;
}
