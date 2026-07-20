// Visualizes actual detected keypoint locations for SIFT vs ORB on a single
// KITTI image, to directly answer the question raised in DEBUGGING.md part
// 56: is SIFT's lower raw keypoint count (~66% of ORB's, part 55) because
// ORB's detector finds real corners/edges that SIFT's detector is missing
// in the same places, or because there genuinely isn't much to detect there
// regardless of detector?
//
// Usage: <this-binary> <settings.yaml> <image-path> <output.png>
//
// Draws every detected keypoint as a circle (radius scaled with the
// keypoint's own .size) on the image and writes the result to output.png.
// Built as two separate executables from this same source (see
// CMakeLists.txt) -- one linked against orbslam3_ext (stock ORB), one
// against orbslam3_sift_ext (this fork's SIFT) -- run both on the SAME
// image for a direct side-by-side comparison.

#include <ORBextractor.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <settings.yaml> <image-path> <output.png>\n", argv[0]);
        return 1;
    }
    const std::string settingsPath = argv[1];
    const std::string imagePath = argv[2];
    const std::string outputPath = argv[3];

    int nFeatures = 2000, nLevels = 3;
    float scaleFactor = 1.2f;
    const int iniThFAST = 20, minThFAST = 7;
    {
        cv::FileStorage fs(settingsPath, cv::FileStorage::READ);
        if (fs.isOpened()) {
            if (!fs["ORBextractor.nFeatures"].empty())
                nFeatures = static_cast<int>(fs["ORBextractor.nFeatures"]);
            if (!fs["ORBextractor.nLevels"].empty())
                nLevels = static_cast<int>(fs["ORBextractor.nLevels"]);
            if (!fs["ORBextractor.scaleFactor"].empty())
                scaleFactor = static_cast<float>(fs["ORBextractor.scaleFactor"]);
        }
    }
    std::fprintf(stderr, "[config] nFeatures=%d nLevels=%d scaleFactor=%.2f\n", nFeatures, nLevels, scaleFactor);

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::fprintf(stderr, "failed to load image: %s\n", imagePath.c_str());
        return 1;
    }

    ORB_SLAM3::ORBextractor extractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<int> vLapping = {0, 0};
    extractor(image, cv::Mat(), keypoints, descriptors, vLapping);

    std::fprintf(stderr, "[result] detected %zu keypoints\n", keypoints.size());

    // Dump every keypoint's (x, y, response, size) to <output>.csv for
    // region-based analysis (e.g. counting/scoring keypoints that fall in
    // a road/ground-plane ROI) -- see DEBUGGING.md part 56.
    {
        const std::string csvPath = outputPath + ".csv";
        FILE* f = std::fopen(csvPath.c_str(), "w");
        if (f) {
            std::fprintf(f, "x,y,response,size,octave\n");
            for (const auto& kp : keypoints)
                std::fprintf(f, "%.2f,%.2f,%.6f,%.3f,%d\n", kp.pt.x, kp.pt.y, kp.response, kp.size, kp.octave);
            std::fclose(f);
            std::fprintf(stderr, "[done] wrote %s\n", csvPath.c_str());
        }
    }

    cv::Mat colorImage;
    cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
    for (const auto& kp : keypoints) {
        const float radius = std::max(2.0f, kp.size / 4.0f);
        cv::circle(colorImage, kp.pt, static_cast<int>(radius), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    cv::putText(colorImage, cv::format("N=%zu", keypoints.size()), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

    cv::imwrite(outputPath, colorImage);
    std::fprintf(stderr, "[done] wrote %s\n", outputPath.c_str());
    return 0;
}
