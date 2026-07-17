// Runs the REAL ORB-SLAM3 system (not this project's own SlamWorker) against
// a raw KITTI image sequence and computes ATE against ground truth, natively
// in C++ -- a project-owned counterpart to analyze/kitti_ate.cpp for
// comparing this project's own pipeline against the actual published system.
//
// Links directly against a prebuilt libORB_SLAM3.so from an existing,
// already-built ORB_SLAM3 checkout (see CMakeLists.txt's own doc comment on
// the orbslam3_kitti_ate target for why this isn't vendored/recompiled the
// way DBoW2 was, and why this whole binary is built against the same conda
// OpenCV/Eigen/Pangolin libORB_SLAM3.so itself uses instead of this
// project's usual system OpenCV).
//
// Tracking loop mirrors ORB_SLAM3's own Examples/Monocular/mono_kitti.cc
// (LoadImages() + the TrackMonocular() loop), minus its real-time playback
// throttle (this is a headless batch tool, same reasoning as kitti_ate.cpp's
// own startUnthrottled()). Ground-truth loading + 2D Umeyama alignment + ATE
// computation mirror kitti_ate.cpp's own loadGroundTruth()/umeyama2D()
// exactly, so numbers from the two tools are directly comparable -- ORB-SLAM3
// saves a per-KEYFRAME TUM-format trajectory (timestamp tx ty tz qx qy qz qw,
// already Twc/world-frame camera-center translation, see
// System::SaveKeyFrameTrajectoryTUM()), so each line is matched to a KITTI
// frame index via nearest timestamp in times.txt rather than by frame index
// directly.

#include <System.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Point2
{
    double x = 0.0, z = 0.0;
};

void LoadImages(const std::string &sequencePath, std::vector<std::string> &imageFiles,
                 std::vector<double> &timestamps)
{
    std::ifstream fTimes(sequencePath + "/times.txt");
    std::string line;
    while (std::getline(fTimes, line)) {
        if (line.empty())
            continue;
        timestamps.push_back(std::stod(line));
    }

    const std::string prefix = sequencePath + "/image_0/";
    imageFiles.resize(timestamps.size());
    char buf[16];
    for (size_t i = 0; i < timestamps.size(); ++i) {
        std::snprintf(buf, sizeof(buf), "%06zu.png", i);
        imageFiles[i] = prefix + buf;
    }
}

std::vector<Point2> loadGroundTruth(const std::string &posesPath)
{
    std::vector<Point2> gt;
    std::ifstream f(posesPath);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::vector<double> vals;
        double v;
        while (ss >> v)
            vals.push_back(v);
        if (vals.size() < 12)
            continue;
        gt.push_back({vals[3], vals[11]});
    }
    return gt;
}

// Identical math to kitti_ate.cpp's own umeyama2D().
bool umeyama2D(const std::vector<Point2> &src, const std::vector<Point2> &dst, double &scale, double &cosTheta,
               double &sinTheta, double &tx, double &tz)
{
    if (src.size() < 8 || src.size() != dst.size())
        return false;

    const double n = static_cast<double>(src.size());
    double meanSrcX = 0.0, meanSrcZ = 0.0, meanDstX = 0.0, meanDstZ = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        meanSrcX += src[i].x;
        meanSrcZ += src[i].z;
        meanDstX += dst[i].x;
        meanDstZ += dst[i].z;
    }
    meanSrcX /= n;
    meanSrcZ /= n;
    meanDstX /= n;
    meanDstZ /= n;

    double C = 0.0, D = 0.0, srcSqSum = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        const double px = src[i].x - meanSrcX, pz = src[i].z - meanSrcZ;
        const double qx = dst[i].x - meanDstX, qz = dst[i].z - meanDstZ;
        C += px * qx + pz * qz;
        D += px * qz - pz * qx;
        srcSqSum += px * px + pz * pz;
    }
    if (srcSqSum < 1e-9)
        return false;

    const double theta = std::atan2(D, C);
    cosTheta = std::cos(theta);
    sinTheta = std::sin(theta);
    scale = std::sqrt(C * C + D * D) / srcSqSum;
    if (!(scale > 0.0) || !std::isfinite(scale))
        return false;

    const double rx = cosTheta * meanSrcX - sinTheta * meanSrcZ;
    const double rz = sinTheta * meanSrcX + cosTheta * meanSrcZ;
    tx = meanDstX - scale * rx;
    tz = meanDstZ - scale * rz;
    return true;
}

struct TumPose
{
    double timestamp = 0.0, x = 0.0, z = 0.0;
};

std::vector<TumPose> loadTumTrajectory(const std::string &path)
{
    std::vector<TumPose> traj;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream ss(line);
        TumPose p;
        double ty, qx, qy, qz, qw;
        if (ss >> p.timestamp >> p.x >> ty >> p.z >> qx >> qy >> qz >> qw)
            traj.push_back(p);
    }
    return traj;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 5) {
        std::fprintf(stderr,
                      "usage: %s <path_to_vocabulary> <path_to_settings> <kitti-sequence-dir> "
                      "<kitti-poses.txt> [out-prefix]\n"
                      "  kitti-sequence-dir: e.g. .../sequences/00 (must contain times.txt, image_0/)\n"
                      "  kitti-poses.txt: e.g. .../poses/00.txt\n"
                      "  out-prefix: writes <prefix>_KeyFrameTrajectory.txt, default 'orbslam3_kitti_ate'\n",
                      argv[0]);
        return 1;
    }
    const std::string vocabPath = argv[1];
    const std::string settingsPath = argv[2];
    const std::string sequencePath = argv[3];
    const std::string posesPath = argv[4];
    const std::string outPrefix = argc > 5 ? argv[5] : "orbslam3_kitti_ate";

    std::vector<std::string> imageFiles;
    std::vector<double> timestamps;
    LoadImages(sequencePath, imageFiles, timestamps);
    if (imageFiles.empty()) {
        std::fprintf(stderr, "Failed to load images from %s\n", sequencePath.c_str());
        return 1;
    }
    std::fprintf(stderr, "[config] %zu images loaded from %s\n", imageFiles.size(), sequencePath.c_str());

    // useViewer=false: this is a headless batch tool, not the interactive
    // GUI mono_kitti.cc is meant for -- Pangolin is still linked (System
    // unconditionally depends on it), it just never opens a window.
    ORB_SLAM3::System SLAM(vocabPath, settingsPath, ORB_SLAM3::System::MONOCULAR, false);
    const float imageScale = SLAM.GetImageScale();

    std::fprintf(stderr, "[config] tracking %zu frames (unthrottled, no real-time pacing)...\n", imageFiles.size());
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        cv::Mat im = cv::imread(imageFiles[i], cv::IMREAD_UNCHANGED);
        if (im.empty()) {
            std::fprintf(stderr, "Failed to load image at: %s\n", imageFiles[i].c_str());
            return 1;
        }
        if (imageScale != 1.f)
            cv::resize(im, im, cv::Size(static_cast<int>(im.cols * imageScale), static_cast<int>(im.rows * imageScale)));

        SLAM.TrackMonocular(im, timestamps[i], std::vector<ORB_SLAM3::IMU::Point>(), imageFiles[i]);

        if (i % 500 == 0) {
            std::fprintf(stderr, "[stats] frame %zu / %zu\n", i, imageFiles.size());
            std::fflush(stderr);
        }
    }

    SLAM.Shutdown();

    const std::string trajPath = outPrefix + "_KeyFrameTrajectory.txt";
    SLAM.SaveKeyFrameTrajectoryTUM(trajPath);

    // --- Evaluation: identical methodology to kitti_ate.cpp ---
    const std::vector<Point2> gt = loadGroundTruth(posesPath);
    if (gt.empty()) {
        std::fprintf(stderr, "Failed to load ground truth from %s\n", posesPath.c_str());
        return 1;
    }
    const std::vector<TumPose> traj = loadTumTrajectory(trajPath);
    if (traj.empty()) {
        std::fprintf(stderr, "No trajectory produced -- did tracking ever start?\n");
        return 1;
    }

    // Match each keyframe's timestamp to the nearest KITTI frame index.
    std::vector<Point2> src, dst;
    src.reserve(traj.size());
    dst.reserve(traj.size());
    for (const TumPose &p : traj) {
        size_t bestIdx = 0;
        double bestDiff = std::abs(timestamps[0] - p.timestamp);
        for (size_t i = 1; i < timestamps.size(); ++i) {
            const double diff = std::abs(timestamps[i] - p.timestamp);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIdx = i;
            }
        }
        if (bestDiff > 1e-3 || bestIdx >= gt.size())
            continue;
        src.push_back({p.x, p.z});
        dst.push_back(gt[bestIdx]);
    }

    double scale = 1.0, cosT = 1.0, sinT = 0.0, tx = 0.0, tz = 0.0;
    if (!umeyama2D(src, dst, scale, cosT, sinT, tx, tz)) {
        std::fprintf(stderr, "Alignment failed -- too few matched points (%zu) or degenerate estimate.\n",
                      src.size());
        return 1;
    }

    double sumSq = 0.0, sumAbs = 0.0, maxErr = 0.0;
    std::vector<double> errors(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        const double ax = scale * (cosT * src[i].x - sinT * src[i].z) + tx;
        const double az = scale * (sinT * src[i].x + cosT * src[i].z) + tz;
        const double dx = ax - dst[i].x, dz = az - dst[i].z;
        const double err = std::hypot(dx, dz);
        errors[i] = err;
        sumSq += err * err;
        sumAbs += err;
        maxErr = std::max(maxErr, err);
    }
    std::vector<double> sortedErrors = errors;
    std::sort(sortedErrors.begin(), sortedErrors.end());
    const double ateRmse = std::sqrt(sumSq / static_cast<double>(src.size()));
    const double ateMean = sumAbs / static_cast<double>(src.size());
    const double ateMedian = sortedErrors[sortedErrors.size() / 2];

    double pathLength = 0.0;
    for (size_t i = 1; i < gt.size(); ++i)
        pathLength += std::hypot(gt[i].x - gt[i - 1].x, gt[i].z - gt[i - 1].z);

    std::printf("Matched keyframes:    %zu / %zu\n", src.size(), traj.size());
    std::printf("Recovered scale:      %.4f (estimated-units -> ground-truth-units)\n", scale);
    std::printf("GT path length:       %.1f m\n", pathLength);
    std::printf("ATE RMSE:             %.3f m\n", ateRmse);
    std::printf("ATE mean:             %.3f m\n", ateMean);
    std::printf("ATE median:           %.3f m\n", ateMedian);
    std::printf("ATE max:              %.3f m\n", maxErr);
    std::printf("ATE RMSE / path len:  %.2f%%\n", 100.0 * ateRmse / pathLength);

    return 0;
}
