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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
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

// Matches each trajectory pose to the nearest-timestamp ground-truth point,
// aligns with a 2D Umeyama similarity fit, and reports ATE -- factored out of
// main() so it can be run once for the "official" (current-map-only) export
// and once per Atlas map fragment (see the [atlas-coverage] diagnostic and
// DEBUGGING.md: SaveKeyFrameTrajectoryTUM() only exports the CURRENT map,
// silently discarding earlier abandoned fragments that may have tracked
// perfectly well). pathLength is computed from the MATCHED ground-truth
// points only, in trajectory order -- not the full probed region's ground
// truth -- so "ATE RMSE / path len" can't be diluted by an irrelevant longer
// path the way the misleading "0.129m" result was earlier this session.
bool EvaluateTrajectory(const std::vector<TumPose> &traj, const std::vector<double> &timestamps,
                         const std::vector<Point2> &gt, const std::string &label)
{
    if (traj.empty()) {
        std::printf("[%s] No trajectory -- did tracking ever start?\n", label.c_str());
        return false;
    }

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
        std::printf("[%s] Alignment failed -- too few matched points (%zu) or degenerate estimate.\n",
                     label.c_str(), src.size());
        return false;
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

    // Path length from the MATCHED ground-truth points only, in the order
    // they were matched (== trajectory/timestamp order) -- not the full
    // probed region, which is what made the earlier "0.129m" result
    // meaningless (an 8-point, ~30m fragment scored against a 3722m
    // denominator that had nothing to do with what was actually compared).
    double pathLength = 0.0;
    for (size_t i = 1; i < dst.size(); ++i)
        pathLength += std::hypot(dst[i].x - dst[i - 1].x, dst[i].z - dst[i - 1].z);

    // Ground-truth bounding box of the matched points -- lets a later pass
    // check whether two fragments' physical footprints overlap (a real
    // revisit map-merge could exploit) or are disjoint (purely sequential
    // new road, nothing for place-recognition-based merge to find). See
    // DEBUGGING.md part 16.
    double gtMinX = dst[0].x, gtMaxX = dst[0].x, gtMinZ = dst[0].z, gtMaxZ = dst[0].z;
    for (const auto &p : dst) {
        gtMinX = std::min(gtMinX, p.x);
        gtMaxX = std::max(gtMaxX, p.x);
        gtMinZ = std::min(gtMinZ, p.z);
        gtMaxZ = std::max(gtMaxZ, p.z);
    }

    std::printf("[%s] matched=%zu/%zu scale=%.3f pathLen=%.1fm ATE(rmse/mean/median/max)=%.3f/%.3f/%.3f/%.3fm (%.2f%% of path) "
                "gtBBox=x[%.1f,%.1f] z[%.1f,%.1f] start=(%.1f,%.1f) end=(%.1f,%.1f)\n",
                label.c_str(), src.size(), traj.size(), scale, pathLength, ateRmse, ateMean, ateMedian, maxErr,
                pathLength > 1e-6 ? 100.0 * ateRmse / pathLength : 0.0,
                gtMinX, gtMaxX, gtMinZ, gtMaxZ, dst.front().x, dst.front().z, dst.back().x, dst.back().z);
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 5) {
        std::fprintf(stderr,
                      "usage: %s <path_to_vocabulary> <path_to_settings> <kitti-sequence-dir> "
                      "<kitti-poses.txt> [out-prefix] [start-frame] [realtime] [max-frames]\n"
                      "  kitti-sequence-dir: e.g. .../sequences/00 (must contain times.txt, image_0/)\n"
                      "  kitti-poses.txt: e.g. .../poses/00.txt\n"
                      "  out-prefix: writes <prefix>_KeyFrameTrajectory.txt, default 'orbslam3_kitti_ate'\n"
                      "  start-frame: skip this many leading frames, re-initializing tracking from\n"
                      "               scratch at that point instead of replaying the whole sequence --\n"
                      "               DIAGNOSTIC ONLY, ATE against ground truth is still meaningful for\n"
                      "               the remaining frames but ignores the skipped ones. Default 0.\n"
                      "  realtime: 1 to pace frames to match each frame's recorded KITTI timestamp\n"
                      "            delta (like mono_kitti.cc's own throttle, deliberately stripped from\n"
                      "            this tool by default -- see the file's top comment), instead of\n"
                      "            feeding frames as fast as the CPU can process them. Default 0.\n"
                      "  max-frames: stop after this many frames (counted after start-frame is applied) --\n"
                      "              for fast checks on a specific stretch instead of replaying to the end\n"
                      "              of the sequence. Default 0 (no limit, run to the end).\n",
                      argv[0]);
        return 1;
    }
    const std::string vocabPath = argv[1];
    const std::string settingsPath = argv[2];
    const std::string sequencePath = argv[3];
    const std::string posesPath = argv[4];
    const std::string outPrefix = argc > 5 ? argv[5] : "orbslam3_kitti_ate";
    const size_t startFrame = argc > 6 ? static_cast<size_t>(std::stoul(argv[6])) : 0;
    const bool realtime = argc > 7 && std::atoi(argv[7]) != 0;
    const size_t maxFrames = argc > 8 ? static_cast<size_t>(std::stoul(argv[8])) : 0;

    // Part 53: tried cv::setNumThreads(1) here to rule out OpenCV-internal
    // parallelism as a nondeterminism source on top of the DUtils::Random
    // mutex fix (kept, see Random.h/.cpp) -- measured INCONCLUSIVE (two
    // otherwise-identical single-threaded runs still gave 85 vs 109 fails on
    // the same 541-frame hot zone, roughly the same spread as before this
    // change). The remaining nondeterminism is almost certainly rooted in
    // this project's own multi-threaded architecture (Tracking/LocalMapping/
    // LoopClosing racing on real wall-clock-timing decisions like "is local
    // mapping idle") -- not practical to eliminate without full per-frame
    // thread synchronization, which would defeat the point of having
    // separate threads. Reverted (this cost ~50% CPU throughput without a
    // clear determinism payoff). See DEBUGGING.md part 53 -- comparisons
    // going forward use repeat trials, not single runs.

    std::vector<std::string> imageFiles;
    std::vector<double> timestamps;
    LoadImages(sequencePath, imageFiles, timestamps);
    if (imageFiles.empty()) {
        std::fprintf(stderr, "Failed to load images from %s\n", sequencePath.c_str());
        return 1;
    }
    if (startFrame > 0) {
        if (startFrame >= imageFiles.size()) {
            std::fprintf(stderr, "start-frame %zu >= %zu total frames\n", startFrame, imageFiles.size());
            return 1;
        }
        imageFiles.erase(imageFiles.begin(), imageFiles.begin() + startFrame);
        timestamps.erase(timestamps.begin(), timestamps.begin() + startFrame);
        std::fprintf(stderr, "[config] skipped %zu leading frames (diagnostic start-frame)\n", startFrame);
    }
    if (maxFrames > 0 && maxFrames < imageFiles.size()) {
        imageFiles.resize(maxFrames);
        timestamps.resize(maxFrames);
        std::fprintf(stderr, "[config] capped to %zu frames (diagnostic max-frames)\n", maxFrames);
    }
    std::fprintf(stderr, "[config] %zu images loaded from %s\n", imageFiles.size(), sequencePath.c_str());

    // useViewer=false: this is a headless batch tool, not the interactive
    // GUI mono_kitti.cc is meant for -- Pangolin is still linked (System
    // unconditionally depends on it), it just never opens a window.
    ORB_SLAM3::System SLAM(vocabPath, settingsPath, ORB_SLAM3::System::MONOCULAR, false);
    const float imageScale = SLAM.GetImageScale();

    std::fprintf(stderr, "[config] tracking %zu frames (%s)...\n", imageFiles.size(),
                  realtime ? "real-time paced to KITTI timestamps" : "unthrottled, no real-time pacing");
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        cv::Mat im = cv::imread(imageFiles[i], cv::IMREAD_UNCHANGED);
        if (im.empty()) {
            std::fprintf(stderr, "Failed to load image at: %s\n", imageFiles[i].c_str());
            return 1;
        }
        if (imageScale != 1.f)
            cv::resize(im, im, cv::Size(static_cast<int>(im.cols * imageScale), static_cast<int>(im.rows * imageScale)));

        const auto trackStart = std::chrono::steady_clock::now();
        SLAM.TrackMonocular(im, timestamps[i], std::vector<ORB_SLAM3::IMU::Point>(), imageFiles[i]);

        // Real-time pacing (opt-in, see usage text): mirrors mono_kitti.cc's
        // own throttle, which this tool otherwise deliberately strips -- lets
        // LocalMapping/LoopClosing's background threads see the same
        // wall-clock cadence they'd get from a live camera instead of being
        // flooded as fast as Tracking alone can go.
        if (realtime && i + 1 < imageFiles.size()) {
            const double trackSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - trackStart).count();
            const double frameSec = timestamps[i + 1] - timestamps[i];
            if (trackSec < frameSec)
                std::this_thread::sleep_for(std::chrono::duration<double>(frameSec - trackSec));
        }

        if (i % 500 == 0) {
            std::fprintf(stderr, "[stats] frame %zu / %zu\n", i, imageFiles.size());
            std::fflush(stderr);
        }
    }

    SLAM.Shutdown();

    const std::vector<Point2> gt = loadGroundTruth(posesPath);
    if (gt.empty()) {
        std::fprintf(stderr, "Failed to load ground truth from %s\n", posesPath.c_str());
        return 1;
    }

    // Per-fragment evaluation, not just the current-map-only export: see
    // DEBUGGING.md -- SaveKeyFrameTrajectoryTUM() (via
    // Atlas::GetAllKeyFrames(), which returns ONLY mpCurrentMap's keyframes,
    // confirmed by reading Atlas.cc directly) silently discards every
    // earlier map fragment the tracker abandoned, whether via a destructive
    // ResetActiveMap() or the non-destructive CreateMapInAtlas() path. A run
    // can track perfectly well for hundreds of frames in an earlier
    // fragment and still show "no trajectory" if the LAST fragment at
    // shutdown time is young. Evaluate every fragment separately (each is
    // its own arbitrary monocular scale/coordinate frame, so they can't be
    // concatenated into one trajectory -- each needs its own alignment).
    std::vector<ORB_SLAM3::Map*> maps = SLAM.GetAtlas()->GetAllMaps();
    std::fprintf(stderr, "[atlas-coverage] %zu map fragment(s) in Atlas at shutdown\n", maps.size());
    size_t totalKFs = 0;
    bool anyScored = false;
    for (size_t m = 0; m < maps.size(); ++m) {
        std::vector<ORB_SLAM3::KeyFrame*> kfs = maps[m]->GetAllKeyFrames();
        std::sort(kfs.begin(), kfs.end(), ORB_SLAM3::KeyFrame::lId);
        totalKFs += kfs.size();

        std::vector<TumPose> traj;
        traj.reserve(kfs.size());
        for (auto *kf : kfs) {
            if (kf->isBad())
                continue;
            Sophus::SE3f Twc = kf->GetPoseInverse();
            Eigen::Vector3f t = Twc.translation();
            traj.push_back({kf->mTimeStamp, t(0), t(2)});
        }

        const std::string label = "fragment " + std::to_string(m) + " (" + std::to_string(traj.size()) + " KFs)";
        if (EvaluateTrajectory(traj, timestamps, gt, label))
            anyScored = true;
    }
    std::fprintf(stderr, "[atlas-coverage] total keyframes across all fragments: %zu\n", totalKFs);

    // Also still write the "official" current-map-only export other tooling
    // may expect (unchanged behavior/path from before this session).
    const std::string trajPath = outPrefix + "_KeyFrameTrajectory.txt";
    SLAM.SaveKeyFrameTrajectoryTUM(trajPath);

    return anyScored ? 0 : 1;
}
