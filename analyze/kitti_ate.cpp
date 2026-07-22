// Runs SlamWorker directly against a raw KITTI image sequence (no video
// re-encoding involved) and computes Absolute Trajectory Error (ATE)
// against KITTI's ground-truth poses.txt, natively in C++ -- no Python.
//
// Alignment math mirrors MapView::computeAlignment() (src/widgets/MapView.cpp):
// monocular SLAM's world frame has an arbitrary origin/rotation/scale, so the
// estimated trajectory is fit to ground truth via a least-squares 2D
// similarity transform (Umeyama 1991: scale + rotation + translation) before
// computing error.

#include <QCoreApplication>
#include <QFile>
#include <QRegularExpression>
#include <QTextStream>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv2/calib3d.hpp>

#include "vision/LoopEstimator.h"
#include "vision/PoseGraphOptimizer.h"
#include "vision/SlamWorker.h"

namespace {

struct Point2
{
    double x = 0.0, z = 0.0;
};

std::vector<Point2> loadGroundTruth(const QString &path)
{
    std::vector<Point2> gt;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return gt;

    QTextStream stream(&file);
    static const QRegularExpression whitespace(QStringLiteral("\\s+"));
    while (!stream.atEnd()) {
        const QStringList parts = stream.readLine().split(whitespace, Qt::SkipEmptyParts);
        if (parts.size() < 12)
            continue;
        // Row-major 3x4 [R|t]: columns 3, 11 are tx, tz -- the camera
        // center in the ground-truth world frame's X/Z (top-down) plane.
        gt.push_back({parts[3].toDouble(), parts[11].toDouble()});
    }
    return gt;
}

// Closed-form least-squares similarity fit (2D Umeyama 1991): finds
// (scale, rotation, translation) minimizing sum||s*R(theta)*src_i + t -
// dst_i||^2 over paired points. Returns false if too few pairs or a
// degenerate (near-zero-spread) source set.
bool umeyama2D(const std::vector<Point2> &src, const std::vector<Point2> &dst, double &scale,
               double &cosTheta, double &sinTheta, double &tx, double &tz)
{
    if (src.size() < 8 || src.size() != dst.size())
        return false;

    const double n = static_cast<double>(src.size());
    double meanSrcX = 0.0, meanSrcZ = 0.0, meanDstX = 0.0, meanDstZ = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        meanSrcX += src[i].x; meanSrcZ += src[i].z;
        meanDstX += dst[i].x; meanDstZ += dst[i].z;
    }
    meanSrcX /= n; meanSrcZ /= n; meanDstX /= n; meanDstZ /= n;

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

} // namespace

int main(int argc, char *argv[])
{
    // NOTE: this session (2026-07-21) confirmed the SAME binary/config/
    // input on full KITTI seq00 produced materially different ATE across
    // separate runs (93.851m, 127.412m, 138.464m, 150.514m). Root cause:
    // OpenCV's SIFT detection runs cv::parallel_for_ internally (across
    // pyramid octaves), so the ORDER kps/descriptors come back in was
    // run-dependent even though the SET found was identical -- a different
    // order fed a different actual correspondence subset to the
    // fixed-seed (kRansacSeed) RANSAC, which could flip a threshold-
    // boundary loop-closure accept/reject decision and cascade through
    // thousands of downstream frames. Fixed at the source instead of here:
    // SlamWorker.cpp's processNext() now sorts kps/descriptors into a
    // fixed, position-based order right after detectAndCompute(), so
    // multi-threaded detection speed is kept (a blanket
    // cv::setNumThreads(1) here measured >2x slower per frame and was
    // reverted). ceres::Solver::Options::num_threads is still pinned to 1
    // in SlamWorker.cpp/PoseGraphOptimizer.cpp as cheap extra insurance
    // against BA's own reduction-order sensitivity, which costs
    // negligible speed since BA isn't the per-frame bottleneck.
    QCoreApplication app(argc, argv);

    if (argc < 3) {
        std::fprintf(stderr,
                      "usage: %s <kitti-image-pattern> <kitti-poses.txt> [seconds] [pnp-method] "
                      "[out-prefix] [estimator] [oxts-dir] [calib-dir]\n",
                      argv[0]);
        std::fprintf(stderr,
                      "  kitti-image-pattern: e.g. .../sequences/00/image_0/%%06d.png\n"
                      "  pnp-method: p3p (default), dlt, iterative, epnp, ap3p, sqpnp\n"
                      "  out-prefix: writes <prefix>_trajectory.txt (frame x z x_aligned z_aligned "
                      "gt_x gt_z), default 'kitti_ate'\n"
                      "  estimator: fivepoint (default, direct calibrated Nister solver) or "
                      "eightpoint (legacy 8-point+Gold-Standard, see EightPointLegacy.h)\n"
                      "  oxts-dir: path to a KITTI OXTS folder (timestamps.txt + data/*.txt) to "
                      "replace vision-only scale heuristics with real forward-velocity-derived "
                      "metric distance; omit to disable\n"
                      "  calib-dir: path to a KITTI raw-data per-date calibration folder "
                      "(calib_imu_to_velo.txt, calib_velo_to_cam.txt, calib_cam_to_cam.txt) -- "
                      "MUST be the same date as oxts-dir's drive. Enables using real IMU "
                      "orientation instead of homography decomposition for near-pure-rotation "
                      "frames (see ImuRotation.h). Requires oxts-dir to also be set; omit to "
                      "disable\n"
                      "  groundplane: pass the literal word 'groundplane' to enable VISO2-M-style "
                      "vision-only ground-plane scale correction as a fallback wherever OXTS isn't "
                      "available (see GroundPlaneScale.h). Use '-' for oxts-dir/calib-dir to skip "
                      "them while still passing this.\n"
                      "  ba: pass the literal word 'ba' to enable real bundle adjustment after loop "
                      "closure (see SlamWorker::runLoopBundleAdjustment()) instead of the default "
                      "interpolated correction. Use '-' for groundplane to skip it while still "
                      "passing this.\n"
                      "  oxtsimupnp: pass the literal word 'oxtsimupnp' to feed OXTS/IMU into "
                      "trackFrame()'s plausibility check too (see "
                      "SlamWorker::setOxtsImuInPnpEnabled()). Use '-' for ba to skip it while still "
                      "passing this.\n"
                      "  posegraph: pass the literal word 'posegraph' to run an offline, one-shot "
                      "pose-graph optimization (sequential + loop-closure edges, Dynamic Covariance "
                      "Scaling on loop edges -- see PoseGraphOptimizer.h) after tracking finishes, and "
                      "print its corrected ATE alongside the raw-live one. Use '-' for oxtsimupnp/ba/"
                      "groundplane to skip them while still passing this.\n"
                      "  [argv14] detector: pass the literal word 'orb' to use ORB instead of SIFT (see "
                      "FeatureDetector.h) -- default is SIFT if omitted or any other value is passed.\n"
                      "  [argv15] mutualmatch: pass the literal word 'mutualmatch' to require a B->A "
                      "nearest-neighbor cross-check on top of the existing A->B ratio test (see "
                      "SlamWorker::setMutualMatchEnabled()) -- default off.\n"
                      "  [argv16] orb-features: ORB only, overrides OrbSettings::nFeatures (default 2000).\n"
                      "  [argv17]/[argv18] dbow <vocab-path>: ORB only, pass the literal word 'dbow' "
                      "followed by a DBoW2 ORB vocabulary path (e.g. ORBvoc.txt) to enable DBoW2 "
                      "place-recognition scoring for the loop-closure candidate search (see "
                      "SlamWorker::setDbowLoopClosureEnabled()) instead of the default raw-match-count "
                      "search.\n"
                      "  [argv19] sim3: requires 'posegraph' to also be passed -- pass the literal word "
                      "'sim3' to switch the offline pose-graph solve from rigid SE(3) edges to Sim(3) "
                      "(adds a free per-keyframe scale DOF, see PoseGraphOptions::useSim3) instead of "
                      "the default SE(3) edges.\n"
                      "  [argv20] cull: pass the literal word 'cull' to enable covisibility-graph-based "
                      "keyframe culling (see SlamWorker::setKeyframeCullingEnabled()) -- detector-"
                      "agnostic, works with either SIFT or ORB.\n"
                      "  [argv22]/[argv23] pnp-reproj-error/pnp-iterations: overrides "
                      "PnpSettings::reprojectionError (default 8.0px) / iterationsCount (default 100) "
                      "for trackFrame()'s live PnP RANSAC.\n"
                      "  [argv24] min-track-inliers: overrides trackFrame()'s minimum-accepted-inlier-"
                      "count gate (default 10, see SlamWorker::setMinTrackInliers()).\n"
                      "  [argv30]/[argv31] vlad <codebook-path>: SIFT only, pass the literal word 'vlad' "
                      "followed by a VLAD codebook path (e.g. "
                      "vocabulary_sift/vlad_codebook_all_rootsift.yml) to enable VLAD place-recognition "
                      "scoring for the loop-closure candidate search (see "
                      "SlamWorker::setVladLoopClosureEnabled()) instead of the default raw-match-count "
                      "search.\n"
                      "  [argv33] sift-features: SIFT only, overrides SiftSettings::nFeatures (default "
                      "2000).\n"
                      "  [argv34] sift-detection-scale: SIFT only, overrides SlamWorker's detection "
                      "resolution fraction (default 0.5, half-res; 1.0 = full-res).\n"
                      "  [argv36]/[argv37] siftdbow <vocab-path>: SIFT only, pass the literal word "
                      "'siftdbow' followed by a DBoW2 vocabulary path trained via "
                      "analyze/train_sift_dbow_vocabulary.cpp to enable real TF-IDF place-recognition "
                      "scoring for the loop-closure candidate search (see "
                      "SlamWorker::setSiftDbowLoopClosureEnabled()) -- checked before 'vlad' when both "
                      "are passed.\n"
                      "  [argv38] loopconsistency: pass the literal word 'loopconsistency' to require a "
                      "loop candidate be re-verified across several consecutive-ish keyframe insertions "
                      "before its correction is applied (see "
                      "SlamWorker::setLoopConsistencyGroupEnabled()).\n"
                      "  [argv39] fuse: pass the literal word 'fuse' to extend landmark observation "
                      "coverage into nearby keyframes via real projection+keypoint matching (see "
                      "SlamWorker::setLandmarkFuseEnabled()) -- RECOMMENDED, measured +29%% ATE win.\n"
                      "  [argv40] fusemerge: pass the literal word 'fusemerge' (requires 'fuse') to ALSO "
                      "merge genuine keypoint-conflicts (see SlamWorker::setLandmarkFuseMergeEnabled()) "
                      "-- NOT RECOMMENDED, measured negative, see DEBUGGING.md item 20.\n"
                      "  [argv41] loopspatialconsensus: pass the literal word 'loopspatialconsensus' to "
                      "require within-call consensus among independently-scored loop candidates (see "
                      "SlamWorker::setLoopSpatialConsensusEnabled()).\n"
                      "  [argv42] local-ba-window: overrides SlamWorker::setLocalBaWindowKeyframes() "
                      "(default 8).\n"
                      "  [argv43] globalbaasync: requires 'globalba' -- pass the literal word "
                      "'globalbaasync' to enable SlamWorker::setGlobalBundleAdjustmentAsyncEnabled(), a "
                      "deterministic stand-in for real ORB-SLAM3's background-thread global BA + "
                      "spanning-tree correction propagation (see DEBUGGING.md item 29).\n");
        return 1;
    }
    const QString imagePattern = QString::fromLocal8Bit(argv[1]);
    const QString posesPath = QString::fromLocal8Bit(argv[2]);
    const int seconds = argc > 3 ? std::atoi(argv[3]) : 600;
    const QString outPrefix = argc > 5 ? QString::fromLocal8Bit(argv[5]) : QStringLiteral("kitti_ate");

    SlamWorker worker;

    if (argc > 4) {
        PnpSettings settings;
        const char *m = argv[4];
        if (std::strcmp(m, "dlt") == 0)
            settings.method = kPnpMethodDlt;
        else if (std::strcmp(m, "epnp") == 0)
            settings.method = cv::SOLVEPNP_EPNP;
        else if (std::strcmp(m, "iterative") == 0)
            settings.method = cv::SOLVEPNP_ITERATIVE;
        else if (std::strcmp(m, "ap3p") == 0)
            settings.method = cv::SOLVEPNP_AP3P;
        else if (std::strcmp(m, "sqpnp") == 0)
            settings.method = cv::SOLVEPNP_SQPNP;
        else
            settings.method = cv::SOLVEPNP_P3P;

        // argv[22]/argv[23]: optional overrides for trackFrame()'s live PnP
        // RANSAC (PnpSettings::reprojectionError default 8.0px,
        // iterationsCount default 100) -- tightening these attacks the
        // actual front-end tracking-accuracy bottleneck directly (this
        // session's back-end work -- DBoW2, culling, Sim(3)/g2o -- can only
        // clean up drift after the fact, it can't fix bad per-frame poses
        // trackFrame() itself produced). Left as opt-in CLI overrides
        // (default PnpSettings unchanged) so the documented SIFT+P3P
        // 17.141m baseline stays exactly reproducible.
        if (argc > 22) {
            const double reprojErr = std::atof(argv[22]);
            if (reprojErr > 0.0) {
                settings.reprojectionError = reprojErr;
                std::fprintf(stderr, "[config] PnP reprojectionError=%.2f\n", reprojErr);
            }
        }
        if (argc > 23) {
            const int iters = std::atoi(argv[23]);
            if (iters > 0) {
                settings.iterationsCount = iters;
                std::fprintf(stderr, "[config] PnP iterationsCount=%d\n", iters);
            }
        }

        worker.setPnpSettings(settings);
        std::fprintf(stderr, "[config] pnp method=%s\n", m);
    }

    if (argc > 6) {
        const char *est = argv[6];
        if (std::strcmp(est, "eightpoint") == 0) {
            worker.setTwoViewEstimator(SlamWorker::TwoViewEstimator::EightPointLegacy);
            std::fprintf(stderr, "[config] two-view estimator=eightpoint (legacy)\n");
        } else {
            std::fprintf(stderr, "[config] two-view estimator=fivepoint (default)\n");
        }
    }

    if (argc > 14 && std::strcmp(argv[14], "orb") == 0) {
        worker.setDetectorType(feature_detector::DetectorType::Orb);
        std::fprintf(stderr, "[config] feature detector=ORB\n");

        // argv[16], ORB only: override OrbSettings::nFeatures (default 2000).
        // Added after observing two-view (E/H) recovery repeatedly starving
        // below kMinInitInliers/kMinInitMatches (min=20) on this sequence --
        // a plain candidate-count shortage, not a step-scale or match-
        // quality problem -- so more candidate keypoints per frame is the
        // direct lever to try before anything more invasive.
        if (argc > 16) {
            const int nFeatures = std::atoi(argv[16]);
            if (nFeatures > 0) {
                feature_detector::OrbSettings orbSettings;
                orbSettings.nFeatures = nFeatures;
                worker.setOrbSettings(orbSettings);
                std::fprintf(stderr, "[config] ORB nFeatures=%d\n", nFeatures);
            }
        }

        // argv[17]/argv[18], ORB only: 'dbow' + a vocabulary path enables
        // DBoW2 place-recognition scoring for tryLoopClosure()'s candidate
        // search (see SlamWorker::setDbowLoopClosureEnabled()) instead of
        // the original raw-descriptor-match-count search.
        if (argc > 18 && std::strcmp(argv[17], "dbow") == 0) {
            if (worker.loadOrbVocabulary(QString::fromLocal8Bit(argv[18]))) {
                worker.setDbowLoopClosureEnabled(true);
                std::fprintf(stderr, "[config] DBoW2 loop-closure candidate search enabled\n");
            } else {
                std::fprintf(stderr, "[config] WARNING: failed to load ORB vocabulary from %s -- "
                                      "continuing with the raw-match-count loop search\n", argv[18]);
            }
        }
    } else {
        std::fprintf(stderr, "[config] feature detector=SIFT (default)\n");

        // argv[33], SIFT only: override SiftSettings::nFeatures (default
        // 2000). Same lever as argv[16]'s ORB equivalent -- denser
        // per-frame keypoints feed denser PnP/BA constraints (queued next
        // step after Session 15's local-BA/loop-BA observation-density
        // fixes, which unlocked real multi-view constraint density but are
        // still capped by how many candidate keypoints exist per frame in
        // the first place).
        if (argc > 33) {
            const int nFeatures = std::atoi(argv[33]);
            if (nFeatures > 0) {
                SiftSettings siftSettings;
                siftSettings.nFeatures = nFeatures;
                worker.setSiftSettings(siftSettings);
                std::fprintf(stderr, "[config] SIFT nFeatures=%d\n", nFeatures);
            }
        }

        // argv[34], SIFT only (ORB's own detector already runs at full
        // resolution -- see FeatureDetector.h): overrides
        // SlamWorker::setDetectionScale() (default 0.5, half-resolution).
        // argv[33]'s nFeatures bump was confirmed inert -- per-frame
        // detection never approached even the 2000 default cap (max 1591,
        // avg 754.9 over full seq00) -- so this is the actual lever for
        // denser keypoints: raising resolution toward 1.0 (full-res) gives
        // the pyramid more real image detail to find candidates in. Only
        // safe to try in an unthrottled harness like this one (see
        // startUnthrottled() below) -- the live GUI's default stays 0.5
        // for its own real-time budget, untouched by this override.
        if (argc > 34) {
            const double detectionScale = std::atof(argv[34]);
            if (detectionScale > 0.0 && detectionScale <= 1.0) {
                worker.setDetectionScale(detectionScale);
                std::fprintf(stderr, "[config] SIFT detectionScale=%.2f\n", detectionScale);
            }
        }
    }

    if (argc > 15 && std::strcmp(argv[15], "mutualmatch") == 0) {
        worker.setMutualMatchEnabled(true);
        std::fprintf(stderr, "[config] mutual (cross-check) descriptor matching enabled\n");
    }

    // argv[20]: detector-agnostic (covisibility is derived from
    // m_landmarkObservations, not from descriptor type) -- pass the
    // literal word 'cull' to enable covisibility-graph-based keyframe
    // culling (see SlamWorker::setKeyframeCullingEnabled()).
    if (argc > 20 && std::strcmp(argv[20], "cull") == 0) {
        worker.setKeyframeCullingEnabled(true);
        std::fprintf(stderr, "[config] covisibility-graph keyframe culling enabled\n");
    }

    // argv[24]: overrides trackFrame()'s minimum-accepted-inlier-count gate
    // (default 10, see SlamWorker::setMinTrackInliers()) -- detector-agnostic.
    if (argc > 24) {
        const int minInliers = std::atoi(argv[24]);
        if (minInliers > 0) {
            worker.setMinTrackInliers(minInliers);
            std::fprintf(stderr, "[config] minTrackInliers=%d\n", minInliers);
        }
    }

    // argv[25]: pass the literal word 'localba' to enable continuous joint
    // local BA on every keyframe insertion (see
    // SlamWorker::setLocalBundleAdjustmentEnabled()) instead of the default
    // per-keyframe-only refineLocalKeyframes() polish.
    if (argc > 25 && std::strcmp(argv[25], "localba") == 0) {
        worker.setLocalBundleAdjustmentEnabled(true);
        std::fprintf(stderr, "[config] continuous local bundle adjustment enabled\n");
    }

    // argv[26]: pass the literal word 'globalba' to enable full global BA
    // (every keyframe/landmark from 0 to the loop-closure keyframe) after
    // loop closure (see SlamWorker::setGlobalBundleAdjustmentEnabled()) --
    // takes priority over 'ba' (the windowed loop BA) when both are passed.
    if (argc > 26 && std::strcmp(argv[26], "globalba") == 0) {
        worker.setGlobalBundleAdjustmentEnabled(true);
        std::fprintf(stderr, "[config] full global bundle adjustment after loop closure enabled\n");
    }

    // argv[27]: pass the literal word 'covismap' to enable the
    // covisibility-driven local map for tracking (see
    // SlamWorker::setCovisibilityLocalMapEnabled()) instead of the default
    // flat rolling map.
    if (argc > 27 && std::strcmp(argv[27], "covismap") == 0) {
        worker.setCovisibilityLocalMapEnabled(true);
        std::fprintf(stderr, "[config] covisibility-driven local map for tracking enabled\n");
    }

    // argv[28]: pass the literal word 'guided' to enable projection-based
    // match filtering (see SlamWorker::setGuidedSearchEnabled()).
    if (argc > 28 && std::strcmp(argv[28], "guided") == 0) {
        worker.setGuidedSearchEnabled(true);
        std::fprintf(stderr, "[config] guided/projection-based search filter enabled\n");
    }

    // argv[29]: pass the literal word 'qualitykf' to enable quality-driven
    // keyframe insertion (see SlamWorker::setQualityDrivenKeyframesEnabled())
    // instead of the default fixed kKeyframeEveryNFrames interval.
    if (argc > 29 && std::strcmp(argv[29], "qualitykf") == 0) {
        worker.setQualityDrivenKeyframesEnabled(true);
        std::fprintf(stderr, "[config] quality-driven keyframe insertion enabled\n");
    }

    // argv[30]/argv[31]: pass the literal word 'vlad' followed by a VLAD
    // codebook path (e.g. vocabulary_sift/vlad_codebook_all_rootsift.yml,
    // see SlamWorker::loadVladVocabulary()) to enable VLAD place-recognition
    // scoring for tryLoopClosure()'s candidate search (see
    // SlamWorker::setVladLoopClosureEnabled()) -- the SIFT-compatible
    // counterpart to 'dbow' above. Detector-specific like dbow: meaningful
    // only when SIFT (the default) is the active detector, and only if the
    // codebook was trained on the same descriptor space this build produces
    // (RootSIFT after FeatureDetector.h's toRootSift() -- the
    // "_rootsift" codebook, not the plain vlad_codebook_all.yml).
    if (argc > 31 && std::strcmp(argv[30], "vlad") == 0) {
        if (worker.loadVladVocabulary(QString::fromLocal8Bit(argv[31]))) {
            worker.setVladLoopClosureEnabled(true);
            std::fprintf(stderr, "[config] VLAD loop-closure candidate search enabled\n");
        } else {
            std::fprintf(stderr, "[config] WARNING: failed to load VLAD codebook from %s -- "
                                  "continuing with the raw-match-count loop search\n", argv[31]);
        }
    }

    // argv[36]/argv[37]: pass the literal word 'siftdbow' followed by a
    // vocabulary path (see analyze/train_sift_dbow_vocabulary.cpp,
    // SlamWorker::loadSiftVocabulary()) to enable real DBoW2/TF-IDF
    // place-recognition scoring for tryLoopClosure()'s candidate search,
    // as a second SIFT-compatible alternative to 'vlad' above (checked
    // first when both are passed, see setSiftDbowLoopClosureEnabled()'s
    // own doc comment). Detector-specific like vlad/dbow: meaningful only
    // when SIFT is the active detector, and only if the vocabulary was
    // trained on this build's own RootSIFT descriptor space.
    if (argc > 37 && std::strcmp(argv[36], "siftdbow") == 0) {
        if (worker.loadSiftVocabulary(QString::fromLocal8Bit(argv[37]))) {
            worker.setSiftDbowLoopClosureEnabled(true);
            std::fprintf(stderr, "[config] SIFT DBoW2 loop-closure candidate search enabled\n");
        } else {
            std::fprintf(stderr, "[config] WARNING: failed to load SIFT DBoW2 vocabulary from %s -- "
                                  "continuing with VLAD/raw-match-count loop search\n", argv[37]);
        }
    }

    // argv[38]: pass the literal word 'loopconsistency' to require a loop
    // candidate be independently re-verified across several consecutive-ish
    // keyframe insertions before its correction is actually applied (see
    // SlamWorker::setLoopConsistencyGroupEnabled() -- a simplified
    // adaptation of real ORB-SLAM3's mnLoopNumCoincidences>=3 gate).
    // Detector-agnostic: applies to whichever candidate-search backend
    // (siftdbow/vlad/dbow/raw-match-count) is active.
    if (argc > 38 && std::strcmp(argv[38], "loopconsistency") == 0) {
        worker.setLoopConsistencyGroupEnabled(true);
        std::fprintf(stderr, "[config] loop-closure temporal-consistency gate enabled\n");
    }

    // argv[39]: pass the literal word 'fuse' to enable fuseWindowLandmarks()
    // Phase A -- a simplified adaptation of real ORB-SLAM3's
    // LocalMapping::SearchInNeighbors()/ORBmatcher::Fuse() (see
    // SlamWorker::setLandmarkFuseEnabled()): extends a just-triangulated
    // landmark's observation coverage into nearby keyframes via real
    // projection + reprojection-gated descriptor matching against their
    // own detected keypoints. Measured real win (DEBUGGING.md item 19):
    // 72.550m -> 51.273m. Detector-agnostic, RECOMMENDED.
    if (argc > 39 && std::strcmp(argv[39], "fuse") == 0) {
        worker.setLandmarkFuseEnabled(true);
        std::fprintf(stderr, "[config] landmark fuse (Phase A: coverage extension) enabled\n");
    }

    // argv[40]: pass the literal word 'fusemerge' to ALSO enable Phase B
    // (see SlamWorker::setLandmarkFuseMergeEnabled()) -- genuine-conflict
    // merging on top of Phase A's coverage extension. Requires 'fuse' to
    // also be passed. NOT RECOMMENDED: measured negative (DEBUGGING.md
    // item 20), 51.273m -> 161.117m, loop closures dropped 71->33 (merging
    // leaves Keyframe::localMapPoints/localMapPointIds stale, corrupting
    // tryLoopClosure()'s own PnP/Sim3Solver measurement). Kept available
    // for anyone who wants to fix that synchronization gap and re-measure.
    if (argc > 40 && std::strcmp(argv[40], "fusemerge") == 0) {
        worker.setLandmarkFuseMergeEnabled(true);
        std::fprintf(stderr, "[config] landmark fuse Phase B (genuine-conflict merging) enabled -- "
                              "NOT RECOMMENDED, see DEBUGGING.md item 20\n");
    }

    // argv[41]: pass the literal word 'loopspatialconsensus' to enable
    // setLoopSpatialConsensusEnabled() -- a DIFFERENT mechanism from
    // 'loopconsistency' (see its own doc comment): checks consensus among
    // several independently-scored candidates within a SINGLE
    // tryLoopClosure() call, instead of requiring re-confirmation ACROSS
    // separate calls (which items 15/21/22 showed doesn't work on this
    // pipeline's own detection cadence). Detector-agnostic.
    if (argc > 41 && std::strcmp(argv[41], "loopspatialconsensus") == 0) {
        worker.setLoopSpatialConsensusEnabled(true);
        std::fprintf(stderr, "[config] loop-closure spatial-consensus gate enabled\n");
    }

    // argv[42]: overrides SlamWorker::setLocalBaWindowKeyframes() (default
    // 8) -- untested at any other value before this session.
    if (argc > 42) {
        const int windowKeyframes = std::atoi(argv[42]);
        if (windowKeyframes > 0) {
            worker.setLocalBaWindowKeyframes(windowKeyframes);
            std::fprintf(stderr, "[config] local BA window keyframes=%d\n", windowKeyframes);
        }
    }

    // argv[43]: pass the literal word 'globalbaasync' (requires 'globalba')
    // to enable SlamWorker::setGlobalBundleAdjustmentAsyncEnabled() --
    // DEBUGGING.md item 29's deterministic stand-in for real ORB-SLAM3's
    // background-thread global BA + spanning-tree correction propagation.
    if (argc > 43 && std::strcmp(argv[43], "globalbaasync") == 0) {
        worker.setGlobalBundleAdjustmentAsyncEnabled(true);
        std::fprintf(stderr, "[config] global BA async (deferred-integration simulation) enabled\n");
    }

    if (argc > 9 && std::strcmp(argv[9], "groundplane") == 0) {
        worker.setGroundPlaneEnabled(true);
        std::fprintf(stderr, "[config] ground-plane scale correction enabled (VISO2-M-style fallback)\n");
    }

    if (argc > 10 && std::strcmp(argv[10], "ba") == 0) {
        worker.setLoopBundleAdjustmentEnabled(true);
        std::fprintf(stderr, "[config] real bundle adjustment after loop closure enabled\n");
    }

    if (argc > 11 && std::strcmp(argv[11], "oxtsimupnp") == 0) {
        worker.setOxtsImuInPnpEnabled(true);
        std::fprintf(stderr, "[config] OXTS/IMU-aware PnP tracking plausibility check enabled\n");
    }

    if (argc > 12 && std::strcmp(argv[12], "loopestimate") == 0) {
        std::fprintf(stderr, "[config] background loop-estimate diagnostic enabled\n");
        QObject::connect(&worker, &SlamWorker::loopClosureDetected, [](LoopEstimateSnapshot snapshot) {
            const int oldF = snapshot.oldFrameIndex, newF = snapshot.newFrameIndex;
            const LoopEstimateResult r = computeLoopEstimate(std::move(snapshot));
            std::fprintf(stderr,
                         "[loopestimate] frame %d<->%d: landmarks %d->%d obs %d->%d verified=%d ba=%s "
                         "cost %.1f->%.1f",
                         oldF, newF, r.landmarksBefore, r.landmarksAfter, r.observationsBefore,
                         r.observationsAfter, r.loopVerifiedResidualCount, r.baConverged ? "ok" : "fail",
                         r.baInitialCost, r.baFinalCost);
            if (r.ok && r.ateMatchedPoints > 0) {
                std::fprintf(stderr, " ATE_RMSE=%.3f mean=%.3f median=%.3f max=%.3f n=%d\n", r.ateRmse,
                             r.ateMean, r.ateMedian, r.ateMax, r.ateMatchedPoints);
            } else {
                std::fprintf(stderr, " (%s)\n", qPrintable(r.message));
            }
            std::fflush(stderr);
        });
    }

    QObject::connect(&worker, &SlamWorker::trackingStateChanged, [](const QString &s) {
        std::fprintf(stderr, "[state] %s\n", s.toLocal8Bit().constData());
        std::fflush(stderr);
    });
    QObject::connect(&worker, &SlamWorker::statsUpdated, [&app](const QString &s) {
        std::fprintf(stderr, "[stats] %s\n", s.toLocal8Bit().constData());
        std::fflush(stderr);
        // Unthrottled processing can finish the whole sequence in well
        // under `seconds` of wall-clock time -- quit as soon as the stream
        // naturally ends instead of idling until the fallback timeout.
        if (s == QStringLiteral("Stream ended"))
            app.quit();
    });
    QObject::connect(&worker, &SlamWorker::sourceOpened, [](bool ok, const QString &msg) {
        std::fprintf(stderr, "[open] ok=%d %s\n", ok, msg.toLocal8Bit().constData());
        std::fflush(stderr);
    });

    // Only needed for the AR overlay and the loopestimate diagnostic's ATE
    // computation -- this tool's own ATE summary at the end reads
    // poses.txt itself via loadGroundTruth(), independent of this.
    if (worker.loadGroundTruthPoses(posesPath))
        std::fprintf(stderr, "[config] ground-truth poses loaded into SlamWorker (for loopestimate ATE)\n");

    worker.openVideoFile(imagePattern);

    // OXTS/IMU must be loaded AFTER openVideoFile(): openVideoFile() calls
    // clearOxtsImuData() unconditionally (added so the GUI never carries
    // stale OXTS/IMU from a previously-opened, unrelated sequence -- see
    // DEBUGGING.md Session 9 item 9), which wipes out anything loaded
    // earlier. This tool used to load OXTS/IMU before opening the video,
    // so it was silently running with none at all regardless of the CLI
    // flags below -- confirmed by a bisection run where OXTS-on and
    // OXTS-off produced byte-identical output.
    // argv[21] 'imuonly': load ONLY IMU orientation (loadImuOrientation()
    // has no dependency on loadOxtsSpeeds() -- verified independent), skipping
    // OXTS speed entirely so it can't influence trackFrame()'s own
    // vision-scale heuristics/plausibility gate at all. For when IMU-measured
    // rotation should feed pose_graph edges (see SlamWorker::insertKeyframe()/
    // tryLoopClosure()'s IMU-rotation substitution) without any OXTS speed
    // aiding live.
    const bool imuOnly = argc > 21 && std::strcmp(argv[21], "imuonly") == 0;
    if (argc > 7 && std::strcmp(argv[7], "-") != 0) {
        const QString oxtsDir = QString::fromLocal8Bit(argv[7]);
        if (!imuOnly) {
            if (worker.loadOxtsSpeeds(oxtsDir))
                std::fprintf(stderr, "[config] OXTS speed data loaded from %s\n", argv[7]);
            else
                std::fprintf(stderr, "[config] WARNING: failed to load OXTS speed data from %s -- "
                                      "continuing with vision-only scale heuristics\n", argv[7]);
        }

        if (argc > 8 && std::strcmp(argv[8], "-") != 0) {
            const QString calibDir = QString::fromLocal8Bit(argv[8]);
            if (worker.loadImuOrientation(oxtsDir, calibDir))
                std::fprintf(stderr, "[config] IMU orientation + calibration loaded from %s%s\n", argv[8],
                             imuOnly ? " (IMU only, no OXTS speed)" : "");
            else
                std::fprintf(stderr,
                              "[config] WARNING: failed to load IMU orientation/calibration from %s -- "
                              "continuing without IMU-based rotation\n",
                              argv[8]);
        }
    }

    worker.startUnthrottled(); // no reason to wait on real-time playback pacing headlessly

    QTimer::singleShot(seconds * 1000, &app, &QCoreApplication::quit);
    app.exec();

    std::fprintf(stderr, "[fuse] landmarks merged this run: %lld\n", worker.fusedLandmarkCount());

    const QVector<QPointF> &traj = worker.trajectoryPoints();
    const QVector<int> &frames = worker.trajectoryFrameIndices();
    const std::vector<Point2> gt = loadGroundTruth(posesPath);

    if (gt.empty()) {
        std::fprintf(stderr, "Failed to load ground truth from %s\n", qPrintable(posesPath));
        return 1;
    }
    if (traj.isEmpty()) {
        std::fprintf(stderr, "No trajectory produced -- did tracking ever start?\n");
        return 1;
    }

    // Pairs estimated points with ground truth by frame index (frame=1 is
    // the first processed frame -> ground truth line 0), fits a 2D Umeyama
    // similarity transform, prints the ATE summary, and dumps the aligned
    // trajectory to trajOutPath. Extracted into a lambda so it can be
    // called a second time for the pose-graph-corrected trajectory below,
    // without duplicating this logic -- called exactly as before for the
    // raw-live trajectory, so that output is unchanged when `posegraph`
    // isn't passed.
    auto computeAndPrintAte = [&](const QVector<QPointF> &trajIn, const QVector<int> &framesIn,
                                   const QString &trajOutPath) -> bool {
        std::vector<Point2> src, dst;
        std::vector<int> matchedFrames;
        src.reserve(trajIn.size());
        dst.reserve(trajIn.size());
        for (int i = 0; i < trajIn.size(); ++i) {
            const int gtIdx = framesIn[i] - 1;
            if (gtIdx < 0 || gtIdx >= static_cast<int>(gt.size()))
                continue;
            src.push_back({trajIn[i].x(), trajIn[i].y()});
            dst.push_back(gt[gtIdx]);
            matchedFrames.push_back(framesIn[i]);
        }

        double scale = 1.0, cosT = 1.0, sinT = 0.0, tx = 0.0, tz = 0.0;
        if (!umeyama2D(src, dst, scale, cosT, sinT, tx, tz)) {
            std::fprintf(stderr, "Alignment failed -- too few matched points (%zu) or degenerate estimate.\n",
                          src.size());
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

        double pathLength = 0.0;
        for (size_t i = 1; i < gt.size(); ++i)
            pathLength += std::hypot(gt[i].x - gt[i - 1].x, gt[i].z - gt[i - 1].z);

        std::printf("Matched points:       %zu / %zu ground-truth frames\n", src.size(), gt.size());
        std::printf("Recovered scale:      %.4f (estimated-units -> ground-truth-units)\n", scale);
        std::printf("GT path length:       %.1f m\n", pathLength);
        std::printf("ATE RMSE:             %.3f m\n", ateRmse);
        std::printf("ATE mean:             %.3f m\n", ateMean);
        std::printf("ATE median:           %.3f m\n", ateMedian);
        std::printf("ATE max:              %.3f m\n", maxErr);
        std::printf("ATE RMSE / path len:  %.2f%%\n", 100.0 * ateRmse / pathLength);

        QFile out(trajOutPath);
        if (out.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream stream(&out);
            stream << "# frame x z x_aligned z_aligned gt_x gt_z\n";
            for (size_t i = 0; i < src.size(); ++i) {
                const double ax = scale * (cosT * src[i].x - sinT * src[i].z) + tx;
                const double az = scale * (sinT * src[i].x + cosT * src[i].z) + tz;
                stream << matchedFrames[i] << ' ' << src[i].x << ' ' << src[i].z << ' ' << ax << ' ' << az
                       << ' ' << dst[i].x << ' ' << dst[i].z << '\n';
            }
            std::fprintf(stderr, "wrote %s\n", qPrintable(trajOutPath));
        }
        return true;
    };

    if (!computeAndPrintAte(traj, frames, outPrefix + QStringLiteral("_trajectory.txt")))
        return 1;

    if (argc > 13 && std::strcmp(argv[13], "posegraph") == 0) {
        // [argv35] covis: pass the literal word 'covis' to add Essential-
        // Graph-style covisibility edges (any two keyframes sharing >=100
        // jointly-observed landmarks, see SlamWorker::covisibilityEdgeRecords())
        // alongside the plain sequential chain -- see that function's own
        // doc comment for why a sequential-only graph has essentially zero
        // internal constraint (every sequential edge's chi2 was measured
        // at exactly 0.000 this session).
        std::vector<pose_graph::SequentialEdgeRecord> sequential = worker.sequentialEdgeRecords();
        const size_t sequentialOnlyCount = sequential.size();
        if (argc > 35 && std::strcmp(argv[35], "covis") == 0) {
            const std::vector<pose_graph::SequentialEdgeRecord> covis = worker.covisibilityEdgeRecords();
            sequential.insert(sequential.end(), covis.begin(), covis.end());
        }
        const std::vector<pose_graph::LoopClosureRecord> &loops = worker.loopClosureRecords();
        std::fprintf(stderr,
                     "[posegraph] starting: %zu keyframes, %zu sequential edges (%zu covisibility), %zu loop closures\n",
                     worker.keyframePoses().size(), sequential.size(), sequential.size() - sequentialOnlyCount,
                     loops.size());

        // argv[32] 'sweep': instead of a single solve, try a grid of
        // (dcsPhi, scaleWeight) combinations against the SAME already-
        // tracked keyframes/edges (re-fetching a fresh worker.keyframePoses()
        // copy each time, since optimizePoseGraph() mutates its `keyframes`
        // argument in place) -- avoids a full ~4-13 minute re-track per
        // parameter combination tried. Sim3 always on (a sweep of SE(3)'s
        // rigid path wouldn't need dcsPhi/scaleWeight tuning the same way).
        // Prints one compact line per combo instead of the usual full
        // multi-line ATE report; does not write any trajectory files.
        if (argc > 32 && std::strcmp(argv[32], "sweep") == 0) {
            auto quickAte = [&](const QVector<QPointF> &trajIn, const QVector<int> &framesIn) -> double {
                std::vector<Point2> src, dst;
                for (int i = 0; i < trajIn.size(); ++i) {
                    const int gtIdx = framesIn[i] - 1;
                    if (gtIdx < 0 || gtIdx >= static_cast<int>(gt.size()))
                        continue;
                    src.push_back({trajIn[i].x(), trajIn[i].y()});
                    dst.push_back(gt[gtIdx]);
                }
                double scale = 1.0, cosT = 1.0, sinT = 0.0, tx = 0.0, tz = 0.0;
                if (!umeyama2D(src, dst, scale, cosT, sinT, tx, tz))
                    return -1.0;
                double sumSq = 0.0;
                for (size_t i = 0; i < src.size(); ++i) {
                    const double ax = scale * (cosT * src[i].x - sinT * src[i].z) + tx;
                    const double az = scale * (sinT * src[i].x + cosT * src[i].z) + tz;
                    const double dx = ax - dst[i].x, dz = az - dst[i].z;
                    sumSq += dx * dx + dz * dz;
                }
                return std::sqrt(sumSq / static_cast<double>(src.size()));
            };

            const double liveAte = quickAte(traj, frames);
            std::fprintf(stderr, "[posegraph][sweep] live (uncorrected) ATE RMSE=%.3f\n", liveAte);

            const double dcsPhiValues[] = {1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0};
            const double scaleWeightValues[] = {8.0, 50.0, 200.0, 1000.0};
            // sequentialWeightMultiplier: added 2026-07-21 alongside the
            // covisibility edges themselves -- see PoseGraphOptions's own
            // doc comment for why (loop-edge chi2 outweighed sequential+
            // covisibility chi2 ~18x even with covisibility edges present,
            // 1034.8 vs 57.6 on a real run, so the graph was still almost
            // entirely loop-edge-driven). Swept alongside dcsPhi/scaleWeight
            // to see whether rebalancing this fixes what covisibility edges
            // alone didn't.
            const double seqWeightValues[] = {1.0, 5.0, 20.0, 100.0, 500.0};
            for (double seqWeight : seqWeightValues) {
                for (double dcsPhi : dcsPhiValues) {
                    for (double scaleWeight : scaleWeightValues) {
                        std::vector<pose_graph::KeyframePose> trial = worker.keyframePoses();
                        std::vector<pose_graph::KeyframePose> trialWarmStart;
                        pose_graph::PoseGraphOptions trialOptions;
                        trialOptions.useSim3 = true;
                        trialOptions.dcsPhi = dcsPhi;
                        trialOptions.scaleWeight = scaleWeight;
                        trialOptions.sequentialWeightMultiplier = seqWeight;
                        if (!pose_graph::optimizePoseGraph(trial, sequential, loops, trialOptions, &trialWarmStart)) {
                            std::fprintf(stderr,
                                         "[posegraph][sweep] seqWeight=%.1f dcsPhi=%.1f scaleWeight=%.1f solve FAILED\n",
                                         seqWeight, dcsPhi, scaleWeight);
                            continue;
                        }
                        const QVector<QPointF> trajCorrected =
                            pose_graph::applyPoseGraphCorrection(trialWarmStart, trial, traj, frames, loops);
                        const double ate = quickAte(trajCorrected, frames);
                        std::fprintf(stderr, "[posegraph][sweep] seqWeight=%.1f dcsPhi=%.1f scaleWeight=%.1f ATE RMSE=%.3f%s\n",
                                     seqWeight, dcsPhi, scaleWeight, ate,
                                     (liveAte > 0 && ate > 0 && ate < liveAte) ? "  <-- BETTER" : "");
                    }
                }
            }
            return 0;
        }

        std::vector<pose_graph::KeyframePose> optimized = worker.keyframePoses(); // mutated in place on success
        std::vector<pose_graph::KeyframePose> warmStart; // the poses actually optimized FROM -- see
                                                           // optimizePoseGraph()'s doc comment for why this,
                                                           // not worker.keyframePoses(), is the correct
                                                           // "pre-optimization" baseline for the delta below
        pose_graph::PoseGraphOptions pgOptions;
        // argv[19]: 'sim3' switches the offline pose-graph solve from rigid
        // SE(3) edges to Sim(3) (adds a free per-keyframe scale DOF -- see
        // PoseGraphOptions::useSim3's doc comment). Default (omitted/any
        // other value) keeps the existing, already-measured SE(3) path.
        if (argc > 19 && std::strcmp(argv[19], "sim3") == 0) {
            pgOptions.useSim3 = true;
            std::fprintf(stderr, "[posegraph] using Sim(3) edges (scale-aware)\n");
        }
        if (pose_graph::optimizePoseGraph(optimized, sequential, loops, pgOptions, &warmStart)) {
            const QVector<QPointF> trajCorrected =
                pose_graph::applyPoseGraphCorrection(warmStart, optimized, traj, frames, loops);
            std::printf("\n--- posegraph-corrected ---\n");
            computeAndPrintAte(trajCorrected, frames, outPrefix + QStringLiteral("_posegraph_trajectory.txt"));
        } else {
            std::fprintf(stderr, "[posegraph] optimization failed -- no corrected trajectory to report\n");
        }
    }

    return 0;
}
