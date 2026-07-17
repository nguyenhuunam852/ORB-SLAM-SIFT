#include "LoopEstimator.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <utility>

namespace {

constexpr int kBaMaxIterations = 50;
constexpr double kBaHuberDeltaPixels = 4.0;
constexpr int kBaMinObservationsPerLandmark = 2;
// Mirrors SlamWorker.cpp's kLoopVerifiedResidualWeight -- see that constant's
// doc comment for why the actual PnP-RANSAC-verified loop correspondence
// needs to be trusted more than an ordinary Huber-lossed observation.
constexpr double kLoopVerifiedResidualWeight = 25.0;
// Mirrors SlamWorker.cpp's kMaxObservationReprojErrorPixels -- the
// enrichment pass's descriptor ratio-test matches get the same geometric
// reprojection gate before being accepted as observations.
constexpr double kEnrichmentMaxReprojErrorPixels = 8.0;
// Mirrors SlamWorker.cpp's kBaMaxWindowKeyframes, but raised from 200 to
// 400 after the geometric reprojection gate above was added (per-user
// request -- capping at 200 silently skipped every loop closure in the
// back half of a full-sequence run, defeating the point of a per-loop
// visualization). The "large windows scored worse" evidence that motivated
// 200 originally (see DEBUGGING.md Session 8) predates the reprojection
// gate, which removes most of the ungated-match contamination that
// evidence was actually blaming -- not re-validated at this new cap, but
// this session's own full-sequence run never exceeded ~320 keyframes for
// any window, so 400 covers everything actually observed with headroom.
// This diagnostic also has no live-thread time budget to protect in the
// GUI (runs off-thread via QtConcurrent) -- only kitti_ate's synchronous
// `loopestimate` mode pays this cost directly, and only for a truly
// pathological loop (Session 7 saw one 400+-keyframe, 20831-landmark
// end-of-sequence case that blew a 900s budget) would this cap matter for
// wall-clock at all.
constexpr int kEnrichmentMaxWindowKeyframes = 400;

// Same reprojection-error residual as SlamWorker's own runLoopBundleAdjustment()
// (see that function's doc comment for the pose/point parameterization):
// pose = angle-axis rotation[0..2] + translation[3..5], world-to-camera.
struct ReprojectionCost
{
    ReprojectionCost(double observedU, double observedV, double fx, double fy, double cx, double cy)
        : observedU(observedU), observedV(observedV), fx(fx), fy(fy), cx(cx), cy(cy)
    {
    }

    template <typename T>
    bool operator()(const T *const pose, const T *const point, T *residuals) const
    {
        T camPoint[3];
        ceres::AngleAxisRotatePoint(pose, point, camPoint);
        camPoint[0] += pose[3];
        camPoint[1] += pose[4];
        camPoint[2] += pose[5];

        T z = camPoint[2];
        if (z < T(1e-3))
            z = T(1e-3);

        const T predictedU = T(fx) * camPoint[0] / z + T(cx);
        const T predictedV = T(fy) * camPoint[1] / z + T(cy);
        residuals[0] = predictedU - T(observedU);
        residuals[1] = predictedV - T(observedV);
        return true;
    }

    static ceres::CostFunction *Create(double observedU, double observedV, double fx, double fy, double cx,
                                        double cy)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 6, 3>(
            new ReprojectionCost(observedU, observedV, fx, fy, cx, cy));
    }

    double observedU, observedV, fx, fy, cx, cy;
};

bool matchDescriptors(const cv::Mat &descA, const cv::Mat &descB, std::vector<cv::DMatch> &goodMatches,
                       float ratio = 0.75f)
{
    goodMatches.clear();
    if (descA.empty() || descB.empty() || descA.rows < 2 || descB.rows < 2)
        return false;

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(descA, descB, knn, 2);

    for (const auto &m : knn) {
        if (m.size() == 2 && m[0].distance < ratio * m[1].distance)
            goodMatches.push_back(m[0]);
    }
    return !goodMatches.empty();
}

void fillPose(std::array<double, 6> &block, const cv::Mat &R, const cv::Mat &t)
{
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    for (int k = 0; k < 3; ++k)
        block[static_cast<size_t>(k)] = rvec.at<double>(k);
    for (int k = 0; k < 3; ++k)
        block[static_cast<size_t>(3 + k)] = t.at<double>(k);
}

void poseToRT(const std::array<double, 6> &block, cv::Mat &R, cv::Mat &t)
{
    const cv::Mat rvec = (cv::Mat_<double>(3, 1) << block[0], block[1], block[2]);
    cv::Rodrigues(rvec, R);
    t = (cv::Mat_<double>(3, 1) << block[3], block[4], block[5]);
}

// Closed-form 2D Umeyama similarity fit -- identical math to
// MapView::computeAlignment()/kitti_ate.cpp's umeyama2D(), reused here so
// this background ATE number means the same thing as those.
bool umeyama2D(const std::vector<QPointF> &src, const std::vector<QPointF> &dst, double &scale, double &cosTheta,
               double &sinTheta, double &tx, double &tz)
{
    if (src.size() < 8 || src.size() != dst.size())
        return false;

    const double n = static_cast<double>(src.size());
    double meanSrcX = 0.0, meanSrcZ = 0.0, meanDstX = 0.0, meanDstZ = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        meanSrcX += src[i].x();
        meanSrcZ += src[i].y();
        meanDstX += dst[i].x();
        meanDstZ += dst[i].y();
    }
    meanSrcX /= n;
    meanSrcZ /= n;
    meanDstX /= n;
    meanDstZ /= n;

    double C = 0.0, D = 0.0, srcSqSum = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        const double px = src[i].x() - meanSrcX, pz = src[i].y() - meanSrcZ;
        const double qx = dst[i].x() - meanDstX, qz = dst[i].y() - meanDstZ;
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

LoopEstimateResult computeLoopEstimate(LoopEstimateSnapshot snapshot)
{
    LoopEstimateResult result;
    result.oldFrameIndex = snapshot.oldFrameIndex;
    result.newFrameIndex = snapshot.newFrameIndex;

    const int windowSize = static_cast<int>(snapshot.keyframes.size());
    if (windowSize < 3) {
        result.message = QStringLiteral("window too small");
        return result;
    }
    if (windowSize > kEnrichmentMaxWindowKeyframes) {
        // Mirrors SlamWorker.cpp's kBaMaxWindowKeyframes -- this session's
        // own data showed large windows (21000+ landmarks observed) both
        // cost more to re-match/solve *and* score worse, not better, on
        // ATE (more room for the enrichment's descriptor-only matching to
        // accumulate wrong associations). Not worth attempting until (1)
        // that's addressed, so skip rather than spend a long time on a
        // result likely to be thrown away anyway.
        result.message = QStringLiteral("window too large for background re-estimate (%1 keyframes > %2)")
                              .arg(windowSize)
                              .arg(kEnrichmentMaxWindowKeyframes);
        return result;
    }

    // --- Step 1: landmark observations as live tracking already recorded
    // them (each keyframe's own triangulated points, one observation
    // each) -- the "before" picture.
    std::unordered_map<long long, std::vector<std::pair<int, cv::Point2f>>> observations;
    std::unordered_map<long long, cv::Point3f> positions;
    for (int i = 0; i < windowSize; ++i) {
        const auto &kf = snapshot.keyframes[static_cast<size_t>(i)];
        for (size_t j = 0; j < kf.localMapPointIds.size(); ++j) {
            const long long id = kf.localMapPointIds[j];
            positions[id] = kf.localMapPoints[j];
            observations[id].emplace_back(i, kf.localMapImagePoints[j]);
        }
    }
    int observationsBefore = 0;
    for (const auto &entry : observations)
        observationsBefore += static_cast<int>(entry.second.size());
    const int landmarksBefore = static_cast<int>(observations.size());

    // --- Step 2: enrichment re-matching. Accumulate every keyframe's full
    // descriptors into one growing, NEVER-evicted pool (unlike live
    // tracking's kMaxMapPoints-capped rolling map), matching each new
    // keyframe against everything accumulated from earlier keyframes in
    // the window. This is exactly what gives a landmark created near the
    // window's start a chance to be re-observed near its end, the
    // "full picture" of tracks live tracking's eviction never allowed.
    // BUG FIXED THIS SESSION (see DEBUGGING.md Session 8): this pool must be
    // built from each keyframe's localMapDescriptors (parallel to
    // localMapPointIds, one row per triangulated landmark), NOT its full
    // `descriptors` (every detected keypoint, triangulated or not). The
    // previous version accumulated `accumulatedIds` by localMapPointIds
    // count but `accumulatedDescriptors` by the full (much larger)
    // descriptors count, so `accumulatedIds[m.queryIdx]` silently read out
    // of bounds/misaligned the moment a keyframe's raw keypoint count
    // exceeded its triangulated-point count (i.e. almost always) --
    // matches got attributed to essentially random landmark IDs. This is
    // why enrichment "worked" (found more landmarks/observations, BA
    // converged) while ATE got dramatically worse: garbage data
    // associations, not a dilution/weighting problem. The new keyframe's
    // FULL descriptor set is still the right thing to match *against*
    // (train side) -- a landmark can reappear at any detected keypoint in
    // a later frame, not just one that happened to get triangulated there.
    cv::Mat accumulatedDescriptors;
    std::vector<long long> accumulatedIds;
    for (int i = 0; i < windowSize; ++i) {
        const auto &kf = snapshot.keyframes[static_cast<size_t>(i)];
        if (!accumulatedDescriptors.empty() && !kf.descriptors.empty()) {
            std::vector<cv::DMatch> matches;
            if (matchDescriptors(accumulatedDescriptors, kf.descriptors, matches)) {
                for (const auto &m : matches) {
                    const long long id = accumulatedIds[static_cast<size_t>(m.queryIdx)];
                    // A plain ratio-test match has no geometric check at
                    // all -- reproject the landmark's already-triangulated
                    // 3D position (from `positions`, step 1 above) through
                    // this keyframe's own pose and reject if it lands far
                    // from the matched keypoint. Mirrors
                    // SlamWorker::recordLandmarkObservations()'s gate (see
                    // kMaxObservationReprojErrorPixels there) -- this is the
                    // fix for the "unverified matches corrupt BA" problem
                    // diagnosed live in Session 8/9, not just the earlier
                    // index-misalignment bug.
                    const auto posIt = positions.find(id);
                    if (posIt == positions.end())
                        continue;
                    const cv::Point3f &p = posIt->second;
                    const cv::Mat P = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
                    const cv::Mat camPt = kf.R * P + kf.t;
                    const double z = camPt.at<double>(2);
                    if (z < 1e-3)
                        continue; // behind the camera
                    const double u = snapshot.intrinsics.fx * camPt.at<double>(0) / z + snapshot.intrinsics.cx;
                    const double v = snapshot.intrinsics.fy * camPt.at<double>(1) / z + snapshot.intrinsics.cy;
                    const cv::Point2f &obs = kf.keypoints[static_cast<size_t>(m.trainIdx)].pt;
                    const double du = u - obs.x, dv = v - obs.y;
                    if (du * du + dv * dv > kEnrichmentMaxReprojErrorPixels * kEnrichmentMaxReprojErrorPixels)
                        continue;
                    observations[id].emplace_back(i, obs);
                }
            }
        }
        if (!kf.localMapDescriptors.empty()) {
            for (long long id : kf.localMapPointIds)
                accumulatedIds.push_back(id);
            if (accumulatedDescriptors.empty())
                accumulatedDescriptors = kf.localMapDescriptors.clone();
            else
                cv::vconcat(accumulatedDescriptors, kf.localMapDescriptors, accumulatedDescriptors);
        }
    }
    int observationsAfter = 0;
    for (const auto &entry : observations)
        observationsAfter += static_cast<int>(entry.second.size());
    result.landmarksBefore = landmarksBefore;
    result.observationsBefore = observationsBefore;
    result.landmarksAfter = static_cast<int>(observations.size());
    result.observationsAfter = observationsAfter;

    // --- Step 3: bundle adjustment over the enriched observation set.
    // Endpoints fixed exactly as SlamWorker::runLoopBundleAdjustment()
    // does: keyframe 0 at its existing (trusted) pose, the last keyframe
    // at the loop-measured (loopR, loopT), everything between free.
    std::vector<std::array<double, 6>> poses(static_cast<size_t>(windowSize));
    for (int i = 0; i < windowSize; ++i)
        fillPose(poses[static_cast<size_t>(i)], snapshot.keyframes[static_cast<size_t>(i)].R,
                 snapshot.keyframes[static_cast<size_t>(i)].t);
    fillPose(poses.back(), snapshot.loopR, snapshot.loopT);

    std::unordered_map<long long, std::array<double, 3>> landmarks;
    for (const auto &entry : observations) {
        if (entry.second.size() < static_cast<size_t>(kBaMinObservationsPerLandmark))
            continue;
        const auto posIt = positions.find(entry.first);
        if (posIt == positions.end())
            continue;
        landmarks[entry.first] = {static_cast<double>(posIt->second.x), static_cast<double>(posIt->second.y),
                                   static_cast<double>(posIt->second.z)};
    }

    ceres::Problem problem;
    for (int i = 0; i < windowSize; ++i)
        problem.AddParameterBlock(poses[static_cast<size_t>(i)].data(), 6);

    int residualCount = 0;
    int verifiedResidualCount = 0;
    const int loopEndpointIdx = windowSize - 1;
    const CameraIntrinsics &intr = snapshot.intrinsics;
    for (auto &entry : landmarks) {
        for (const auto &obs : observations.at(entry.first)) {
            ceres::CostFunction *cost =
                ReprojectionCost::Create(obs.second.x, obs.second.y, intr.fx, intr.fy, intr.cx, intr.cy);
            // Same special trust runLoopBundleAdjustment() gives the actual
            // PnP-RANSAC-verified loop correspondence (see
            // kLoopVerifiedResidualWeight's doc comment) -- everything else
            // here is at best a re-matched (not geometrically re-verified)
            // pair.
            const bool isLoopVerified =
                obs.first == loopEndpointIdx && snapshot.loopVerifiedLandmarkIds.count(entry.first) > 0;
            ceres::LossFunction *loss =
                isLoopVerified ? new ceres::ScaledLoss(nullptr, kLoopVerifiedResidualWeight, ceres::TAKE_OWNERSHIP)
                                : static_cast<ceres::LossFunction *>(new ceres::HuberLoss(kBaHuberDeltaPixels));
            problem.AddResidualBlock(cost, loss, poses[static_cast<size_t>(obs.first)].data(), entry.second.data());
            ++residualCount;
            verifiedResidualCount += isLoopVerified ? 1 : 0;
        }
    }
    result.loopVerifiedResidualCount = verifiedResidualCount;

    if (landmarks.empty() || residualCount == 0) {
        result.message = QStringLiteral("no jointly-observed landmarks even after enrichment");
        return result;
    }

    problem.SetParameterBlockConstant(poses.front().data());
    problem.SetParameterBlockConstant(poses.back().data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = kBaMaxIterations;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::JACOBI;
        ceres::Solve(options, &problem, &summary);
    }
    result.baConverged = summary.IsSolutionUsable();
    result.baInitialCost = summary.initial_cost;
    result.baFinalCost = summary.final_cost;
    if (!result.baConverged) {
        result.message = QStringLiteral("bundle adjustment did not converge");
        return result;
    }

    // --- Step 4: propagate the optimized poses to a corrected copy of the
    // trajectory (piecewise between consecutive keyframes, same yaw+
    // translation-only style as SlamWorker::tryLoopClosure() -- see that
    // function's doc comment for why m_trajectory only ever gets a
    // yaw-only projection of the full correction), purely for computing
    // ATE here -- never written back to anything live.
    std::vector<cv::Mat> newR(static_cast<size_t>(windowSize)), newT(static_cast<size_t>(windowSize));
    for (int i = 0; i < windowSize; ++i)
        poseToRT(poses[static_cast<size_t>(i)], newR[static_cast<size_t>(i)], newT[static_cast<size_t>(i)]);

    struct Delta
    {
        double yawRad = 0.0;
        double tx = 0.0, tz = 0.0;
    };
    std::vector<Delta> deltas(static_cast<size_t>(windowSize));
    for (int i = 0; i < windowSize; ++i) {
        const cv::Mat &oldR = snapshot.keyframes[static_cast<size_t>(i)].R;
        const cv::Mat &oldT = snapshot.keyframes[static_cast<size_t>(i)].t;
        const cv::Mat RcwOld = oldR.t();
        const cv::Mat COld = -RcwOld * oldT;
        const cv::Mat RcwNew = newR[static_cast<size_t>(i)].t();
        const cv::Mat CNew = -RcwNew * newT[static_cast<size_t>(i)];
        const cv::Mat Rdelta = RcwNew * RcwOld.t();
        const cv::Mat tDelta = CNew - Rdelta * COld;
        deltas[static_cast<size_t>(i)].yawRad = std::atan2(-Rdelta.at<double>(2, 0), Rdelta.at<double>(0, 0));
        deltas[static_cast<size_t>(i)].tx = tDelta.at<double>(0);
        deltas[static_cast<size_t>(i)].tz = tDelta.at<double>(2);
    }

    QVector<QPointF> correctedTrajectory = snapshot.trajectory;
    for (int p = 0; p < correctedTrajectory.size(); ++p) {
        const int f = snapshot.trajectoryFrameIndex[p];
        if (f < snapshot.oldFrameIndex || f > snapshot.newFrameIndex)
            continue;
        // Find the bounding keyframe segment [a, a+1] covering frame f.
        int a = 0;
        while (a + 1 < windowSize && snapshot.keyframes[static_cast<size_t>(a + 1)].frameIndex <= f)
            ++a;
        const int b = std::min(a + 1, windowSize - 1);
        const int fa = snapshot.keyframes[static_cast<size_t>(a)].frameIndex;
        const int fb = snapshot.keyframes[static_cast<size_t>(b)].frameIndex;
        const double alpha = (fb > fa) ? static_cast<double>(f - fa) / static_cast<double>(fb - fa) : 0.0;
        const double yaw = (1.0 - alpha) * deltas[static_cast<size_t>(a)].yawRad +
                            alpha * deltas[static_cast<size_t>(b)].yawRad;
        const double tx = (1.0 - alpha) * deltas[static_cast<size_t>(a)].tx + alpha * deltas[static_cast<size_t>(b)].tx;
        const double tz = (1.0 - alpha) * deltas[static_cast<size_t>(a)].tz + alpha * deltas[static_cast<size_t>(b)].tz;
        const double c = std::cos(yaw), s = std::sin(yaw);
        const QPointF orig = correctedTrajectory[p];
        const double x = orig.x(), z = orig.y();
        correctedTrajectory[p] = QPointF(c * x + s * z + tx, -s * x + c * z + tz);
    }

    // --- Step 5: ATE against ground truth, if loaded.
    if (!snapshot.hasGroundTruth || snapshot.groundTruth.isEmpty()) {
        result.message = QStringLiteral("no ground truth loaded -- BA ran, ATE unavailable");
        result.ok = true;
        return result;
    }

    std::vector<QPointF> src, dst;
    src.reserve(correctedTrajectory.size());
    dst.reserve(correctedTrajectory.size());
    for (int i = 0; i < correctedTrajectory.size(); ++i) {
        const int gtIdx = snapshot.trajectoryFrameIndex[i] - 1;
        if (gtIdx < 0 || gtIdx >= snapshot.groundTruth.size())
            continue;
        src.push_back(correctedTrajectory[i]);
        dst.push_back(snapshot.groundTruth[gtIdx]);
    }

    double scale = 1.0, cosT = 1.0, sinT = 0.0, tx = 0.0, tz = 0.0;
    if (!umeyama2D(src, dst, scale, cosT, sinT, tx, tz)) {
        result.message = QStringLiteral("ATE alignment failed (too few matched points)");
        return result;
    }

    double sumSq = 0.0, sumAbs = 0.0, maxErr = 0.0;
    std::vector<double> errors(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        const double ax = scale * (cosT * src[i].x() - sinT * src[i].y()) + tx;
        const double az = scale * (sinT * src[i].x() + cosT * src[i].y()) + tz;
        const double dx = ax - dst[i].x(), dz = az - dst[i].y();
        const double err = std::hypot(dx, dz);
        errors[i] = err;
        sumSq += err * err;
        sumAbs += err;
        maxErr = std::max(maxErr, err);
    }
    std::vector<double> sorted = errors;
    std::sort(sorted.begin(), sorted.end());

    result.ok = true;
    result.ateMatchedPoints = static_cast<int>(src.size());
    result.ateRmse = std::sqrt(sumSq / static_cast<double>(src.size()));
    result.ateMean = sumAbs / static_cast<double>(src.size());
    result.ateMedian = sorted[sorted.size() / 2];
    result.ateMax = maxErr;
    result.recoveredScale = scale;

    // --- Step 6: map data for LoopEstimatePanel's per-loop mini-map, using
    // this exact (scale, cosT, sinT, tx, tz) so the thumbnail and the ATE
    // number above always agree. Scoped to just this loop's own window
    // (not the whole trajectory `src`/`dst` above was fit over) so each
    // thumbnail stays a readable, self-contained picture of one loop.
    auto alignPoint = [&](double x, double z) -> QPointF {
        return QPointF(scale * (cosT * x - sinT * z) + tx, scale * (sinT * x + cosT * z) + tz);
    };
    result.alignedLandmarks.reserve(static_cast<int>(landmarks.size()));
    for (const auto &entry : landmarks) {
        const QPointF p = alignPoint(entry.second[0], entry.second[2]);
        if (std::isfinite(p.x()) && std::isfinite(p.y()))
            result.alignedLandmarks.append(p);
    }
    double windowMaxErr = -1.0;
    for (int i = 0; i < correctedTrajectory.size(); ++i) {
        const int f = snapshot.trajectoryFrameIndex[i];
        if (f < snapshot.oldFrameIndex || f > snapshot.newFrameIndex)
            continue;
        const int gtIdx = f - 1;
        if (gtIdx < 0 || gtIdx >= snapshot.groundTruth.size())
            continue;
        const QPointF aligned = alignPoint(correctedTrajectory[i].x(), correctedTrajectory[i].y());
        const QPointF gt = snapshot.groundTruth[gtIdx];
        result.alignedTrajectory.append(aligned);
        result.alignedGroundTruth.append(gt);
        const double err = std::hypot(aligned.x() - gt.x(), aligned.y() - gt.y());
        if (err > windowMaxErr) {
            windowMaxErr = err;
            result.maxErrorIndex = result.alignedTrajectory.size() - 1;
        }
    }

    return result;
}
