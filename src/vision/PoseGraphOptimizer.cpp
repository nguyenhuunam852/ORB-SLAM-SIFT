#include "vision/PoseGraphOptimizer.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

// g2o -- ORB-SLAM2/3's own Sim(3) essential-graph optimization backend, used
// here for optimizePoseGraphSim3() only (the SE(3) path above stays on
// Ceres, untouched). See CMakeLists.txt's g2o_ext target doc comment for
// why this links a prebuilt libg2o.so rather than vendoring/recompiling it
// the way DBoW2's own prebuilt .so (ABI-incompatible OpenCV) could not be.
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_seven_dof_expmap.h"

#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace pose_graph {

namespace {

// Relative-pose residual between two world-to-camera poses (Xcam = R*Xworld
// + t, matching ReprojectionCost/runLoopBundleAdjustment()'s own convention
// -- deliberately NOT copied from Ceres's own pose_graph_3d example, which
// assumes camera-to-world pose blocks; re-derived here for this codebase's
// actual convention).
//
// For poses (R_i, t_i), (R_j, t_j): Xcam_j = R_j*Xworld + t_j =
// (R_j*R_i^T)*Xcam_i + (t_j - R_j*R_i^T*t_i), so the predicted relative
// transform from camera i's frame into camera j's frame is
//   R_pred = R_j * R_i^T
//   t_pred = t_j - R_pred * t_i
// The measured relative transform (Rmeas, tmeas) is built once, before
// optimization, via the same formula from the pre-optimization poses (see
// optimizePoseGraph()) and never changes during the DCS outer loop -- only
// each edge's weight does.
//
// Uses Eigen::Quaternion<T> for rotation composition/error rather than
// hand-derived matrix Jacobians -- well-tested, Jet-compatible primitives,
// matching the spirit of Ceres's own pose_graph_3d example.
struct RelativePoseCost
{
    RelativePoseCost(const cv::Mat &Rmeas, const cv::Mat &tmeas, double rotationWeight)
        : rotationWeight(rotationWeight)
    {
        cv::Mat rvec;
        cv::Rodrigues(Rmeas, rvec);
        const double aa[3] = {rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)};
        ceres::AngleAxisToQuaternion(aa, qMeas); // qMeas = [w, x, y, z]
        tMeas[0] = tmeas.at<double>(0);
        tMeas[1] = tmeas.at<double>(1);
        tMeas[2] = tmeas.at<double>(2);
    }

    template <typename T>
    bool operator()(const T *const poseI, const T *const poseJ, T *residuals) const
    {
        T qIarr[4], qJarr[4];
        ceres::AngleAxisToQuaternion(poseI, qIarr);
        ceres::AngleAxisToQuaternion(poseJ, qJarr);

        const Eigen::Quaternion<T> eqI(qIarr[0], qIarr[1], qIarr[2], qIarr[3]); // Eigen ctor order (w,x,y,z)
        const Eigen::Quaternion<T> eqJ(qJarr[0], qJarr[1], qJarr[2], qJarr[3]);
        const Eigen::Matrix<T, 3, 1> tI(poseI[3], poseI[4], poseI[5]);
        const Eigen::Matrix<T, 3, 1> tJ(poseJ[3], poseJ[4], poseJ[5]);

        Eigen::Quaternion<T> qPred = eqJ * eqI.conjugate(); // R_pred = R_j * R_i^T
        qPred.normalize();
        const Eigen::Matrix<T, 3, 1> tPred = tJ - (qPred * tI); // t_pred = t_j - R_pred*t_i

        const T qm0 = T(qMeas[0]), qm1 = T(qMeas[1]), qm2 = T(qMeas[2]), qm3 = T(qMeas[3]);
        const Eigen::Quaternion<T> qMeasT(qm0, qm1, qm2, qm3);
        Eigen::Quaternion<T> qErr = qMeasT.conjugate() * qPred; // rotation from measured to predicted
        if (qErr.w() < T(0.0))
            qErr.coeffs() = -qErr.coeffs(); // shortest path -- q and -q are the same rotation, but the
                                             // small-angle vec()*2 approximation below only holds near
                                             // identity (w close to +1)

        residuals[0] = T(rotationWeight) * T(2.0) * qErr.x();
        residuals[1] = T(rotationWeight) * T(2.0) * qErr.y();
        residuals[2] = T(rotationWeight) * T(2.0) * qErr.z();
        residuals[3] = tPred.x() - T(tMeas[0]);
        residuals[4] = tPred.y() - T(tMeas[1]);
        residuals[5] = tPred.z() - T(tMeas[2]);
        return true;
    }

    static ceres::CostFunction *Create(const cv::Mat &Rmeas, const cv::Mat &tmeas, double rotationWeight)
    {
        return new ceres::AutoDiffCostFunction<RelativePoseCost, 6, 6, 6>(
            new RelativePoseCost(Rmeas, tmeas, rotationWeight));
    }

    double qMeas[4];
    double tMeas[3];
    double rotationWeight;
};

struct EdgeMeasurement
{
    int i = 0, j = 0;
    cv::Mat R, t;
};

struct LoopEdge : EdgeMeasurement
{
    double weight = 1.0; // DCS s^2, recomputed every outer iteration
    double scaleMeas = 1.0; // Sim3 path only -- see LoopClosureRecord::scale's own doc comment.
                             // Unused by the SE(3)/Ceres path above (SimilarityPoseCost isn't
                             // reachable from rigid SE(3) edges at all).
};

void fillParamsFromPose(std::array<double, 6> &block, const cv::Mat &R, const cv::Mat &t)
{
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    for (int k = 0; k < 3; ++k)
        block[static_cast<size_t>(k)] = rvec.at<double>(k);
    for (int k = 0; k < 3; ++k)
        block[static_cast<size_t>(3 + k)] = t.at<double>(k);
}

void poseFromParams(const std::array<double, 6> &block, cv::Mat &R, cv::Mat &t)
{
    const cv::Mat rvec = (cv::Mat_<double>(3, 1) << block[0], block[1], block[2]);
    cv::Rodrigues(rvec, R);
    t = (cv::Mat_<double>(3, 1) << block[3], block[4], block[5]);
}

// Sim(3) relative-pose residual (see PoseGraphOptions::useSim3's doc
// comment) -- same convention and derivation as RelativePoseCost above,
// with an added free per-node log-scale term. For similarity poses
// (s_i, R_i, t_i), (s_j, R_j, t_j) with Xcam = s*R*Xworld + t:
//   Xworld = (1/s_i) * R_i^T * (Xcam_i - t_i)
//   Xcam_j = s_j*R_j*Xworld + t_j
//          = (s_j/s_i)*(R_j*R_i^T)*Xcam_i - (s_j/s_i)*(R_j*R_i^T)*t_i + t_j
// so the predicted relative similarity transform from camera i to camera j is
//   s_pred = s_j / s_i
//   R_pred = R_j * R_i^T
//   t_pred = t_j - s_pred * R_pred * t_i
// sMeas is fixed at 1.0 for EVERY edge (both sequential and loop) -- see
// PoseGraphOptions::useSim3's doc comment for why: this codebase has no
// independent per-keyframe scale measurement, so sMeas=1 is a soft "no
// unexplained scale jump" prior between connected nodes, not a real
// measurement. The actual absolute-magnitude constraint still comes
// through tMeas (captured in the ORIGINAL, already-scaled units, exactly
// as RelativePoseCost uses it) -- s_pred only gives the solver an extra,
// separate degree of freedom to reconcile a translation-magnitude
// mismatch without having to distort rotation/direction to compensate.
struct SimilarityPoseCost
{
    SimilarityPoseCost(const cv::Mat &Rmeas, const cv::Mat &tmeas, double rotationWeight, double scaleWeight)
        : rotationWeight(rotationWeight), scaleWeight(scaleWeight)
    {
        cv::Mat rvec;
        cv::Rodrigues(Rmeas, rvec);
        const double aa[3] = {rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)};
        ceres::AngleAxisToQuaternion(aa, qMeas);
        tMeas[0] = tmeas.at<double>(0);
        tMeas[1] = tmeas.at<double>(1);
        tMeas[2] = tmeas.at<double>(2);
    }

    template <typename T>
    bool operator()(const T *const poseI, const T *const poseJ, T *residuals) const
    {
        T qIarr[4], qJarr[4];
        ceres::AngleAxisToQuaternion(poseI, qIarr);
        ceres::AngleAxisToQuaternion(poseJ, qJarr);

        const Eigen::Quaternion<T> eqI(qIarr[0], qIarr[1], qIarr[2], qIarr[3]);
        const Eigen::Quaternion<T> eqJ(qJarr[0], qJarr[1], qJarr[2], qJarr[3]);
        const Eigen::Matrix<T, 3, 1> tI(poseI[3], poseI[4], poseI[5]);
        const Eigen::Matrix<T, 3, 1> tJ(poseJ[3], poseJ[4], poseJ[5]);
        const T &logSI = poseI[6];
        const T &logSJ = poseJ[6];

        Eigen::Quaternion<T> qPred = eqJ * eqI.conjugate();
        qPred.normalize();
        const T sPred = exp(logSJ - logSI);
        const Eigen::Matrix<T, 3, 1> tPred = tJ - sPred * (qPred * tI);

        const T qm0 = T(qMeas[0]), qm1 = T(qMeas[1]), qm2 = T(qMeas[2]), qm3 = T(qMeas[3]);
        const Eigen::Quaternion<T> qMeasT(qm0, qm1, qm2, qm3);
        Eigen::Quaternion<T> qErr = qMeasT.conjugate() * qPred;
        if (qErr.w() < T(0.0))
            qErr.coeffs() = -qErr.coeffs();

        residuals[0] = T(rotationWeight) * T(2.0) * qErr.x();
        residuals[1] = T(rotationWeight) * T(2.0) * qErr.y();
        residuals[2] = T(rotationWeight) * T(2.0) * qErr.z();
        residuals[3] = tPred.x() - T(tMeas[0]);
        residuals[4] = tPred.y() - T(tMeas[1]);
        residuals[5] = tPred.z() - T(tMeas[2]);
        residuals[6] = T(scaleWeight) * (logSJ - logSI); // sMeas = 1 -> log(sMeas) = 0
        return true;
    }

    static ceres::CostFunction *Create(const cv::Mat &Rmeas, const cv::Mat &tmeas, double rotationWeight,
                                        double scaleWeight)
    {
        return new ceres::AutoDiffCostFunction<SimilarityPoseCost, 7, 7, 7>(
            new SimilarityPoseCost(Rmeas, tmeas, rotationWeight, scaleWeight));
    }

    double qMeas[4];
    double tMeas[3];
    double rotationWeight;
    double scaleWeight;
};

void fillParamsFromPoseSim3(std::array<double, 7> &block, const cv::Mat &R, const cv::Mat &t)
{
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    for (int k = 0; k < 3; ++k)
        block[static_cast<size_t>(k)] = rvec.at<double>(k);
    for (int k = 0; k < 3; ++k)
        block[static_cast<size_t>(3 + k)] = t.at<double>(k);
    block[6] = 0.0; // logScale -- every node starts at scale 1 (see PoseGraphOptions::useSim3)
}

void poseFromParamsSim3(const std::array<double, 7> &block, cv::Mat &R, cv::Mat &t)
{
    const cv::Mat rvec = (cv::Mat_<double>(3, 1) << block[0], block[1], block[2]);
    cv::Rodrigues(rvec, R);
    t = (cv::Mat_<double>(3, 1) << block[3], block[4], block[5]);
    // block[6] (logScale) deliberately dropped here: KeyframePose has no
    // scale field (see its own doc comment) -- t itself, a free parameter
    // the solver adjusts directly, IS the corrected output translation;
    // logScale is purely an internal fitting aid for SimilarityPoseCost's
    // residual, not a separate transform to re-apply afterward.
}

// Sim(3) counterpart to optimizePoseGraph()'s SE(3) solve below -- see
// PoseGraphOptions::useSim3's doc comment for the motivation. Deliberately
// a fully separate function (not a parameterized branch inside the same
// one) rather than risk altering the already-measured, already-tuned SE(3)
// numeric path in the process of adding this.
bool optimizePoseGraphSim3(std::vector<KeyframePose> &keyframes,
                            const std::vector<SequentialEdgeRecord> &sequentialRecords,
                            const std::vector<LoopClosureRecord> &loopClosures, const PoseGraphOptions &opts,
                            std::vector<KeyframePose> *warmStartOut)
{
    const int n = static_cast<int>(keyframes.size());
    if (n < 2)
        return false;

    std::vector<KeyframePose> original = keyframes;
    for (auto &kf : original) {
        kf.R = kf.R.clone();
        kf.t = kf.t.clone();
    }

    std::vector<EdgeMeasurement> sequentialEdges;
    sequentialEdges.reserve(sequentialRecords.size());
    for (const SequentialEdgeRecord &sr : sequentialRecords) {
        if (sr.i < 0 || sr.j < 0 || sr.i >= n || sr.j >= n || sr.i == sr.j)
            continue;
        sequentialEdges.push_back({sr.i, sr.j, sr.R, sr.t});
    }

    std::vector<std::array<double, 7>> params(static_cast<size_t>(n));
    fillParamsFromPoseSim3(params[0], original[0].R, original[0].t);
    std::vector<const EdgeMeasurement *> edgeFromIndex(static_cast<size_t>(n), nullptr);
    for (const EdgeMeasurement &e : sequentialEdges) {
        if (e.j == e.i + 1)
            edgeFromIndex[static_cast<size_t>(e.i)] = &e;
    }
    for (int i = 1; i < n; ++i) {
        cv::Mat Rprev, tprev;
        poseFromParamsSim3(params[static_cast<size_t>(i - 1)], Rprev, tprev);
        const EdgeMeasurement *e = edgeFromIndex[static_cast<size_t>(i - 1)];
        cv::Mat Ri, ti;
        if (e) {
            Ri = e->R * Rprev;
            ti = e->t + e->R * tprev;
        } else {
            Ri = Rprev;
            ti = tprev;
        }
        fillParamsFromPoseSim3(params[static_cast<size_t>(i)], Ri, ti);
    }

    if (warmStartOut) {
        warmStartOut->assign(static_cast<size_t>(n), KeyframePose());
        for (int i = 0; i < n; ++i) {
            (*warmStartOut)[static_cast<size_t>(i)].frameIndex = original[static_cast<size_t>(i)].frameIndex;
            poseFromParamsSim3(params[static_cast<size_t>(i)], (*warmStartOut)[static_cast<size_t>(i)].R,
                                (*warmStartOut)[static_cast<size_t>(i)].t);
        }
    }

    std::vector<LoopEdge> loopEdges;
    loopEdges.reserve(loopClosures.size());
    for (const LoopClosureRecord &lc : loopClosures) {
        if (lc.oldKfIdx < 0 || lc.newKfIdx < 0 || lc.oldKfIdx >= n || lc.newKfIdx >= n || lc.oldKfIdx == lc.newKfIdx)
            continue;
        LoopEdge e;
        static_cast<EdgeMeasurement &>(e) = {lc.oldKfIdx, lc.newKfIdx, lc.R, lc.t};
        e.scaleMeas = lc.scale; // real scale-drift measurement -- see LoopClosureRecord::scale's doc comment
        loopEdges.push_back(std::move(e));
    }

    // g2o::VertexSim3Expmap/EdgeSim3 (types_seven_dof_expmap.h) -- the exact
    // types real ORB-SLAM2/3 uses for its own Sim(3) essential-graph
    // optimization (Optimizer::OptimizeEssentialGraph()) -- instead of the
    // hand-rolled Ceres SimilarityPoseCost functor above. Vertex convention
    // (Siw = world-to-camera Sim3) and edge convention (measurement = Sji,
    // vertex(0)=i, vertex(1)=j, error = measurement*Siw*Sjw^-1) both match
    // this codebase's own (R,t) convention and edge-record construction
    // directly -- no adaptation needed, see EdgeMeasurement's use below.
    // information diagonal mirrors SimilarityPoseCost's own residual
    // weighting (rotationWeight^2 on the 3 rotation dims, scaleWeight^2 on
    // the scale dim, unweighted translation) since g2o's Sim3::log() is
    // ALSO [rotation(3), translation(3), scale(1)] -- same reasoning
    // rotationWeight existed for in the Ceres version: these three
    // quantities have unrelated natural units (radians, world-units,
    // log-scale) and aren't comparable in one combined chi-square without it.
    //
    // translationScale: root-cause fix for a real bug found by direct
    // reproduction (see [posegraph][sim3][g2o] logs) -- EdgeSim3 has no
    // analytic linearizeOplus(), so g2o falls back to numeric
    // differentiation with a FIXED absolute perturbation (delta=1e-9, see
    // g2o's base_binary_edge.hpp) applied directly to the manifold update,
    // i.e. to translation in this codebase's own raw "world units" (NOT
    // metric -- recovered scale this session was ~0.15, so world-units run
    // roughly 6-7x larger than meters, and a monocular trajectory's own
    // vertex translations reach into the thousands to tens of thousands by
    // the end of a long sequence). At that magnitude, delta/|t| approaches
    // double-precision round-off (~2.2e-16) closely enough that the
    // finite-difference translation columns of the Jacobian are dominated
    // by floating-point noise, not signal -- confirmed via reproduction:
    // g2o's Levenberg-Marquardt solver rejected every single proposed step
    // (chi2 never improved, lambda ballooned to ~1e17, exhausted
    // maxTrialsAfterFailure=10, returned with vertices completely
    // unmoved), regardless of dcsPhi tuning (also tried, ruled out
    // separately). Real ORB-SLAM2/3 never hits this because ITS Essential
    // Graph runs on metric-scale keyframe translations (already real
    // meters, typically O(1-100) for the scenes it targets) where a 1e-9
    // absolute perturbation is comfortably resolvable. Fix: rescale every
    // translation fed into g2o (vertices AND edge measurements) into a
    // similarly well-conditioned range before optimizing, then rescale the
    // result back -- a pure unit-system change (like meters vs
    // millimeters), mathematically inert for a similarity transform since
    // rotation/scale don't depend on the translation unit. infoDiag's
    // translation weighting is scaled by translationScale^2 to exactly
    // compensate, so the same real-world discrepancy still contributes the
    // same chi2 as before this fix -- only the numeric conditioning changes.
    double translationScale = 1.0;
    for (const auto &p : params)
        translationScale = std::max({translationScale, std::fabs(p[3]), std::fabs(p[4]), std::fabs(p[5])});
    translationScale /= 100.0; // land typical translation magnitudes around O(1-100)
    std::fprintf(stderr, "[posegraph][sim3][diag] translationScale=%.6f (n=%d vertices)\n", translationScale, n);

    Eigen::Matrix<double, 7, 7> infoDiag = Eigen::Matrix<double, 7, 7>::Zero();
    for (int d = 0; d < 3; ++d)
        infoDiag(d, d) = opts.rotationWeight * opts.rotationWeight;
    for (int d = 3; d < 6; ++d)
        infoDiag(d, d) = translationScale * translationScale;
    infoDiag(6, 6) = opts.scaleWeight * opts.scaleWeight;

    auto toEigenR = [](const cv::Mat &R) {
        Eigen::Matrix3d Reig;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Reig(r, c) = R.at<double>(r, c);
        return Reig;
    };
    auto toEigenT = [translationScale](const cv::Mat &t) {
        return Eigen::Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2)) / translationScale;
    };

    // Heap-allocated and deliberately never destroyed (leaked): g2o's
    // SparseOptimizer/OptimizableGraph destructor (HyperGraph::clear())
    // corrupts the heap when it runs on this build (confirmed via a
    // synthetic reproduction + gdb backtrace -- crashes inside
    // g2o::HyperGraph::clear(), called from ~OptimizableGraph(), with
    // "free(): invalid pointer"; the solve itself runs and produces correct
    // results before this). Root cause not fully chased down (candidate: an
    // Eigen/g2o ABI mismatch between this prebuilt libg2o.so and this
    // project's own compilation flags) -- for a short-lived batch CLI
    // process that solves once and exits, leaking this is a pragmatic,
    // harmless workaround rather than a real fix; revisit if this code is
    // ever used somewhere long-running.
    auto *optimizer = new g2o::SparseOptimizer();
    optimizer->setVerbose(false);
    auto *linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    auto *solverPtr = new g2o::BlockSolver_7_3(linearSolver);
    auto *algorithm = new g2o::OptimizationAlgorithmLevenberg(solverPtr);
    optimizer->setAlgorithm(algorithm);

    std::vector<g2o::VertexSim3Expmap *> vertices(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        cv::Mat Ri, ti;
        poseFromParamsSim3(params[static_cast<size_t>(i)], Ri, ti);
        auto *v = new g2o::VertexSim3Expmap();
        v->setEstimate(g2o::Sim3(toEigenR(Ri), toEigenT(ti), std::exp(params[static_cast<size_t>(i)][6])));
        v->setId(i);
        v->_fix_scale = false; // free scale is the entire point of Sim(3) here
        v->setMarginalized(false);
        if (i == 0)
            v->setFixed(true); // gauge fix (rotation+translation+scale all anchored)
        optimizer->addVertex(v);
        vertices[static_cast<size_t>(i)] = v;
    }

    // sMeas defaults to 1.0 (sequential edges, and any loop edge with no
    // real scale evidence -- see PoseGraphOptions::useSim3's doc comment
    // for why sMeas=1 was the ONLY option before LoopClosureRecord::scale
    // existed). Loop edges now pass their own real measured value (see the
    // loop-edge call site below and LoopClosureRecord::scale's own doc
    // comment) -- this is what actually gives Sim(3)'s extra scale DOF
    // something to correct, instead of it sitting inert at exactly 1.0
    // everywhere (confirmed empirically this session: with every edge at
    // sMeas=1, scaleWeight had ZERO effect on the solve, any value from
    // 8 to 1000 produced byte-identical output).
    auto makeEdge = [&](const EdgeMeasurement &e, double sMeas = 1.0, double weightMultiplier = 1.0) {
        g2o::Sim3 measurement(toEigenR(e.R), toEigenT(e.t), sMeas);
        auto *edge = new g2o::EdgeSim3();
        edge->setVertex(0, vertices[static_cast<size_t>(e.i)]);
        edge->setVertex(1, vertices[static_cast<size_t>(e.j)]);
        edge->setMeasurement(measurement);
        edge->information() = infoDiag * weightMultiplier;
        return edge;
    };

    for (const EdgeMeasurement &e : sequentialEdges) {
        g2o::EdgeSim3 *edge = makeEdge(e, 1.0, opts.sequentialWeightMultiplier);
        auto *rk = new g2o::RobustKernelHuber();
        rk->setDelta(opts.sequentialHuberDelta);
        edge->setRobustKernel(rk);
        optimizer->addEdge(edge);
    }
    // g2o's own g2o::RobustKernelDCS (Dynamic Covariance Scaling, Agarwal et
    // al. ICRA 2013 -- the exact paper this file's own top-of-file doc
    // comment already cites) replaces this file's previous hand-rolled
    // outer-iteration reweighting loop: rho = min(1, 2*dcsPhi/(dcsPhi+chi2))
    // is g2o's OWN built-in formula for this kernel (verified against
    // g2o's robust_kernel_impl.cpp), identical to what this file computed
    // manually before. g2o applies and re-evaluates it internally during
    // optimize()'s own iterations (standard IRLS-within-Levenberg-Marquardt),
    // so a single optimize() call now does what used to take
    // opts.outerIterations full rebuild-and-resolve passes.
    for (const LoopEdge &e : loopEdges) {
        g2o::EdgeSim3 *edge = makeEdge(e, e.scaleMeas);
        auto *rk = new g2o::RobustKernelDCS();
        rk->setDelta(opts.dcsPhi);
        edge->setRobustKernel(rk);
        optimizer->addEdge(edge);
    }

    optimizer->initializeOptimization();
    optimizer->computeActiveErrors(); // must run before activeChi2() below has anything to read --
                                       // without it activeChi2() reports a stale/inf value

    // Non-finite-edge guard: found by direct reproduction (see git history/
    // DEBUGGING.md for the full investigation trail) that a small,
    // non-deterministic fraction of runs produce ONE edge whose measurement
    // and vertex estimates are individually finite and well-formed, but
    // whose g2o::Sim3::log() error vector still comes out non-finite --
    // e.g. a rotation/translation pairing that lands exactly on (or very
    // near) one of log()'s internal piecewise-branch boundaries (see
    // sim3.h's own `if (fabs(sigma)<eps)` / `if (d>1-eps)` branches and the
    // `W.lu().solve(t)` linear solve inside them). Root-caused down to
    // g2o's own numerics, not this file's edge construction (confirmed:
    // e.R/e.t and the warm-start chain both check finite right up to the
    // point of construction). Left unresolved at that level (a genuine g2o
    // library edge case, not something worth patching upstream for a
    // batch offline tool) -- instead, guard defensively: any edge whose
    // computed error is non-finite gets DROPPED from the graph before
    // solving, exactly like DCS/Huber already drop the WEIGHT of an edge
    // whose residual is merely large. One bad edge silently poisoning
    // chi2/lambda for the entire graph (confirmed: this alone made the
    // whole solve return "1 iteration, cost unchanged" every time it hit)
    // is worse than just not using that one edge's evidence this run.
    std::vector<g2o::EdgeSim3 *> badEdges;
    {
        double seqSum = 0.0, loopSum = 0.0;
        double seqMax = 0.0, loopMax = 0.0;
        int nanCount = 0;
        for (auto *e : optimizer->edges()) {
            auto *se = dynamic_cast<g2o::EdgeSim3 *>(e);
            if (!se)
                continue;
            se->computeError();
            const double c = se->chi2();
            const bool finite = std::isfinite(c);
            if (!finite) {
                ++nanCount;
                badEdges.push_back(se);
                continue;
            }
            // Loop edges use RobustKernelDCS, sequential use RobustKernelHuber --
            // distinguish by robust kernel type since that's how they were added.
            if (dynamic_cast<g2o::RobustKernelDCS *>(se->robustKernel()) != nullptr) {
                loopSum += c;
                loopMax = std::max(loopMax, c);
            } else {
                seqSum += c;
                seqMax = std::max(seqMax, c);
            }
        }
        std::fprintf(stderr,
                      "[posegraph][sim3][diag] seqEdges chi2 sum=%.3f max=%.3f | loopEdges chi2 sum=%.3f "
                      "max=%.3f | nonFiniteEdges=%d (dropped)\n",
                      seqSum, seqMax, loopSum, loopMax, nanCount);
    }
    for (g2o::EdgeSim3 *se : badEdges)
        optimizer->removeEdge(se);
    if (!badEdges.empty())
        optimizer->initializeOptimization(); // active-edge set changed -- must rebuild before solving

    const double initialChi2 = optimizer->activeChi2();
    const int totalIterations = std::max(1, opts.outerIterations) * opts.maxSolverIterations;
    const int iterationsRun = optimizer->optimize(totalIterations);
    if (iterationsRun <= 0) {
        std::fprintf(stderr, "[posegraph][sim3][g2o] solve failed (0 iterations)\n");
        return false;
    }
    std::fprintf(stderr, "[posegraph][sim3][g2o] cost %.3f -> %.3f (%d iterations, RobustKernelDCS phi=%.3f)\n",
                 initialChi2, optimizer->activeChi2(), iterationsRun, opts.dcsPhi);

    for (int i = 0; i < n; ++i) {
        const g2o::Sim3 &est = vertices[static_cast<size_t>(i)]->estimate();
        const Eigen::Matrix3d Reig = est.rotation().toRotationMatrix();
        cv::Mat R(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R.at<double>(r, c) = Reig(r, c);
        // Undo translationScale (see its own doc comment above) -- g2o's
        // vertex estimate is in the rescaled unit system, not this
        // codebase's own world units.
        const cv::Mat t = (cv::Mat_<double>(3, 1) << est.translation()(0) * translationScale,
                            est.translation()(1) * translationScale, est.translation()(2) * translationScale);
        keyframes[static_cast<size_t>(i)].R = R;
        keyframes[static_cast<size_t>(i)].t = t;
    }
    return true;
}

} // namespace

bool optimizePoseGraph(std::vector<KeyframePose> &keyframes, const std::vector<SequentialEdgeRecord> &sequentialRecords,
                        const std::vector<LoopClosureRecord> &loopClosures, const PoseGraphOptions &opts,
                        std::vector<KeyframePose> *warmStartOut)
{
    // See PoseGraphOptions::useSim3's doc comment -- a fully separate solve
    // path, not a branch further down inside this one, so opts.useSim3
    // defaulting to false is a hard guarantee this function's existing
    // (already-measured) SE(3) behavior is completely untouched.
    if (opts.useSim3)
        return optimizePoseGraphSim3(keyframes, sequentialRecords, loopClosures, opts, warmStartOut);

    const int n = static_cast<int>(keyframes.size());
    if (n < 2)
        return false;

    // Kept only for warm-starting parameter blocks and for
    // applyPoseGraphCorrection()'s pre-vs-post delta afterward -- edge
    // MEASUREMENTS themselves now come entirely from sequentialRecords/
    // loopClosures (captured at observation time, see PoseGraphOptimizer.h),
    // never reconstructed from these poses. Reconstructing measurements
    // from final poses was tried and confirmed wrong this session: an edge
    // whose own raw correction was trivial produced a chi2 of 18105 because
    // its endpoint's absolute pose had since been perturbed by a different,
    // later closure.
    std::vector<KeyframePose> original = keyframes;
    for (auto &kf : original) {
        kf.R = kf.R.clone();
        kf.t = kf.t.clone();
    }

    std::vector<EdgeMeasurement> sequentialEdges;
    sequentialEdges.reserve(sequentialRecords.size());
    for (const SequentialEdgeRecord &sr : sequentialRecords) {
        if (sr.i < 0 || sr.j < 0 || sr.i >= n || sr.j >= n || sr.i == sr.j)
            continue; // defensive -- should always be valid, but never corrupt the graph silently
        sequentialEdges.push_back({sr.i, sr.j, sr.R, sr.t});
    }

    // Warm-start by CHAINING the sequential edges forward from keyframe 0's
    // original pose -- NOT from every keyframe's own final `original` pose.
    // Sequential edges are captured once, at insertion time; a keyframe's
    // FINAL absolute pose (in `original`) may since have been overwritten
    // in place by any number of later, unrelated loop-closure corrections
    // live tracking applied. Warm-starting from those final poses evaluates
    // fresh, uncorrected relative measurements against an incompatible,
    // already-corrected reference frame -- confirmed empirically: doing so
    // produced an initial cost in the quadrillions and a subsequent
    // catastrophic scale collapse. Chaining keeps every consecutive pair
    // exactly self-consistent with its own measurement from the start;
    // indexed by edge start so a missing/out-of-order edge (shouldn't
    // normally happen) falls back to an identity step rather than crashing.
    std::vector<std::array<double, 6>> params(static_cast<size_t>(n));
    fillParamsFromPose(params[0], original[0].R, original[0].t);
    std::vector<const EdgeMeasurement *> edgeFromIndex(static_cast<size_t>(n), nullptr);
    for (const EdgeMeasurement &e : sequentialEdges) {
        if (e.j == e.i + 1)
            edgeFromIndex[static_cast<size_t>(e.i)] = &e;
    }
    for (int i = 1; i < n; ++i) {
        cv::Mat Rprev, tprev;
        poseFromParams(params[static_cast<size_t>(i - 1)], Rprev, tprev);
        const EdgeMeasurement *e = edgeFromIndex[static_cast<size_t>(i - 1)];
        cv::Mat Ri, ti;
        if (e) {
            Ri = e->R * Rprev;
            ti = e->t + e->R * tprev;
        } else {
            Ri = Rprev; // identity step -- no persisted edge for this pair (defensive fallback)
            ti = tprev;
        }
        fillParamsFromPose(params[static_cast<size_t>(i)], Ri, ti);
    }

    if (warmStartOut) {
        warmStartOut->assign(static_cast<size_t>(n), KeyframePose());
        for (int i = 0; i < n; ++i) {
            (*warmStartOut)[static_cast<size_t>(i)].frameIndex = original[static_cast<size_t>(i)].frameIndex;
            poseFromParams(params[static_cast<size_t>(i)], (*warmStartOut)[static_cast<size_t>(i)].R,
                            (*warmStartOut)[static_cast<size_t>(i)].t);
        }
    }

    std::vector<LoopEdge> loopEdges;
    loopEdges.reserve(loopClosures.size());
    for (const LoopClosureRecord &lc : loopClosures) {
        if (lc.oldKfIdx < 0 || lc.newKfIdx < 0 || lc.oldKfIdx >= n || lc.newKfIdx >= n || lc.oldKfIdx == lc.newKfIdx)
            continue; // defensive -- indices should always be valid, but never corrupt the graph silently
        LoopEdge e;
        static_cast<EdgeMeasurement &>(e) = {lc.oldKfIdx, lc.newKfIdx, lc.R, lc.t};
        loopEdges.push_back(std::move(e));
    }

    bool everSucceeded = false;
    for (int outer = 0; outer < opts.outerIterations; ++outer) {
        ceres::Problem problem;
        for (int i = 0; i < n; ++i)
            problem.AddParameterBlock(params[static_cast<size_t>(i)].data(), 6);

        for (const EdgeMeasurement &e : sequentialEdges)
            problem.AddResidualBlock(RelativePoseCost::Create(e.R, e.t, opts.rotationWeight),
                                      new ceres::HuberLoss(opts.sequentialHuberDelta),
                                      params[static_cast<size_t>(e.i)].data(), params[static_cast<size_t>(e.j)].data());

        for (const LoopEdge &e : loopEdges)
            problem.AddResidualBlock(RelativePoseCost::Create(e.R, e.t, opts.rotationWeight),
                                      new ceres::ScaledLoss(nullptr, e.weight, ceres::TAKE_OWNERSHIP),
                                      params[static_cast<size_t>(e.i)].data(), params[static_cast<size_t>(e.j)].data());

        problem.SetParameterBlockConstant(params[0].data()); // gauge fix -- anchor the whole graph to keyframe 0

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = opts.maxSolverIterations;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1; // pinned for run-to-run reproducibility -- see kitti_ate.cpp's
        // own cv::setNumThreads(1) call for the matching OpenCV-side fix
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (!summary.IsSolutionUsable()) {
            options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            options.preconditioner_type = ceres::JACOBI;
            ceres::Solve(options, &problem, &summary);
        }
        if (!summary.IsSolutionUsable()) {
            std::fprintf(stderr, "[posegraph] outer %d: solve failed, stopping: %s\n", outer,
                         summary.BriefReport().c_str());
            break;
        }
        everSucceeded = true;
        std::fprintf(stderr, "[posegraph] outer %d: cost %.3f -> %.3f\n", outer, summary.initial_cost,
                     summary.final_cost);

        // Summary (not per-edge -- there are thousands) of the sequential
        // edges' own chi2 distribution at the just-solved poses, so
        // sequentialHuberDelta can be calibrated the same way dcsPhi is:
        // by inspecting where the natural distribution actually sits
        // rather than guessing.
        {
            double maxChi2 = 0.0, sumChi2 = 0.0;
            int aboveHuber = 0;
            for (const EdgeMeasurement &e : sequentialEdges) {
                const RelativePoseCost functor(e.R, e.t, opts.rotationWeight);
                double residuals[6];
                functor(params[static_cast<size_t>(e.i)].data(), params[static_cast<size_t>(e.j)].data(), residuals);
                double chi2 = 0.0;
                for (double r : residuals)
                    chi2 += r * r;
                maxChi2 = std::max(maxChi2, chi2);
                sumChi2 += chi2;
                if (chi2 > opts.sequentialHuberDelta)
                    ++aboveHuber;
            }
            std::fprintf(stderr,
                         "[posegraph][seq] outer %d: %zu edges, mean chi2=%.4f max=%.4f, %d above "
                         "sequentialHuberDelta=%.2f\n",
                         outer, sequentialEdges.size(), sumChi2 / static_cast<double>(sequentialEdges.size()),
                         maxChi2, aboveHuber, opts.sequentialHuberDelta);
        }

        // Recompute each loop edge's chi2 at the just-solved poses, then its
        // DCS weight for the NEXT outer iteration.
        for (LoopEdge &e : loopEdges) {
            const RelativePoseCost functor(e.R, e.t, opts.rotationWeight);
            double residuals[6];
            functor(params[static_cast<size_t>(e.i)].data(), params[static_cast<size_t>(e.j)].data(), residuals);
            double chi2 = 0.0;
            for (double r : residuals)
                chi2 += r * r;
            const double s = std::min(1.0, 2.0 * opts.dcsPhi / (opts.dcsPhi + chi2));
            std::fprintf(stderr, "[posegraph][dcs] outer %d loop kf#%d<->kf#%d chi2=%.3f s=%.5f weight=%.6f\n", outer,
                         e.i, e.j, chi2, s, s * s);
            e.weight = s * s;
        }
    }

    if (!everSucceeded)
        return false;

    for (int i = 0; i < n; ++i)
        poseFromParams(params[static_cast<size_t>(i)], keyframes[static_cast<size_t>(i)].R,
                        keyframes[static_cast<size_t>(i)].t);
    return true;
}

QVector<QPointF> applyPoseGraphCorrection(const std::vector<KeyframePose> &originalKeyframes,
                                           const std::vector<KeyframePose> &optimizedKeyframes,
                                           const QVector<QPointF> &trajectory, const QVector<int> &trajectoryFrameIndex,
                                           const std::vector<LoopClosureRecord> &loopClosures)
{
    const size_t n = originalKeyframes.size();
    QVector<QPointF> corrected = trajectory;
    if (n == 0 || n != optimizedKeyframes.size() || trajectory.size() != trajectoryFrameIndex.size())
        return corrected;

    // A keyframe is "covered" by a loop closure whose span [oldKfIdx,
    // newKfIdx] is "local" (<= kMaxLocalSpan keyframes, the same threshold
    // SlamWorker.cpp's own kBaMaxWindowKeyframes uses to decide a loop
    // window is small enough to treat as one continuous, correctable
    // stretch) if it falls ANYWHERE in that span. For a "return to start"
    // closure whose two endpoints are numerically far apart (e.g. observed
    // this session: kf#0<->kf#330, spanning nearly the whole sequence),
    // only keyframes within kCoverageWindow of EITHER endpoint are
    // covered -- treating the whole numeric range between two such
    // endpoints as covered would mark almost the entire sequence covered,
    // including long genuine gaps with zero loop-closure evidence
    // (confirmed empirically: a real ~600-frame, kf#299-to-kf#330 gap was
    // incorrectly "corrected" this way with no actual improvement). The
    // opposite mistake (treating EVERY closure with only a bounded window)
    // was also confirmed empirically to under-cover genuinely broad, local
    // revisit windows like kf#10-17<->kf#124-130, leaving a new,
    // previously-fine stretch (frame ~1181-1227) uncorrected instead.
    constexpr int kMaxLocalSpan = 200; // == SlamWorker.cpp's kBaMaxWindowKeyframes
    constexpr int kCoverageWindow = 15; // matches this codebase's own kLocalRefineWindow-scale choices
    std::vector<bool> covered(n, false);
    for (const LoopClosureRecord &lc : loopClosures) {
        if (lc.oldKfIdx < 0 || lc.newKfIdx < 0 || static_cast<size_t>(lc.oldKfIdx) >= n ||
            static_cast<size_t>(lc.newKfIdx) >= n)
            continue;
        const int lo = std::min(lc.oldKfIdx, lc.newKfIdx);
        const int hi = std::max(lc.oldKfIdx, lc.newKfIdx);
        if (hi - lo + 1 <= kMaxLocalSpan) {
            for (size_t k = static_cast<size_t>(lo); k <= static_cast<size_t>(hi); ++k)
                covered[k] = true;
        } else {
            for (int endpoint : {lc.oldKfIdx, lc.newKfIdx}) {
                const size_t wlo = static_cast<size_t>(std::max(0, endpoint - kCoverageWindow));
                const size_t whi = std::min(n - 1, static_cast<size_t>(endpoint + kCoverageWindow));
                for (size_t k = wlo; k <= whi; ++k)
                    covered[k] = true;
            }
        }
    }

    // Per-keyframe world-frame delta between its original (pre-optimization)
    // and optimized pose -- same construction SlamWorker::tryLoopClosure()
    // uses for its own single shared delta, just one per keyframe here.
    // Reduced to yaw + planar translation since the trajectory only ever
    // stores (world X, world Z).
    std::vector<double> yaw(n), tdx(n), tdz(n);
    for (size_t i = 0; i < n; ++i) {
        const cv::Mat &Rorig = originalKeyframes[i].R;
        const cv::Mat &torig = originalKeyframes[i].t;
        const cv::Mat &Ropt = optimizedKeyframes[i].R;
        const cv::Mat &topt = optimizedKeyframes[i].t;

        const cv::Mat RcwOrig = Rorig.t();
        const cv::Mat COrig = -RcwOrig * torig;
        const cv::Mat RcwOpt = Ropt.t();
        const cv::Mat COpt = -RcwOpt * topt;

        const cv::Mat Rdelta = RcwOpt * RcwOrig.t();
        const cv::Mat tDelta = COpt - Rdelta * COrig;

        yaw[i] = std::atan2(-Rdelta.at<double>(2, 0), Rdelta.at<double>(0, 0));
        tdx[i] = tDelta.at<double>(0);
        tdz[i] = tDelta.at<double>(2);
    }

    // Two-pointer sweep: both the trajectory's frame indices and the
    // keyframes' frame indices are non-decreasing, so this is O(N) total
    // rather than a binary search per point.
    size_t kfIdx = 0;
    for (int ti = 0; ti < trajectory.size(); ++ti) {
        const int f = trajectoryFrameIndex[ti];
        while (kfIdx + 1 < n && originalKeyframes[kfIdx + 1].frameIndex <= f)
            ++kfIdx;

        // Not bracketed by any loop closure -- leave this point exactly as
        // `trajectory` already had it (already the default via the
        // `corrected = trajectory` copy above).
        const bool bracketCovered = covered[kfIdx] && (kfIdx + 1 >= n || covered[kfIdx + 1]);
        if (!bracketCovered)
            continue;

        double blendYaw, blendTx, blendTz;
        if (kfIdx + 1 >= n || originalKeyframes[kfIdx + 1].frameIndex <= originalKeyframes[kfIdx].frameIndex) {
            // At or past the last keyframe -- no bracketing pair available, use its own delta directly.
            blendYaw = yaw[kfIdx];
            blendTx = tdx[kfIdx];
            blendTz = tdz[kfIdx];
        } else {
            const int fA = originalKeyframes[kfIdx].frameIndex;
            const int fB = originalKeyframes[kfIdx + 1].frameIndex;
            const double beta =
                std::clamp(static_cast<double>(f - fA) / static_cast<double>(fB - fA), 0.0, 1.0);
            blendYaw = (1.0 - beta) * yaw[kfIdx] + beta * yaw[kfIdx + 1];
            blendTx = (1.0 - beta) * tdx[kfIdx] + beta * tdx[kfIdx + 1];
            blendTz = (1.0 - beta) * tdz[kfIdx] + beta * tdz[kfIdx + 1];
        }

        const double c = std::cos(blendYaw), s = std::sin(blendYaw);
        const QPointF &p = trajectory[ti];
        const double x = p.x(), z = p.y();
        corrected[ti] = QPointF(c * x + s * z + blendTx, -s * x + c * z + blendTz);
    }

    return corrected;
}

} // namespace pose_graph
