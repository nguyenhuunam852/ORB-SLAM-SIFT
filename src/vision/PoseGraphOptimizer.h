#pragma once

// Offline pose-graph optimizer with Dynamic Covariance Scaling (DCS) on
// loop-closure edges -- Agarwal, Tipaldi, Spinello, Stachniss, Burgard,
// "Robust Map Optimization using Dynamic Covariance Scaling", ICRA 2013.
//
// Why this exists: this codebase's only two attempts (2026-07-15) at
// improving on the documented 17.141m ATE baseline both regressed for the
// same root cause -- an occasional bad/wrong loop-closure measurement
// (confirmed example: kf#34 (frame 402) <-> kf#202 (frame 2452), a
// 106.789-world-unit/11.991-degree correction, ~30x every neighboring
// closure that run) gets trusted unconditionally by every existing
// correction path (interpolation, per-window BA). DCS gives each
// loop-closure edge a closed-form weight s = min(1, 2*Phi/(Phi+chi2))
// applied (as s^2) to its information matrix, recomputed iteratively as the
// graph solves -- a bad edge's own large residual causes it to be
// automatically down-weighted, with no extra switch variables needed. This
// is also the vehicle for the *other* previously-failed goal (continuous
// inter-loop drift correction): the graph spans every keyframe via
// sequential edges, not just one loop's local window.
//
// Deliberately standalone: takes plain pose/edge data, not SlamWorker's own
// private Keyframe struct (no landmarks/descriptors/reprojection at all --
// this is pure 6-DOF relative-pose consistency, a fundamentally simpler and
// more robust problem than the existing reprojection-error BA). Runs once,
// offline, as an opt-in post-processing step (see analyze/kitti_ate.cpp's
// `posegraph` flag) -- never wired into live/continuous tracking, so it
// cannot regress the live GUI/default path the way this session's two
// earlier in-place attempts risked.

#include <opencv2/core.hpp>

#include <QPointF>
#include <QVector>

#include <vector>

namespace pose_graph {

// World-to-camera, exactly matching SlamWorker::Keyframe's own (R, t)
// convention (Xcam = R*Xworld + t). frameIndex mirrors Keyframe::frameIndex
// (the m_frameCount value at insertion).
struct KeyframePose
{
    cv::Mat R, t;
    int frameIndex = 0;
};

// The RELATIVE transform from keyframe oldKfIdx to keyframe newKfIdx, as
// independently PnP-measured by SlamWorker::tryLoopClosure() (verified
// against oldKfIdx's own triangulated landmarks) at the moment the loop
// fired: R = loopR * oldKf.R_at_that_moment.t(), t = loopT - R*oldKf.t_at_that_moment
// (world-to-camera convention, same as buildEdge() below). Deliberately a
// RELATIVE measurement, not the absolute (loopR, loopT) paired with
// oldKfIdx's CURRENT/final pose -- oldKfIdx's own absolute pose can be
// further modified by later, unrelated loop closures before a full run
// finishes, which would silently corrupt a naive "measurement vs. final
// pose" reconstruction (confirmed empirically: an edge whose own raw
// correction was a trivial 3.55 world units / 1.09 degrees produced a
// chi2 of 18105 when built that way, because its OLD endpoint's absolute
// pose had since been perturbed by a different, later closure). A
// relative measurement captured once, at observation time, is immune to
// this by construction -- exactly the point of pose-graph SLAM.
struct LoopClosureRecord
{
    int oldKfIdx = -1;
    int newKfIdx = -1;
    cv::Mat R, t;

    // Real scale-drift measurement: the ratio (PnP-loop-measured
    // oldKf<->newKf camera-center distance) / (live/drifted oldKf<->newKf
    // camera-center distance), both computed at observation time in
    // SlamWorker::tryLoopClosure(). loopR/loopT (which R/t above derive
    // from) is a PnP solve against oldKf's own already-triangulated,
    // real-scale 3D points -- PnP has no scale ambiguity of its own, so
    // this ratio isolates exactly how far the live trajectory's own
    // internal scale has drifted from real scale by the time it looped
    // back around, without needing a separate re-triangulated 3D-3D
    // similarity solve (which would only re-derive a noisier version of
    // the rotation/translation this file already trusts). Only loop edges
    // carry a real value here -- sequential edges have no independent
    // scale evidence (see SequentialEdgeRecord, which has no such field)
    // and PoseGraphOptimizer.cpp's Sim(3) path keeps sMeas=1 for those,
    // same as before. Defaults to 1.0 (scale-neutral) so any caller that
    // doesn't set it explicitly gets the old SE(3)-equivalent behavior.
    double scale = 1.0;
};

// The RELATIVE transform from keyframe i to keyframe i+1 (i = index into
// SlamWorker::keyframePoses()' result), captured in SlamWorker::insertKeyframe()
// at the moment the new keyframe is inserted -- i.e. from both keyframes'
// poses as they stood at that instant, before any future loop closure could
// touch either one. Same rationale as LoopClosureRecord above: this must be
// a relative measurement, not reconstructed later from (possibly
// multiply-corrected) final absolute poses.
struct SequentialEdgeRecord
{
    int i = -1;
    int j = -1;
    cv::Mat R, t;
};

struct PoseGraphOptions
{
    int outerIterations = 5; // DCS/IRLS re-weight-then-re-solve rounds
    double dcsPhi = 1000.0;     // DCS tuning constant -- NOT chi-square-calibrated by default, see
                             // PoseGraphOptimizer.cpp's calibration note; must be tuned empirically
    double rotationWeight = 8.0;  // world-units-per-radian scale so rotation/translation residuals
                                  // are comparable in one combined chi-square
    int maxSolverIterations = 50; // matches SlamWorker.cpp's kBaMaxIterations convention
    double sequentialHuberDelta = 3.0; // Huber threshold (combined rotation+translation residual
                                       // units, see rotationWeight) for sequential edges -- unlike
                                       // loop edges (DCS-scaled), sequential edges are always
                                       // present and were originally given zero robustification at
                                       // all, letting one difficult tracking segment (fast turn,
                                       // sparse features) inject unbounded, permanent drift into
                                       // everything downstream until the next loop edge; confirmed
                                       // empirically this session (see DEBUGGING.md Session 10 continued)

    // Default off (behavior-preserving): every edge (sequential AND loop)
    // currently constrains keyframe j's translation directly against a
    // FIXED measured t_meas, with no way to explain a translation-magnitude
    // mismatch except by moving t_i/t_j themselves -- fine for rotation
    // (which doesn't drift the way monocular translation SCALE does), but
    // rigid for translation: this codebase's own per-step scale
    // (m_avgStepScale) drifts continuously along the sequential chain (see
    // SlamWorker.h's own doc comment on it), so a long chain of
    // individually-reasonable sequential edges can still accumulate a
    // scale error a rigid model can only correct by fighting the
    // (well-constrained) rotation/direction of each edge to make the
    // magnitudes work out. Real ORB-SLAM2/3 solves exactly this with a
    // Sim(3) (7-DOF: rotation + translation + scale) essential graph
    // (Strasdat et al., "Scale Drift-Aware Large-Scale Monocular SLAM",
    // RSS 2010) instead of SE(3) -- when true, optimizePoseGraph() adds a
    // free per-keyframe log-scale parameter (see PoseGraphOptimizer.cpp's
    // SimilarityPoseCost) that lets translation MAGNITUDE drift smoothly
    // across the chain, constrained only by scaleWeight's soft "no
    // unexplained jump" prior between adjacent nodes -- while a loop
    // edge's own measurement (independently PnP-measured against the OLD
    // keyframe's own already-scaled local map, so its translation
    // magnitude is NOT subject to the same drift) still anchors the
    // absolute scale wherever a loop closure provides one. Implemented as
    // an entirely separate solve path from the existing SE(3) one (not a
    // parameterized unification of the same code) specifically so this
    // flag defaulting to false is a hard guarantee of zero change to the
    // already-measured SE(3) results, not just an intended one.
    bool useSim3 = false;
    double scaleWeight = 8.0; // useSim3 only: weight on the log-scale-ratio residual between two
                              // nodes connected by an edge (both sequential and loop) -- larger
                              // values push per-keyframe scale harder toward matching its neighbors
                              // (closer to the rigid SE(3) behavior); smaller values let scale drift
                              // more freely between what the loop edges' own absolute measurements
                              // pin down. Not chi-square-calibrated, same caveat as rotationWeight.

    // useSim3 only: multiplies the information matrix (i.e. inverse
    // covariance -- NOT chi2 directly, chi2 scales with the SQUARE of this)
    // for every sequential AND covisibility edge (SlamWorker::
    // covisibilityEdgeRecords(), see its own doc comment), relative to loop
    // edges which always use weight 1.0. Added 2026-07-21 after confirming
    // Essential-Graph-style covisibility edges give sequential+covisibility
    // edges real, nonzero internal constraint for the first time (previously
    // every sequential edge's chi2 measured exactly 0.000 with no
    // covisibility edges at all) -- but even so, loop-edge chi2 still
    // outweighed sequential+covisibility chi2 by ~18x (1034.8 vs 57.6 on a
    // real run), meaning the graph was still almost entirely loop-edge-
    // driven and posegraph correction still underperformed live tracking
    // (152.480m vs 125.195m). This lets that imbalance be corrected without
    // touching dcsPhi/scaleWeight (which only affect the loop-edge side).
    // Defaults to 1.0 (no change from before this option existed).
    double sequentialWeightMultiplier = 1.0;
};

// Mutates keyframes[i].R/.t in place on success (left untouched on failure).
// sequentialEdges/loopClosures' indices must index into `keyframes` (i.e.
// m_keyframeHistory-index-aligned -- see SlamWorker::keyframePoses()/
// sequentialEdgeRecords()/loopClosureRecords()). If warmStartOut is
// non-null, it's filled with the CHAINED warm-start poses the solver
// actually started from (see .cpp: chained forward from keyframes[0]'s
// incoming pose via the sequential edges, NOT keyframes[i]'s own incoming
// values, which may reflect live tracking's own unrelated in-place loop
// corrections and are inconsistent with these fresh relative measurements)
// -- callers doing a pre-vs-post delta (e.g. applyPoseGraphCorrection())
// MUST use this as the "pre" reference, not the poses `keyframes` held on
// entry, or the delta conflates the optimizer's real refinement with an
// arbitrary difference between two unrelated reference chains (confirmed
// empirically: doing it the wrong way produced the exact same catastrophic
// scale collapse before and after fixing the warm-start itself).
bool optimizePoseGraph(std::vector<KeyframePose> &keyframes, const std::vector<SequentialEdgeRecord> &sequentialEdges,
                        const std::vector<LoopClosureRecord> &loopClosures,
                        const PoseGraphOptions &options = PoseGraphOptions(),
                        std::vector<KeyframePose> *warmStartOut = nullptr);

// Re-derives the full trajectory by blending each keyframe's own
// pre-vs-post-optimization world-frame delta onto each trajectory point's
// OWN EXISTING (X, Z) position -- never re-synthesizing a fresh position
// from bracketing keyframes alone, which would flatten real per-frame
// motion between keyframes (confirmed regression in this session's first
// reverted attempt at this exact problem). originalKeyframes MUST be
// optimizePoseGraph()'s warmStartOut, not whatever `keyframes` held before
// calling it -- see that function's own doc comment for why.
//
// Only touches trajectory points whose BOTH bracketing keyframes fall
// inside the keyframe-index span of at least one loopClosures entry --
// everywhere else (e.g. a long stretch with no loop closure at all) is left
// EXACTLY as `trajectory` already had it. Confirmed empirically this
// session (DEBUGGING.md Session 10 continued): a pure sequential-edge chain
// has no shared-landmark stabilization the way live tracking's rolling map
// does, so it accumulates MORE drift than live tracking over any stretch
// with no loop-closure evidence to correct it -- there is nothing for this
// optimizer to legitimately improve there, so it must not touch it.
QVector<QPointF> applyPoseGraphCorrection(const std::vector<KeyframePose> &originalKeyframes,
                                           const std::vector<KeyframePose> &optimizedKeyframes,
                                           const QVector<QPointF> &trajectory, const QVector<int> &trajectoryFrameIndex,
                                           const std::vector<LoopClosureRecord> &loopClosures);

} // namespace pose_graph
