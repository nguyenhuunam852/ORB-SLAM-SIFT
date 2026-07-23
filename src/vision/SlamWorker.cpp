#include "SlamWorker.h"

#include "EightPointLegacy.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMap>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QTemporaryFile>
#include <QTextStream>
#include <QTimer>
#include <QUuid>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

// Full ORB_SLAM3::System definition -- only forward-declared in
// SlamWorker.h (see its own doc comment on why). Brings in the whole
// vendored third_party/ORB_SLAM3 header set (Sophus, Eigen, etc.), which is
// why this stays confined to this .cpp rather than the header.
#include <System.h>
// MapPoint's full definition (GetWorldPos()) -- System.h only exposes
// std::vector<MapPoint*> via GetTrackedMapPoints(), not the class body.
#include <MapPoint.h>
// Atlas/KeyFrame full definitions -- needed to rebuild the live trajectory
// from GetAtlas()->GetAllKeyFrames() every frame (see trackFrameOrbSlam3()).
#include <Atlas.h>
#include <KeyFrame.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>

namespace {
constexpr int kProcessIntervalMs = 100; // ~10 fps, close to KITTI's native ~9.65 Hz capture rate
constexpr int kMinInitMatches = 20; // lower = more chances for the F/E pipeline to even attempt a
                                     // solution at a low-texture/hard-geometry spot; the strict
                                     // plausibility check is what guards against a bad one, not this
constexpr int kMinInitInliers = 20;
constexpr int kMinInitMapPoints = 15;
constexpr int kMinTrackMatches = 15;
// kMinTrackInliers moved to SlamWorker::m_minTrackInliers (see setMinTrackInliers()) -- was a fixed
// constexpr here, now runtime-overridable; default value (10) unchanged.
constexpr int kKeyframeEveryNFrames = 8; // lower = fresher reference keyframe, which matters most
                                          // during fast viewpoint change (e.g. a turn), where an
                                          // old keyframe quickly loses overlap with the current view
constexpr int kLostDisplayThreshold = 5; // consecutive failed frames before showing "Lost" in the UI
constexpr int kStepScaleResetStreak = 50; // after this many CONSECUTIVE total-tracking-failure frames
                                           // (both trackFrame() and recoverViaEpipolar() failing), reset
                                           // m_avgStepScale/m_longTermStepScale/m_recentStepDistances back
                                           // to "unknown" (same state resetSlamState() leaves them in) --
                                           // isPlausibleStep() returns true unconditionally whenever
                                           // m_avgStepScale <= 0, so this re-opens the plausibility gate
                                           // for the next candidate instead of leaving it permanently
                                           // shut. m_longTermStepScale's own floor guards against a
                                           // *sudden single-frame* collapse, but not a *sustained* run of
                                           // many legitimately-accepted-looking small steps dragging both
                                           // the median and the slow EMA down together -- confirmed this
                                           // session with the ORB detector (see FeatureDetector.h),
                                           // independent of PnP method: avgStepScale collapsed to ~0.001
                                           // around frame ~1560-1580 on KITTI seq00, after which every
                                           // subsequent candidate step (however correct) was rejected as
                                           // implausible forever, freezing the trajectory for the rest of
                                           // the run with no recovery mechanism at all. Retried every
                                           // kStepScaleResetStreak frames, not just once, in case the
                                           // difficult stretch outlasts a single reset. Large enough
                                           // relative to kLostDisplayThreshold (5) to not fire on a brief,
                                           // ordinary recovery blip.
constexpr size_t kMaxMapPoints = 2000; // rolling window: oldest points are evicted past this
constexpr int kScaleWindowSize = 20; // m_avgStepScale is the median of the last this-many
                                      // independently-measured step distances -- locally
                                      // responsive to real speed changes and outlier-robust,
                                      // unlike a single whole-video EMA (see m_avgStepScale's
                                      // doc comment in SlamWorker.h)
constexpr double kLongTermScaleAlpha = 0.02; // slow EMA rate for m_longTermStepScale
constexpr double kMinScaleFraction = 0.15; // m_avgStepScale is never allowed below this fraction of
                                            // m_longTermStepScale -- see m_longTermStepScale's doc
                                            // comment in SlamWorker.h for the deadlock this prevents
constexpr double kMaxStepMultiplier = 10.0; // reject a candidate pose implying a step further than
                                             // this many times the recent average -- loose enough to
                                             // tolerate some zigzag through a hard corner, still
                                             // bounded enough to catch a degenerate RANSAC solve that
                                             // passed the inlier check (no bypass past this, ever --
                                             // see isPlausibleStep())
constexpr double kMaxAvgStepUpdateMultiplier = 2.0; // separate, much tighter cap on how far a single
                                                     // *accepted* step is allowed to move
                                                     // m_avgStepScale itself (clamped before the EMA
                                                     // update, not just before accept/reject). Without
                                                     // this, kMaxStepMultiplier's generous 10x accept
                                                     // window lets a single genuinely-large-but-still-
                                                     // "plausible" step (e.g. during a fast turn) nearly
                                                     // double the running average in one EMA update
                                                     // (0.9*avg + 0.1*(10*avg) ~= 1.9*avg); the new,
                                                     // larger average then makes the *next* 10x window
                                                     // even more permissive, producing unbounded
                                                     // runaway growth over a long run with no downward
                                                     // correction (confirmed via a full-video run this
                                                     // session -- avgStepScale grew from ~1 to over
                                                     // 20000 -- see DEBUGGING.md). The pose itself is
                                                     // still accepted/used at its real (unclamped) size
                                                     // for tracking continuity; only what feeds the
                                                     // *estimator* is clamped.
constexpr unsigned kRansacSeed = 42; // fixed (not std::random_device) so that a given input video
                                      // produces byte-identical RANSAC sample sequences on every
                                      // run -- debugging a marginal-frame failure is impossible if
                                      // the outcome is a coin flip that changes between runs of the
                                      // exact same code and video (confirmed this was happening --
                                      // see DEBUGGING.md, session 2026-07-11)
constexpr int kMaxFramesElapsedForRescale = 15; // cap how many frames' worth of average motion a
                                                 // stale reference (frozen because matches kept
                                                 // failing quality checks, not the match-count
                                                 // check, so it never slid forward) is assumed to
                                                 // cover -- otherwise a long stuck stretch inflates
                                                 // the rescaled step to an arbitrarily huge, made-up
                                                 // distance once something finally passes again
// kDetectionScale moved to SlamWorker::m_detectionScale (see setDetectionScale()) -- was a fixed
// constexpr here, default value (0.5) unchanged. SIFT cost is dominated by scale-space pyramid
// construction over the image, not by nFeatures -- detecting on a half-resolution copy cuts it
// from ~70ms/frame to ~18ms (measured on KITTI's 1240x376 frames), which is what was starving the
// live GUI's ~100ms per-frame real-time budget. Keypoints are rescaled back to full-resolution
// pixel coordinates right after detection. That real-time constraint doesn't apply to an offline,
// unthrottled harness like kitti_ate (see its own startUnthrottled() call), which is why this
// became overridable instead of staying fixed.
constexpr double kMaxTriangulationDepth = 1000.0;
constexpr double kFRansacSampsonThreshold = 1.0; // squared-pixel Sampson error -- still used by
                                                  // estimateEssentialRansac() to score E candidates
                                                  // (converted back through K), not just the old F path
constexpr int kERansacIterations = 300; // 5-point's minimal sample (5 vs F's 8) needs far fewer
                                         // RANSAC iterations for the same outlier-tolerance/confidence
                                         // (standard RANSAC sample-count theory), which is what offsets
                                         // its per-iteration cost being ~4x F's (measured: 0.054ms vs
                                         // 0.014ms per minimal solve, due to the polynomial elimination
                                         // + eigendecomposition a minimal-5-point solve requires that an
                                         // 8-point linear solve doesn't -- see DEBUGGING.md)
constexpr int kHRansacIterations = 500;
constexpr double kHRansacThreshold = 4.0; // squared-pixel symmetric transfer error (2 * 1px budget)
constexpr double kHomographyPreferenceRatio = 0.45; // ORB-SLAM's R_H threshold: prefer H over F/E
                                                     // once H explains this fraction of the combined
                                                     // inlier support -- catches the (near-)planar or
                                                     // (near-)pure-rotation case where F/E is
                                                     // ill-conditioned (e.g. mid-turn at an intersection)
constexpr int kLoopExclusionWindow = 30; // don't search the most recent N keyframes for a loop match --
                                          // otherwise the ordinary previous keyframe (which always
                                          // shares heavy overlap with the current one) would trivially
                                          // "close a loop" with itself every time
constexpr int kLoopMinMatches = 60; // raw descriptor match count against a candidate keyframe needed to
                                     // call it a revisit -- deliberately stricter than kMinInitMatches
                                     // (20): a false-positive loop closure corrupts the trajectory, so
                                     // this favors missing real loops over accepting fake ones
constexpr int kLoopMinPnpInliers = 20; // minimum inlier count for the loop pose PnP solve to be trusted
constexpr double kDbowMinScore = 0.015; // tryLoopClosure()'s DBoW2 branch (see setDbowLoopClosureEnabled()):
                                         // minimum vocabulary score() between two keyframes' BowVectors to
                                         // treat one as a loop candidate for the other. DBoW2's L1-based
                                         // score() ranges roughly 0 (nothing in common) to ~1 (identical
                                         // image); unrelated KITTI frames typically score well under 0.01,
                                         // genuine revisits noticeably higher -- this is an initial,
                                         // untuned-per-sequence estimate (ORB-SLAM2 instead derives its own
                                         // threshold per-query from covisible-neighbor scores, which this
                                         // simpler integration doesn't replicate), kept low deliberately so
                                         // a missed threshold shows up as too many false positives (visible
                                         // in the [loop] log) rather than silently finding zero loops at all.
constexpr double kSiftDbowMinScore = 0.015; // tryLoopClosure()'s SIFT-DBoW2 branch (see
                                             // setSiftDbowLoopClosureEnabled()): same L1-based score() range
                                             // and same untuned-per-sequence-estimate spirit as kDbowMinScore
                                             // (the ORB-DBoW2 threshold) -- kept as its own constant rather
                                             // than reused directly since it's a different vocabulary trained
                                             // on different data (RootSIFT, not ORB), so the two may need to
                                             // diverge once real measurements exist.
constexpr double kVladMinScore = 0.05; // tryLoopClosure()'s VLAD branch (see setVladLoopClosureEnabled()):
                                        // minimum cosine-similarity score() between two keyframes' VLAD
                                        // vectors to treat one as a loop candidate for the other. Reference
                                        // data from this codebase's own VLAD sanity check (RootSIFT
                                        // codebook, see vocabulary_sift/vlad_codebook_all_rootsift.yml's
                                        // training log): adjacent-frame similarity ~0.45, distant-frame
                                        // (frame0<->frame200) similarity ~0.006 -- 0.05 sits well above the
                                        // distant-frame noise floor while staying well below a genuine
                                        // revisit's score, same untuned-per-sequence-estimate spirit as
                                        // kDbowMinScore above.
constexpr double kLoopMaxCorrectionMagnitude = 100000.0; // tryLoopClosure()'s degenerate-solve guard
                                                           // (see its own doc comment at the check site
                                                           // for the full reasoning) -- world units,
                                                           // ~50x the largest genuine loop correction
                                                           // observed across sessions (~2000 units), so
                                                           // it only ever engages for numerically
                                                           // pathological PnP solves, never real drift.
constexpr double kLoopMaxCorrectionAngleDeg = 90.0; // same guard, rotation component -- ~3x the largest
                                                     // genuine correction observed (~30 degrees).
// Loop-closure QUALITY gate (item 40, setLoopQualityGateEnabled()): reject a
// closure whose Sim3-measured scale is extreme (far from 1.0) AND supported by
// too few Sim3 inliers -- the signature of an UNRELIABLE scale measurement (garbage
// loop) rather than real drift, which map-compresses and degrades tracking. Unlike
// the deliberately-loose degenerate-solve caps above, this targets the "few-inlier
// large-scale" case specifically, and ONLY when both conditions hold, to avoid the
// documented vicious cycle of rejecting real large-drift corrections (a genuine
// large drift has MANY inliers, so it passes). Untuned.
constexpr int kLoopQualityMinSim3Inliers = 15; // below this, an extreme scaleMeas is untrustworthy
constexpr double kLoopQualityScaleLo = 0.70; // extreme-scale band: reject only OUTSIDE [lo, hi]
constexpr double kLoopQualityScaleHi = 1.45;
constexpr int kLoopConsistencyRequiredCount = 2; // setLoopConsistencyGroupEnabled()'s own gate --
                                                  // ORB-SLAM3's own mnLoopNumCoincidences>=3
                                                  // (LoopClosing.cc) was tried literally twice (items
                                                  // 15/21, both measured negative -- most streaks died at
                                                  // exactly 2/3, one confirmation short) and lowered to 2
                                                  // here to test whether this pipeline's own re-detection
                                                  // cadence (sparser/noisier than ORB-SLAM3's) simply can't
                                                  // sustain 3 -- see DEBUGGING.md item 22.
constexpr int kLoopConsistencyOldIdxWindow = 5; // FALLBACK-only "same place" test (see
                                                 // kLoopConsistencyPlaceMinScore below for the primary
                                                 // one) -- only used when no place-recognition vector is
                                                 // available for either old keyframe (raw-match-count
                                                 // candidate search). Untuned-per-sequence estimate.
constexpr double kLoopConsistencyPlaceMinScore = 0.3; // setLoopConsistencyGroupEnabled()'s primary "same
                                                       // place" test: place-recognition score (VLAD/
                                                       // SIFT-DBoW2/DBoW2 -- whichever candidate-search
                                                       // backend is active) between the two OLD keyframes
                                                       // being confirmed against each other, grounding the
                                                       // test in real appearance evidence instead of the
                                                       // 1D keyframe-index proxy this replaced. Same lesson
                                                       // item 19 already confirmed for fuseWindowLandmarks()
                                                       // (grounding matches in real evidence fixed an
                                                       // analogous problem there) -- the pure index-window
                                                       // version was measured (item 15) to lose far more
                                                       // real corrections than it filtered false positives,
                                                       // because viewing-angle drift shifts the best-
                                                       // matching old-keyframe index faster than a small
                                                       // window tolerates. Two OLD keyframes from the SAME
                                                       // physical revisit episode should score well above
                                                       // kVladMinScore's distant-frame noise floor (~0.05) --
                                                       // 0.3 is an initial, untuned-per-sequence estimate,
                                                       // same spirit as kVladMinScore/kDbowMinScore/
                                                       // kSiftDbowMinScore.
constexpr size_t kLoopConsistencyMaxGapKeyframes = 4; // how many new-keyframe insertions may pass between
                                                       // two confirmations of the same pending candidate
                                                       // before treating the streak as stale and
                                                       // restarting it -- roughly half a kKeyframeEveryNFrames
                                                       // cycle's worth of slack, untuned.
constexpr int kLoopSpatialConsensusWindow = 5; // setLoopSpatialConsensusEnabled()'s own gate (DEBUGGING.md
                                                // item 23): among all candidates whose place-recognition
                                                // score independently exceeds the primary acceptance
                                                // threshold in a SINGLE tryLoopClosure() call, the top-
                                                // scoring one is only accepted if at least one OTHER
                                                // qualifying candidate lies within this many keyframe
                                                // indices of it -- checks WITHIN-call consensus among
                                                // several independently-ranked candidates instead of
                                                // ACROSS-call temporal recurrence (see
                                                // setLoopConsistencyGroupEnabled()'s own doc comment for
                                                // why that approach was closed as a structural mismatch,
                                                // items 15/21/22). Untuned-per-sequence estimate.
constexpr int kFuseWindowKeyframes = 15; // fuseWindowLandmarks()'s own recent-activity window --
                                          // roughly 2x kLocalBaWindowKeyframes, wide enough to catch a
                                          // duplicate triangulated a handful of keyframes apart, still
                                          // narrow enough that the O(new candidates x window keyframes)
                                          // projection search stays cheap. Untuned.
constexpr double kFuseMergeMaxReprojErrorPixels = 3.0; // fuseWindowLandmarks()'s Phase B (see
                                                        // setLandmarkFuseMergeEnabled()) applies this
                                                        // STRICTER reprojection-error gate on TOP of the
                                                        // shared kMaxObservationReprojErrorPixels (8px)
                                                        // check every candidate match already passed --
                                                        // merging changes a landmark's identity/history
                                                        // permanently (items 18/20/25 all found real,
                                                        // measurable harm from merges that were only
                                                        // individually "good enough"), so it deserves a
                                                        // materially tighter bar than Phase A's coverage-
                                                        // extension (which only ever adds a new
                                                        // observation to an unambiguous, still-alive
                                                        // landmark, never changes any existing identity).
                                                        // Untuned-per-sequence estimate, item 26. Reused
                                                        // by triangulateMultiView() as the mean-
                                                        // reprojection-error acceptance gate for Phase B
                                                        // v5's re-triangulated merge position (same
                                                        // "how good does a merge decision need to be"
                                                        // bar, now applied to the ACTUAL geometric
                                                        // consistency of the combined observation set
                                                        // instead of just one single-view pixel offset).
constexpr int kSim3SolverMinCorrespondences = 8; // solveSim3Ransac()'s own call site in
                                                  // tryLoopClosure(): below this many descriptor-matched
                                                  // 3D-3D correspondences, don't even attempt RANSAC (the
                                                  // minimal solve itself only needs 3, but a meaningful
                                                  // inlier-consensus vote needs real headroom above that)
constexpr int kSim3SolverMinInliers = 6; // matches ORB-SLAM3's own Sim3Solver default (minInliers=6,
                                          // see Sim3Solver::SetRansacParameters()) -- below this many
                                          // RANSAC inliers, don't trust the found Sim3 at all, fall back
                                          // to the single-point ratio instead
constexpr double kMinScaleMeasBaseline = 20.0; // world units -- tryLoopClosure()'s scaleMeas guard.
                                                // scaleMeas = distLoop/distDrifted is a ratio of two
                                                // camera-center distances that are often genuinely tiny
                                                // for a real revisit (the whole point of a loop closure
                                                // is the two keyframes are spatially close), which makes
                                                // the ratio numerically ill-conditioned right when it
                                                // matters most -- confirmed empirically (2026-07-21):
                                                // measured scaleMeas swung 0.0058 to 16.27 across early
                                                // closures, and the wild values consistently paired with
                                                // at least one of distDrifted/distLoop being small, while
                                                // measurements with both distances comfortably above this
                                                // floor stayed tightly clustered near a plausible 0.99-1.23.
                                                // Below this floor there simply isn't enough baseline to
                                                // trust the ratio, so scaleMeas falls back to the neutral
                                                // 1.0 (same as before this measurement existed) rather than
                                                // propagating noise into the Sim3 solve.
constexpr double kScaleMeasClampMin = 0.3; // even above the baseline floor, bound the trusted ratio to a
constexpr double kScaleMeasClampMax = 3.0; // plausible range -- real monocular scale drift observed in
                                            // this pipeline's own "clean" measurements never approached
                                            // this, so anything beyond it is far more likely leftover
                                            // numerical noise (e.g. from an already-huge PnP correction,
                                            // see kLoopMaxCorrectionMagnitude) than genuine drift, and an
                                            // unclamped outlier can otherwise dominate the DCS-weighted
                                            // Sim3 solve (confirmed: an unclamped run produced a single
                                            // loop edge with chi2=39M, swamping the rest of the graph).
constexpr int kLocalRefineWindow = 6; // how many of the most-recent keyframes refineLocalKeyframes()
                                       // re-polishes via nonlinear reprojection-error minimization on
                                       // every new keyframe insertion
constexpr int kCullingCheckIntervalKeyframes = 10; // cullRedundantKeyframes() rebuilds the whole
                                                    // covisibility graph from m_landmarkObservations
                                                    // every time it runs -- only run it this often
                                                    // (not every insertion) to bound that cost
constexpr int kCullingExclusionWindow = 15; // never consider the most recent N keyframes for
                                             // culling -- mirrors kLoopExclusionWindow's spirit;
                                             // a very recent keyframe's redundancy isn't
                                             // reliably known yet (later keyframes haven't had a
                                             // chance to re-observe its landmarks)
constexpr int kCullingMinOwnLandmarks = 5; // don't judge a keyframe against this criterion at all
                                            // if it observes too few landmarks to make the
                                            // redundancy ratio below statistically meaningful
constexpr int kCullingMinObservers = 3; // ORB-SLAM2's own KeyFrameCulling() threshold: a landmark
                                         // counts as "redundant" for this keyframe if at least
                                         // this many OTHER keyframes also observe it
constexpr double kCullingRedundancyRatio = 0.9; // ORB-SLAM2's own threshold: a keyframe is culled
                                                 // once at least this fraction of its own observed
                                                 // landmarks are individually redundant
constexpr int kBaMinObservationsPerLandmark = 2; // a landmark needs observations from at least this
                                                  // many distinct in-window keyframes to constrain
                                                  // anything -- with only 1, BA can always drive its
                                                  // reprojection error to zero by moving the (free)
                                                  // landmark itself, contributing nothing to the poses
constexpr double kBaHuberDeltaPixels = 4.0; // robust-loss threshold: a mismatched/outlier
                                             // correspondence beyond this many pixels of reprojection
                                             // error gets down-weighted instead of dominating the fit
constexpr double kPoseOnlyLoopSuppressScaleLo = 0.80; // item 39: only suppress pose-only BA after a loop
constexpr double kPoseOnlyLoopSuppressScaleHi = 1.25; // closure whose scaleMeas falls OUTSIDE [lo,hi] --
                                                       // i.e. one that applied a real scale correction
                                                       // (the map-compressing case that causes the
                                                       // collapse). Near-unit closures leave pose-only BA
                                                       // on. Untuned.
constexpr int kPoseOnlyLoopSuppressFrames = 30; // DEBUGGING.md item 39: after a loop closure commits, use
                                                 // pure SQPnP (skip pose-only BA) for this many frames while
                                                 // the loop-BA-perturbed map re-settles -- pose-only BA holds
                                                 // the map fixed and fits the camera rigidly to it, so it
                                                 // follows a momentarily-compressed post-loop map straight
                                                 // into a local scale collapse (the diagnosed frames-2500-3300
                                                 // failure); SQPnP's RANSAC is robust to it. ~4 keyframes at
                                                 // the default keyframe spacing. Untuned.
constexpr double kPoseOnlyMinStepFraction = 0.35; // item 41 #1: reject a pose-only-BA frame whose
                                                   // camera-center step is below this fraction of the
                                                   // running avg step (m_avgStepScale) -- the scale-collapse
                                                   // signature -- and fall back to SQPnP for that frame.
                                                   // Untuned.
constexpr double kPoseOnlyLeashRotWeight = 30.0;  // item 41 #2: soft-prior leash weights anchoring pose-only
constexpr double kPoseOnlyLeashTransWeight = 5.0; // BA to the SQPnP solution (see optimizePoseOnly's prior).
                                                   // Same spirit/scale as kLocalBaPosePriorRotWeight/Trans.
                                                   // Untuned.
constexpr double kOctaveWeightRefSize = 2.0; // octave/scale information weighting reference (item 38, see
                                              // optimizePoseOnly()): a SIFT keypoint at/below this SIZE
                                              // (diameter, ~the finest detectable scale) keeps full unit
                                              // weight; coarser ones get invSigma2=(ref/size)^2 < 1. Not
                                              // per-sequence tuned.
constexpr double kMaxObservationReprojErrorPixels = 8.0; // recordLandmarkObservations() gate: a plain
                                                          // descriptor ratio-test match (unlike every
                                                          // other point-to-3D-structure step in this
                                                          // file, e.g. tryLoopClosure()'s PnP+RANSAC or
                                                          // trackFrame()'s solvePnPRansac()) has no
                                                          // geometric check at all -- reject a match if
                                                          // reprojecting the landmark's known 3D
                                                          // position through this keyframe's own pose
                                                          // lands more than this many pixels from the
                                                          // matched keypoint (same threshold
                                                          // tryLoopClosure()'s own PnP RANSAC already
                                                          // uses, see kLoopMinPnpInliers's call site).
                                                          // See DEBUGGING.md Session 8/9 for why
                                                          // unverified matches silently corrupting BA's
                                                          // data associations, not just diluting it, was
                                                          // the real remaining problem.
constexpr int kRetriangulateMinViews = 3; // item 41 Backend #2 (setRetriangulateEnabled()): only re-
                                          // triangulate a landmark once it has at least this many
                                          // distinct-keyframe observations. 2 is the trivial creation
                                          // case (already what triangulate() did); the first genuine
                                          // multi-view improvement is at 3. Untuned.
constexpr double kMinTriangulationParallaxCos = 0.99985; // item 41 Backend #2 (setParallaxGateEnabled()):
                                                          // reject a newly-triangulated landmark whose two
                                                          // viewing rays are more parallel than this cosine
                                                          // (~1.0 deg). Low-parallax points triangulate to
                                                          // noisy, far, unreliable depths that pollute the
                                                          // map and make tight map-fitting backfire (item
                                                          // 40); real ORB-SLAM3 culls with the same
                                                          // cosParallax test. Untuned.
constexpr double kLoopVerifiedResidualWeight = 25.0; // squared-cost multiplier for the one observation
                                                      // per loop closure that's the actual PnP-RANSAC-
                                                      // verified correspondence proving the loop (see
                                                      // tryLoopClosure()) -- everything else in a BA
                                                      // window is typically a bare 2-observation local
                                                      // track (see DEBUGGING.md Session 7's diagnosis:
                                                      // BA measurably hurt ATE because this one real
                                                      // long-baseline signal got diluted by thousands of
                                                      // weakly-connected short ones under identical
                                                      // Huber-lossed treatment). No robust loss + a 25x
                                                      // cost weight (~5x tighter effective noise sigma)
                                                      // makes the optimizer trust it fully rather than
                                                      // down-weighting it like an ordinary outlier
                                                      // candidate, without being a hard equality
                                                      // constraint that would break if the loop
                                                      // measurement itself has any of its own noise.
constexpr int kBaMaxIterations = 50;
constexpr int kBaMaxWindowKeyframes = 600; // skip BA (fall back to the interpolated correction) for a
                                            // loop window wider than this -- raised from 200 to 600
                                            // (2026-07-21): confirmed the ORIGINAL 200 cap made EVERY
                                            // end-of-sequence "return near start" closure (observed:
                                            // 5 in a row, kf#0-9 <-> kf#471-479, span ~470) fall
                                            // through to interpolation even after fixing the separate
                                            // global-BA-doesn't-fall-back-to-windowed bug, since these
                                            // spans exceed 200 too -- exactly the kind of correction
                                            // where a real reprojection-error solve matters most (the
                                            // whole sequence's accumulated drift, not a local
                                            // discrepancy). 200 was chosen purely for kitti_ate's time
                                            // budget, not solve quality -- see git history/this
                                            // comment's prior text for the specific timeout observed;
                                            // that's a harness concern (raise --seconds), not a reason
                                            // to leave real drift uncorrected.
// Full global BA (see setGlobalBundleAdjustmentEnabled()): unlike
// kBaMaxWindowKeyframes's windowed loop BA (bounded to [oldKfIdx, newKfIdx]),
// this spans [0, newKfIdx] -- every keyframe and every landmark in the
// entire map, not just the loop's own span. Safe from continuous local BA's
// scale-collapse failure mode for two structural reasons, not just luck:
// (1) it fires once per loop closure, not hundreds of times over
// heavily-overlapping windows, so there's no repeated-compounding
// mechanism; (2) it has TWO real hard anchors spanning the whole
// optimization -- keyframe 0 (the world origin) AND newKfIdx (fixed at the
// independently PnP-verified loop pose) -- rather than local BA's single
// anchor that was itself just whatever an earlier unverified free solve
// left it at. Still capped by kGlobalBaMaxWindowKeyframes for the same
// runtime reason kBaMaxWindowKeyframes exists (a late-sequence loop can
// span 400+ keyframes/20000+ landmarks).
constexpr int kGlobalBaMaxWindowKeyframes = 400;
constexpr int kGlobalBaIntegrationDelayKeyframes = 15; // setGlobalBundleAdjustmentAsyncEnabled()'s
                                                        // simulated "still solving in the background"
                                                        // window -- same scale as kFuseWindowKeyframes'
                                                        // own "recent window" (15), chosen for the same
                                                        // reason (long enough to be a meaningfully
                                                        // different scenario from immediate synchronous
                                                        // write-back, short enough that the gap-keyframe
                                                        // propagation's single-rigid-delta approximation
                                                        // stays plausible). Untuned.

// Covisibility-driven local map for tracking (see
// setCovisibilityLocalMapEnabled()): trackFrame() matches against the union
// of map points owned by keyframes covisible with the current reference
// keyframe, rebuilt once per keyframe insertion (see
// buildCovisibilityLocalMap()), instead of the flat rolling m_mapPoints/
// m_mapDescriptors (capped at kMaxMapPoints, oldest points evicted
// regardless of relevance). ORB-SLAM2's own convention: two keyframes are
// "covisible" once they jointly observe at least this many landmarks.
constexpr int kCovisibilityMinSharedLandmarks = 15;

// Guided/projection-based search (see setGuidedSearchEnabled()): real
// ORB-SLAM2's SearchByProjection() only searches keypoints within a small
// radius of each map point's predicted projection, instead of matching
// every descriptor against every other one. This is a lighter-weight
// version of the same idea: keep the existing brute-force
// matchDescriptors() pass, then reject any match whose keypoint lands
// farther than this many pixels from where a zeroth-order (constant-
// position) motion prediction says it should be -- cheap to add on top of
// the existing matcher rather than a full from-scratch guided search, but
// captures the same core benefit (rejecting spatially-implausible matches
// a descriptor-only comparison let through).
constexpr double kGuidedSearchRadiusPixels = 60.0; // originally widened from 40px to 200px when the
                                                     // prediction was constant-POSITION (a poor motion
                                                     // model for a forward-driving car at ~10fps -- match
                                                     // counts collapsed to single digits at 40px). Now
                                                     // that the prediction is constant-VELOCITY (see
                                                     // m_velocityR/m_velocityT), a tighter radius may work
                                                     // better -- left at 200 (i.e. still just a loose
                                                     // sanity filter, not yet re-tuned) until measured;
                                                     // see DEBUGGING.md for whichever session re-tunes it.

// Quality-driven keyframe insertion (see setQualityDrivenKeyframesEnabled()):
// real ORB-SLAM2 decides where to keyframe based on tracking quality
// (fraction of currently-tracked points relative to the reference), not a
// fixed frame interval the way kKeyframeEveryNFrames does. kKeyframeMinInterval
// still rate-limits it (no point inserting every single frame even if
// quality is dropping every frame); kKeyframeMaxInterval is a safety net
// so an easy, high-quality stretch still eventually gets a fresh keyframe.
constexpr int kKeyframeMinInterval = 3;
constexpr int kKeyframeMaxInterval = 20;
constexpr double kKeyframeQualityRatioThreshold = 0.5; // insert once inlierCount/matches.size() drops
                                                        // below this fraction -- tracking quality against
                                                        // the current reference has degraded enough to
                                                        // warrant a fresh one
constexpr int kCovisibilityMapStaleFrames = 24; // see m_framesSinceCovisibilityMapRebuild's own doc
                                                 // comment (SlamWorker.h) -- 3x kKeyframeEveryNFrames,
                                                 // long enough to not fall back on an ordinary single
                                                 // missed keyframe insertion, short enough to not track
                                                 // against a badly stale local map for long

// Continuous local BA (see setLocalBundleAdjustmentEnabled()): a joint Ceres
// BA over the last kLocalBaWindowKeyframes, run on EVERY keyframe insertion
// -- unlike runLoopBundleAdjustment() above (which only fires at a
// confirmed loop closure and anchors to an independently PnP-verified loop
// pose), this has no independent measurement anywhere in the window to
// anchor scale/gauge with. A prior attempt at exactly this (see
// DEBUGGING.md, "Attempted: continuous local bundle adjustment (reverted,
// catastrophic failure)") gauge-fixed only the single oldest keyframe in
// each sliding window -- but that "anchor" was itself just whatever a
// PREVIOUS window's own free, unverified optimization left it at, so small
// systematic biases compounded across hundreds of ~93%-overlapping windows,
// collapsing scale to 0.0001 (193.4m ATE). kLocalBaPosePriorRotWeight/
// kLocalBaPosePriorTransWeight fix this differently: instead of one
// (unverified) hard-fixed anchor per window, EVERY pose in the window gets
// a soft prior pulling it back toward its own pre-BA (PnP-tracked) value --
// caps how far any single keyframe can drift from what live tracking
// actually measured, directly bounding the runaway-drift mechanism that
// broke the earlier attempt, without needing OXTS or ground truth as an
// external anchor (both off the table -- see this session's own earlier
// discussion).
// kLocalBaWindowKeyframes moved to SlamWorker::m_localBaWindowKeyframes (see setLocalBaWindowKeyframes())
// -- was a fixed constexpr here, now runtime-overridable; default value (8) unchanged. Queued since item
// 8/10's observation-density fix unlocked real multi-view constraint in a window that previously only
// had ownership-only landmarks to work with -- a bigger window may now capture meaningfully more of it,
// untested before this session (see DEBUGGING.md).
// kLocalBaPosePriorRotWeight/kLocalBaPosePriorTransWeight moved to
// SlamWorker::m_localBaPosePriorRotWeight/m_localBaPosePriorTransWeight (see
// setLocalBaPosePriorWeights()) -- were fixed constexpr here, now runtime-
// overridable (item 41 Backend #1, argv60/61); default values (20 radians /
// 3 world-units) unchanged. Same soft-prior-on-every-window-pose principle the
// FE pose-only leash (setPoseOnlyLeashWeights) just validated at the per-frame
// level: rotation kept deliberately tight (well-observed even in a small window,
// unlike scale), translation looser (real local corrections need room to move it
// a bit) but still firmly bounded -- unlike the reverted attempt's
// unconstrained-until-window-exit scheme. Untuned until this sweep.

constexpr double kGlobalBaLoopPosePriorRotWeight = 50.0; // runGlobalBundleAdjustment()'s v3 (DEBUGGING.md
                                                          // item 32): SOFT prior pulling newKfIdx toward
                                                          // the independently PnP/Sim3Solver-verified loop
                                                          // pose, replacing the original HARD anchor
                                                          // (SetParameterBlockConstant) -- real ORB-SLAM3
                                                          // never hard-pins a loop pose inside its global
                                                          // BA (only the map's origin keyframe is fixed;
                                                          // the loop correction is a separate essential-
                                                          // graph propagation step this codebase doesn't
                                                          // have). Stronger than kLocalBaPosePriorRotWeight
                                                          // (20.0) since this is an independently verified
                                                          // measurement, not just "stay close to your own
                                                          // last live estimate" -- but finite, unlike the
                                                          // old hard constant. Untuned.
constexpr double kGlobalBaLoopPosePriorTransWeight = 50.0; // see kGlobalBaLoopPosePriorRotWeight. Unlike
                                                            // kLocalBaPosePriorTransWeight's deliberately
                                                            // LOOSER translation weight (real local
                                                            // corrections need room to move translation),
                                                            // kept equal to the rotation weight here since
                                                            // the loop measurement's translation component
                                                            // (Sim3Solver-derived) is just as directly
                                                            // trusted as its rotation. Untuned.

constexpr double kOxtsPnpStepToleranceMultiplier = 3.0; // trackFrame()'s OXTS-aware plausibility check
                                                          // (see setOxtsImuInPnpEnabled()): accept a PnP
                                                          // step within this factor of the OXTS-measured
                                                          // real distance, either direction. Tighter than
                                                          // kMaxStepMultiplier (10x) since oxtsDist is a
                                                          // real measurement, not a rough heuristic --
                                                          // the PnP solve should track it closely if
                                                          // correct.
constexpr double kImuPnpMaxAngleDeg = 15.0; // trackFrame()'s IMU-aware plausibility check: reject a PnP
                                             // solve whose rotation disagrees with the independently
                                             // IMU-measured rotation by more than this many degrees.
                                             // Loose enough to tolerate a real sharp turn between
                                             // frames (loop-closure corrections seen this session ran
                                             // up to ~20 degrees over much larger keyframe gaps, so a
                                             // single-frame step should rarely need anywhere near that).

// Reprojection-error residual for runLoopBundleAdjustment(): pose is a
// 6-vector (angle-axis rotation[0..2], translation[3..5]) for this
// class's usual world-to-camera convention (Xcam = R*Xworld + t), matching
// ceres::AngleAxisRotatePoint's own R*point convention directly. point is
// the landmark's 3D world position. Camera intrinsics are fixed (not
// optimized), consistent with the rest of this codebase treating K as
// known/calibrated.
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

        // Clamp rather than reject outright: keeps the residual finite and
        // differentiable if a candidate landmark position briefly swings
        // behind the camera mid-optimization, instead of NaN-ing the solve.
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

// Pose-only reprojection residual (see SlamWorker::optimizePoseOnly(),
// DEBUGGING.md item 36): the 3D landmark position is BAKED IN as a
// constant (X/Y/Z), so the only free parameter block is the 6-DOF camera
// pose [rvec(3), t(3)]. This is exactly real ORB-SLAM3's
// Optimizer::PoseOptimization() structure -- refine the current frame's
// pose against all its matched map points with the map held fixed, far
// more accurate than trusting RANSAC PnP's minimal-sample winner, which is
// the actual per-frame drift source vs ORB-SLAM3 (see trackFrame()).
struct PoseOnlyReprojectionCost
{
    PoseOnlyReprojectionCost(double X, double Y, double Z, double observedU, double observedV, double fx,
                              double fy, double cx, double cy)
        : X(X), Y(Y), Z(Z), observedU(observedU), observedV(observedV), fx(fx), fy(fy), cx(cx), cy(cy)
    {
    }

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {
        T point[3] = {T(X), T(Y), T(Z)};
        T camPoint[3];
        ceres::AngleAxisRotatePoint(pose, point, camPoint);
        camPoint[0] += pose[3];
        camPoint[1] += pose[4];
        camPoint[2] += pose[5];
        T z = camPoint[2];
        if (z < T(1e-3))
            z = T(1e-3);
        residuals[0] = T(fx) * camPoint[0] / z + T(cx) - T(observedU);
        residuals[1] = T(fy) * camPoint[1] / z + T(cy) - T(observedV);
        return true;
    }

    static ceres::CostFunction *Create(double X, double Y, double Z, double observedU, double observedV,
                                        double fx, double fy, double cx, double cy)
    {
        return new ceres::AutoDiffCostFunction<PoseOnlyReprojectionCost, 2, 6>(
            new PoseOnlyReprojectionCost(X, Y, Z, observedU, observedV, fx, fy, cx, cy));
    }

    double X, Y, Z, observedU, observedV, fx, fy, cx, cy;
};

// Soft prior pulling a [rvec(3), t(3)] pose parameter block back toward a
// fixed reference value -- see runLocalBundleAdjustment()'s doc comment for
// why this exists (bounding drift in place of an unverified hard-fixed
// window anchor). A plain elementwise angle-axis difference, not a proper
// geodesic rotation distance -- only accurate for small deviations, but
// that's exactly the regime this is meant to operate in: it's a leash
// against runaway drift, not a precise rotational-distance metric.
struct PosePriorCost
{
    PosePriorCost(const std::array<double, 6> &prior, double rotWeight, double transWeight)
        : prior(prior), rotWeight(rotWeight), transWeight(transWeight)
    {
    }

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {
        for (int i = 0; i < 3; ++i)
            residuals[i] = T(rotWeight) * (pose[i] - T(prior[static_cast<size_t>(i)]));
        for (int i = 0; i < 3; ++i)
            residuals[3 + i] = T(transWeight) * (pose[3 + i] - T(prior[static_cast<size_t>(3 + i)]));
        return true;
    }

    static ceres::CostFunction *Create(const std::array<double, 6> &prior, double rotWeight, double transWeight)
    {
        return new ceres::AutoDiffCostFunction<PosePriorCost, 6, 6>(new PosePriorCost(prior, rotWeight, transWeight));
    }

    std::array<double, 6> prior;
    double rotWeight, transWeight;
};

// Horn 1987 ("Closed-form solution of absolute orientation using unit
// quaternions") closed-form solution for the minimal-3-point similarity
// (Sim3) registration problem -- the exact algorithm ORB-SLAM2/3's own
// Sim3Solver::ComputeSim3() uses (third_party/ORB_SLAM3/src/Sim3Solver.cc,
// consulted directly rather than reconstructed from memory), ported here
// from Eigen to this codebase's cv::Mat convention. P1/P2 are 3x3 (CV_64F),
// each COLUMN one of exactly 3 points -- camera1's own local-frame
// coordinates for P1, camera2's own local-frame coordinates for P2, both
// columns referring to the SAME 3 real-world points. Solves for R12
// (3x3 rotation), t12 (3x1), s12 (scalar) such that P1 ~= s12*R12*P2 + t12.
void computeSim3Horn(const cv::Mat &P1, const cv::Mat &P2, cv::Mat &R12, cv::Mat &t12, double &s12)
{
    // Step 1: centroids + relative (centered) coordinates.
    cv::Mat O1, O2;
    cv::reduce(P1, O1, 1, cv::REDUCE_AVG);
    cv::reduce(P2, O2, 1, cv::REDUCE_AVG);
    const cv::Mat Pr1 = P1 - cv::repeat(O1, 1, 3);
    const cv::Mat Pr2 = P2 - cv::repeat(O2, 1, 3);

    // Step 2/3: cross-covariance M, then the 4x4 symmetric N built from M's
    // entries -- N's eigenvector of largest eigenvalue is the quaternion of
    // the optimal rotation (Horn's own closed-form result).
    const cv::Mat M = Pr2 * Pr1.t();
    const double m00 = M.at<double>(0, 0), m01 = M.at<double>(0, 1), m02 = M.at<double>(0, 2);
    const double m10 = M.at<double>(1, 0), m11 = M.at<double>(1, 1), m12 = M.at<double>(1, 2);
    const double m20 = M.at<double>(2, 0), m21 = M.at<double>(2, 1), m22 = M.at<double>(2, 2);
    const cv::Mat N = (cv::Mat_<double>(4, 4) << m00 + m11 + m22, m12 - m21, m20 - m02, m01 - m10, m12 - m21,
                        m00 - m11 - m22, m01 + m10, m20 + m02, m20 - m02, m01 + m10, -m00 + m11 - m22, m12 + m21,
                        m01 - m10, m20 + m02, m12 + m21, -m00 - m11 + m22);

    // cv::eigen() requires (and here gets, by construction) a real
    // symmetric matrix, and returns eigenvalues/vectors in DESCENDING
    // order -- row 0 of the eigenvector matrix is exactly the quaternion
    // ORB-SLAM3's own eval.maxCoeff()-based selection picks out.
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(N, eigenvalues, eigenvectors);
    const double qw = eigenvectors.at<double>(0, 0), qx = eigenvectors.at<double>(0, 1),
                 qy = eigenvectors.at<double>(0, 2), qz = eigenvectors.at<double>(0, 3);
    R12 = (cv::Mat_<double>(3, 3) << 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
           2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx), 2 * (qx * qz - qw * qy),
           2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy));

    // Step 5/6: rotate set 2 by the found rotation, then the closed-form
    // least-squares scale between the two (now co-rotated) point sets.
    const cv::Mat P3 = R12 * Pr2;
    const double nom = Pr1.dot(P3);
    const double den = P3.dot(P3);
    s12 = (den > 1e-9) ? (nom / den) : 1.0;

    // Step 7: translation.
    t12 = O1 - s12 * R12 * O2;
}

// RANSAC wrapper around computeSim3Horn(), matching ORB-SLAM3's own
// Sim3Solver::iterate() -- see that function's doc comment for why a
// minimal-3-point closed-form solve plus RANSAC (rather than, say, a
// least-squares fit over all correspondences at once) is the right
// approach: a handful of the descriptor-matched 3D-3D correspondences
// below are typically wrong (visually similar but different real points),
// and Horn's method has no built-in robustness against that on its own.
// X3Dc1/X3Dc2 are PARALLEL arrays of matched points, each already expressed
// in its own camera's local frame (see tryLoopClosure()'s own call site for
// how these get built). Returns true with R12/t12/s12/inlierIndices filled
// in if a solution with at least minInliers support was found.
bool solveSim3Ransac(const std::vector<cv::Point3f> &X3Dc1, const std::vector<cv::Point3f> &X3Dc2, cv::Mat &R12,
                      cv::Mat &t12, double &s12, std::vector<int> &inlierIndices, int minInliers = 6,
                      int maxIterations = 300, double inlierRelativeThreshold = 0.1)
{
    const int n = static_cast<int>(X3Dc1.size());
    if (n < 3 || n < minInliers)
        return false;

    std::mt19937 rng(kRansacSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    cv::Mat bestR, bestT;
    double bestS = 1.0;
    std::vector<int> bestInliers;

    for (int iter = 0; iter < maxIterations; ++iter) {
        std::array<int, 3> sample{};
        int filled = 0;
        while (filled < 3) {
            const int idx = dist(rng);
            if (std::find(sample.begin(), sample.begin() + filled, idx) == sample.begin() + filled)
                sample[static_cast<size_t>(filled++)] = idx;
        }

        cv::Mat P1(3, 3, CV_64F), P2(3, 3, CV_64F);
        for (int c = 0; c < 3; ++c) {
            const cv::Point3f &p1 = X3Dc1[static_cast<size_t>(sample[static_cast<size_t>(c)])];
            const cv::Point3f &p2 = X3Dc2[static_cast<size_t>(sample[static_cast<size_t>(c)])];
            P1.at<double>(0, c) = p1.x;
            P1.at<double>(1, c) = p1.y;
            P1.at<double>(2, c) = p1.z;
            P2.at<double>(0, c) = p2.x;
            P2.at<double>(1, c) = p2.y;
            P2.at<double>(2, c) = p2.z;
        }

        cv::Mat R, t;
        double s = 1.0;
        computeSim3Horn(P1, P2, R, t, s);
        if (s <= 0.0 || !std::isfinite(s))
            continue;

        // Inlier check: 3D distance between camera1's own observation and
        // camera2's point mapped into camera1's frame, relative to that
        // point's own distance from the origin (a stand-in for ORB-SLAM3's
        // own pixel-reprojection check -- this codebase's loop 3D-3D
        // correspondences don't carry a clean per-point 2D pixel the way
        // ORB-SLAM3's KeyFrame-indexed ones do, since one side is matched
        // against the rolling map, not a single keyframe's own keypoints).
        std::vector<int> inliers;
        inliers.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const cv::Mat X1 = (cv::Mat_<double>(3, 1) << X3Dc1[static_cast<size_t>(i)].x,
                                 X3Dc1[static_cast<size_t>(i)].y, X3Dc1[static_cast<size_t>(i)].z);
            const cv::Mat X2 = (cv::Mat_<double>(3, 1) << X3Dc2[static_cast<size_t>(i)].x,
                                 X3Dc2[static_cast<size_t>(i)].y, X3Dc2[static_cast<size_t>(i)].z);
            const cv::Mat predicted1 = s * R * X2 + t;
            const double err = cv::norm(X1 - predicted1);
            const double scale = std::max(cv::norm(X1), 1e-6);
            if (err < inlierRelativeThreshold * scale)
                inliers.push_back(i);
        }

        if (inliers.size() > bestInliers.size()) {
            bestInliers = inliers;
            bestR = R;
            bestT = t;
            bestS = s;
        }
    }

    if (static_cast<int>(bestInliers.size()) < minInliers)
        return false;

    R12 = bestR;
    t12 = bestT;
    s12 = bestS;
    inlierIndices = bestInliers;
    return true;
}

// Hartley normalization: a similarity transform moving the point set's
// centroid to the origin with mean distance sqrt(2) from it.
cv::Mat computeNormalizationTransform(const std::vector<cv::Point2f> &pts)
{
    double meanX = 0.0, meanY = 0.0;
    for (const auto &p : pts) {
        meanX += p.x;
        meanY += p.y;
    }
    meanX /= static_cast<double>(pts.size());
    meanY /= static_cast<double>(pts.size());

    double meanDist = 0.0;
    for (const auto &p : pts) {
        const double dx = p.x - meanX;
        const double dy = p.y - meanY;
        meanDist += std::sqrt(dx * dx + dy * dy);
    }
    meanDist /= static_cast<double>(pts.size());
    const double scale = (meanDist > 1e-8) ? std::sqrt(2.0) / meanDist : 1.0;

    return (cv::Mat_<double>(3, 3) << scale, 0.0, -scale * meanX,
                                       0.0, scale, -scale * meanY,
                                       0.0, 0.0, 1.0);
}

std::vector<cv::Point2f> applyTransform(const cv::Mat &T, const std::vector<cv::Point2f> &pts)
{
    std::vector<cv::Point2f> out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        const double x = T.at<double>(0, 0) * pts[i].x + T.at<double>(0, 1) * pts[i].y + T.at<double>(0, 2);
        const double y = T.at<double>(1, 0) * pts[i].x + T.at<double>(1, 1) * pts[i].y + T.at<double>(1, 2);
        out[i] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    }
    return out;
}

// Normalized DLT (Hartley & Zisserman, Alg. 4.2) on already normalized
// correspondences: x2 ~ H x1. Requires pts1.size() == pts2.size() >= 4.
cv::Mat solveHomographyDLT(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2)
{
    const int n = static_cast<int>(pts1.size());
    cv::Mat A = cv::Mat::zeros(2 * n, 9, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double x1 = pts1[i].x, y1 = pts1[i].y;
        const double x2 = pts2[i].x, y2 = pts2[i].y;

        double *row0 = A.ptr<double>(2 * i);
        row0[0] = -x1;
        row0[1] = -y1;
        row0[2] = -1.0;
        row0[6] = x2 * x1;
        row0[7] = x2 * y1;
        row0[8] = x2;

        double *row1 = A.ptr<double>(2 * i + 1);
        row1[3] = -x1;
        row1[4] = -y1;
        row1[5] = -1.0;
        row1[6] = y2 * x1;
        row1[7] = y2 * y1;
        row1[8] = y2;
    }

    // H is the singular vector of A associated with its smallest singular value.
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);
    return vt.row(8).clone().reshape(0, 3);
}

// 3D analog of computeNormalizationTransform/applyTransform above: translate
// the centroid to the origin, isotropic scale so the mean distance from the
// origin is sqrt(3) (Hartley normalization extended to 3D) -- conditions
// the pose-DLT system below the same way the 2D normalization conditions
// solveEightPoint/solveHomographyDLT.
cv::Mat computeNormalizationTransform3D(const std::vector<cv::Point3f> &pts)
{
    double meanX = 0.0, meanY = 0.0, meanZ = 0.0;
    for (const auto &p : pts) {
        meanX += p.x;
        meanY += p.y;
        meanZ += p.z;
    }
    const double n = static_cast<double>(pts.size());
    meanX /= n; meanY /= n; meanZ /= n;

    double meanDist = 0.0;
    for (const auto &p : pts) {
        const double dx = p.x - meanX, dy = p.y - meanY, dz = p.z - meanZ;
        meanDist += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    meanDist /= n;
    const double scale = (meanDist > 1e-8) ? std::sqrt(3.0) / meanDist : 1.0;

    return (cv::Mat_<double>(4, 4) << scale, 0.0, 0.0, -scale * meanX,
                                       0.0, scale, 0.0, -scale * meanY,
                                       0.0, 0.0, scale, -scale * meanZ,
                                       0.0, 0.0, 0.0, 1.0);
}

std::vector<cv::Point3f> applyTransform3D(const cv::Mat &T, const std::vector<cv::Point3f> &pts)
{
    std::vector<cv::Point3f> out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        const double x = T.at<double>(0, 0) * pts[i].x + T.at<double>(0, 1) * pts[i].y +
                          T.at<double>(0, 2) * pts[i].z + T.at<double>(0, 3);
        const double y = T.at<double>(1, 0) * pts[i].x + T.at<double>(1, 1) * pts[i].y +
                          T.at<double>(1, 2) * pts[i].z + T.at<double>(1, 3);
        const double z = T.at<double>(2, 0) * pts[i].x + T.at<double>(2, 1) * pts[i].y +
                          T.at<double>(2, 2) * pts[i].z + T.at<double>(2, 3);
        out[i] = cv::Point3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    }
    return out;
}

// Linear DLT for a 3x4 pose matrix [R|t] (up to an unknown positive scale)
// from >=6 correspondences between K-calibrated image rays (x,y) = ((u-cx)/fx,
// (v-cy)/fy) and 3D object points: (x,y,1) ~ [R|t] (X,Y,Z,1). Same structure
// as solveEightPoint/solveHomographyDLT -- stack the DLT constraints, take
// the right singular vector of smallest singular value. Points should
// already be similarity-normalized (see computeNormalizationTransform(3D))
// for numerical conditioning; the raw solve here doesn't care either way.
cv::Mat solvePoseDLT(const std::vector<cv::Point3f> &objPts, const std::vector<cv::Point2f> &imgPts)
{
    const int n = static_cast<int>(objPts.size());
    cv::Mat A = cv::Mat::zeros(2 * n, 12, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double X = objPts[i].x, Y = objPts[i].y, Z = objPts[i].z;
        const double x = imgPts[i].x, y = imgPts[i].y;

        double *row0 = A.ptr<double>(2 * i);
        row0[0] = -X; row0[1] = -Y; row0[2] = -Z; row0[3] = -1.0;
        row0[8] = x * X; row0[9] = x * Y; row0[10] = x * Z; row0[11] = x;

        double *row1 = A.ptr<double>(2 * i + 1);
        row1[4] = -X; row1[5] = -Y; row1[6] = -Z; row1[7] = -1.0;
        row1[8] = y * X; row1[9] = y * Y; row1[10] = y * Z; row1[11] = y;
    }

    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);
    return vt.row(11).clone().reshape(0, 3); // 3x4
}

// Extracts a proper (R, t) from a raw (not-necessarily-orthonormal, unknown
// scale) 3x4 pose matrix produced by solvePoseDLT once denormalized back to
// calibrated-ray space. M = P(:, 0:3) should be a rotation scaled by some
// positive s; take the closest orthonormal matrix to M (Procrustes, via
// SVD) as R, and the mean of M's singular values as an estimate of s to
// unscale the translation. Resolves the DLT null-space sign ambiguity via
// det(M): a proper rotation scaled by s > 0 always has positive
// determinant. Returns false if M is (near-)singular (degenerate sample).
bool decomposeDltPose(const cv::Mat &P34, cv::Mat &R, cv::Mat &t)
{
    cv::Mat M = P34(cv::Range(0, 3), cv::Range(0, 3)).clone();
    cv::Mat tRaw = P34(cv::Range(0, 3), cv::Range(3, 4)).clone();
    if (cv::determinant(M) < 0.0) {
        M = -M;
        tRaw = -tRaw;
    }

    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt, cv::SVD::FULL_UV);
    const double s = (w.at<double>(0) + w.at<double>(1) + w.at<double>(2)) / 3.0;
    if (s < 1e-9)
        return false;

    R = u * vt;
    if (cv::determinant(R) < 0.0) {
        u.col(2) *= -1.0;
        R = u * vt;
    }
    t = tRaw / s;
    return true;
}

// Symmetric transfer error (squared pixels) between a homography-mapped
// point and its actual match, checked in both directions. H must be a
// contiguous 3x3 CV_64F matrix; Hinv is its inverse (passed in since it's
// constant per RANSAC candidate, not per correspondence).
double homographyTransferErrorSq(const cv::Mat &H, const cv::Mat &Hinv, const cv::Point2f &p1, const cv::Point2f &p2)
{
    const double *h = H.ptr<double>();
    const double w1 = h[6] * p1.x + h[7] * p1.y + h[8];
    if (std::abs(w1) < 1e-12)
        return std::numeric_limits<double>::max();
    const double px2 = (h[0] * p1.x + h[1] * p1.y + h[2]) / w1;
    const double py2 = (h[3] * p1.x + h[4] * p1.y + h[5]) / w1;
    const double d1 = (px2 - p2.x) * (px2 - p2.x) + (py2 - p2.y) * (py2 - p2.y);

    const double *hi = Hinv.ptr<double>();
    const double w2 = hi[6] * p2.x + hi[7] * p2.y + hi[8];
    if (std::abs(w2) < 1e-12)
        return std::numeric_limits<double>::max();
    const double px1 = (hi[0] * p2.x + hi[1] * p2.y + hi[2]) / w2;
    const double py1 = (hi[3] * p2.x + hi[4] * p2.y + hi[5]) / w2;
    const double d2 = (px1 - p1.x) * (px1 - p1.x) + (py1 - p1.y) * (py1 - p1.y);

    return d1 + d2;
}

// Plain scalar coefficients of a 3x3 F, pulled out once per RANSAC candidate
// so the per-correspondence inlier test below never touches cv::Mat (each
// cv::Mat construction heap-allocates; doing that inside an
// iterations x correspondences loop is what was causing multi-second stalls).
struct FCoeffs
{
    double f00, f01, f02;
    double f10, f11, f12;
    double f20, f21, f22;
};

FCoeffs extractF(const cv::Mat &F)
{
    return FCoeffs{F.at<double>(0, 0), F.at<double>(0, 1), F.at<double>(0, 2),
                    F.at<double>(1, 0), F.at<double>(1, 1), F.at<double>(1, 2),
                    F.at<double>(2, 0), F.at<double>(2, 1), F.at<double>(2, 2)};
}

// First-order (Sampson) approximation of squared geometric distance to the
// epipolar line, in squared pixels. Pure scalar arithmetic, no allocation.
double sampsonDistance(const FCoeffs &f, const cv::Point2f &p1, const cv::Point2f &p2)
{
    const double x1 = p1.x, y1 = p1.y;
    const double x2 = p2.x, y2 = p2.y;

    const double Fx1_0 = f.f00 * x1 + f.f01 * y1 + f.f02;
    const double Fx1_1 = f.f10 * x1 + f.f11 * y1 + f.f12;
    const double Fx1_2 = f.f20 * x1 + f.f21 * y1 + f.f22;

    const double Ftx2_0 = f.f00 * x2 + f.f10 * y2 + f.f20;
    const double Ftx2_1 = f.f01 * x2 + f.f11 * y2 + f.f21;

    const double x2tFx1 = x2 * Fx1_0 + y2 * Fx1_1 + Fx1_2;

    const double denom = Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 + Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1;
    if (denom < 1e-12)
        return std::numeric_limits<double>::max();
    return (x2tFx1 * x2tFx1) / denom;
}

// ---------------------------------------------------------------------
// Nister-style 5-point essential-matrix solver (calibrated coordinates).
//
// Unlike the F-then-convert path above (8-point on pixel coordinates,
// E = K^T F K), this solves for E directly under the calibrated
// constraint (equal nonzero singular values), which the 8-point solve
// never enforces. Minimal case: 5 correspondences give a 4-dimensional
// null space for E; imposing E's two defining matrix identities
// (det(E)=0, and 2*E*E^T*E - trace(E*E^T)*E = 0) cuts that down to (up
// to) 10 solutions. Verified against 500+ synthetic random 5-point
// problems during development (see chat history / session notes) before
// ever being wired in here.
//
// Method: build the 10 constraint equations via exact polynomial algebra
// (not a memorized published elimination recipe), then reduce them
// mechanically -- expand by multiplying by every monomial up to degree 1
// (giving 40 equations in the 35 monomials of degree <= 4), Gauss-Jordan
// eliminate with high-degree columns preferred as pivots, and read off a
// 10-dimensional "quotient basis" of low-degree monomials (empirically
// always {1,x,y,z,x^2,xy,xz,y^2,yz,z^2} for this problem). The
// multiplication-by-z operator on that basis is a 10x10 matrix; its
// eigenvalues are the z-coordinates of the (up to 10) solutions
// (cv::eigenNonSymmetric -- a generic linear-algebra primitive, the same
// category as the cv::SVD::compute/cv::solve calls already used
// throughout this file, not something specific to this algorithm).
// ---------------------------------------------------------------------

using Mono = std::array<int, 3>; // (i, j, k) exponents of x, y, z

int numMonomialsUpTo(int D) { return (D + 1) * (D + 2) * (D + 3) / 6; }

std::vector<Mono> monomialList(int D)
{
    std::vector<Mono> out;
    out.reserve(numMonomialsUpTo(D));
    for (int d = 0; d <= D; ++d) {
        for (int i = d; i >= 0; --i) {
            for (int j = d - i; j >= 0; --j)
                out.push_back({i, j, d - i - j});
        }
    }
    return out;
}

// Flat-array (not map) monomial index: exponents never exceed kMaxExp here
// (working degree caps at 4), so O(1) direct lookup beats a tree/hash map,
// which matters since this sits on the RANSAC hot path.
struct MonoIndex
{
    static constexpr int kMaxExp = 8;
    int maxDeg;
    std::vector<Mono> list;
    std::array<int, kMaxExp * kMaxExp * kMaxExp> lut;

    explicit MonoIndex(int D) : maxDeg(D), list(monomialList(D))
    {
        lut.fill(-1);
        for (size_t idx = 0; idx < list.size(); ++idx) {
            const Mono &m = list[idx];
            lut[(m[0] * kMaxExp + m[1]) * kMaxExp + m[2]] = static_cast<int>(idx);
        }
    }

    int indexOf(int i, int j, int k) const
    {
        if (i < 0 || j < 0 || k < 0 || i >= kMaxExp || j >= kMaxExp || k >= kMaxExp)
            return -1;
        return lut[(i * kMaxExp + j) * kMaxExp + k];
    }
};

// MonoIndex is a pure function of the degree, never of point data -- build
// each one once and reuse for the program's lifetime instead of rebuilding
// (heap-allocated vector + array) on every RANSAC iteration.
const MonoIndex &monoIndexFor(int D)
{
    static std::vector<std::unique_ptr<MonoIndex>> cache;
    if (D >= static_cast<int>(cache.size()))
        cache.resize(D + 1);
    if (!cache[D])
        cache[D] = std::make_unique<MonoIndex>(D);
    return *cache[D];
}

struct Poly
{
    int maxDeg = 0;
    std::vector<double> c;
    static Poly zero(int D) { return Poly{D, std::vector<double>(numMonomialsUpTo(D), 0.0)}; }
};

Poly polyExtend(const Poly &p, int newMaxDeg)
{
    if (p.maxDeg == newMaxDeg)
        return p;
    Poly out = Poly::zero(newMaxDeg);
    std::copy(p.c.begin(), p.c.end(), out.c.begin());
    return out;
}

Poly polyAdd(const Poly &a, const Poly &b)
{
    const int D = std::max(a.maxDeg, b.maxDeg);
    Poly ae = polyExtend(a, D), be = polyExtend(b, D);
    Poly out = Poly::zero(D);
    for (size_t i = 0; i < out.c.size(); ++i)
        out.c[i] = ae.c[i] + be.c[i];
    return out;
}

Poly polySub(const Poly &a, const Poly &b)
{
    const int D = std::max(a.maxDeg, b.maxDeg);
    Poly ae = polyExtend(a, D), be = polyExtend(b, D);
    Poly out = Poly::zero(D);
    for (size_t i = 0; i < out.c.size(); ++i)
        out.c[i] = ae.c[i] - be.c[i];
    return out;
}

Poly polyScale(const Poly &a, double s)
{
    Poly out = a;
    for (double &v : out.c)
        v *= s;
    return out;
}

Poly polyMul(const Poly &a, const Poly &b)
{
    const int Da = a.maxDeg, Db = b.maxDeg;
    const int Dout = Da + Db;
    const MonoIndex &idxA = monoIndexFor(Da);
    const MonoIndex &idxB = monoIndexFor(Db);
    const MonoIndex &idxOut = monoIndexFor(Dout);
    Poly out = Poly::zero(Dout);
    for (int ia = 0; ia < static_cast<int>(idxA.list.size()); ++ia) {
        if (a.c[ia] == 0.0)
            continue;
        const Mono &ma = idxA.list[ia];
        for (int ib = 0; ib < static_cast<int>(idxB.list.size()); ++ib) {
            if (b.c[ib] == 0.0)
                continue;
            const Mono &mb = idxB.list[ib];
            const int oi = idxOut.indexOf(ma[0] + mb[0], ma[1] + mb[1], ma[2] + mb[2]);
            out.c[oi] += a.c[ia] * b.c[ib];
        }
    }
    return out;
}

Poly polyConst(double v) { Poly p = Poly::zero(0); p.c[0] = v; return p; }
Poly polyX() { Poly p = Poly::zero(1); p.c[monoIndexFor(1).indexOf(1, 0, 0)] = 1.0; return p; }
Poly polyY() { Poly p = Poly::zero(1); p.c[monoIndexFor(1).indexOf(0, 1, 0)] = 1.0; return p; }
Poly polyZ() { Poly p = Poly::zero(1); p.c[monoIndexFor(1).indexOf(0, 0, 1)] = 1.0; return p; }

struct PolyMat3
{
    Poly e[3][3];
};

PolyMat3 buildEParam(const cv::Mat &X, const cv::Mat &Y, const cv::Mat &Z, const cv::Mat &W)
{
    PolyMat3 E;
    Poly px = polyX(), py = polyY(), pz = polyZ();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            E.e[r][c] = polyAdd(polyAdd(polyScale(px, X.at<double>(r, c)), polyScale(py, Y.at<double>(r, c))),
                                 polyAdd(polyScale(pz, Z.at<double>(r, c)), polyConst(W.at<double>(r, c))));
        }
    }
    return E;
}

Poly poly3Det(const PolyMat3 &M)
{
    auto &e = M.e;
    Poly t1 = polySub(polyMul(e[1][1], e[2][2]), polyMul(e[1][2], e[2][1]));
    Poly t2 = polySub(polyMul(e[1][0], e[2][2]), polyMul(e[1][2], e[2][0]));
    Poly t3 = polySub(polyMul(e[1][0], e[2][1]), polyMul(e[1][1], e[2][0]));
    return polyAdd(polySub(polyMul(e[0][0], t1), polyMul(e[0][1], t2)), polyMul(e[0][2], t3));
}

PolyMat3 poly3MatMul(const PolyMat3 &A, const PolyMat3 &B)
{
    PolyMat3 C;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            Poly s = polyConst(0.0);
            for (int k = 0; k < 3; ++k)
                s = polyAdd(s, polyMul(A.e[r][k], B.e[k][c]));
            C.e[r][c] = s;
        }
    }
    return C;
}

PolyMat3 poly3Transpose(const PolyMat3 &A)
{
    PolyMat3 T;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            T.e[r][c] = A.e[c][r];
    return T;
}

Poly poly3Trace(const PolyMat3 &A) { return polyAdd(polyAdd(A.e[0][0], A.e[1][1]), A.e[2][2]); }

PolyMat3 poly3ScaleMat(const PolyMat3 &A, const Poly &s)
{
    PolyMat3 out;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out.e[r][c] = polyMul(A.e[r][c], s);
    return out;
}

// The 10 base constraint equations (1 det + 9 from the trace identity) for
// E(x,y,z) = xX + yY + zZ + W, each a cubic polynomial in x,y,z.
std::vector<Poly> buildFivePointEquations(const cv::Mat &X, const cv::Mat &Y, const cv::Mat &Z, const cv::Mat &W)
{
    PolyMat3 E = buildEParam(X, Y, Z, W);
    PolyMat3 Et = poly3Transpose(E);
    PolyMat3 EEt = poly3MatMul(E, Et);
    Poly halfTrace = polyScale(poly3Trace(EEt), 0.5);
    PolyMat3 EEtE = poly3MatMul(EEt, E);
    PolyMat3 rhs = poly3ScaleMat(E, halfTrace);

    std::vector<Poly> eqs;
    eqs.push_back(poly3Det(E));
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            eqs.push_back(polySub(EEtE.e[r][c], rhs.e[r][c]));
    return eqs;
}

// Raw-pointer Gauss-Jordan elimination with partial pivoting, columns
// visited in the given order. M must be CV_64F and continuous (true for a
// freshly-allocated cv::Mat). Using M.row(i)-based cv::Mat expressions here
// instead of raw pointers measured ~15x slower in profiling (temporary
// Mat/MatExpr construction per row op dominates when rows/cols are this
// small) -- see DEBUGGING.md for the benchmark.
void fivePointGaussJordan(cv::Mat &M, const std::vector<int> &colOrder, std::vector<int> &pivotColOfRow,
                           std::vector<char> &isPivotCol)
{
    CV_Assert(M.isContinuous());
    const int rows = M.rows, cols = M.cols;
    double *const base = M.ptr<double>();
    auto rowPtr = [&](int r) { return base + static_cast<size_t>(r) * cols; };

    isPivotCol.assign(cols, 0);
    pivotColOfRow.assign(rows, -1);
    int pivotRow = 0;
    for (int col : colOrder) {
        if (pivotRow >= rows)
            break;
        int best = -1;
        double bestVal = 1e-9;
        for (int r = pivotRow; r < rows; ++r) {
            const double v = std::abs(rowPtr(r)[col]);
            if (v > bestVal) {
                bestVal = v;
                best = r;
            }
        }
        if (best < 0)
            continue;
        if (best != pivotRow) {
            double *rb = rowPtr(best), *rp = rowPtr(pivotRow);
            for (int c = 0; c < cols; ++c)
                std::swap(rb[c], rp[c]);
        }
        double *pr = rowPtr(pivotRow);
        const double invPv = 1.0 / pr[col];
        for (int c = 0; c < cols; ++c)
            pr[c] *= invPv;
        for (int r = 0; r < rows; ++r) {
            if (r == pivotRow)
                continue;
            double *rr = rowPtr(r);
            const double factor = rr[col];
            if (factor != 0.0) {
                for (int c = 0; c < cols; ++c)
                    rr[c] -= factor * pr[c];
            }
        }
        isPivotCol[col] = 1;
        pivotColOfRow[pivotRow] = col;
        ++pivotRow;
    }
}

struct FivePointReducedSystem
{
    MonoIndex idx{0};
    std::vector<int> pivotOfCol;
    cv::Mat R;
    std::vector<int> freeCols;
};

// Expand the 10 generators by every monomial up to degree (Ds-3), reduce,
// and check the result is the expected clean 10-dimensional, degree<=Ds-1
// quotient basis. Ds=4 is what this specific problem's structure needs
// (empirically verified, not assumed) in the ~99.6% non-degenerate case;
// returns false (caller should treat the sample as degenerate and move on,
// exactly like the existing estimateHomographyRansac()'s
// `if (!cv::invert(H, Hinv)) continue;` pattern) rather than silently
// proceeding on an unverified assumption otherwise.
bool buildFivePointReducedSystem(const std::vector<Poly> &generators, int Ds, FivePointReducedSystem &out)
{
    out.idx = monoIndexFor(Ds);
    const int numCols = static_cast<int>(out.idx.list.size());

    std::vector<Poly> equations;
    const std::vector<Mono> multipliers = monomialList(Ds - 3);
    equations.reserve(generators.size() * multipliers.size());
    for (const Poly &g : generators) {
        for (const Mono &m : multipliers) {
            Poly mono = Poly::zero(m[0] + m[1] + m[2]);
            const MonoIndex &mi = monoIndexFor(m[0] + m[1] + m[2]);
            mono.c[mi.indexOf(m[0], m[1], m[2])] = 1.0;
            equations.push_back(polyExtend(polyMul(mono, g), Ds));
        }
    }

    cv::Mat M(static_cast<int>(equations.size()), numCols, CV_64F);
    for (size_t r = 0; r < equations.size(); ++r)
        for (int c = 0; c < numCols; ++c)
            M.at<double>(static_cast<int>(r), c) = equations[r].c[c];

    // Column order: descending total degree, so high-degree monomials are
    // preferred as pivots, leaving low-degree ones as the quotient basis.
    std::vector<int> colOrder(numCols);
    for (int c = 0; c < numCols; ++c)
        colOrder[c] = c;
    std::sort(colOrder.begin(), colOrder.end(), [&](int a, int b) {
        const int da = out.idx.list[a][0] + out.idx.list[a][1] + out.idx.list[a][2];
        const int db = out.idx.list[b][0] + out.idx.list[b][1] + out.idx.list[b][2];
        return da > db;
    });

    std::vector<int> pivotColOfRow;
    std::vector<char> isPivotCol;
    fivePointGaussJordan(M, colOrder, pivotColOfRow, isPivotCol);
    out.R = M;

    out.freeCols.clear();
    int maxFreeDeg = -1;
    for (int c = 0; c < numCols; ++c) {
        if (!isPivotCol[c]) {
            out.freeCols.push_back(c);
            const Mono &mo = out.idx.list[c];
            maxFreeDeg = std::max(maxFreeDeg, mo[0] + mo[1] + mo[2]);
        }
    }
    out.pivotOfCol.assign(numCols, -1);
    for (int r = 0; r < M.rows; ++r)
        if (pivotColOfRow[r] >= 0)
            out.pivotOfCol[pivotColOfRow[r]] = r;

    return static_cast<int>(out.freeCols.size()) == 10 && maxFreeDeg <= Ds - 1;
}

std::vector<double> fivePointReduceMonomial(const FivePointReducedSystem &sys, int i, int j, int k)
{
    std::vector<double> result(sys.freeCols.size(), 0.0);
    const int col = sys.idx.indexOf(i, j, k);
    for (size_t f = 0; f < sys.freeCols.size(); ++f) {
        if (sys.freeCols[f] == col) {
            result[f] = 1.0;
            return result;
        }
    }
    const int row = sys.pivotOfCol[col];
    for (size_t f = 0; f < sys.freeCols.size(); ++f)
        result[f] = -sys.R.at<double>(row, sys.freeCols[f]);
    return result;
}

cv::Mat buildFivePointMulZMatrix(const FivePointReducedSystem &sys)
{
    const int K = static_cast<int>(sys.freeCols.size());
    cv::Mat Mz(K, K, CV_64F);
    for (int f = 0; f < K; ++f) {
        const Mono &m = sys.idx.list[sys.freeCols[f]];
        std::vector<double> col = fivePointReduceMonomial(sys, m[0], m[1], m[2] + 1);
        for (int r = 0; r < K; ++r)
            Mz.at<double>(r, f) = col[r];
    }
    return Mz;
}

// Solves the minimal 5-point problem for E given 5 *calibrated* (K^-1
// applied) correspondences. Returns up to 10 real candidate essential
// matrices, already filtered for numerical self-consistency (does NOT yet
// check against a larger inlier set -- that's estimateEssentialRansac()'s
// job). Returns an empty vector for a degenerate 5-point sample (rare,
// ~0.4% of random configurations in testing) rather than guessing.
std::vector<cv::Mat> solveFivePoint(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2)
{
    std::vector<cv::Mat> candidates;
    if (pts1.size() != 5 || pts2.size() != 5)
        return candidates;

    cv::Mat Q(5, 9, CV_64F);
    for (int i = 0; i < 5; ++i) {
        const double x1 = pts1[i].x, y1 = pts1[i].y, x2 = pts2[i].x, y2 = pts2[i].y;
        Q.at<double>(i, 0) = x2 * x1;
        Q.at<double>(i, 1) = x2 * y1;
        Q.at<double>(i, 2) = x2;
        Q.at<double>(i, 3) = y2 * x1;
        Q.at<double>(i, 4) = y2 * y1;
        Q.at<double>(i, 5) = y2;
        Q.at<double>(i, 6) = x1;
        Q.at<double>(i, 7) = y1;
        Q.at<double>(i, 8) = 1.0;
    }
    cv::Mat w, u, vt;
    cv::SVD::compute(Q, w, u, vt, cv::SVD::FULL_UV);
    const cv::Mat X = vt.row(5).clone().reshape(0, 3);
    const cv::Mat Y = vt.row(6).clone().reshape(0, 3);
    const cv::Mat Z = vt.row(7).clone().reshape(0, 3);
    const cv::Mat W = vt.row(8).clone().reshape(0, 3);

    std::vector<Poly> eqs = buildFivePointEquations(X, Y, Z, W);

    FivePointReducedSystem sys;
    if (!buildFivePointReducedSystem(eqs, 4, sys))
        return candidates; // degenerate sample -- caller (RANSAC) just skips it

    // Positions of the '1','x','y','z' free monomials -- needed to read
    // (x,y,z) back out of an eigenvector. Computed here (not assumed fixed)
    // since it's cheap and this guards against ever silently misreading a
    // different quotient basis ordering.
    int idx1 = -1, idxX = -1, idxY = -1, idxZ = -1;
    for (size_t f = 0; f < sys.freeCols.size(); ++f) {
        const Mono &m = sys.idx.list[sys.freeCols[f]];
        if (m == Mono{0, 0, 0}) idx1 = static_cast<int>(f);
        if (m == Mono{1, 0, 0}) idxX = static_cast<int>(f);
        if (m == Mono{0, 1, 0}) idxY = static_cast<int>(f);
        if (m == Mono{0, 0, 1}) idxZ = static_cast<int>(f);
    }
    if (idx1 < 0 || idxX < 0 || idxY < 0 || idxZ < 0)
        return candidates;

    cv::Mat Mz = buildFivePointMulZMatrix(sys);
    // Column i of Mz is the reduction of z*basis_i, so the per-solution
    // evaluation vector is a left eigenvector of Mz (right eigenvector of
    // Mz^T) -- see the derivation in session notes/DEBUGGING.md.
    cv::Mat eigenvalues, eigenvectors;
    cv::eigenNonSymmetric(Mz.t(), eigenvalues, eigenvectors);

    for (int e = 0; e < eigenvalues.rows; ++e) {
        const double z = eigenvalues.at<double>(e);
        cv::Mat v = eigenvectors.row(e);
        const double wv = v.at<double>(idx1);
        if (std::abs(wv) < 1e-9)
            continue;
        const double x = v.at<double>(idxX) / wv;
        const double y = v.at<double>(idxY) / wv;
        const double zFromVec = v.at<double>(idxZ) / wv;
        // Internal self-consistency check: the eigenvalue and the 'z'
        // component read from its own eigenvector must agree for a genuine
        // real solution -- cv::eigenNonSymmetric assumes real eigenvalues,
        // so this is what catches (and discards) a complex-conjugate pair
        // it can't represent correctly.
        if (std::abs(zFromVec - z) > 1e-3 * (std::abs(z) + 1.0))
            continue;

        cv::Mat Ecand = X * x + Y * y + Z * z + W;
        // Strong independent validation against the actual 5 correspondences
        // (not just the equations derived from them) before ever handing
        // this candidate back to the caller.
        double maxEpi = 0.0;
        for (int p = 0; p < 5; ++p) {
            cv::Mat p1 = (cv::Mat_<double>(3, 1) << pts1[p].x, pts1[p].y, 1.0);
            cv::Mat p2 = (cv::Mat_<double>(3, 1) << pts2[p].x, pts2[p].y, 1.0);
            cv::Mat r = p2.t() * Ecand * p1;
            maxEpi = std::max(maxEpi, std::abs(r.at<double>(0)));
        }
        if (maxEpi < 1e-4)
            candidates.push_back(Ecand);
    }
    return candidates;
}

// Projects an arbitrary 3x3 matrix onto the essential-matrix manifold: SVD,
// force the two nonzero singular values equal (average them), zero the
// third, renormalize. Unlike this file's earlier (removed) F-only
// projectFundamentalRank2 -- which only zeroed the smallest singular value
// -- this enforces E's *stronger* calibrated constraint. Needed because,
// unlike the no-op finding for cv::recoverPose on an *already-decomposed*
// E (see DEBUGGING.md's Session 3 note: recoverPose only reads U/Vt, not
// the singular values), the linear refit below produces a genuinely
// different U/Vt than the minimal-5-point estimate and isn't guaranteed to
// satisfy the constraint at all -- this projection is doing real work here,
// not repeating that earlier no-op.
cv::Mat projectEssentialManifold(const cv::Mat &E)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(E, w, u, vt, cv::SVD::FULL_UV);
    const double avgSv = (w.at<double>(0) + w.at<double>(1)) / 2.0;
    cv::Mat sv = (cv::Mat_<double>(3, 1) << avgSv, avgSv, 0.0);
    cv::Mat Ep = u * cv::Mat::diag(sv) * vt;
    const double n = cv::norm(Ep);
    return (n > 1e-12) ? Ep / n : Ep;
}

// Linear refit of E over the full inlier set (calibrated coordinates),
// analogous to solveEightPoint's refit over F's inlier set: same
// epipolar-constraint construction as the 5-point minimal solver's null-
// space matrix, just with all inliers instead of exactly 5 rows, giving an
// over-determined least-squares solve. The raw result won't generally
// satisfy E's constraint (only the 5-point minimal case is guaranteed
// rank-2; a general linear least-squares fit isn't even guaranteed that
// much), so it's projected onto the essential-matrix manifold before
// being returned.
cv::Mat refitEssentialLinear(const std::vector<cv::Point2f> &calib1, const std::vector<cv::Point2f> &calib2)
{
    const int n = static_cast<int>(calib1.size());
    cv::Mat Q(n, 9, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double x1 = calib1[i].x, y1 = calib1[i].y, x2 = calib2[i].x, y2 = calib2[i].y;
        Q.at<double>(i, 0) = x2 * x1;
        Q.at<double>(i, 1) = x2 * y1;
        Q.at<double>(i, 2) = x2;
        Q.at<double>(i, 3) = y2 * x1;
        Q.at<double>(i, 4) = y2 * y1;
        Q.at<double>(i, 5) = y2;
        Q.at<double>(i, 6) = x1;
        Q.at<double>(i, 7) = y1;
        Q.at<double>(i, 8) = 1.0;
    }
    cv::Mat w, u, vt;
    cv::SVD::compute(Q, w, u, vt, cv::SVD::FULL_UV);
    const cv::Mat Eraw = vt.row(8).clone().reshape(0, 3);
    return projectEssentialManifold(Eraw);
}

// Signed Sampson residual (calibrated coordinates) -- same formula as
// sampsonDistance but signed (r*r == sampsonDistance(...)), needed because
// the Gauss-Newton/LM refinement below builds a proper per-point Jacobian,
// not just a gradient of the summed squared cost. Same structure as the
// removed refineFundamentalSampson()'s sampsonResidual (see
// EightPointLegacy.cpp), duplicated here rather than shared since that
// module is deliberately SlamWorker-independent.
double sampsonResidualCalib(const FCoeffs &f, const cv::Point2f &p1, const cv::Point2f &p2)
{
    const double x1 = p1.x, y1 = p1.y;
    const double x2 = p2.x, y2 = p2.y;

    const double Fx1_0 = f.f00 * x1 + f.f01 * y1 + f.f02;
    const double Fx1_1 = f.f10 * x1 + f.f11 * y1 + f.f12;
    const double Fx1_2 = f.f20 * x1 + f.f21 * y1 + f.f22;

    const double Ftx2_0 = f.f00 * x2 + f.f10 * y2 + f.f20;
    const double Ftx2_1 = f.f01 * x2 + f.f11 * y2 + f.f21;

    const double e = x2 * Fx1_0 + y2 * Fx1_1 + Fx1_2;
    const double denom = Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 + Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1;
    if (denom < 1e-12)
        return 0.0;
    return e / std::sqrt(denom);
}

// Gold Standard (Sampson-distance-minimizing Levenberg-Marquardt) nonlinear
// refinement of E over the full inlier set, in calibrated coordinates.
// Same structure as the removed refineFundamentalSampson() (see
// EightPointLegacy.cpp for the byte-identical F-only original), with one
// essential difference: projects onto the *essential-matrix* manifold
// (projectEssentialManifold(), equal singular values) after every step
// instead of just rank-2, since E carries that stronger calibrated
// constraint and the whole point of the 5-point approach is to actually
// enforce it end-to-end rather than lose it again during refinement.
cv::Mat refineEssentialSampson(const cv::Mat &Einit, const std::vector<cv::Point2f> &calib1,
                                const std::vector<cv::Point2f> &calib2)
{
    constexpr int kMaxIters = 10;
    constexpr int kMaxLmTries = 10;
    constexpr double kFdStep = 1e-6;

    const int n = static_cast<int>(calib1.size());
    cv::Mat E = projectEssentialManifold(Einit);

    auto residuals = [&](const cv::Mat &Ec) {
        std::vector<double> r(n);
        const FCoeffs fc = extractF(Ec);
        for (int i = 0; i < n; ++i)
            r[i] = sampsonResidualCalib(fc, calib1[i], calib2[i]);
        return r;
    };
    auto costOf = [](const std::vector<double> &r) {
        double c = 0.0;
        for (double v : r)
            c += v * v;
        return c;
    };

    std::vector<double> r0 = residuals(E);
    double cost = costOf(r0);
    double lambda = 1e-3;

    for (int iter = 0; iter < kMaxIters; ++iter) {
        cv::Mat J(n, 9, CV_64F);
        double *fptr = E.ptr<double>();
        for (int j = 0; j < 9; ++j) {
            const double orig = fptr[j];
            fptr[j] = orig + kFdStep;
            const std::vector<double> rPlus = residuals(E);
            fptr[j] = orig - kFdStep;
            const std::vector<double> rMinus = residuals(E);
            fptr[j] = orig;
            for (int i = 0; i < n; ++i)
                J.at<double>(i, j) = (rPlus[i] - rMinus[i]) / (2.0 * kFdStep);
        }

        cv::Mat rVec(n, 1, CV_64F);
        for (int i = 0; i < n; ++i)
            rVec.at<double>(i) = r0[i];

        const cv::Mat JtJ = J.t() * J;
        const cv::Mat Jtr = J.t() * rVec;

        bool improved = false;
        for (int lmTry = 0; lmTry < kMaxLmTries; ++lmTry) {
            const cv::Mat A = JtJ + lambda * cv::Mat::diag(JtJ.diag());
            cv::Mat delta;
            if (!cv::solve(A, -Jtr, delta, cv::DECOMP_SVD)) {
                lambda *= 10.0;
                continue;
            }

            cv::Mat Ecand = E.clone();
            double *ecand = Ecand.ptr<double>();
            for (int j = 0; j < 9; ++j)
                ecand[j] += delta.at<double>(j);
            Ecand = projectEssentialManifold(Ecand);

            const std::vector<double> rCand = residuals(Ecand);
            const double candCost = costOf(rCand);
            if (candCost < cost) {
                E = Ecand;
                r0 = rCand;
                cost = candCost;
                lambda = std::max(lambda * 0.3, 1e-8);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if (!improved)
            break;
    }

    return E;
}
}

SlamWorker::SlamWorker(QObject *parent)
    : QObject(parent)
{
    rebuildDetector();
    resetSlamState();
}

SlamWorker::~SlamWorker()
{
    // ORB_SLAM3::System has no destructor of its own that calls Shutdown()
    // (see Shutdown()'s own implementation in System.cc -- it stops
    // LocalMapping/LoopClosing's background threads and, if configured,
    // saves the Atlas) -- so this must be done explicitly before
    // m_orbSlam3System is destroyed, or those threads are abandoned rather
    // than cleanly stopped. Defined here (not left implicit) specifically
    // because ORB_SLAM3::System is only forward-declared in SlamWorker.h --
    // std::unique_ptr's destructor needs the complete type, which is only
    // visible here, in this .cpp, via <System.h>.
    if (m_orbSlam3System)
        m_orbSlam3System->Shutdown();
}

void SlamWorker::rebuildDetector()
{
    m_detector = feature_detector::createDetector(m_detectorType, m_siftSettings, m_orbSettings);
    m_matcherNorm = feature_detector::normTypeFor(m_detectorType);
}

void SlamWorker::resetSlamState()
{
    m_state = State::Idle;
    m_refKeypoints.clear();
    m_refDescriptors.release();
    m_refR = cv::Mat::eye(3, 3, CV_64F);
    m_refT = cv::Mat::zeros(3, 1, CV_64F);
    m_refFrameIndex = 0;
    m_avgStepScale = -1.0;
    m_recentStepDistances.clear();
    m_longTermStepScale = -1.0;
    m_mapPoints.clear();
    m_mapDescriptors.release();
    m_mapPointIds.clear();
    m_nextLandmarkId = 0;
    m_landmarkPositions.clear();
    m_landmarkObservations.clear();
    m_currR = cv::Mat::eye(3, 3, CV_64F);
    m_currT = cv::Mat::zeros(3, 1, CV_64F);
    m_velocityR = cv::Mat::eye(3, 3, CV_64F);
    m_velocityT = cv::Mat::zeros(3, 1, CV_64F);
    m_trajectory.clear();
    m_trajectoryFrameIndex.clear();
    m_keyframeHistory.clear();
    m_keyframeObservedLandmarkIds.clear();
    m_frameCount = 0;
    m_framesSinceKeyframe = 0;
    m_trackFailStreak = 0;

    // Tear down any live ORB-SLAM3 System -- called from openVideoFile()/
    // openCamera()/reset()/start() alike, so a new source or an explicit
    // reset never leaves a previous run's map/threads alive underneath a
    // fresh one. start() lazily reconstructs it right after this call when
    // m_orbSlam3Enabled is set (see start()) -- System has no live-
    // reconfigure or restart API, so "reset" for ORB-SLAM3 mode necessarily
    // means "throw the whole System away and build a new one".
    if (m_orbSlam3System) {
        m_orbSlam3System->Shutdown();
        m_orbSlam3System.reset();
    }
}

void SlamWorker::setReferenceFrame(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors,
                                    const cv::Mat &R, const cv::Mat &t)
{
    m_refKeypoints = kps;
    m_refDescriptors = descriptors.clone();
    m_refR = R.clone();
    m_refT = t.clone();
    m_refFrameIndex = m_frameCount;
}

bool SlamWorker::loadOxtsSpeeds(const QString &oxtsDir)
{
    QFile tsFile(oxtsDir + QStringLiteral("/timestamps.txt"));
    if (!tsFile.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;

    // KITTI OXTS timestamps look like "2011-10-03 12:55:34.997992704" --
    // only the time-of-day and only *differences* between consecutive
    // frames matter here, so parse HH:MM:SS.fractional directly into
    // seconds rather than dealing with a full date/epoch conversion
    // (validated this parsing + the vf field choice below against the
    // known-correct ground-truth path length, 3722.3m -- integrating
    // vf*dt over the whole sequence gave 3716.56m, 0.15% off).
    std::vector<double> timestamps;
    {
        QTextStream tsStream(&tsFile);
        while (!tsStream.atEnd()) {
            const QString line = tsStream.readLine().trimmed();
            if (line.isEmpty())
                continue;
            const QStringList parts = line.split(QLatin1Char(' '));
            if (parts.size() < 2)
                return false;
            const QStringList hms = parts[1].split(QLatin1Char(':'));
            if (hms.size() < 3)
                return false;
            timestamps.push_back(hms[0].toDouble() * 3600.0 + hms[1].toDouble() * 60.0 + hms[2].toDouble());
        }
    }
    if (timestamps.size() < 2)
        return false;

    QVector<double> cumulative;
    cumulative.reserve(static_cast<int>(timestamps.size()));
    cumulative.push_back(0.0);

    double cum = 0.0;
    for (int i = 1; i < static_cast<int>(timestamps.size()); ++i) {
        const QString dataPath =
            oxtsDir + QStringLiteral("/data/") + QString::number(i).rightJustified(10, QLatin1Char('0')) +
            QStringLiteral(".txt");
        QFile dataFile(dataPath);
        if (!dataFile.open(QIODevice::ReadOnly | QIODevice::Text))
            return false;
        QTextStream dataStream(&dataFile);
        const QStringList tokens = dataStream.readLine().split(QLatin1Char(' '), Qt::SkipEmptyParts);
        if (tokens.size() < 9)
            return false;

        const double vf = tokens[8].toDouble(); // field index 8 == "vf: forward velocity (m/s)"
        const double dt = timestamps[i] - timestamps[i - 1];
        cum += vf * dt;
        cumulative.push_back(cum);
    }

    m_oxtsCumulativeDistance = std::move(cumulative);
    return true;
}

double SlamWorker::oxtsDistanceBetween(int frameA, int frameB) const
{
    if (!m_oxtsEnabled || m_oxtsCumulativeDistance.isEmpty())
        return -1.0;
    const int last = m_oxtsCumulativeDistance.size() - 1;
    const int a = std::clamp(frameA, 0, last);
    const int b = std::clamp(frameB, 0, last);
    return std::abs(m_oxtsCumulativeDistance[b] - m_oxtsCumulativeDistance[a]);
}

bool SlamWorker::loadImuOrientation(const QString &oxtsDir, const QString &calibDir)
{
    std::vector<cv::Mat> navFromBody;
    if (!imu_rotation::loadOxtsOrientations(oxtsDir, navFromBody))
        return false;

    imu_rotation::ImuToCameraCalib calib;
    if (!imu_rotation::loadImuToCameraCalib(calibDir, calib))
        return false;

    m_oxtsNavFromBody = std::move(navFromBody);
    m_imuToCameraCalib = calib;
    return true;
}

void SlamWorker::loadOxtsDir(const QString &oxtsDir)
{
    if (loadOxtsSpeeds(oxtsDir))
        emit statsUpdated(QStringLiteral("OXTS speed data loaded from %1").arg(oxtsDir));
    else
        emit statsUpdated(QStringLiteral("Failed to load OXTS speed data from %1 (need timestamps.txt + "
                                          "data/<frame>.txt, KITTI raw-data format)")
                               .arg(oxtsDir));
    // Reflects the actual resulting state, not just whether *this* call
    // succeeded -- a failed load leaves any previously loaded data (and
    // its availability) untouched, per loadOxtsSpeeds()'s own contract.
    emit oxtsAvailabilityChanged(!m_oxtsCumulativeDistance.isEmpty());
}

void SlamWorker::loadImuDirs(const QString &oxtsDir, const QString &calibDir)
{
    if (loadImuOrientation(oxtsDir, calibDir))
        emit statsUpdated(
            QStringLiteral("IMU orientation + calibration loaded (oxts: %1, calib: %2)").arg(oxtsDir, calibDir));
    else
        emit statsUpdated(QStringLiteral("Failed to load IMU orientation (oxts: %1, calib: %2) -- calibDir must "
                                          "be the same date as oxtsDir's drive")
                               .arg(oxtsDir, calibDir));
    emit imuAvailabilityChanged(!m_oxtsNavFromBody.empty() && !m_imuToCameraCalib.R.empty());
}

bool SlamWorker::loadGroundTruthPoses(const QString &posesPath)
{
    QFile file(posesPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;

    std::vector<cv::Mat> Rs, Ts;
    QTextStream stream(&file);
    while (!stream.atEnd()) {
        const QString line = stream.readLine();
        const QStringList parts = line.split(QLatin1Char(' '), Qt::SkipEmptyParts);
        if (parts.size() < 12)
            continue;
        cv::Mat R(3, 3, CV_64F);
        cv::Mat t(3, 1, CV_64F);
        bool ok = true;
        for (int row = 0; row < 3 && ok; ++row) {
            for (int col = 0; col < 3 && ok; ++col)
                R.at<double>(row, col) = parts[row * 4 + col].toDouble(&ok);
            if (ok)
                t.at<double>(row) = parts[row * 4 + 3].toDouble(&ok);
        }
        if (!ok)
            continue;
        Rs.push_back(R);
        Ts.push_back(t);
    }
    if (Rs.empty())
        return false;

    m_groundTruthR = std::move(Rs);
    m_groundTruthT = std::move(Ts);
    return true;
}

bool SlamWorker::loadOrbVocabulary(const QString &path)
{
    auto vocab = std::make_unique<OrbVocabulary>();
    if (!vocab->loadFromTextFile(path.toStdString())) {
        std::fprintf(stderr, "[config] failed to load ORB vocabulary from %s\n", qPrintable(path));
        return false;
    }
    m_orbVocabulary = std::move(vocab);
    std::fprintf(stderr, "[config] ORB vocabulary loaded from %s (%u words)\n", qPrintable(path),
                 static_cast<unsigned>(m_orbVocabulary->size()));
    return true;
}

bool SlamWorker::loadVladVocabulary(const QString &path)
{
    auto vocab = std::make_unique<vlad::VladVocabulary>();
    if (!vocab->loadFromTextFile(path.toStdString())) {
        std::fprintf(stderr, "[config] failed to load VLAD codebook from %s\n", qPrintable(path));
        return false;
    }
    m_vladVocabulary = std::move(vocab);
    std::fprintf(stderr, "[config] VLAD codebook loaded from %s (%u centroids)\n", qPrintable(path),
                 static_cast<unsigned>(m_vladVocabulary->size()));
    return true;
}

bool SlamWorker::loadSiftVocabulary(const QString &path)
{
    auto vocab = std::make_unique<SiftVocabulary>();
    if (!vocab->loadFromTextFile(path.toStdString())) {
        std::fprintf(stderr, "[config] failed to load SIFT (RootSIFT) DBoW2 vocabulary from %s\n",
                     qPrintable(path));
        return false;
    }
    m_siftVocabulary = std::move(vocab);
    std::fprintf(stderr, "[config] SIFT DBoW2 vocabulary loaded from %s (%u words)\n", qPrintable(path),
                 static_cast<unsigned>(m_siftVocabulary->size()));
    return true;
}

namespace {
struct KnownKittiSequence
{
    QString oxtsDir;
    QString calibDir;
    QString posesPath;
};

// Known local OXTS/calibration/poses paths, keyed by KITTI sequence number
// -- BUG FIX (see DEBUGGING.md): this used to be a single hardcoded
// sequence-00 path set applied unconditionally regardless of what was
// actually opened, silently overwriting a manually-loaded different
// sequence's OXTS/IMU (e.g. sequence 01's) the moment its own video was
// (re)opened, since autoLoadKittiExtras() fires on every sourceOpened.
// Add an entry here whenever another sequence's data gets extracted
// locally; an unrecognized sequence is simply left alone (see
// autoLoadKittiExtras() below).
const QMap<QString, KnownKittiSequence> &knownKittiSequences()
{
    static const QMap<QString, KnownKittiSequence> table = {
        {QStringLiteral("00"),
         {QStringLiteral("/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/oxts_seq00/2011_10_03/"
                          "2011_10_03_drive_0027_sync/oxts"),
          QStringLiteral("/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/calib_2011_10_03/2011_10_03"),
          QStringLiteral("/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/dataset/poses/00.txt")}},
        {QStringLiteral("01"),
         {QStringLiteral("/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/oxts_seq01/2011_10_03/"
                          "2011_10_03_drive_0042_sync/oxts"),
          QStringLiteral("/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/oxts_seq01/2011_10_03/"
                          "2011_10_03_drive_0042_sync"),
          QStringLiteral("/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/dataset/poses/01.txt")}},
    };
    return table;
}
} // namespace

void SlamWorker::autoLoadKittiExtras()
{
    // Detect the sequence number from the opened video's filename (trailing
    // digits, e.g. "kitti_01.mp4" -> "01" -- same heuristic
    // MainWindow::tryAutoLoadGroundTruth() uses for its own ground-truth
    // lookup). No match, or a sequence this session has no known local
    // data for, is a silent no-op -- never overwrite whatever's already
    // loaded (e.g. a manual browse) with something unrelated.
    if (m_lastOpenedVideoPath.isEmpty())
        return;
    const QFileInfo videoInfo(m_lastOpenedVideoPath);
    static const QRegularExpression trailingDigits(QStringLiteral("(\\d+)$"));
    const QRegularExpressionMatch match = trailingDigits.match(videoInfo.completeBaseName());
    if (!match.hasMatch())
        return;
    const QString seq = match.captured(1);
    const auto &table = knownKittiSequences();
    const auto it = table.find(seq);
    if (it == table.end())
        return;
    const KnownKittiSequence &known = *it;

    if (QFileInfo::exists(known.posesPath) && loadGroundTruthPoses(known.posesPath))
        emit statsUpdated(QStringLiteral("Auto-loaded ground-truth poses for KITTI sequence %1").arg(seq));

    if (!QFileInfo::exists(known.oxtsDir))
        return;

    if (loadOxtsSpeeds(known.oxtsDir))
        emit statsUpdated(QStringLiteral("Auto-loaded OXTS speed data for KITTI sequence %1").arg(seq));
    emit oxtsAvailabilityChanged(!m_oxtsCumulativeDistance.isEmpty());

    if (QFileInfo::exists(known.calibDir) && loadImuOrientation(known.oxtsDir, known.calibDir))
        emit statsUpdated(QStringLiteral("Auto-loaded IMU orientation + calibration for KITTI sequence %1").arg(seq));
    emit imuAvailabilityChanged(!m_oxtsNavFromBody.empty() && !m_imuToCameraCalib.R.empty());
}

void SlamWorker::pushTrajectoryPoint(const cv::Mat &R, const cv::Mat &t, bool updateAvgStepScale)
{
    const cv::Mat C = -R.t() * t;
    const QPointF point(C.at<double>(0), C.at<double>(2));

    if (!m_trajectory.isEmpty() && updateAvgStepScale) {
        const QPointF &prev = m_trajectory.last();
        const double stepDist = std::hypot(point.x() - prev.x(), point.y() - prev.y());
        if (stepDist > 1e-9) {
            if (m_avgStepScale < 0.0) {
                m_avgStepScale = stepDist;
                m_longTermStepScale = stepDist;
                m_recentStepDistances.push_back(stepDist);
            } else {
                // Slow long-term baseline, clamped relative to itself (not
                // the fast-adapting m_avgStepScale) so it isn't dragged
                // down by the same collapse it exists to guard against --
                // see its doc comment in SlamWorker.h.
                const double clampedForLongTerm =
                    std::min(stepDist, kMaxAvgStepUpdateMultiplier * m_longTermStepScale);
                m_longTermStepScale = (1.0 - kLongTermScaleAlpha) * m_longTermStepScale +
                                       kLongTermScaleAlpha * clampedForLongTerm;

                // Clamp what feeds the estimator (not the pose itself) so a
                // single large-but-plausible step can't dominate the window --
                // see kMaxAvgStepUpdateMultiplier's doc comment in SlamWorker.h.
                const double clampedStep =
                    std::min(stepDist, kMaxAvgStepUpdateMultiplier * m_avgStepScale);
                m_recentStepDistances.push_back(clampedStep);
                while (m_recentStepDistances.size() > kScaleWindowSize)
                    m_recentStepDistances.pop_front();

                QVector<double> sorted = m_recentStepDistances;
                std::sort(sorted.begin(), sorted.end());
                const double median = sorted[sorted.size() / 2];
                m_avgStepScale = std::max(median, kMinScaleFraction * m_longTermStepScale);
            }
        }
    }
    m_trajectory.push_back(point);
    m_trajectoryFrameIndex.push_back(m_frameCount);
}

bool SlamWorker::isPlausibleStep(const cv::Mat &R, const cv::Mat &t, int framesElapsed) const
{
    if (m_trajectory.isEmpty() || m_avgStepScale <= 0.0)
        return true;

    const cv::Mat C = -R.t() * t;
    const QPointF &prev = m_trajectory.last();
    const double stepDist = std::hypot(C.at<double>(0) - prev.x(), C.at<double>(2) - prev.y());
    return stepDist <= kMaxStepMultiplier * m_avgStepScale * std::max(1, framesElapsed);
}

void SlamWorker::appendToMap(std::vector<cv::Point3f> &&newPoints, const cv::Mat &newDescriptors,
                              const std::vector<long long> &newIds)
{
    if (newPoints.empty())
        return;

    m_mapPoints.insert(m_mapPoints.end(), newPoints.begin(), newPoints.end());
    m_mapPointIds.insert(m_mapPointIds.end(), newIds.begin(), newIds.end());
    if (m_mapDescriptors.empty())
        m_mapDescriptors = newDescriptors.clone();
    else
        cv::vconcat(m_mapDescriptors, newDescriptors, m_mapDescriptors);

    // Keep the map a bounded, recent window: evict the oldest points once
    // over the cap instead of freezing it, so matching stays local to where
    // the camera actually is rather than degrading against stale, long
    // out-of-view points as the sequence progresses.
    if (m_mapPoints.size() > kMaxMapPoints) {
        const size_t excess = m_mapPoints.size() - kMaxMapPoints;
        m_mapPoints.erase(m_mapPoints.begin(), m_mapPoints.begin() + static_cast<long>(excess));
        m_mapPointIds.erase(m_mapPointIds.begin(), m_mapPointIds.begin() + static_cast<long>(excess));
        m_mapDescriptors = m_mapDescriptors.rowRange(static_cast<int>(excess), m_mapDescriptors.rows).clone();
    }
}

void SlamWorker::clearOxtsImuData()
{
    m_oxtsCumulativeDistance.clear();
    m_oxtsNavFromBody.clear();
    m_imuToCameraCalib = imu_rotation::ImuToCameraCalib();
    emit oxtsAvailabilityChanged(false);
    emit imuAvailabilityChanged(false);
}

void SlamWorker::openVideoFile(const QString &path)
{
    stop();
    const bool ok = m_source.openFile(path.toStdString());
    resetSlamState();
    m_lastOpenedVideoPath = path;
    clearOxtsImuData();
    if (ok) {
        emit sourceOpened(true, QStringLiteral("Loaded video: %1 (%2x%3)")
                                     .arg(path)
                                     .arg(m_source.frameWidth())
                                     .arg(m_source.frameHeight()));
    } else {
        emit sourceOpened(false, QStringLiteral("Failed to open video: %1").arg(path));
    }
}

void SlamWorker::openCamera(int index)
{
    stop();
    const bool ok = m_source.openCamera(index);
    resetSlamState();
    m_lastOpenedVideoPath.clear();
    clearOxtsImuData();
    if (ok) {
        emit sourceOpened(true, QStringLiteral("Opened camera %1 (%2x%3)")
                                     .arg(index)
                                     .arg(m_source.frameWidth())
                                     .arg(m_source.frameHeight()));
    } else {
        emit sourceOpened(false, QStringLiteral("Failed to open camera %1").arg(index));
    }
}

void SlamWorker::previewFirstFrame()
{
    if (!m_source.isOpened())
        return;
    cv::Mat frame;
    if (!m_source.readFrame(frame))
        return;
    m_source.rewindToStart(); // don't consume this frame -- start() must still see it as frame 1

    m_lastDisplayBase = frame.clone();
    m_lastDisplayFrameCount = 1;
    refreshGroundTruthOverlayDisplay();
}

void SlamWorker::setGroundTruthOverlayEnabled(bool enabled)
{
    m_groundTruthOverlayEnabled = enabled;
    refreshGroundTruthOverlayDisplay();
}

void SlamWorker::setGroundTruthOverlayOffset(int dx, int dy)
{
    m_groundTruthOverlayOffsetX = dx;
    m_groundTruthOverlayOffsetY = dy;
    refreshGroundTruthOverlayDisplay();
}

void SlamWorker::setOldStreetOverlayOffset(int dx, int dy)
{
    m_oldStreetOverlayOffsetX = dx;
    m_oldStreetOverlayOffsetY = dy;
    refreshGroundTruthOverlayDisplay();
}

void SlamWorker::start()
{
    if (!m_source.isOpened()) {
        emit statsUpdated(QStringLiteral("No video source opened"));
        return;
    }
    resetSlamState(); // also tears down any previous m_orbSlam3System -- see its own doc comment

    if (m_orbSlam3Enabled) {
        // Lazily constructed here (not in setOrbSlam3Enabled()) because
        // building its settings YAML needs the video source's resolution,
        // which requires a source to already be open -- guaranteed by the
        // early-return above.
        const QString yamlPath = buildOrbSlam3SettingsYaml();
        if (yamlPath.isEmpty()) {
            emit statsUpdated(QStringLiteral("Failed to write ORB-SLAM3 settings file"));
            return;
        }
        // ORB_SLAM3::System's constructor calls exit(-1) directly (not a
        // throwable exception -- see System.cc) if the vocabulary fails to
        // load, which would silently kill this entire GUI process rather
        // than just fail this one Start() -- checked explicitly here since
        // that's the one hardcoded path (ORBSLAM3_VOCAB_PATH, see
        // CMakeLists.txt) most likely to be wrong on a machine without this
        // exact ORB_SLAM3 checkout.
        if (!QFileInfo::exists(QString::fromUtf8(ORBSLAM3_VOCAB_PATH))) {
            emit statsUpdated(QStringLiteral("ORB-SLAM3 vocabulary not found: %1")
                                   .arg(QString::fromUtf8(ORBSLAM3_VOCAB_PATH)));
            return;
        }
        // Loading the vocabulary text file below blocks this worker thread
        // for several real seconds with no other feedback -- confirmed this
        // session as a real usability trap: the status bar/video view
        // appear frozen during that window, which reads as "Start did
        // nothing", inviting a second Start press that tears down this
        // System mid-construction and restarts the whole (multi-second)
        // load from scratch, indefinitely. Emitted before the blocking call
        // so the status bar shows *something* is happening -- this signal
        // crosses to the GUI thread immediately since MainWindow's own
        // event loop isn't blocked by this (only this worker thread is).
        emit statsUpdated(QStringLiteral("Loading ORB-SLAM3 (vocabulary load takes a few seconds -- "
                                          "please wait, don't press Start again)..."));
        try {
            m_orbSlam3System = std::make_unique<ORB_SLAM3::System>(
                std::string(ORBSLAM3_VOCAB_PATH), yamlPath.toStdString(), ORB_SLAM3::System::MONOCULAR,
                false /* headless -- no Viewer, see third_party/ORB_SLAM3's own doc comments */);
        } catch (const std::exception &e) {
            emit statsUpdated(QStringLiteral("Failed to start ORB-SLAM3: %1").arg(QString::fromUtf8(e.what())));
            m_orbSlam3System.reset();
            return;
        }
        m_orbSlam3Clock.start();
    }

    if (!m_timer) {
        m_timer = new QTimer(this);
        connect(m_timer, &QTimer::timeout, this, &SlamWorker::processNext);
    }
    m_timer->start(m_realtimeThrottle ? kProcessIntervalMs : 0);
    emit trackingStateChanged(QStringLiteral("Idle"));
}

void SlamWorker::startUnthrottled()
{
    m_realtimeThrottle = false;
    start();
}

void SlamWorker::stop()
{
    if (m_timer)
        m_timer->stop();
}

void SlamWorker::reset()
{
    const bool wasRunning = m_timer && m_timer->isActive();
    stop();
    resetSlamState();
    m_source.rewindToStart();
    emit mapUpdated({}, {}, {});
    emit trackingStateChanged(QStringLiteral("Idle"));
    if (wasRunning)
        m_timer->start(m_realtimeThrottle ? kProcessIntervalMs : 0);
}

void SlamWorker::setIntrinsics(CameraIntrinsics intrinsics)
{
    m_intrinsics = intrinsics;
}

void SlamWorker::setSiftSettings(SiftSettings settings)
{
    m_siftSettings = settings;
    rebuildDetector();
}

void SlamWorker::setDetectorType(feature_detector::DetectorType type)
{
    m_detectorType = type;
    rebuildDetector();
}

void SlamWorker::setOrbSettings(feature_detector::OrbSettings settings)
{
    m_orbSettings = settings;
    rebuildDetector();
}

void SlamWorker::setPnpSettings(PnpSettings settings)
{
    m_pnpSettings = settings;
}

void SlamWorker::setOrbSlam3Enabled(bool enabled)
{
    m_orbSlam3Enabled = enabled;
    if (!enabled && m_orbSlam3System) {
        // See resetSlamState()'s teardown doc comment -- same reasoning:
        // System has no live-reconfigure/pause API, so switching back to the
        // custom pipeline means throwing this run away entirely, not
        // pausing it. start() will lazily rebuild it if this is re-enabled.
        m_orbSlam3System->Shutdown();
        m_orbSlam3System.reset();
    }
}

bool SlamWorker::matchDescriptors(const cv::Mat &descA, const cv::Mat &descB,
                                   std::vector<cv::DMatch> &goodMatches, float ratio) const
{
    const float effectiveRatio = ratio > 0.0f ? ratio : feature_detector::defaultRatioFor(m_detectorType);
    return feature_detector::matchDescriptors(m_matcherNorm, descA, descB, goodMatches, effectiveRatio,
                                               m_mutualMatchEnabled);
}

cv::Mat SlamWorker::estimateEssentialRansac(const std::vector<cv::Point2f> &pts1,
                                             const std::vector<cv::Point2f> &pts2, cv::Mat &mask) const
{
    const int n = static_cast<int>(pts1.size());
    mask = cv::Mat::zeros(n, 1, CV_8U);
    if (n < 5)
        return cv::Mat();

    const double fx = m_intrinsics.fx, fy = m_intrinsics.fy, cx = m_intrinsics.cx, cy = m_intrinsics.cy;
    std::vector<cv::Point2f> calib1(n), calib2(n);
    for (int i = 0; i < n; ++i) {
        calib1[i] = cv::Point2f(static_cast<float>((pts1[i].x - cx) / fx), static_cast<float>((pts1[i].y - cy) / fy));
        calib2[i] = cv::Point2f(static_cast<float>((pts2[i].x - cx) / fx), static_cast<float>((pts2[i].y - cy) / fy));
    }

    const cv::Mat K = m_intrinsics.toMat();
    cv::Mat Kinv;
    cv::invert(K, Kinv);

    std::mt19937 rng(kRansacSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    cv::Mat bestE;
    std::vector<int> bestInliers;

    for (int iter = 0; iter < kERansacIterations; ++iter) {
        std::vector<int> sample;
        sample.reserve(5);
        while (sample.size() < 5) {
            const int idx = dist(rng);
            if (std::find(sample.begin(), sample.end(), idx) == sample.end())
                sample.push_back(idx);
        }

        std::vector<cv::Point2f> sample1(5), sample2(5);
        for (int i = 0; i < 5; ++i) {
            sample1[i] = calib1[sample[i]];
            sample2[i] = calib2[sample[i]];
        }

        for (const cv::Mat &Ecandidate : solveFivePoint(sample1, sample2)) {
            // Score in pixel space via the existing Sampson-distance
            // machinery/threshold (same one estimateFundamentalRansac()
            // uses): convert E back through K so the resulting inlier count
            // is directly comparable to nF/nH in estimateTwoViewPose()'s
            // model-selection ratio, without introducing a second,
            // uncalibrated threshold in calibrated-ray units.
            const cv::Mat Fequiv = Kinv.t() * Ecandidate * Kinv;
            const FCoeffs fc = extractF(Fequiv);

            std::vector<int> inliers;
            inliers.reserve(n);
            for (int i = 0; i < n; ++i) {
                if (sampsonDistance(fc, pts1[i], pts2[i]) < kFRansacSampsonThreshold)
                    inliers.push_back(i);
            }

            if (inliers.size() > bestInliers.size()) {
                bestInliers = std::move(inliers);
                bestE = Ecandidate;
            }
        }
    }

    if (bestInliers.size() < 5)
        return cv::Mat();

    // Refit + Gold Standard refinement over the full inlier set (calibrated
    // coordinates), same pattern estimateFundamentalRansac() used to have
    // for F: the minimal-5-point RANSAC loop above only ever returns
    // whichever single 5-correspondence sample's E had the most support --
    // unlike estimateFundamentalRansac()/EightPointLegacy, nothing
    // previously incorporated the *other* inliers into a better final
    // estimate. Both steps enforce the essential-matrix manifold
    // constraint (equal singular values), not just rank-2.
    std::vector<cv::Point2f> inCalib1, inCalib2;
    inCalib1.reserve(bestInliers.size());
    inCalib2.reserve(bestInliers.size());
    for (int idx : bestInliers) {
        inCalib1.push_back(calib1[idx]);
        inCalib2.push_back(calib2[idx]);
    }
    const cv::Mat Erefit = refitEssentialLinear(inCalib1, inCalib2);
    const cv::Mat Erefined = refineEssentialSampson(Erefit, inCalib1, inCalib2);

    for (int idx : bestInliers)
        mask.at<uchar>(idx) = 1;

    return Erefined;
}

cv::Mat SlamWorker::estimateHomographyRansac(const std::vector<cv::Point2f> &pts1,
                                              const std::vector<cv::Point2f> &pts2,
                                              cv::Mat &mask) const
{
    const int n = static_cast<int>(pts1.size());
    mask = cv::Mat::zeros(n, 1, CV_8U);
    if (n < 4)
        return cv::Mat();

    const cv::Mat T1 = computeNormalizationTransform(pts1);
    const cv::Mat T2 = computeNormalizationTransform(pts2);
    const std::vector<cv::Point2f> normPts1 = applyTransform(T1, pts1);
    const std::vector<cv::Point2f> normPts2 = applyTransform(T2, pts2);

    std::mt19937 rng(kRansacSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    std::vector<int> bestInliers;

    for (int iter = 0; iter < kHRansacIterations; ++iter) {
        std::vector<int> sample;
        sample.reserve(4);
        while (sample.size() < 4) {
            const int idx = dist(rng);
            if (std::find(sample.begin(), sample.end(), idx) == sample.end())
                sample.push_back(idx);
        }

        std::vector<cv::Point2f> sample1(4), sample2(4);
        for (int i = 0; i < 4; ++i) {
            sample1[i] = normPts1[sample[i]];
            sample2[i] = normPts2[sample[i]];
        }

        const cv::Mat Hn = solveHomographyDLT(sample1, sample2);
        const cv::Mat H = T2.inv() * Hn * T1;
        cv::Mat Hinv;
        if (!cv::invert(H, Hinv))
            continue;

        std::vector<int> inliers;
        inliers.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (homographyTransferErrorSq(H, Hinv, pts1[i], pts2[i]) < kHRansacThreshold)
                inliers.push_back(i);
        }

        if (inliers.size() > bestInliers.size())
            bestInliers = std::move(inliers);
    }

    if (bestInliers.size() < 4)
        return cv::Mat();

    // Refit H over the full inlier set (over-determined least squares).
    std::vector<cv::Point2f> inNorm1, inNorm2;
    inNorm1.reserve(bestInliers.size());
    inNorm2.reserve(bestInliers.size());
    for (int idx : bestInliers) {
        inNorm1.push_back(normPts1[idx]);
        inNorm2.push_back(normPts2[idx]);
    }
    const cv::Mat HnRefined = solveHomographyDLT(inNorm1, inNorm2);
    const cv::Mat Hrefined = T2.inv() * HnRefined * T1;

    for (int idx : bestInliers)
        mask.at<uchar>(idx) = 1;

    return Hrefined;
}

bool SlamWorker::solvePnPDltRansac(const std::vector<cv::Point3f> &objectPoints,
                                    const std::vector<cv::Point2f> &imagePoints, cv::Mat &R, cv::Mat &t,
                                    std::vector<int> &inlierIndices) const
{
    constexpr int kDltMinSample = 6;
    const int n = static_cast<int>(objectPoints.size());
    if (n < kDltMinSample)
        return false;

    // Calibrate image points into rays (K^-1 applied) once up front, then
    // similarity-normalize both point sets (each computed once from the
    // full correspondence set, same pattern as estimateFundamentalRansac/
    // estimateHomographyRansac) so every RANSAC sample just indexes into
    // already-normalized arrays instead of renormalizing per iteration.
    std::vector<cv::Point2f> calibPts(n);
    for (int i = 0; i < n; ++i) {
        calibPts[i] = cv::Point2f(static_cast<float>((imagePoints[i].x - m_intrinsics.cx) / m_intrinsics.fx),
                                   static_cast<float>((imagePoints[i].y - m_intrinsics.cy) / m_intrinsics.fy));
    }
    const cv::Mat T2 = computeNormalizationTransform(calibPts);
    const cv::Mat T3 = computeNormalizationTransform3D(objectPoints);
    const std::vector<cv::Point2f> normImg = applyTransform(T2, calibPts);
    const std::vector<cv::Point3f> normObj = applyTransform3D(T3, objectPoints);
    const cv::Mat T2inv = T2.inv();

    const double fx = m_intrinsics.fx, fy = m_intrinsics.fy, cx = m_intrinsics.cx, cy = m_intrinsics.cy;
    const double threshSq = m_pnpSettings.reprojectionError * m_pnpSettings.reprojectionError;

    auto scorePose = [&](const cv::Mat &Rc, const cv::Mat &tc) {
        const double r00 = Rc.at<double>(0, 0), r01 = Rc.at<double>(0, 1), r02 = Rc.at<double>(0, 2);
        const double r10 = Rc.at<double>(1, 0), r11 = Rc.at<double>(1, 1), r12 = Rc.at<double>(1, 2);
        const double r20 = Rc.at<double>(2, 0), r21 = Rc.at<double>(2, 1), r22 = Rc.at<double>(2, 2);
        const double tx = tc.at<double>(0), ty = tc.at<double>(1), tz = tc.at<double>(2);

        std::vector<int> inliers;
        inliers.reserve(n);
        for (int i = 0; i < n; ++i) {
            const double X = objectPoints[i].x, Y = objectPoints[i].y, Z = objectPoints[i].z;
            const double zc = r20 * X + r21 * Y + r22 * Z + tz;
            if (zc <= 0.0)
                continue;
            const double xc = r00 * X + r01 * Y + r02 * Z + tx;
            const double yc = r10 * X + r11 * Y + r12 * Z + ty;
            const double u = fx * xc / zc + cx;
            const double v = fy * yc / zc + cy;
            const double du = u - imagePoints[i].x, dv = v - imagePoints[i].y;
            if (du * du + dv * dv < threshSq)
                inliers.push_back(i);
        }
        return inliers;
    };

    std::mt19937 rng(kRansacSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    std::vector<int> bestInliers;
    for (int iter = 0; iter < m_pnpSettings.iterationsCount; ++iter) {
        std::vector<int> sample;
        sample.reserve(kDltMinSample);
        while (static_cast<int>(sample.size()) < kDltMinSample) {
            const int idx = dist(rng);
            if (std::find(sample.begin(), sample.end(), idx) == sample.end())
                sample.push_back(idx);
        }

        std::vector<cv::Point3f> sampleObj(kDltMinSample);
        std::vector<cv::Point2f> sampleImg(kDltMinSample);
        for (int i = 0; i < kDltMinSample; ++i) {
            sampleObj[i] = normObj[sample[i]];
            sampleImg[i] = normImg[sample[i]];
        }

        const cv::Mat Pn = solvePoseDLT(sampleObj, sampleImg);
        const cv::Mat Pcalib = T2inv * Pn * T3;

        cv::Mat Rc, tc;
        if (!decomposeDltPose(Pcalib, Rc, tc))
            continue;

        std::vector<int> inliers = scorePose(Rc, tc);
        if (inliers.size() > bestInliers.size())
            bestInliers = std::move(inliers);
    }

    if (bestInliers.size() < kDltMinSample)
        return false;

    // Refit over the full inlier set (over-determined least squares).
    std::vector<cv::Point3f> inObj;
    std::vector<cv::Point2f> inImgNorm;
    inObj.reserve(bestInliers.size());
    inImgNorm.reserve(bestInliers.size());
    for (int idx : bestInliers) {
        inObj.push_back(normObj[idx]);
        inImgNorm.push_back(normImg[idx]);
    }
    const cv::Mat PnRefined = solvePoseDLT(inObj, inImgNorm);
    const cv::Mat PcalibRefined = T2inv * PnRefined * T3;

    cv::Mat Rrefined, trefined;
    if (!decomposeDltPose(PcalibRefined, Rrefined, trefined))
        return false;

    R = Rrefined;
    t = trefined;
    inlierIndices = std::move(bestInliers);
    return true;
}

bool SlamWorker::estimateTwoViewPose(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2,
                                      cv::Mat &R, cv::Mat &t, cv::Mat &mask) const
{
    const cv::Mat K = m_intrinsics.toMat();

    // Direct calibrated 5-point solve for E (Nister-style) by default, not
    // the old 8-point-F-then-E=K^T-F-K conversion: since K is known/exact
    // for KITTI, E has a strictly stronger constraint than F (equal
    // nonzero singular values, not just rank-2), which 8-point-then-convert
    // never enforces. See DEBUGGING.md's "Full F/E option menu" item 1.
    // m_twoViewEstimator can select the legacy 8-point+Gold-Standard path
    // instead (EightPointLegacy.h) for benchmarking comparisons -- not
    // wired to any UI control. Inlier count/mask always scored via the
    // same pixel-space Sampson distance/threshold either way, so nE is
    // directly comparable to nH below regardless of which estimator ran.
    cv::Mat maskE;
    cv::Mat E;
    if (m_twoViewEstimator == TwoViewEstimator::EightPointLegacy) {
        cv::Mat maskF;
        const cv::Mat F = eight_point_legacy::estimateFundamentalRansac(pts1, pts2, maskF);
        if (!F.empty())
            E = K.t() * F * K;
        maskE = maskF;
    } else {
        E = estimateEssentialRansac(pts1, pts2, maskE);
    }
    const int nE = E.empty() ? 0 : cv::countNonZero(maskE);

    cv::Mat maskH;
    const cv::Mat H = estimateHomographyRansac(pts1, pts2, maskH);
    const int nH = H.empty() ? 0 : cv::countNonZero(maskH);

    if (nE == 0 && nH == 0)
        return false;

    // ORB-SLAM's model-selection heuristic: prefer the homography once it
    // explains a large enough share of the combined inlier support. F/E is
    // ill-conditioned for a (near-)planar scene or (near-)pure rotation
    // (e.g. mid-turn at an intersection) -- exactly the case H handles well.
    const double ratio = static_cast<double>(nH) / static_cast<double>(nH + nE + 1);
    const bool preferHomography = !H.empty() && (ratio > kHomographyPreferenceRatio || E.empty());
    std::fprintf(stderr, "[model] npts=%d nE=%d nH=%d ratio=%.3f -> %s\n", static_cast<int>(pts1.size()), nE, nH,
                 ratio, preferHomography ? "H" : "E");

    if (!preferHomography) {
        if (E.empty() || nE < kMinInitInliers) {
            std::fprintf(stderr, "[model] E rejected: empty=%d nE=%d < min=%d\n", E.empty(), nE, kMinInitInliers);
            return false;
        }
        const int inliers = cv::recoverPose(E, pts1, pts2, K, R, t, maskE);
        if (inliers < kMinInitInliers) {
            std::fprintf(stderr, "[model] E recoverPose rejected: inliers=%d < min=%d\n", inliers, kMinInitInliers);
            return false;
        }
        mask = maskE;
        return true;
    }

    // Prefer IMU-derived rotation over decomposing a homography when OXTS
    // orientation data is loaded: ratio > kHomographyPreferenceRatio is
    // exactly the near-pure-rotation regime where F/E's own rotation
    // estimate is least trustworthy, so real measured orientation
    // sidesteps the ill-conditioning entirely rather than working around
    // it via a homography decomposition. Cross-validated against the
    // actual ground-truth poses before ever being wired in here (mean
    // 0.044deg / max 0.136deg disagreement over 18 real frame-pair test
    // cases spanning the whole sequence -- see DEBUGGING.md). Translation
    // *direction* still comes from whichever F/E estimate is active
    // (approximate, not independently validated the same way -- solving
    // translation given a known rotation is flagged in DEBUGGING.md as the
    // properly-scoped next step, not attempted here) and its *magnitude*
    // still gets the same OXTS-distance rescale every other step already
    // gets (see initializeFromFrame()/recoverViaEpipolar()). Falls through
    // to the homography path unchanged if IMU data was never loaded.
    if (m_imuEnabled && !m_oxtsNavFromBody.empty() && !m_imuToCameraCalib.R.empty() && !E.empty()) {
        const cv::Mat imuR =
            imu_rotation::relativeCameraRotation(m_oxtsNavFromBody, m_imuToCameraCalib, m_refFrameIndex, m_frameCount);
        if (!imuR.empty()) {
            cv::Mat Rfe, tfe;
            const int inliers = cv::recoverPose(E, pts1, pts2, K, Rfe, tfe, maskE);
            if (inliers >= kMinInitInliers) {
                R = imuR;
                t = tfe;
                mask = maskE;
                std::fprintf(stderr, "[model] using IMU rotation (near-rotation case), inliers=%d\n", inliers);
                return true;
            }
        }
    }

    // Homography branch: decompose into up to 4 (R, t, plane-normal)
    // candidates and pick the one with the most triangulated points in
    // front of both cameras (the same cheirality criterion cv::recoverPose
    // uses internally for E, applied manually here since
    // decomposeHomographyMat doesn't do this disambiguation itself).
    std::vector<cv::Mat> Rs, Ts, Ns;
    cv::decomposeHomographyMat(H, K, Rs, Ts, Ns);
    std::fprintf(stderr, "[model] decomposeHomographyMat -> %d candidates\n", static_cast<int>(Rs.size()));
    if (Rs.empty())
        return false;

    int bestIdx = -1;
    int bestSupport = -1;
    for (size_t i = 0; i < Rs.size(); ++i) {
        cv::Mat tCandidate = Ts[i].clone();
        const double tNorm = cv::norm(tCandidate);
        if (tNorm > 1e-9)
            tCandidate /= tNorm; // decomposeHomographyMat's t is scaled by an arbitrary plane
                                 // distance, not unit length like recoverPose's -- normalize so
                                 // both branches hand back the same "unit baseline" convention

        std::vector<uchar> valid;
        triangulate(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F), Rs[i], tCandidate, pts1, pts2, valid);
        const int support = static_cast<int>(std::count(valid.begin(), valid.end(), static_cast<uchar>(1)));
        std::fprintf(stderr, "[model]   candidate %d: support=%d\n", static_cast<int>(i), support);
        if (support > bestSupport) {
            bestSupport = support;
            bestIdx = static_cast<int>(i);
        }
    }
    if (bestIdx < 0 || bestSupport < kMinInitInliers) {
        std::fprintf(stderr, "[model] H rejected: bestIdx=%d bestSupport=%d < min=%d\n", bestIdx, bestSupport,
                     kMinInitInliers);
        return false;
    }

    R = Rs[bestIdx];
    t = Ts[bestIdx].clone();
    const double tNorm = cv::norm(t);
    if (tNorm > 1e-9)
        t /= tNorm;
    mask = maskH;
    std::fprintf(stderr, "[model] H accepted: bestIdx=%d bestSupport=%d\n", bestIdx, bestSupport);
    return true;
}

std::vector<cv::Point3f> SlamWorker::triangulate(const cv::Mat &R1, const cv::Mat &t1,
                                                   const cv::Mat &R2, const cv::Mat &t2,
                                                   const std::vector<cv::Point2f> &pts1,
                                                   const std::vector<cv::Point2f> &pts2,
                                                   std::vector<uchar> &validMask) const
{
    const cv::Mat K = m_intrinsics.toMat();
    cv::Mat Rt1, Rt2;
    cv::hconcat(R1, t1, Rt1);
    cv::hconcat(R2, t2, Rt2);
    const cv::Mat P1 = K * Rt1;
    const cv::Mat P2 = K * Rt2;

    cv::Mat pts4d;
    cv::triangulatePoints(P1, P2, pts1, pts2, pts4d);

    std::vector<cv::Point3f> out(pts1.size());
    validMask.assign(pts1.size(), 0);

    for (size_t i = 0; i < pts1.size(); ++i) {
        const cv::Mat x = pts4d.col(static_cast<int>(i));
        const float w = x.at<float>(3);
        if (std::abs(w) < 1e-6f)
            continue;

        const cv::Mat Xw = (cv::Mat_<double>(4, 1) << x.at<float>(0) / w,
                             x.at<float>(1) / w, x.at<float>(2) / w, 1.0);
        const cv::Mat Xc1 = Rt1 * Xw;
        const cv::Mat Xc2 = Rt2 * Xw;
        const double z1 = Xc1.at<double>(2);
        const double z2 = Xc2.at<double>(2);
        if (z1 <= 0.0 || z2 <= 0.0 || z1 > kMaxTriangulationDepth || z2 > kMaxTriangulationDepth)
            continue;

        out[i] = cv::Point3f(static_cast<float>(Xw.at<double>(0)),
                              static_cast<float>(Xw.at<double>(1)),
                              static_cast<float>(Xw.at<double>(2)));
        validMask[i] = 1;
    }
    return out;
}

cv::Point3f SlamWorker::triangulateMultiView(const std::vector<std::pair<int, cv::Point2f>> &observations,
                                              bool &valid) const
{
    valid = false;
    if (observations.size() < 2)
        return {};

    const cv::Mat K = m_intrinsics.toMat();
    cv::Mat A(2 * static_cast<int>(observations.size()), 4, CV_64F, cv::Scalar(0.0));
    int row = 0;
    for (const auto &ob : observations) {
        const Keyframe &kf = m_keyframeHistory[static_cast<size_t>(ob.first)];
        cv::Mat Rt;
        cv::hconcat(kf.R, kf.t, Rt);
        const cv::Mat P = K * Rt; // 3x4
        const double u = ob.second.x, v = ob.second.y;
        for (int c = 0; c < 4; ++c) {
            A.at<double>(row, c) = u * P.at<double>(2, c) - P.at<double>(0, c);
            A.at<double>(row + 1, c) = v * P.at<double>(2, c) - P.at<double>(1, c);
        }
        row += 2;
    }

    cv::Mat w, u_, vt;
    cv::SVD::compute(A, w, u_, vt, cv::SVD::FULL_UV);
    const cv::Mat X = vt.row(vt.rows - 1).t(); // smallest-singular-value solution
    const double homog = X.at<double>(3);
    if (std::abs(homog) < 1e-9)
        return {}; // point at infinity -- degenerate view configuration

    const cv::Point3f p(static_cast<float>(X.at<double>(0) / homog),
                         static_cast<float>(X.at<double>(1) / homog),
                         static_cast<float>(X.at<double>(2) / homog));

    // Acceptance gate: this is the actual merge decision (see this
    // function's own doc comment in SlamWorker.h), not just a diagnostic --
    // every observing keyframe must see this candidate position at a
    // plausible depth AND within Phase B's own tight reprojection bar,
    // otherwise the two landmarks being merged are geometrically
    // inconsistent (most likely genuinely different physical points that
    // happened to share a similar-enough descriptor) and the merge itself
    // must be rejected, not silently accepted with a worse position.
    double sumSqErr = 0.0;
    int validViews = 0;
    for (const auto &ob : observations) {
        const Keyframe &kf = m_keyframeHistory[static_cast<size_t>(ob.first)];
        const cv::Mat Xw = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
        const cv::Mat camPt = kf.R * Xw + kf.t;
        const double z = camPt.at<double>(2);
        if (z < 1e-3 || z > kMaxTriangulationDepth)
            continue;
        const double u = m_intrinsics.fx * camPt.at<double>(0) / z + m_intrinsics.cx;
        const double v = m_intrinsics.fy * camPt.at<double>(1) / z + m_intrinsics.cy;
        const double du = u - ob.second.x, dv = v - ob.second.y;
        sumSqErr += du * du + dv * dv;
        ++validViews;
    }
    if (validViews < 2)
        return {}; // fewer than 2 views agree on positive depth -- not a real triangulation
    const double meanReprojErr = std::sqrt(sumSqErr / validViews);
    if (meanReprojErr > kFuseMergeMaxReprojErrorPixels)
        return {};

    valid = true;
    return p;
}

bool SlamWorker::initializeFromFrame(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors)
{
    if (m_refDescriptors.empty()) {
        // First frame becomes the (initial) reference keyframe candidate.
        // Not necessarily the world origin yet: the reference may still
        // slide forward several times below before initialization actually
        // succeeds, so the origin trajectory point is recorded then instead.
        setReferenceFrame(kps, descriptors, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
        return false;
    }

    std::vector<cv::DMatch> matches;
    if (!matchDescriptors(m_refDescriptors, descriptors, matches) ||
        static_cast<int>(matches.size()) < kMinInitMatches) {
        // The reference has drifted too far out of view to match reliably
        // (typical a few dozen frames into a moving sequence). Slide the
        // reference forward to the current frame instead of waiting forever
        // on one that can never regain enough overlap again — otherwise
        // initialization (and therefore PnP tracking, which only starts
        // once initialization succeeds) never happens at all.
        setReferenceFrame(kps, descriptors, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
        return false;
    }

    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto &m : matches) {
        pts1.push_back(m_refKeypoints[m.queryIdx].pt);
        pts2.push_back(kps[m.trainIdx].pt);
    }

    // Fundamental-vs-homography model selection (see estimateTwoViewPose()):
    // F/E for general 3D scenes, H for the (near-)planar/(near-)pure-rotation
    // case where F/E would be ill-conditioned (e.g. mid-turn at a corner).
    cv::Mat R, t, mask;
    if (!estimateTwoViewPose(pts1, pts2, R, t, mask))
        return false;

    // Correct the bootstrap's otherwise-arbitrary unit-baseline scale to
    // real metric distance, if OXTS speed data was loaded (loadOxtsSpeeds()).
    // Since the reference camera is at identity here, the baseline distance
    // is exactly norm(t); rescaling t before triangulate() means every
    // initial map point comes out correctly scaled too, with no separate
    // map-rescale step needed. Everything downstream (trackFrame's PnP
    // against this map) inherits this same metric scale automatically.
    const double oxtsDist = oxtsDistanceBetween(m_refFrameIndex, m_frameCount);
    bool scaleCorrected = false;
    if (oxtsDist > 0.0) {
        const double unitDist = cv::norm(t);
        if (unitDist > 1e-9) {
            t = t * (oxtsDist / unitDist);
            scaleCorrected = true;
        }
    }

    std::vector<cv::Point2f> inPts1, inPts2;
    std::vector<int> trainIdx;
    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i)) {
            inPts1.push_back(pts1[i]);
            inPts2.push_back(pts2[i]);
            trainIdx.push_back(matches[i].trainIdx);
        }
    }
    if (inPts1.size() < static_cast<size_t>(kMinInitMapPoints))
        return false;

    std::vector<uchar> valid;
    std::vector<cv::Point3f> triangulated =
        triangulate(m_refR, m_refT, R, t, inPts1, inPts2, valid);

    // Vision-only fallback when OXTS wasn't available: VISO2-M-style
    // ground-plane scale correction (see GroundPlaneScale.h). Needs the
    // points to already exist (unlike OXTS, which rescales t before
    // triangulating), so apply it here instead, uniformly rescaling both t
    // and every just-triangulated point by the same factor -- a similarity
    // transform, so nothing about the reconstruction's shape/directions
    // changes, only its overall scale. The reference camera is at identity
    // here, so the triangulated points are already expressed in its own
    // (camera) frame, exactly what estimateScaleCorrection() expects.
    if (!scaleCorrected && m_groundPlaneEnabled) {
        std::vector<cv::Point3f> validPoints;
        for (size_t i = 0; i < triangulated.size(); ++i) {
            if (valid[i])
                validPoints.push_back(triangulated[i]);
        }
        const double gpFactor = ground_plane_scale::estimateScaleCorrection(validPoints, m_groundPlaneConfig);
        if (gpFactor > 0.0) {
            t = t * gpFactor;
            for (cv::Point3f &p : triangulated) {
                p.x = static_cast<float>(p.x * gpFactor);
                p.y = static_cast<float>(p.y * gpFactor);
                p.z = static_cast<float>(p.z * gpFactor);
            }
        }
    }

    std::vector<cv::Point3f> newMapPoints;
    cv::Mat newMapDescriptors;
    for (size_t i = 0; i < triangulated.size(); ++i) {
        if (!valid[i])
            continue;
        newMapPoints.push_back(triangulated[i]);
        newMapDescriptors.push_back(descriptors.row(trainIdx[i]));
    }
    if (newMapPoints.size() < static_cast<size_t>(kMinInitMapPoints))
        return false;

    m_mapPoints = std::move(newMapPoints);
    m_mapDescriptors = newMapDescriptors;

    // Bypasses appendToMap() (this is the very first assignment, not an
    // append), so it must assign m_mapPointIds itself to keep it parallel
    // to m_mapPoints -- see recordLandmarkObservations(), which indexes
    // into it for every subsequent keyframe. These bootstrap points have no
    // owning Keyframe entry (this function doesn't push one to
    // m_keyframeHistory), so -- like recoverViaEpipolar()'s points -- they
    // can never become bundle-adjustment landmarks; they still need valid
    // IDs so the array stays in sync.
    m_mapPointIds.clear();
    m_mapPointIds.reserve(m_mapPoints.size());
    for (size_t i = 0; i < m_mapPoints.size(); ++i)
        m_mapPointIds.push_back(m_nextLandmarkId++);

    m_currR = R.clone();
    m_currT = t.clone();

    // The reference frame used above is now the fixed world origin; record
    // both it and the second camera as trajectory points (also seeds
    // m_avgStepScale from this bootstrap step's -- arbitrary but
    // thereafter consistent -- baseline distance).
    pushTrajectoryPoint(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
    pushTrajectoryPoint(R, t);

    setReferenceFrame(kps, descriptors, R, t);
    m_framesSinceKeyframe = 0;
    return true;
}

bool SlamWorker::optimizePoseOnly(const std::vector<cv::Point3f> &objectPoints,
                                   const std::vector<cv::Point2f> &imagePoints,
                                   const std::vector<float> &imageScales, cv::Mat &R, cv::Mat &tvec,
                                   int &inlierCountOut, const std::array<double, 6> *priorPose,
                                   double priorRotWeight, double priorTransWeight) const
{
    // Real ORB-SLAM3-style pose-only BA (Optimizer::PoseOptimization): the
    // map is held fixed; only this frame's 6-DOF pose is optimized against
    // ALL its matched map points, with iterative outlier re-rejection
    // (ORB-SLAM3 does 4 passes). Initialized from the SQPnP RANSAC solve
    // (R/tvec on entry) -- SQPnP stays as the robust minimal-sample
    // bootstrap that keeps coverage from collapsing (per the user's
    // explicit requirement), this just refines its result using every
    // consistent observation instead of RANSAC's minimal sample. See
    // trackFrame()'s call site (gated by m_poseOnlyBaEnabled) and
    // DEBUGGING.md item 36.
    const size_t n = objectPoints.size();
    if (n < 4)
        return false;

    std::array<double, 6> pose;
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    for (int k = 0; k < 3; ++k)
        pose[static_cast<size_t>(k)] = rvec.at<double>(k);
    for (int k = 0; k < 3; ++k)
        pose[static_cast<size_t>(3 + k)] = tvec.at<double>(k);

    // Outlier gate in pixels -- reuse trackFrame()'s own PnP RANSAC
    // reprojection-error threshold so "inlier" means the same thing before
    // and after this refinement.
    const double thresh = m_pnpSettings.reprojectionError;
    const double thresh2 = thresh * thresh;
    std::vector<uchar> isInlier(n, 1);

    constexpr int kPoseOnlyPasses = 4; // ORB-SLAM3's own PoseOptimization pass count
    for (int pass = 0; pass < kPoseOnlyPasses; ++pass) {
        ceres::Problem problem;
        problem.AddParameterBlock(pose.data(), 6);
        // Optional leash toward a reference pose (item 41, #2): a soft
        // PosePriorCost anchoring the refined pose near the robust SQPnP
        // solution, so pose-only BA can refine but cannot drift/collapse
        // far from it -- the per-frame analogue of soft-prior local BA's
        // live-pose leash.
        if (priorPose)
            problem.AddResidualBlock(PosePriorCost::Create(*priorPose, priorRotWeight, priorTransWeight),
                                      nullptr, pose.data());
        int used = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!isInlier[i])
                continue;
            const cv::Point3f &p = objectPoints[i];
            ceres::CostFunction *cost = PoseOnlyReprojectionCost::Create(
                p.x, p.y, p.z, imagePoints[i].x, imagePoints[i].y, m_intrinsics.fx, m_intrinsics.fy,
                m_intrinsics.cx, m_intrinsics.cy);
            ceres::LossFunction *loss = new ceres::HuberLoss(kBaHuberDeltaPixels);
            if (m_octaveWeightingEnabled && i < imageScales.size() && imageScales[i] > 1e-3f) {
                // Octave/scale information weighting (item 38), the SIFT-
                // appropriate analogue of ORB-SLAM3's invSigma2 per pyramid
                // level: a SIFT keypoint's SIZE (diameter) is proportional
                // to the scale it was detected at, and its localization
                // uncertainty in the original image grows with that scale --
                // so information = 1/sigma^2 ~ (kOctaveWeightRefSize/size)^2.
                // Fine keypoints (small size) keep ~unit weight; coarse ones
                // are down-weighted. Uses the keypoint's REAL SIFT size, not
                // ORB's discrete 1.2^level formula (which doesn't fit SIFT's
                // sub-pixel/sub-scale-interpolated continuous scale space).
                // ScaledLoss(Huber, invSigma2) mirrors ORB-SLAM3's
                // information-weighted robust edge.
                const double ratio = kOctaveWeightRefSize / static_cast<double>(imageScales[i]);
                const double invSigma2 = std::min(1.0, ratio * ratio); // never UP-weight past unit
                loss = new ceres::ScaledLoss(loss, invSigma2, ceres::TAKE_OWNERSHIP);
            }
            problem.AddResidualBlock(cost, loss, pose.data());
            ++used;
        }
        if (used < 4)
            return false;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR; // tiny problem (one 6-DOF block)
        options.max_num_iterations = kBaMaxIterations;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1; // pinned for run-to-run reproducibility
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (!summary.IsSolutionUsable())
            return false;

        // Re-classify inliers against the refined pose for the next pass
        // (ORB-SLAM3 re-includes points that come back within threshold too,
        // not just permanently discarding -- so re-evaluate ALL points).
        const cv::Mat rv = (cv::Mat_<double>(3, 1) << pose[0], pose[1], pose[2]);
        cv::Mat Rp;
        cv::Rodrigues(rv, Rp);
        int inliers = 0;
        for (size_t i = 0; i < n; ++i) {
            const cv::Point3f &p = objectPoints[i];
            const double cx = Rp.at<double>(0, 0) * p.x + Rp.at<double>(0, 1) * p.y +
                              Rp.at<double>(0, 2) * p.z + pose[3];
            const double cy = Rp.at<double>(1, 0) * p.x + Rp.at<double>(1, 1) * p.y +
                              Rp.at<double>(1, 2) * p.z + pose[4];
            const double cz = Rp.at<double>(2, 0) * p.x + Rp.at<double>(2, 1) * p.y +
                              Rp.at<double>(2, 2) * p.z + pose[5];
            if (cz < 1e-3) {
                isInlier[i] = 0;
                continue;
            }
            const double u = m_intrinsics.fx * cx / cz + m_intrinsics.cx;
            const double v = m_intrinsics.fy * cy / cz + m_intrinsics.cy;
            const double du = u - imagePoints[i].x, dv = v - imagePoints[i].y;
            const bool in = (du * du + dv * dv) <= thresh2;
            isInlier[i] = in ? 1 : 0;
            inliers += in ? 1 : 0;
        }
        inlierCountOut = inliers;
    }

    const cv::Mat rvOut = (cv::Mat_<double>(3, 1) << pose[0], pose[1], pose[2]);
    cv::Rodrigues(rvOut, R);
    tvec = (cv::Mat_<double>(3, 1) << pose[3], pose[4], pose[5]);
    return true;
}

bool SlamWorker::trackFrame(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors)
{
    // Covisibility-driven local map (see setCovisibilityLocalMapEnabled())
    // replaces the flat rolling map when enabled -- falls back to the flat
    // map if the local one hasn't been built yet, OR has gone stale (see
    // m_framesSinceCovisibilityMapRebuild's own doc comment: insertKeyframe()
    // can silently stop rebuilding it during a degraded-tracking patch,
    // which is exactly when tracking against a stale local map hurts most).
    ++m_framesSinceCovisibilityMapRebuild;
    if (m_poseOnlyLoopSuppressFrames > 0)
        --m_poseOnlyLoopSuppressFrames; // see kPoseOnlyLoopSuppressFrames (item 39)
    const bool useLocalMap = m_covisibilityLocalMapEnabled && !m_localMapDescriptors.empty() &&
                              m_framesSinceCovisibilityMapRebuild <= kCovisibilityMapStaleFrames;
    const cv::Mat &trackMapDescriptors = useLocalMap ? m_localMapDescriptors : m_mapDescriptors;
    const std::vector<cv::Point3f> &trackMapPoints = useLocalMap ? m_localMapPoints : m_mapPoints;

    std::vector<cv::DMatch> matches;
    if (!matchDescriptors(trackMapDescriptors, descriptors, matches)) {
        std::fprintf(stderr, "[track] match-count fail: matches=%d min=%d\n",
                     static_cast<int>(matches.size()), kMinTrackMatches);
        return false;
    }

    // Guided/projection-based filter (see setGuidedSearchEnabled()): reject
    // any match whose keypoint lands far from where a constant-velocity
    // motion prediction says the map point should project to -- m_currR/
    // m_currT still hold the PREVIOUS frame's pose here (this frame's own
    // hasn't been computed yet), extrapolated one more step forward by
    // m_velocityR/m_velocityT (the last frame-to-frame motion; identity/
    // zero -- i.e. falls back to constant-position -- until a second step
    // has been tracked).
    if (m_guidedSearchEnabled && !m_currR.empty()) {
        const cv::Mat predR = m_velocityR * m_currR;
        const cv::Mat predT = m_velocityR * m_currT + m_velocityT;
        std::vector<cv::DMatch> filtered;
        filtered.reserve(matches.size());
        for (const auto &m : matches) {
            const cv::Point3f &p = trackMapPoints[m.queryIdx];
            const cv::Mat P = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
            const cv::Mat camPt = predR * P + predT;
            const double z = camPt.at<double>(2);
            if (z < 1e-3)
                continue; // behind the camera under the prediction -- can't be this match
            const double u = m_intrinsics.fx * camPt.at<double>(0) / z + m_intrinsics.cx;
            const double v = m_intrinsics.fy * camPt.at<double>(1) / z + m_intrinsics.cy;
            const double du = u - kps[m.trainIdx].pt.x, dv = v - kps[m.trainIdx].pt.y;
            if (std::sqrt(du * du + dv * dv) <= kGuidedSearchRadiusPixels)
                filtered.push_back(m);
        }
        matches = std::move(filtered);
    }

    if (static_cast<int>(matches.size()) < kMinTrackMatches) {
        std::fprintf(stderr, "[track] match-count fail: matches=%d min=%d\n",
                     static_cast<int>(matches.size()), kMinTrackMatches);
        return false;
    }

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    std::vector<float> imageScales; // parallel: the SIFT keypoint SIZE (diameter) of each match --
                                     // used for octave/scale information weighting in pose-only BA
                                     // (see optimizePoseOnly()/setOctaveWeightingEnabled(), item 38)
    objectPoints.reserve(matches.size());
    imagePoints.reserve(matches.size());
    imageScales.reserve(matches.size());
    for (const auto &m : matches) {
        objectPoints.push_back(trackMapPoints[m.queryIdx]);
        imagePoints.push_back(kps[m.trainIdx].pt);
        imageScales.push_back(kps[m.trainIdx].size);
    }

    cv::Mat R, tvec;
    int inlierCount = 0;
    bool ok = false;
    bool trackedByMotionModel = false;

    // Motion-model-PRIMARY tracking (real ORB-SLAM3 TrackWithMotionModel,
    // see setPoseOnlyBaEnabled()/DEBUGGING.md item 36): when enabled and a
    // previous pose+velocity exist, predict this frame's pose from constant
    // velocity (m_velocityR/m_velocityT) and refine it DIRECTLY with
    // pose-only BA over the guided matches -- no RANSAC. This is the
    // accurate, low-per-frame-drift path ORB-SLAM3 uses on the vast
    // majority of frames. SQPnP RANSAC below runs ONLY if this loses track
    // (too few inliers) -- the recovery path, exactly ORB-SLAM3's
    // fall-through from the motion model to reference-KF/relocalization.
    // Keeping SQPnP as the fallback is what guarantees coverage can't
    // collapse (the user's explicit requirement) on hard frames (sharp
    // turns, where the constant-velocity prediction is worst).
    // Motion-model-primary runs unless leash mode (item 41 #2) is on, which
    // instead does SQPnP-primary + leashed refine below.
    if (m_poseOnlyBaEnabled && !m_poseOnlyLeashEnabled && m_poseOnlyLoopSuppressFrames == 0 &&
        !m_currR.empty()) {
        cv::Mat mmR = m_velocityR * m_currR;
        cv::Mat mmT = m_velocityR * m_currT + m_velocityT;
        int mmInliers = 0;
        if (optimizePoseOnly(objectPoints, imagePoints, imageScales, mmR, mmT, mmInliers) &&
            mmInliers >= m_minTrackInliers) {
            // Per-frame step-consistency gate (item 41 #1): the diagnosed
            // failure is a SCALE collapse -- the refined step shrinks to a
            // tiny fraction of the real motion. Reject the pose-only result
            // and fall through to SQPnP for THIS frame only when its camera-
            // center step is implausibly small vs the running average step
            // (m_avgStepScale). Surgical: keeps pose-only BA on everywhere
            // else, unlike the loop-timing suppression that degraded clean
            // regions. isPlausibleStep() already bounds steps from ABOVE;
            // this adds the missing lower bound, specific to pose-only BA.
            bool stepOk = true;
            if (m_poseOnlyStepGateEnabled && m_avgStepScale > 1e-6) {
                const cv::Mat prevC = -m_currR.t() * m_currT;
                const cv::Mat curC = -mmR.t() * mmT;
                const double step = cv::norm(curC - prevC);
                if (step < kPoseOnlyMinStepFraction * m_avgStepScale) {
                    stepOk = false;
                    std::fprintf(stderr,
                                  "[track] pose-only step-gate: step=%.3f < %.2f*avgStep(%.3f) -- SQPnP "
                                  "recovery this frame\n",
                                  step, kPoseOnlyMinStepFraction, m_avgStepScale);
                }
            }
            if (stepOk) {
                R = mmR;
                tvec = mmT;
                inlierCount = mmInliers;
                ok = true;
                trackedByMotionModel = true;
            }
        } else {
            std::fprintf(stderr, "[track] motion-model tracking lost (inliers=%d min=%d) -- SQPnP recovery\n",
                         mmInliers, m_minTrackInliers);
        }
    }

    if (!ok && m_pnpSettings.method == kPnpMethodDlt) {
        std::vector<int> inlierIndices;
        ok = solvePnPDltRansac(objectPoints, imagePoints, R, tvec, inlierIndices);
        inlierCount = static_cast<int>(inlierIndices.size());
    } else if (!ok) {
        const cv::Mat K = m_intrinsics.toMat();
        cv::Mat rvec;
        std::vector<int> inliers;
        try {
            ok = cv::solvePnPRansac(objectPoints, imagePoints, K, cv::Mat(), rvec, tvec, false,
                                     m_pnpSettings.iterationsCount,
                                     static_cast<float>(m_pnpSettings.reprojectionError),
                                     m_pnpSettings.confidence, inliers, m_pnpSettings.method);
        } catch (const cv::Exception &) {
            ok = false;
        }
        inlierCount = static_cast<int>(inliers.size());
        if (ok && m_pnpFullInlierRefineEnabled) {
            // Refit over the FULL inlier set via nonlinear reprojection-
            // error minimization -- mirrors what solvePnPDltRansac() already
            // does (a linear refit over all inliers, not just its minimal
            // 6-point sample) and what refineLocalKeyframes() does for
            // keyframes, but was missing here. Without it, this pose is
            // whatever RANSAC's minimal-sample winner produced (as few as
            // 4 points for P3P), which is far more exposed to a systematic
            // bias in near-degenerate point geometry (near-planar/narrow-
            // baseline triangulated points, common for a forward-driving
            // dashcam) than an over-determined solve using every inlier --
            // a real, observed effect this session (P3P/Iterative showing a
            // consistent directional drift DLT's own full-inlier refit
            // doesn't). Falls back to the RANSAC-only estimate if
            // refinement throws. Toggle: setPnpFullInlierRefineEnabled().
            std::vector<cv::Point3f> inlierObj;
            std::vector<cv::Point2f> inlierImg;
            inlierObj.reserve(inliers.size());
            inlierImg.reserve(inliers.size());
            for (int idx : inliers) {
                inlierObj.push_back(objectPoints[static_cast<size_t>(idx)]);
                inlierImg.push_back(imagePoints[static_cast<size_t>(idx)]);
            }
            try {
                cv::solvePnPRefineLM(inlierObj, inlierImg, K, cv::Mat(), rvec, tvec);
            } catch (const cv::Exception &) {
                // keep the RANSAC-only rvec/tvec
            }
        }
        if (ok)
            cv::Rodrigues(rvec, R);
    }
    if (!ok || inlierCount < m_minTrackInliers) {
        std::fprintf(stderr, "[track] PnP fail: ok=%d inliers=%d min=%d\n", ok, inlierCount, m_minTrackInliers);
        return false;
    }

    // Pose-only BA refinement on the SQPnP RECOVERY path (see
    // optimizePoseOnly()/setPoseOnlyBaEnabled(), DEBUGGING.md item 36):
    // only when this frame fell back to SQPnP (motion model already ran
    // pose-only BA itself, so re-running it would be redundant). Refines
    // the SQPnP minimal-sample pose against ALL matched map points with the
    // map fixed + iterative outlier rejection. Falls through to the SQPnP
    // result untouched if it declines, so it can only help.
    if (m_poseOnlyBaEnabled && m_poseOnlyLoopSuppressFrames == 0 && !trackedByMotionModel && ok) {
        cv::Mat refinedR = R.clone(), refinedT = tvec.clone();
        int refinedInliers = inlierCount;
        // Leash mode (item 41 #2): anchor the refinement to the robust SQPnP
        // solution (R/tvec here) via a soft PosePriorCost -- pose-only BA
        // sharpens the pose but cannot collapse away from SQPnP. Non-leash
        // mode refines unanchored (item 36's original recovery-path behavior).
        std::array<double, 6> priorPose;
        const std::array<double, 6> *prior = nullptr;
        if (m_poseOnlyLeashEnabled) {
            cv::Mat rv;
            cv::Rodrigues(R, rv);
            priorPose = {rv.at<double>(0), rv.at<double>(1), rv.at<double>(2),
                         tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)};
            prior = &priorPose;
        }
        if (optimizePoseOnly(objectPoints, imagePoints, imageScales, refinedR, refinedT, refinedInliers, prior,
                              m_poseOnlyLeashRotWeight, m_poseOnlyLeashTransWeight) &&
            refinedInliers >= m_minTrackInliers) {
            R = refinedR;
            tvec = refinedT;
            inlierCount = refinedInliers;
        }
    }

    // A degenerate/bad RANSAC solve can still clear the minimum inlier
    // count while producing a wildly wrong pose. Reject implausibly large
    // jumps rather than accepting a solution that would corrupt the
    // trajectory and map -- treat it the same as a failed attempt.
    //
    // When enabled (setOxtsImuInPnpEnabled()) and real OXTS distance data
    // covers this exact frame-to-frame step, use it in place of the
    // m_avgStepScale heuristic bound entirely: recoverViaEpipolar() already
    // got this fix (Session 6) since m_avgStepScale can drift low relative
    // to genuinely fast real motion and then reject every correct step
    // forever; trackFrame() had the identical vulnerability, documented as
    // still open at the time. Unlike recoverViaEpipolar() (whose candidate
    // step distance *is* the OXTS measurement, so any check is circular),
    // trackFrame()'s PnP solve is an independent vision-only measurement --
    // so this compares against oxtsDist with a tolerance rather than
    // skipping the check outright. If IMU orientation data also covers
    // this step, additionally cross-check the PnP solve's own rotation
    // against the independently IMU-measured one. Falls through to the
    // exact original heuristic whenever OXTS doesn't cover this step (or
    // this is disabled) -- zero behavior change for anyone not using it.
    const double oxtsDist = m_oxtsImuInPnpEnabled ? oxtsDistanceBetween(m_frameCount - 1, m_frameCount) : -1.0;
    if (oxtsDist > 0.0) {
        const cv::Mat C = -R.t() * tvec;
        double stepDist = -1.0;
        if (!m_trajectory.isEmpty()) {
            const QPointF &prev = m_trajectory.last();
            stepDist = std::hypot(C.at<double>(0) - prev.x(), C.at<double>(2) - prev.y());
        }
        bool plausible = stepDist >= 0.0 && stepDist >= oxtsDist / kOxtsPnpStepToleranceMultiplier &&
                          stepDist <= oxtsDist * kOxtsPnpStepToleranceMultiplier;

        double imuAngleDeg = -1.0;
        if (plausible && m_imuEnabled && !m_oxtsNavFromBody.empty() && !m_imuToCameraCalib.R.empty()) {
            const cv::Mat imuRrel = imu_rotation::relativeCameraRotation(m_oxtsNavFromBody, m_imuToCameraCalib,
                                                                          m_frameCount - 1, m_frameCount);
            if (!imuRrel.empty()) {
                const cv::Mat pnpRrel = R * m_currR.t(); // m_currR still holds the *previous* frame's
                                                          // pose here -- overwritten just below
                cv::Mat diffRvec;
                cv::Rodrigues(pnpRrel * imuRrel.t(), diffRvec);
                imuAngleDeg = cv::norm(diffRvec) * 180.0 / CV_PI;
                if (imuAngleDeg > kImuPnpMaxAngleDeg)
                    plausible = false;
            }
        }

        if (!plausible) {
            std::fprintf(stderr,
                          "[track] OXTS/IMU-aware plausibility fail: stepDist=%.3f oxtsDist=%.3f "
                          "imuAngleDeg=%.3f\n",
                          stepDist, oxtsDist, imuAngleDeg);
            return false;
        }
    } else if (!isPlausibleStep(R, tvec)) {
        const cv::Mat C = -R.t() * tvec;
        double stepDist = -1.0;
        if (!m_trajectory.isEmpty()) {
            const QPointF &prev = m_trajectory.last();
            stepDist = std::hypot(C.at<double>(0) - prev.x(), C.at<double>(2) - prev.y());
        }
        std::fprintf(stderr, "[track] isPlausibleStep fail: stepDist=%.3f avgStepScale=%.3f bound=%.3f\n",
                     stepDist, m_avgStepScale, kMaxStepMultiplier * m_avgStepScale);
        return false;
    }

    // Update the constant-velocity estimate (see m_velocityR/m_velocityT's
    // doc comment) from the PREVIOUS m_currR/m_currT (this step's starting
    // pose) to the just-accepted (R, tvec) -- i.e. "the motion this step
    // just made", used to predict the NEXT step's guided-search block.
    if (!m_currR.empty()) {
        m_velocityR = R * m_currR.t();
        m_velocityT = tvec - m_velocityR * m_currT;
    }
    m_currR = R;
    m_currT = tvec;

    pushTrajectoryPoint(R, tvec);

    ++m_framesSinceKeyframe;
    bool shouldInsertKeyframe;
    if (m_qualityDrivenKeyframesEnabled) {
        const double inlierRatio =
            matches.empty() ? 0.0 : static_cast<double>(inlierCount) / static_cast<double>(matches.size());
        shouldInsertKeyframe =
            m_framesSinceKeyframe >= kKeyframeMaxInterval ||
            (m_framesSinceKeyframe >= kKeyframeMinInterval && inlierRatio < kKeyframeQualityRatioThreshold);
    } else {
        shouldInsertKeyframe = m_framesSinceKeyframe >= kKeyframeEveryNFrames;
    }
    if (shouldInsertKeyframe) {
        insertKeyframe(kps, descriptors, R, tvec);
        m_framesSinceKeyframe = 0;
    }
    return true;
}

void SlamWorker::insertKeyframe(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors,
                                 const cv::Mat &R, const cv::Mat &t)
{
    tryIntegratePendingGlobalBa(); // see setGlobalBundleAdjustmentAsyncEnabled() -- checked on every
                                    // keyframe-insertion attempt, since kGlobalBaIntegrationDelayKeyframes
                                    // is counted in keyframe-count units

    std::vector<cv::DMatch> matches;
    if (!matchDescriptors(m_refDescriptors, descriptors, matches) ||
        matches.size() < static_cast<size_t>(kMinInitMatches))
        return;

    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto &m : matches) {
        pts1.push_back(m_refKeypoints[m.queryIdx].pt);
        pts2.push_back(kps[m.trainIdx].pt);
    }

    std::vector<uchar> valid;
    const std::vector<cv::Point3f> triangulated = triangulate(m_refR, m_refT, R, t, pts1, pts2, valid);

    std::vector<cv::Point3f> newPoints;
    std::vector<cv::Point2f> newImagePoints;
    cv::Mat newDescriptors;
    std::vector<int> newKeypointIndices; // parallel to newPoints -- this keyframe's own kps/kf.keypoints
                                          // index each new point came from, needed below to seed
                                          // kf.keypointLandmarkId (matches[i].trainIdx is only available
                                          // inside this loop, over the unfiltered triangulated/valid index
                                          // space)
    // Backend #2 parallax gate (setParallaxGateEnabled()): world-space camera
    // centers of the two views this batch was triangulated from (reference and
    // current keyframe, C = -R^T t), used to reject low-parallax points below.
    // Cheap to precompute once outside the per-point loop.
    cv::Mat parallaxC1, parallaxC2;
    if (m_parallaxGateEnabled) {
        parallaxC1 = -m_refR.t() * m_refT;
        parallaxC2 = -R.t() * t;
    }
    for (size_t i = 0; i < triangulated.size(); ++i) {
        if (!valid[i])
            continue;
        // Backend #2 (item 41): cull points whose two viewing rays are nearly
        // parallel -- their triangulated depth is dominated by pixel noise, so
        // admitting them pollutes the map that local BA then tight-fits to
        // (item 40's backfire mechanism). Same cosParallax test real ORB-SLAM3
        // uses at landmark creation. Default off; enable via 'parallaxgate'.
        if (m_parallaxGateEnabled) {
            const cv::Point3f &X = triangulated[i];
            const cv::Vec3d r1(X.x - parallaxC1.at<double>(0), X.y - parallaxC1.at<double>(1),
                               X.z - parallaxC1.at<double>(2));
            const cv::Vec3d r2(X.x - parallaxC2.at<double>(0), X.y - parallaxC2.at<double>(1),
                               X.z - parallaxC2.at<double>(2));
            const double n1 = cv::norm(r1), n2 = cv::norm(r2);
            if (n1 < 1e-9 || n2 < 1e-9)
                continue;
            const double cosParallax = r1.dot(r2) / (n1 * n2);
            if (cosParallax > kMinTriangulationParallaxCos)
                continue; // rays too parallel -- unreliable depth, reject
        }
        newPoints.push_back(triangulated[i]);
        newImagePoints.push_back(pts2[i]); // this keyframe's own 2D observation of the point
        newDescriptors.push_back(descriptors.row(matches[i].trainIdx));
        newKeypointIndices.push_back(matches[i].trainIdx);
    }

    // A fresh, stable landmark ID per new point -- see m_landmarkPositions/
    // m_landmarkObservations' doc comment in SlamWorker.h. Recorded now
    // (this keyframe's own observation) so bundle adjustment has something
    // to work with even if no later keyframe ever re-observes the point.
    std::vector<long long> newIds;
    newIds.reserve(newPoints.size());
    const int newKeyframeIndex = static_cast<int>(m_keyframeHistory.size());
    for (size_t i = 0; i < newPoints.size(); ++i) {
        const long long id = m_nextLandmarkId++;
        newIds.push_back(id);
        m_landmarkPositions[id] = newPoints[i];
        m_landmarkDescriptors[id] = newDescriptors.row(static_cast<int>(i)).clone();
        m_landmarkObservations[id].emplace_back(newKeyframeIndex, newImagePoints[i]);
    }

    // Seed this keyframe's own reverse-index entry (see
    // m_keyframeObservedLandmarkIds's own doc comment) with its own
    // just-triangulated points -- recordLandmarkObservations() below
    // appends any further re-observed landmarks onto the SAME entry.
    m_keyframeObservedLandmarkIds.push_back(newIds);

    // Record this keyframe (with a copy of its own locally-triangulated
    // points, kept independently of the global map's rolling eviction --
    // see the Keyframe comment in SlamWorker.h) before newPoints is moved
    // out below, so a much-later loop closure still has something of this
    // keyframe's own to PnP-solve against. Constructed here (earlier than
    // this function used to build it) specifically so
    // kf.keypointLandmarkId exists for recordLandmarkObservations() to
    // write into next.
    Keyframe kf;
    kf.keypoints = kps;
    kf.descriptors = descriptors.clone();
    kf.R = R.clone();
    kf.t = t.clone();
    kf.localMapPoints = newPoints;
    kf.localMapImagePoints = newImagePoints;
    kf.localMapDescriptors = newDescriptors;
    kf.localMapPointIds = newIds;
    kf.frameIndex = m_frameCount;

    // See Keyframe::keypointLandmarkId's own doc comment. Seeded with this
    // keyframe's own just-triangulated points; recordLandmarkObservations()
    // below fills in any further re-observed (rolling-map) landmarks onto
    // the SAME array.
    kf.keypointLandmarkId.assign(kps.size(), -1);
    for (size_t i = 0; i < newIds.size(); ++i)
        kf.keypointLandmarkId[static_cast<size_t>(newKeypointIndices[i])] = newIds[i];

    // Before this keyframe's own new points join the global map below, see
    // whether this keyframe *also* re-observes any already-known landmark
    // (triangulated by some earlier keyframe) -- the cross-keyframe
    // correspondence bundle adjustment needs and nothing else in this
    // codebase records.
    recordLandmarkObservations(newKeyframeIndex, kps, descriptors, R, t, kf.keypointLandmarkId);

    // DBoW2 place-recognition vector for tryLoopClosure()'s candidate
    // search (see setDbowLoopClosureEnabled()) -- only computed when a
    // vocabulary is loaded and ORB is the active detector (a vocabulary
    // trained on ORB descriptors can't meaningfully score SIFT's float
    // ones). transform()'s levelsup=4 mirrors ORB-SLAM2/3's own
    // KeyFrame::ComputeBoW() call.
    if (m_orbVocabulary && m_detectorType == feature_detector::DetectorType::Orb) {
        std::vector<cv::Mat> descriptorRows;
        descriptorRows.reserve(static_cast<size_t>(kf.descriptors.rows));
        for (int row = 0; row < kf.descriptors.rows; ++row)
            descriptorRows.push_back(kf.descriptors.row(row));
        DBoW2::FeatureVector unusedFeatVec;
        m_orbVocabulary->transform(descriptorRows, kf.bowVec, unusedFeatVec, 4);
    }

    // VLAD place-recognition vector for tryLoopClosure()'s candidate search
    // (see setVladLoopClosureEnabled()) -- the SIFT-compatible counterpart
    // to the DBoW2 block above, only computed when a codebook is loaded and
    // SIFT is the active detector.
    if (m_vladVocabulary && m_detectorType == feature_detector::DetectorType::Sift)
        kf.vladVector = m_vladVocabulary->computeVlad(kf.descriptors);

    // SIFT DBoW2 place-recognition vector for tryLoopClosure()'s candidate
    // search (see setSiftDbowLoopClosureEnabled()) -- a second SIFT-
    // compatible option alongside the VLAD block above. kf.descriptors is
    // already RootSIFT (toRootSift() applied right after detection, see
    // this function's caller), matching what
    // analyze/train_sift_dbow_vocabulary.cpp trained the vocabulary on.
    // Same transform() levelsup=4 convention as the ORB DBoW2 block above.
    if (m_siftVocabulary && m_detectorType == feature_detector::DetectorType::Sift) {
        std::vector<cv::Mat> descriptorRows;
        descriptorRows.reserve(static_cast<size_t>(kf.descriptors.rows));
        for (int row = 0; row < kf.descriptors.rows; ++row)
            descriptorRows.push_back(kf.descriptors.row(row));
        DBoW2::FeatureVector unusedFeatVec;
        m_siftVocabulary->transform(descriptorRows, kf.siftBowVec, unusedFeatVec, 4);
    }

    // Relative-pose measurement for pose_graph::optimizePoseGraph() (see
    // PoseGraphOptimizer.h), captured NOW -- from both keyframes' poses as
    // they stand at this exact instant, before any future loop closure
    // could touch either one -- rather than reconstructed later from
    // (possibly multiply-corrected) final absolute poses.
    if (!m_keyframeHistory.empty()) {
        const Keyframe &prevKf = m_keyframeHistory.back();
        cv::Mat Rrel = R * prevKf.R.t();
        const cv::Mat trel = t - Rrel * prevKf.t;
        // Replace the tracked (possibly rotation-drifted) Rrel with the
        // real IMU-measured relative rotation for this span, when
        // available -- gives pose_graph::optimizePoseGraph() (SE(3) or
        // Sim(3), see PoseGraphOptions::useSim3) a per-edge ROTATION
        // anchored to real gyro/orientation measurement instead of
        // whatever the vision pipeline's own accumulated R drifted to.
        // Same imu_rotation::relativeCameraRotation() this codebase
        // already uses live (see trackFrame()'s own OXTS/IMU-aware
        // plausibility check) -- translation magnitude is untouched here
        // (IMU orientation alone carries no distance information).
        if (m_imuEnabled && !m_oxtsNavFromBody.empty() && !m_imuToCameraCalib.R.empty()) {
            const cv::Mat imuRrel =
                imu_rotation::relativeCameraRotation(m_oxtsNavFromBody, m_imuToCameraCalib, prevKf.frameIndex, m_frameCount);
            if (!imuRrel.empty())
                Rrel = imuRrel;
        }
        m_sequentialEdgeRecords.push_back({static_cast<int>(m_keyframeHistory.size()) - 1,
                                            static_cast<int>(m_keyframeHistory.size()), Rrel.clone(), trel.clone()});
    }

    m_keyframeHistory.push_back(std::move(kf));

    appendToMap(std::move(newPoints), newDescriptors, newIds);

    setReferenceFrame(kps, descriptors, R, t);

    // Before local BA -- see fuseWindowLandmarks()'s own doc comment --
    // mirrors ORB-SLAM3's own LocalMapping::Run() ordering
    // (SearchInNeighbors() before LocalBundleAdjustment()).
    if (m_landmarkFuseEnabled)
        fuseWindowLandmarks(static_cast<int>(m_keyframeHistory.size()) - 1);

    // Backend #2 (item 41): re-triangulate this keyframe's landmarks from all
    // their views now that the keyframe is in m_keyframeHistory and fuse has
    // extended observation coverage -- but BEFORE local BA, so BA refines from
    // the better-conditioned seed. No-op unless 'retriangulate' is set.
    retriangulateKeyframeLandmarks(static_cast<int>(m_keyframeHistory.size()) - 1);

    if (m_localBundleAdjustmentEnabled && m_localBaHardAnchorEnabled)
        runLocalBundleAdjustmentHardAnchor();
    else if (m_localBundleAdjustmentEnabled)
        runLocalBundleAdjustment();
    else
        refineLocalKeyframes();

    tryLoopClosure(m_keyframeHistory.size() - 1);

    if (m_keyframeCullingEnabled && m_keyframeHistory.size() % kCullingCheckIntervalKeyframes == 0)
        cullRedundantKeyframes();

    if (m_covisibilityLocalMapEnabled)
        buildCovisibilityLocalMap();
}

void SlamWorker::fuseWindowLandmarks(int newKeyframeIndex)
{
    // v4 redesign (DEBUGGING.md item 19), Phase A only (coverage
    // extension, no merging -- see setLandmarkFuseEnabled()'s own doc
    // comment for why items 16-18's version, and why merging is deferred).
    // Real ORB-SLAM3-style PROJECTION test this time: for each of the
    // current keyframe's own just-triangulated landmarks, project it into
    // each nearby keyframe's OWN pose and search that keyframe's OWN
    // actually-detected keypoints (not another landmark's stored 3D
    // position) for a reprojection-gated descriptor match -- exactly what
    // recordLandmarkObservations() already does for the rolling map,
    // applied here to nearby PAST keyframes too.
    const int n = static_cast<int>(m_keyframeHistory.size());
    const int windowStart = std::max(0, n - kFuseWindowKeyframes);

    const std::vector<long long> newIds = m_keyframeHistory[static_cast<size_t>(newKeyframeIndex)].localMapPointIds;

    for (long long newId : newIds) {
        const auto posIt = m_landmarkPositions.find(newId);
        const auto descIt = m_landmarkDescriptors.find(newId);
        if (posIt == m_landmarkPositions.end() || descIt == m_landmarkDescriptors.end())
            continue;
        const cv::Point3f &p = posIt->second;
        const cv::Mat pWorld = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
        const cv::Mat &d1 = descIt->second;

        for (int ki = windowStart; ki < n; ++ki) {
            if (ki == newKeyframeIndex)
                continue;
            Keyframe &otherKf = m_keyframeHistory[static_cast<size_t>(ki)];
            if (otherKf.descriptors.empty())
                continue;

            const cv::Mat camPt = otherKf.R * pWorld + otherKf.t;
            const double z = camPt.at<double>(2);
            if (z < 1e-3)
                continue; // behind this keyframe's own camera

            const double u = m_intrinsics.fx * camPt.at<double>(0) / z + m_intrinsics.cx;
            const double v = m_intrinsics.fy * camPt.at<double>(1) / z + m_intrinsics.cy;

            std::vector<cv::DMatch> matches;
            if (!matchDescriptors(d1, otherKf.descriptors, matches) || matches.empty())
                continue;
            const cv::DMatch &m = matches.front(); // d1 is a single-row query -> at most 1 result
            const cv::Point2f &obs = otherKf.keypoints[static_cast<size_t>(m.trainIdx)].pt;
            const double du = u - obs.x, dv = v - obs.y;
            if (du * du + dv * dv > kMaxObservationReprojErrorPixels * kMaxObservationReprojErrorPixels)
                continue; // descriptor-similar but geometrically implausible from this viewpoint

            const long long existing = otherKf.keypointLandmarkId[static_cast<size_t>(m.trainIdx)];
            if (existing == newId)
                continue; // already linked (defensive, shouldn't normally occur for a brand-new id)

            if (existing >= 0 && !m_landmarkFuseMergeEnabled)
                continue; // genuine conflict, but Phase B is off -- see setLandmarkFuseMergeEnabled()'s
                           // own doc comment for why it's NOT recommended (measured negative, item 20)

            if (existing >= 0 && du * du + dv * dv > kFuseMergeMaxReprojErrorPixels * kFuseMergeMaxReprojErrorPixels)
                continue; // passed the shared 8px extension gate but not Phase B's own stricter
                           // merge-specific bar (see kFuseMergeMaxReprojErrorPixels's own doc comment,
                           // item 26) -- too consequential a decision to accept on the same margin as a
                           // plain coverage extension

            if (existing >= 0) {
                // Phase B v5 (untried redesign flagged in DEBUGGING.md's
                // Phase B section, after v1-v4 all measured negative):
                // newId and existing both demonstrably explain this exact
                // detected keypoint in otherKf, confirmed via a real
                // reprojection-gated descriptor match. Richer-evidence-wins
                // decides who nominally "survives" the id, same rule as
                // before -- but instead of then keeping that survivor's own
                // stale, pre-merge 3D position (v1-v4's shared flaw, root-
                // caused across items 20/25/26 as the actual source of
                // every prior regression), re-triangulate a fresh position
                // from the COMBINED observation set of both ids using ALL
                // their accumulated multi-view constraint
                // (triangulateMultiView()). If that combined set is
                // geometrically inconsistent (its own internal reprojection
                // gate fails), the merge is REJECTED outright -- newId and
                // existing are left exactly as they were, as if this
                // otherKf candidate had never matched. This makes
                // triangulation success itself the real merge criterion,
                // not just the earlier single-view pixel gate that only
                // narrows candidates.
                const auto newObsIt = m_landmarkObservations.find(newId);
                const auto existingObsIt = m_landmarkObservations.find(existing);
                const size_t newObsCount = newObsIt == m_landmarkObservations.end() ? 0 : newObsIt->second.size();
                const size_t existingObsCount =
                    existingObsIt == m_landmarkObservations.end() ? 0 : existingObsIt->second.size();
                const long long survivor = existingObsCount >= newObsCount ? existing : newId;
                const long long loser = survivor == existing ? newId : existing;

                const auto loserIt = m_landmarkObservations.find(loser);
                const auto survivorIt = m_landmarkObservations.find(survivor);
                if (loserIt != m_landmarkObservations.end() && survivorIt != m_landmarkObservations.end()) {
                    std::vector<std::pair<int, cv::Point2f>> combinedObs = survivorIt->second;
                    combinedObs.insert(combinedObs.end(), loserIt->second.begin(), loserIt->second.end());

                    bool triangulationValid = false;
                    const cv::Point3f newPos = triangulateMultiView(combinedObs, triangulationValid);
                    if (!triangulationValid)
                        continue; // geometrically inconsistent merge -- reject, leave both ids untouched

                    m_landmarkPositions[survivor] = newPos;

                    // Resync BOTH owning keyframes' own local-map copies
                    // (Keyframe::localMapPointIds/localMapPoints/
                    // localMapDescriptors) -- loser's, because its id is
                    // being retired and tryLoopClosure()'s own PnP/
                    // Sim3Solver measurement reads these arrays directly,
                    // not m_landmarkObservations/m_landmarkPositions (item
                    // 20's original finding); AND survivor's own, because
                    // unlike v1-v4, survivor's position now genuinely
                    // changes at merge time and its owning keyframe's copy
                    // would otherwise go stale too -- a gap v1-v4 never had
                    // to handle since they never moved the survivor. The
                    // owning keyframe is always the FIRST (oldest) entry in
                    // the observation list -- seeded at triangulation time
                    // in insertKeyframe() and never reordered, only ever
                    // appended to afterward.
                    const auto survivorDescIt = m_landmarkDescriptors.find(survivor);
                    auto syncOwnerLocalMap = [this, &survivorDescIt](long long ownedId, long long newOwnedId,
                                                                      const cv::Point3f &pos, int ownerKfIdx) {
                        Keyframe &ownerKf = m_keyframeHistory[static_cast<size_t>(ownerKfIdx)];
                        const auto ownIt = std::find(ownerKf.localMapPointIds.begin(),
                                                      ownerKf.localMapPointIds.end(), ownedId);
                        if (ownIt == ownerKf.localMapPointIds.end())
                            return;
                        const size_t localIdx = static_cast<size_t>(ownIt - ownerKf.localMapPointIds.begin());
                        ownerKf.localMapPointIds[localIdx] = newOwnedId;
                        ownerKf.localMapPoints[localIdx] = pos;
                        if (survivorDescIt != m_landmarkDescriptors.end())
                            survivorDescIt->second.copyTo(ownerKf.localMapDescriptors.row(static_cast<int>(localIdx)));
                    };
                    syncOwnerLocalMap(loser, survivor, newPos, loserIt->second.front().first);
                    syncOwnerLocalMap(survivor, survivor, newPos, survivorIt->second.front().first);

                    std::vector<std::pair<int, cv::Point2f>> loserObs = std::move(loserIt->second);
                    m_landmarkObservations.erase(loserIt);
                    auto &survivorObs = m_landmarkObservations[survivor];
                    survivorObs.insert(survivorObs.end(), loserObs.begin(), loserObs.end());
                }
                // Only this one now-resolved slot is fixed up -- any OTHER
                // keyframe that historically linked to loser (via an
                // earlier Phase A coverage-extension) keeps a stale
                // keypointLandmarkId entry; the next fuse pass that
                // touches it will find loser's m_landmarkObservations
                // entry gone and treat it as inert, self-healing onto
                // whatever survivor is current by then. Same soft-
                // consistency approach this codebase already uses for
                // keyframe culling.
                otherKf.keypointLandmarkId[static_cast<size_t>(m.trainIdx)] = survivor;
                ++m_fusedLandmarkCount;

                if (loser == newId)
                    break; // newId itself is gone -- nothing left to check against the rest of the window
                continue;
            }

            // Unclaimed keypoint: extend coverage. A real, reprojection-
            // gated new observation of newId -- structurally identical to
            // what recordLandmarkObservations() already does safely for
            // the rolling map, just reaching into a nearby PAST keyframe.
            m_landmarkObservations[newId].emplace_back(ki, obs);
            m_keyframeObservedLandmarkIds[static_cast<size_t>(ki)].push_back(newId);
            otherKf.keypointLandmarkId[static_cast<size_t>(m.trainIdx)] = newId;
            ++m_fusedLandmarkCount;
        }
    }
}

void SlamWorker::recordLandmarkObservations(int keyframeIndex, const std::vector<cv::KeyPoint> &kps,
                                             const cv::Mat &descriptors, const cv::Mat &R, const cv::Mat &t,
                                             std::vector<long long> &keypointLandmarkId)
{
    if (m_mapDescriptors.empty())
        return;

    std::vector<cv::DMatch> matches;
    if (!matchDescriptors(m_mapDescriptors, descriptors, matches))
        return;

    for (const auto &m : matches) {
        const long long id = m_mapPointIds[m.queryIdx];
        // Descriptor ratio-test matches alone aren't trusted here -- see
        // kMaxObservationReprojErrorPixels' doc comment (this constant's
        // definition, near the top of this file) for why: a plain ratio
        // test has no geometric check at all, and this observation feeds
        // runLoopBundleAdjustment() as if it were as reliable as a
        // triangulation-time observation. Reproject the landmark's already-
        // known 3D position through this keyframe's own pose and reject the
        // match if it lands far from where the descriptor match claims it
        // is -- cheap (no extra RANSAC solve) and catches descriptor-
        // similar-but-geometrically-wrong matches before they ever reach BA.
        const auto posIt = m_landmarkPositions.find(id);
        if (posIt == m_landmarkPositions.end())
            continue;
        const cv::Point3f &p = posIt->second;
        const cv::Mat P = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
        const cv::Mat camPt = R * P + t;
        const double z = camPt.at<double>(2);
        if (z < 1e-3)
            continue; // behind the camera -- can't be a real re-observation
        const double u = m_intrinsics.fx * camPt.at<double>(0) / z + m_intrinsics.cx;
        const double v = m_intrinsics.fy * camPt.at<double>(1) / z + m_intrinsics.cy;
        const cv::Point2f &obs = kps[m.trainIdx].pt;
        const double du = u - obs.x, dv = v - obs.y;
        if (du * du + dv * dv > kMaxObservationReprojErrorPixels * kMaxObservationReprojErrorPixels)
            continue;
        m_landmarkObservations[id].emplace_back(keyframeIndex, obs);
        m_keyframeObservedLandmarkIds[static_cast<size_t>(keyframeIndex)].push_back(id);
        keypointLandmarkId[static_cast<size_t>(m.trainIdx)] = id;
    }
}

// Backend #2 (item 41, setRetriangulateEnabled()): re-triangulate every
// landmark this just-inserted keyframe observes that now has enough cross-
// keyframe views, from ALL of them (multi-view DLT via triangulateMultiView()),
// replacing the original 2-view estimate. Better map quality is the item-40
// lever that lets local BA's soft-prior fit tightly without backfiring.
//
// MUST run AFTER the keyframe is appended to m_keyframeHistory: the current
// keyframe's own observations carry its history index, and triangulateMultiView()
// dereferences m_keyframeHistory[obs.first] for every view -- calling this
// before the push (e.g. from inside recordLandmarkObservations) would index
// one past the end.
void SlamWorker::retriangulateKeyframeLandmarks(int keyframeIndex)
{
    if (!m_retriangulateEnabled)
        return;
    if (keyframeIndex < 0 || static_cast<size_t>(keyframeIndex) >= m_keyframeObservedLandmarkIds.size())
        return;

    // id -> rolling-map slot, built once for this keyframe's touched landmarks
    // so the accepted position can be synced into m_mapPoints (parallel to
    // m_mapPointIds) without a per-landmark scan -- same lookup local BA's
    // write-back uses.
    std::unordered_map<long long, size_t> mapIndexById;
    mapIndexById.reserve(m_mapPointIds.size());
    for (size_t i = 0; i < m_mapPointIds.size(); ++i)
        mapIndexById[m_mapPointIds[i]] = i;

    const auto meanReproj = [&](const cv::Point3f &X,
                                const std::vector<std::pair<int, cv::Point2f>> &obs) -> double {
        double sumSq = 0.0;
        int views = 0;
        for (const auto &o : obs) {
            if (o.first < 0 || static_cast<size_t>(o.first) >= m_keyframeHistory.size())
                continue;
            const Keyframe &kf = m_keyframeHistory[static_cast<size_t>(o.first)];
            const cv::Mat Xw = (cv::Mat_<double>(3, 1) << X.x, X.y, X.z);
            const cv::Mat cp = kf.R * Xw + kf.t;
            const double zz = cp.at<double>(2);
            if (zz < 1e-3)
                continue;
            const double pu = m_intrinsics.fx * cp.at<double>(0) / zz + m_intrinsics.cx - o.second.x;
            const double pv = m_intrinsics.fy * cp.at<double>(1) / zz + m_intrinsics.cy - o.second.y;
            sumSq += pu * pu + pv * pv;
            ++views;
        }
        return views ? std::sqrt(sumSq / views) : std::numeric_limits<double>::max();
    };

    // Copy the id list: the accepted-position write-backs below don't touch
    // m_keyframeObservedLandmarkIds, but iterate a stable snapshot regardless.
    const std::vector<long long> touched = m_keyframeObservedLandmarkIds[static_cast<size_t>(keyframeIndex)];
    for (const long long id : touched) {
        const auto obsIt = m_landmarkObservations.find(id);
        if (obsIt == m_landmarkObservations.end() ||
            obsIt->second.size() < static_cast<size_t>(kRetriangulateMinViews))
            continue;
        const auto posIt = m_landmarkPositions.find(id);
        if (posIt == m_landmarkPositions.end())
            continue;
        bool triOk = false;
        const cv::Point3f cand = triangulateMultiView(obsIt->second, triOk);
        if (!triOk)
            continue; // fails triangulateMultiView()'s own depth/reprojection gate
        // Strict-improvement guard: triangulateMultiView() is a raw DLT with no
        // robust loss, so accept it only if it reprojects BETTER than the
        // current position across the SAME observations -- this can never
        // regress a landmark local BA (Huber) already refined below the DLT
        // optimum, making re-triangulation monotone and BA-safe.
        if (meanReproj(cand, obsIt->second) >= meanReproj(posIt->second, obsIt->second))
            continue;
        posIt->second = cand;
        const auto mapIt = mapIndexById.find(id);
        if (mapIt != mapIndexById.end())
            m_mapPoints[mapIt->second] = cand; // keep the rolling map in sync
        // The owning keyframe's localMapPoints copy is left for the following
        // local BA to re-sync -- the same out-of-window staleness tolerance
        // local BA's own write-back already has.
        ++m_retriangulatedLandmarkCount;
    }
}

void SlamWorker::refineLocalKeyframes()
{
    const int start = std::max(0, static_cast<int>(m_keyframeHistory.size()) - kLocalRefineWindow);
    for (int i = start; i < static_cast<int>(m_keyframeHistory.size()); ++i) {
        Keyframe &kf = m_keyframeHistory[i];
        if (kf.localMapPoints.size() < 6)
            continue;

        cv::Mat rvec, tvec;
        cv::Rodrigues(kf.R, rvec);
        tvec = kf.t.clone();
        try {
            cv::solvePnPRefineLM(kf.localMapPoints, kf.localMapImagePoints, m_intrinsics.toMat(),
                                  cv::Mat(), rvec, tvec);
        } catch (const cv::Exception &) {
            continue;
        }

        cv::Mat refinedR;
        cv::Rodrigues(rvec, refinedR);
        kf.R = refinedR;
        kf.t = tvec;
    }

    // setReferenceFrame() was called with the pre-refinement pose -- keep
    // the live pose and short-term reference frame in sync with whatever
    // the newest keyframe's pose ended up as above.
    if (!m_keyframeHistory.empty()) {
        const Keyframe &latest = m_keyframeHistory.back();
        m_refR = latest.R.clone();
        m_refT = latest.t.clone();
        m_currR = latest.R.clone();
        m_currT = latest.t.clone();
        if (!m_trajectory.isEmpty()) {
            const cv::Mat C = -latest.R.t() * latest.t;
            m_trajectory.last() = QPointF(C.at<double>(0), C.at<double>(2));
        }
    }
}

bool SlamWorker::runLocalBundleAdjustment()
{
    const int n = static_cast<int>(m_keyframeHistory.size());
    const int windowStart = std::max(0, n - m_localBaWindowKeyframes);
    const int windowSize = n - windowStart;
    if (windowSize < 2)
        return false;

    // Any landmark with AT LEAST ONE observation from an in-window keyframe
    // -- not just ones some in-window keyframe happens to be the ORIGINAL
    // triangulator of (see m_keyframeObservedLandmarkIds's own doc comment
    // for why the ownership-only rule this used to use silently starved
    // this of real, already-recorded multi-view constraint density).
    // Observations themselves are still strictly bounded to the window
    // (the inner loop below), so this doesn't pull in any extra keyframe
    // pose parameter blocks, only more per-landmark residuals among the
    // ones already in the problem. processedLandmarkIds deduplicates: the
    // SAME landmark can now legitimately appear in several different
    // in-window keyframes' own observed-landmark lists (that's the whole
    // point), so without this guard it would get its observations
    // gathered -- and turned into duplicate residual blocks -- once per
    // keyframe that lists it, not once overall.
    std::unordered_map<long long, std::vector<std::pair<int, cv::Point2f>>> windowObservations;
    std::unordered_set<long long> processedLandmarkIds;
    for (int i = windowStart; i < n; ++i) {
        for (long long id : m_keyframeObservedLandmarkIds[static_cast<size_t>(i)]) {
            if (!processedLandmarkIds.insert(id).second)
                continue;
            const auto it = m_landmarkObservations.find(id);
            if (it == m_landmarkObservations.end())
                continue;
            auto &dst = windowObservations[id];
            for (const auto &obs : it->second) {
                if (obs.first >= windowStart && obs.first < n)
                    dst.push_back(obs);
            }
        }
    }

    auto fillPose = [](std::array<double, 6> &block, const cv::Mat &R, const cv::Mat &t) {
        cv::Mat rvec;
        cv::Rodrigues(R, rvec);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(k)] = rvec.at<double>(k);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(3 + k)] = t.at<double>(k);
    };
    std::vector<std::array<double, 6>> poses(static_cast<size_t>(windowSize));
    std::vector<std::array<double, 6>> priors(static_cast<size_t>(windowSize));
    for (int i = windowStart; i < n; ++i) {
        const Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
        fillPose(poses[static_cast<size_t>(i - windowStart)], kf.R, kf.t);
        priors[static_cast<size_t>(i - windowStart)] = poses[static_cast<size_t>(i - windowStart)];
    }

    std::unordered_map<long long, std::array<double, 3>> landmarks;
    for (auto &entry : windowObservations) {
        if (entry.second.size() < static_cast<size_t>(kBaMinObservationsPerLandmark))
            continue;
        const cv::Point3f &p = m_landmarkPositions.at(entry.first);
        landmarks[entry.first] = {static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)};
    }
    if (landmarks.empty())
        return false;

    ceres::Problem problem;
    for (int i = 0; i < windowSize; ++i) {
        problem.AddParameterBlock(poses[static_cast<size_t>(i)].data(), 6);
        // The pose-prior residual on EVERY window pose -- not a single
        // hard-fixed anchor -- is what bounds drift here; see
        // kLocalBaWindowKeyframes's doc comment for why a single
        // hard-anchor scheme (the reverted prior attempt) let scale
        // collapse over hundreds of highly-overlapping windows.
        problem.AddResidualBlock(PosePriorCost::Create(priors[static_cast<size_t>(i)], m_localBaPosePriorRotWeight,
                                                         m_localBaPosePriorTransWeight),
                                  nullptr, poses[static_cast<size_t>(i)].data());
    }

    const CameraIntrinsics &intr = m_intrinsics;
    int residualCount = 0;
    for (auto &entry : landmarks) {
        for (const auto &obs : windowObservations.at(entry.first)) {
            ceres::CostFunction *cost =
                ReprojectionCost::Create(obs.second.x, obs.second.y, intr.fx, intr.fy, intr.cx, intr.cy);
            problem.AddResidualBlock(cost, new ceres::HuberLoss(kBaHuberDeltaPixels),
                                      poses[static_cast<size_t>(obs.first - windowStart)].data(), entry.second.data());
            ++residualCount;
        }
    }
    if (residualCount == 0)
        return false;

    // keyframe 0 (the world anchor) stays hard-fixed on the rare window
    // that includes it (only true near the very start of a run) -- every
    // other pose relies on its own prior above, not a hard fix.
    if (windowStart == 0)
        problem.SetParameterBlockConstant(poses.front().data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = kBaMaxIterations;
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
        std::fprintf(stderr, "[localba] did not converge, falling back: %s\n", summary.BriefReport().c_str());
        return false;
    }

    auto poseToRT = [](const std::array<double, 6> &block, cv::Mat &R, cv::Mat &t) {
        const cv::Mat rvec = (cv::Mat_<double>(3, 1) << block[0], block[1], block[2]);
        cv::Rodrigues(rvec, R);
        t = (cv::Mat_<double>(3, 1) << block[3], block[4], block[5]);
    };
    for (int i = windowStart; i < n; ++i) {
        cv::Mat R, t;
        poseToRT(poses[static_cast<size_t>(i - windowStart)], R, t);
        m_keyframeHistory[static_cast<size_t>(i)].R = R;
        m_keyframeHistory[static_cast<size_t>(i)].t = t;
    }

    // Write optimized landmark positions back -- mirrors
    // runLoopBundleAdjustment()'s own write-back exactly.
    std::unordered_map<long long, size_t> mapIndexById;
    mapIndexById.reserve(m_mapPointIds.size());
    for (size_t i = 0; i < m_mapPointIds.size(); ++i)
        mapIndexById[m_mapPointIds[i]] = i;
    for (const auto &entry : landmarks) {
        const cv::Point3f newPos(static_cast<float>(entry.second[0]), static_cast<float>(entry.second[1]),
                                  static_cast<float>(entry.second[2]));
        m_landmarkPositions[entry.first] = newPos;
        const auto mapIt = mapIndexById.find(entry.first);
        if (mapIt != mapIndexById.end())
            m_mapPoints[mapIt->second] = newPos;
        for (int i = windowStart; i < n; ++i) {
            Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
            const auto idIt = std::find(kf.localMapPointIds.begin(), kf.localMapPointIds.end(), entry.first);
            if (idIt != kf.localMapPointIds.end()) {
                const size_t localIdx = static_cast<size_t>(idIt - kf.localMapPointIds.begin());
                kf.localMapPoints[localIdx] = newPos;
                break; // a landmark has exactly one owning keyframe
            }
        }
    }

    // Same m_refR/m_refT/m_currR/m_currT/trajectory-tail sync
    // refineLocalKeyframes() does -- setReferenceFrame() was called with
    // the pre-BA pose, so it needs re-syncing to whatever the newest
    // keyframe's pose ended up as above.
    const Keyframe &latest = m_keyframeHistory.back();
    m_refR = latest.R.clone();
    m_refT = latest.t.clone();
    m_currR = latest.R.clone();
    m_currT = latest.t.clone();
    if (!m_trajectory.isEmpty()) {
        const cv::Mat C = -latest.R.t() * latest.t;
        m_trajectory.last() = QPointF(C.at<double>(0), C.at<double>(2));
    }

    std::fprintf(stderr, "[localba] kf#%d..kf#%d, %d landmarks, %d observations, initial cost=%.3f final cost=%.3f\n",
                  windowStart, n - 1, static_cast<int>(landmarks.size()), residualCount, summary.initial_cost,
                  summary.final_cost);
    return true;
}

bool SlamWorker::runLocalBundleAdjustmentHardAnchor()
{
    // Real ORB-SLAM3 LocalBundleAdjustment (Optimizer.cc): the LOCAL
    // keyframes (the recent window) and the map points they observe are
    // optimized freely; every OTHER keyframe that ALSO observes one of
    // those local map points is added to the problem but held CONSTANT --
    // these "border"/fixed keyframes are the hard anchors, and because they
    // co-observe the local map points their fixed poses+observations pin
    // the absolute scale/gauge of the whole local solve. This is the
    // structurally-correct hard-anchor scheme, NOT the reverted single-
    // hard-anchor attempt that let scale collapse (187m, see
    // runLocalBundleAdjustment()'s soft-prior comment): there, one lone
    // anchor left the rest of the window free to drift/rescale as a group;
    // here, multiple fixed border keyframes distributed around the window
    // co-constrain the same points, so the group cannot rescale. See
    // setLocalBaHardAnchorEnabled(), DEBUGGING.md item 37.
    const int n = static_cast<int>(m_keyframeHistory.size());
    const int windowStart = std::max(0, n - m_localBaWindowKeyframes);
    if (n - windowStart < 2)
        return false;

    // Local map points: every landmark observed by any in-window keyframe.
    std::unordered_set<long long> localMpIds;
    for (int i = windowStart; i < n; ++i)
        for (long long id : m_keyframeObservedLandmarkIds[static_cast<size_t>(i)])
            localMpIds.insert(id);

    auto fillPose = [](std::array<double, 6> &block, const cv::Mat &R, const cv::Mat &t) {
        cv::Mat rvec;
        cv::Rodrigues(R, rvec);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(k)] = rvec.at<double>(k);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(3 + k)] = t.at<double>(k);
    };

    // Pose blocks, created lazily for every keyframe that actually observes
    // a local map point (in-window OR fixed border). Stored in a
    // deque/map so their addresses stay stable for Ceres.
    std::unordered_map<int, std::array<double, 6>> poseBlocks;
    std::unordered_set<int> fixedKfs; // keyframes < windowStart that co-observe -> hard anchors
    std::unordered_map<long long, std::array<double, 3>> landmarks;
    std::unordered_map<long long, std::vector<std::pair<int, cv::Point2f>>> mpObs;

    auto ensurePose = [&](int kfIdx) {
        if (poseBlocks.find(kfIdx) == poseBlocks.end()) {
            const Keyframe &kf = m_keyframeHistory[static_cast<size_t>(kfIdx)];
            std::array<double, 6> b;
            fillPose(b, kf.R, kf.t);
            poseBlocks[kfIdx] = b;
        }
    };

    for (long long id : localMpIds) {
        const auto it = m_landmarkObservations.find(id);
        if (it == m_landmarkObservations.end() ||
            it->second.size() < static_cast<size_t>(kBaMinObservationsPerLandmark))
            continue;
        const auto posIt = m_landmarkPositions.find(id);
        if (posIt == m_landmarkPositions.end())
            continue;
        auto &dst = mpObs[id];
        for (const auto &obs : it->second) {
            const int kf = obs.first;
            if (kf < 0 || kf >= n)
                continue;
            ensurePose(kf);
            if (kf < windowStart)
                fixedKfs.insert(kf); // border keyframe -> fixed anchor
            dst.push_back(obs);
        }
        if (dst.empty()) {
            mpObs.erase(id);
            continue;
        }
        landmarks[id] = {static_cast<double>(posIt->second.x), static_cast<double>(posIt->second.y),
                          static_cast<double>(posIt->second.z)};
    }
    if (landmarks.empty())
        return false;

    const CameraIntrinsics &intr = m_intrinsics;
    ceres::Problem problem;
    for (auto &entry : poseBlocks)
        problem.AddParameterBlock(entry.second.data(), 6);

    int residualCount = 0;
    for (auto &entry : landmarks) {
        for (const auto &obs : mpObs.at(entry.first)) {
            ceres::CostFunction *cost =
                ReprojectionCost::Create(obs.second.x, obs.second.y, intr.fx, intr.fy, intr.cx, intr.cy);
            problem.AddResidualBlock(cost, new ceres::HuberLoss(kBaHuberDeltaPixels),
                                      poseBlocks.at(obs.first).data(), entry.second.data());
            ++residualCount;
        }
    }
    if (residualCount == 0)
        return false;

    // Hard-fix every border keyframe (the anchors). Also fix kf#0 whenever
    // it's in the problem (world origin). If somehow no border keyframe
    // co-observes (rare, very early in a run), fix the oldest in-window
    // keyframe so the gauge is still pinned.
    for (int kf : fixedKfs)
        problem.SetParameterBlockConstant(poseBlocks.at(kf).data());
    if (poseBlocks.find(0) != poseBlocks.end())
        problem.SetParameterBlockConstant(poseBlocks.at(0).data());
    if (fixedKfs.empty() && poseBlocks.find(0) == poseBlocks.end() &&
        poseBlocks.find(windowStart) != poseBlocks.end())
        problem.SetParameterBlockConstant(poseBlocks.at(windowStart).data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = kBaMaxIterations;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1; // pinned for run-to-run reproducibility
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::JACOBI;
        ceres::Solve(options, &problem, &summary);
    }
    if (!summary.IsSolutionUsable()) {
        std::fprintf(stderr, "[localba][hardanchor] did not converge, falling back: %s\n",
                     summary.BriefReport().c_str());
        return false;
    }

    auto poseToRT = [](const std::array<double, 6> &block, cv::Mat &R, cv::Mat &t) {
        const cv::Mat rvec = (cv::Mat_<double>(3, 1) << block[0], block[1], block[2]);
        cv::Rodrigues(rvec, R);
        t = (cv::Mat_<double>(3, 1) << block[3], block[4], block[5]);
    };
    // Write back ONLY the in-window (free) keyframe poses -- fixed anchors
    // were held constant and must stay byte-identical.
    for (int i = windowStart; i < n; ++i) {
        const auto it = poseBlocks.find(i);
        if (it == poseBlocks.end())
            continue;
        cv::Mat R, t;
        poseToRT(it->second, R, t);
        m_keyframeHistory[static_cast<size_t>(i)].R = R;
        m_keyframeHistory[static_cast<size_t>(i)].t = t;
    }

    std::unordered_map<long long, size_t> mapIndexById;
    mapIndexById.reserve(m_mapPointIds.size());
    for (size_t i = 0; i < m_mapPointIds.size(); ++i)
        mapIndexById[m_mapPointIds[i]] = i;
    for (const auto &entry : landmarks) {
        const cv::Point3f newPos(static_cast<float>(entry.second[0]), static_cast<float>(entry.second[1]),
                                  static_cast<float>(entry.second[2]));
        m_landmarkPositions[entry.first] = newPos;
        const auto mapIt = mapIndexById.find(entry.first);
        if (mapIt != mapIndexById.end())
            m_mapPoints[mapIt->second] = newPos;
        for (int i = windowStart; i < n; ++i) {
            Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
            const auto idIt = std::find(kf.localMapPointIds.begin(), kf.localMapPointIds.end(), entry.first);
            if (idIt != kf.localMapPointIds.end()) {
                const size_t localIdx = static_cast<size_t>(idIt - kf.localMapPointIds.begin());
                kf.localMapPoints[localIdx] = newPos;
                break;
            }
        }
    }

    const Keyframe &latest = m_keyframeHistory.back();
    m_refR = latest.R.clone();
    m_refT = latest.t.clone();
    m_currR = latest.R.clone();
    m_currT = latest.t.clone();
    if (!m_trajectory.isEmpty()) {
        const cv::Mat C = -latest.R.t() * latest.t;
        m_trajectory.last() = QPointF(C.at<double>(0), C.at<double>(2));
    }

    std::fprintf(stderr,
                  "[localba][hardanchor] window kf#%d..kf#%d, %d fixed-anchor kfs, %d landmarks, %d "
                  "observations, initial cost=%.3f final cost=%.3f\n",
                  windowStart, n - 1, static_cast<int>(fixedKfs.size()), static_cast<int>(landmarks.size()),
                  residualCount, summary.initial_cost, summary.final_cost);
    return true;
}

bool SlamWorker::runLoopBundleAdjustment(int oldKfIdx, int newKfIdx, const cv::Mat &loopR, const cv::Mat &loopT,
                                          const std::unordered_set<long long> &loopVerifiedIds)
{
    if (newKfIdx <= oldKfIdx + 1)
        return false; // nothing strictly between the two endpoints to optimize
    if (newKfIdx - oldKfIdx + 1 > kBaMaxWindowKeyframes)
        return false; // too large to solve in reasonable time -- see kBaMaxWindowKeyframes

    // Any landmark with AT LEAST ONE observation from an in-window keyframe
    // -- not just ones some in-window keyframe happens to be the ORIGINAL
    // triangulator of. This function used to use the ownership-only rule
    // (triangulated by a keyframe inside [oldKfIdx, newKfIdx]) that Session
    // 15 item 8 found silently starving runLocalBundleAdjustment() of real,
    // already-recorded multi-view constraint density -- same bug, same
    // fix, applied here since loop closures are the highest-leverage
    // moment for correcting drift (see DEBUGGING.md's queued next steps).
    // Observations themselves are still strictly bounded to the window
    // (the inner loop below), so this doesn't pull in any extra keyframe
    // pose parameter blocks, only more per-landmark residuals among the
    // ones already in the problem. processedLandmarkIds dedups: the SAME
    // landmark can now legitimately appear in several in-window keyframes'
    // own observed-landmark lists.
    std::unordered_map<long long, std::vector<std::pair<int, cv::Point2f>>> windowObservations;
    std::unordered_set<long long> processedLandmarkIds;
    for (int i = oldKfIdx; i <= newKfIdx; ++i) {
        for (long long id : m_keyframeObservedLandmarkIds[static_cast<size_t>(i)]) {
            if (!processedLandmarkIds.insert(id).second)
                continue;
            const auto it = m_landmarkObservations.find(id);
            if (it == m_landmarkObservations.end())
                continue;
            auto &dst = windowObservations[id];
            for (const auto &obs : it->second) {
                if (obs.first >= oldKfIdx && obs.first <= newKfIdx)
                    dst.push_back(obs);
            }
        }
    }

    const int windowSize = newKfIdx - oldKfIdx + 1;
    std::vector<std::array<double, 6>> poses(static_cast<size_t>(windowSize));
    auto fillPose = [](std::array<double, 6> &block, const cv::Mat &R, const cv::Mat &t) {
        cv::Mat rvec;
        cv::Rodrigues(R, rvec);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(k)] = rvec.at<double>(k);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(3 + k)] = t.at<double>(k);
    };
    // NOTE: a "smooth warm-start" variant was tried here (spreading a
    // fractional version of the endpoint's rigid correction across every
    // intermediate keyframe's initial guess, matching
    // runGlobalBundleAdjustment()'s own copy of this idea) and measured
    // WORSE (142.586m vs this function's own 93.851m baseline) -- reverted.
    // Root cause: for this function's SHORT window, each intermediate
    // keyframe's raw live pose is already a good warm start (continuous
    // local BA/guided search during live tracking keeps neighboring poses
    // locally consistent); replacing it with a single global rigid-drift
    // interpolation discards that real, already-accurate local structure
    // in favor of a cruder uniform-drift assumption, which only pays off
    // when the ORIGINAL warm start is itself badly inconsistent (confirmed
    // true for runGlobalBundleAdjustment()'s much longer span, not here).
    for (int i = oldKfIdx; i <= newKfIdx; ++i) {
        const Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
        fillPose(poses[static_cast<size_t>(i - oldKfIdx)], kf.R, kf.t);
    }
    fillPose(poses.back(), loopR, loopT); // endpoint anchored to the loop measurement, not its drifted pose

    const CameraIntrinsics &intr = m_intrinsics;
    std::unordered_map<long long, std::array<double, 3>> landmarks;
    for (auto &entry : windowObservations) {
        if (entry.second.size() < static_cast<size_t>(kBaMinObservationsPerLandmark))
            continue;
        const cv::Point3f &p = m_landmarkPositions.at(entry.first);
        landmarks[entry.first] = {static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)};
    }
    if (landmarks.empty())
        return false; // nothing to jointly constrain -- let the caller fall back

    ceres::Problem problem;
    for (int i = 0; i < windowSize; ++i)
        problem.AddParameterBlock(poses[static_cast<size_t>(i)].data(), 6);

    int residualCount = 0;
    int verifiedResidualCount = 0;
    for (auto &entry : landmarks) {
        const auto &obsList = windowObservations.at(entry.first);
        for (const auto &obs : obsList) {
            ceres::CostFunction *cost =
                ReprojectionCost::Create(obs.second.x, obs.second.y, intr.fx, intr.fy, intr.cx, intr.cy);
            // The one observation that's the actual PnP-RANSAC-verified
            // loop correspondence (oldKf's landmark re-observed at newKfIdx,
            // exactly what measured loopR/loopT) gets trusted fully instead
            // of being just another Huber-lossed observation among
            // thousands of short local tracks -- see kLoopVerifiedResidualWeight's
            // doc comment for why.
            const bool isLoopVerified = obs.first == newKfIdx && loopVerifiedIds.count(entry.first) > 0;
            ceres::LossFunction *loss =
                isLoopVerified ? new ceres::ScaledLoss(nullptr, kLoopVerifiedResidualWeight, ceres::TAKE_OWNERSHIP)
                                : static_cast<ceres::LossFunction *>(new ceres::HuberLoss(kBaHuberDeltaPixels));
            problem.AddResidualBlock(cost, loss, poses[static_cast<size_t>(obs.first - oldKfIdx)].data(),
                                      entry.second.data());
            ++residualCount;
            verifiedResidualCount += isLoopVerified ? 1 : 0;
        }
    }
    if (residualCount == 0)
        return false;

    problem.SetParameterBlockConstant(poses.front().data());
    problem.SetParameterBlockConstant(poses.back().data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = kBaMaxIterations;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1; // pinned for run-to-run reproducibility -- see kitti_ate.cpp's
    // own cv::setNumThreads(1) call for the matching OpenCV-side fix
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        // Ceres may not have been built with a sparse direct solver --
        // iterative Schur + Jacobi preconditioning needs no optional
        // dependency and always works, just converges a bit slower.
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::JACOBI;
        ceres::Solve(options, &problem, &summary);
    }
    if (!summary.IsSolutionUsable()) {
        std::fprintf(stderr, "[ba] loop bundle adjustment did not converge, falling back: %s\n",
                     summary.BriefReport().c_str());
        return false;
    }

    std::fprintf(stderr,
                  "[ba] loop bundle adjustment: kf#%d..kf#%d, %d landmarks, %d observations "
                  "(%d loop-verified), initial cost=%.3f final cost=%.3f\n",
                  oldKfIdx, newKfIdx, static_cast<int>(landmarks.size()), residualCount, verifiedResidualCount,
                  summary.initial_cost, summary.final_cost);

    // Write the optimized poses back -- oldKfIdx is untouched (it was held
    // constant at its own already-trusted pose), newKfIdx is set to the
    // loop-measured pose (also held constant, but the keyframe's stored
    // pose still needs updating from its old drifted value), everything
    // strictly between gets its jointly-optimized pose.
    auto poseToRT = [](const std::array<double, 6> &block, cv::Mat &R, cv::Mat &t) {
        const cv::Mat rvec = (cv::Mat_<double>(3, 1) << block[0], block[1], block[2]);
        cv::Rodrigues(rvec, R);
        t = (cv::Mat_<double>(3, 1) << block[3], block[4], block[5]);
    };
    for (int i = oldKfIdx + 1; i <= newKfIdx; ++i) {
        cv::Mat R, t;
        poseToRT(poses[static_cast<size_t>(i - oldKfIdx)], R, t);
        m_keyframeHistory[static_cast<size_t>(i)].R = R;
        m_keyframeHistory[static_cast<size_t>(i)].t = t;
    }

    // Write optimized landmark positions back: the authoritative registry,
    // the owning keyframe's own copy (so a future loop closure's PnP
    // against it -- see tryLoopClosure() -- uses the refined position),
    // and the live rolling map for any landmark still present there.
    std::unordered_map<long long, size_t> mapIndexById;
    mapIndexById.reserve(m_mapPointIds.size());
    for (size_t i = 0; i < m_mapPointIds.size(); ++i)
        mapIndexById[m_mapPointIds[i]] = i;

    for (const auto &entry : landmarks) {
        const cv::Point3f newPos(static_cast<float>(entry.second[0]), static_cast<float>(entry.second[1]),
                                  static_cast<float>(entry.second[2]));
        m_landmarkPositions[entry.first] = newPos;

        const auto mapIt = mapIndexById.find(entry.first);
        if (mapIt != mapIndexById.end())
            m_mapPoints[mapIt->second] = newPos;

        for (int i = oldKfIdx; i <= newKfIdx; ++i) {
            Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
            const auto idIt = std::find(kf.localMapPointIds.begin(), kf.localMapPointIds.end(), entry.first);
            if (idIt != kf.localMapPointIds.end()) {
                const size_t localIdx = static_cast<size_t>(idIt - kf.localMapPointIds.begin());
                kf.localMapPoints[localIdx] = newPos;
                break; // a landmark has exactly one owning keyframe
            }
        }
    }

    return true;
}

bool SlamWorker::runPoseGraphThenGlobalBundleAdjustment(int newKfIdx, const cv::Mat &loopR, const cv::Mat &loopT,
                                                          const std::unordered_set<long long> &loopVerifiedIds)
{
    const std::vector<pose_graph::LoopClosureRecord> &loops = loopClosureRecords();
    if (loops.empty())
        return runGlobalBundleAdjustment(newKfIdx, loopR, loopT, loopVerifiedIds); // nothing for the
                                                                                     // pose graph to
                                                                                     // correct yet

    std::vector<pose_graph::KeyframePose> keyframes = keyframePoses();
    std::vector<pose_graph::SequentialEdgeRecord> sequential = sequentialEdgeRecords();
    const std::vector<pose_graph::SequentialEdgeRecord> covis = covisibilityEdgeRecords(m_covisibilityMinShared);
    sequential.insert(sequential.end(), covis.begin(), covis.end());

    pose_graph::PoseGraphOptions pgOptions;
    pgOptions.useSim3 = true; // matches ORB-SLAM3's real essential-graph Sim3 correction (7-DOF,
                              // scale-aware) -- see PoseGraphOptions::useSim3's own doc comment
    std::vector<pose_graph::KeyframePose> warmStart;
    if (pose_graph::optimizePoseGraph(keyframes, sequential, loops, pgOptions, &warmStart)) {
        for (size_t i = 0; i < keyframes.size() && i < m_keyframeHistory.size(); ++i) {
            m_keyframeHistory[i].R = keyframes[i].R.clone();
            m_keyframeHistory[i].t = keyframes[i].t.clone();
        }
        std::fprintf(stderr,
                      "[globalba][posegraph] essential-graph-style Sim3 correction applied to %zu keyframes "
                      "(%zu sequential+covisibility edges, %zu loop edges), now polishing with global BA\n",
                      keyframes.size(), sequential.size(), loops.size());
    } else {
        std::fprintf(stderr, "[globalba][posegraph] pose-graph solve failed, skipping straight to global BA\n");
    }

    // Global BA's own warm-start (item 30's v2 fix) reads m_keyframeHistory's
    // CURRENT poses -- now the pose-graph-corrected ones if the solve above
    // succeeded, so it polishes an already-corrected map instead of a raw
    // drifted one, mirroring real ORB-SLAM3's CorrectLoop()-then-GBA order.
    return runGlobalBundleAdjustment(newKfIdx, loopR, loopT, loopVerifiedIds);
}

bool SlamWorker::runGlobalBundleAdjustment(int newKfIdx, const cv::Mat &loopR, const cv::Mat &loopT,
                                            const std::unordered_set<long long> &loopVerifiedIds)
{
    if (newKfIdx < 1)
        return false; // nothing before it to jointly optimize against
    if (!m_globalBaSchurEnabled && newKfIdx + 1 > kGlobalBaMaxWindowKeyframes)
        return false; // too large for the plain SPARSE_NORMAL_CHOLESKY solver -- see
                       // kGlobalBaMaxWindowKeyframes. With Schur marginalization on
                       // (m_globalBaSchurEnabled, item 35) this cap is lifted: the whole map
                       // becomes tractable in one solve, which is the entire point of Schur here.
    if (m_globalBundleAdjustmentAsyncEnabled && m_pendingGlobalBaValid)
        return false; // a previous solve hasn't been integrated yet -- mirrors ORB-SLAM3's own
                       // mbRunningGBA guard against overlapping background solves (LoopClosing.cc)

    // Same rule as runLoopBundleAdjustment() (see its own doc comment,
    // DEBUGGING.md item 10/27's queued item 7): any landmark with AT LEAST
    // ONE observation from an in-window keyframe, not just ones some
    // in-window keyframe happens to be the ORIGINAL triangulator of
    // (m_keyframeHistory[i].localMapPointIds -- the stale ownership-only
    // rule local/loop BA both used to have, fixed there in items 8/10 with
    // a real measured win each time, never applied here). Spans the WHOLE
    // map ([0, newKfIdx]) instead of one loop's window.
    // processedLandmarkIds dedups: the SAME landmark can legitimately
    // appear in several in-window keyframes' own observed-landmark lists.
    std::unordered_map<long long, std::vector<std::pair<int, cv::Point2f>>> windowObservations;
    std::unordered_set<long long> processedLandmarkIds;
    for (int i = 0; i <= newKfIdx; ++i) {
        for (long long id : m_keyframeObservedLandmarkIds[static_cast<size_t>(i)]) {
            if (!processedLandmarkIds.insert(id).second)
                continue;
            const auto it = m_landmarkObservations.find(id);
            if (it == m_landmarkObservations.end())
                continue;
            auto &dst = windowObservations[id];
            for (const auto &obs : it->second) {
                if (obs.first >= 0 && obs.first <= newKfIdx)
                    dst.push_back(obs);
            }
        }
    }

    const int windowSize = newKfIdx + 1;
    std::vector<std::array<double, 6>> poses(static_cast<size_t>(windowSize));
    auto fillPose = [](std::array<double, 6> &block, const cv::Mat &R, const cv::Mat &t) {
        cv::Mat rvec;
        cv::Rodrigues(R, rvec);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(k)] = rvec.at<double>(k);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(3 + k)] = t.at<double>(k);
    };
    // Live-pose warm-start (v2, see DEBUGGING.md item 29) -- replaces the
    // earlier "smooth" warm-start (a single rigid drift transform spread
    // LINEARLY across every intermediate keyframe, alpha=i/newKfIdx),
    // which real ORB-SLAM3's own Optimizer::BundleAdjustment() (see
    // third_party/ORB_SLAM3/src/Optimizer.cc) does NOT do: it fixes only
    // the map's origin keyframe and initializes every OTHER keyframe from
    // its own current pose estimate, trusting Levenberg-Marquardt to
    // converge from there instead of assuming drift accrued as one uniform
    // rigid transform (false in general -- real drift accrues unevenly,
    // faster through turns/low-texture stretches than straight, well-
    // textured ones). Each intermediate keyframe here is now warm-started
    // from its own live R/t verbatim, same principle. The two-hard-anchor
    // scheme below (kf#0 AND newKfIdx both held constant, not just kf#0
    // like ORB-SLAM3) is intentionally kept as-is -- it has its own
    // separate justification (monocular global BA's scale/gauge ambiguity
    // without an external anchor) and its own measured history, not part
    // of this change.
    for (int i = 0; i < newKfIdx; ++i) {
        const Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
        fillPose(poses[static_cast<size_t>(i)], kf.R, kf.t);
    }
    fillPose(poses[static_cast<size_t>(newKfIdx)], loopR, loopT);

    const CameraIntrinsics &intr = m_intrinsics;
    std::unordered_map<long long, std::array<double, 3>> landmarks;
    for (auto &entry : windowObservations) {
        if (entry.second.size() < static_cast<size_t>(kBaMinObservationsPerLandmark))
            continue;
        const cv::Point3f &p = m_landmarkPositions.at(entry.first);
        landmarks[entry.first] = {static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)};
    }
    if (landmarks.empty())
        return false;

    ceres::Problem problem;
    for (int i = 0; i < windowSize; ++i)
        problem.AddParameterBlock(poses[static_cast<size_t>(i)].data(), 6);

    int residualCount = 0;
    int verifiedResidualCount = 0;
    for (auto &entry : landmarks) {
        const auto &obsList = windowObservations.at(entry.first);
        for (const auto &obs : obsList) {
            ceres::CostFunction *cost =
                ReprojectionCost::Create(obs.second.x, obs.second.y, intr.fx, intr.fy, intr.cx, intr.cy);
            const bool isLoopVerified = obs.first == newKfIdx && loopVerifiedIds.count(entry.first) > 0;
            ceres::LossFunction *loss =
                isLoopVerified ? new ceres::ScaledLoss(nullptr, kLoopVerifiedResidualWeight, ceres::TAKE_OWNERSHIP)
                                : static_cast<ceres::LossFunction *>(new ceres::HuberLoss(kBaHuberDeltaPixels));
            problem.AddResidualBlock(cost, loss, poses[static_cast<size_t>(obs.first)].data(), entry.second.data());
            ++residualCount;
            verifiedResidualCount += isLoopVerified ? 1 : 0;
        }
    }
    if (residualCount == 0)
        return false;

    // Hard anchor at keyframe 0 (world origin) always -- pure gauge-fixing,
    // no ORB-SLAM3 analogue to diverge from. newKfIdx (the independently
    // PnP-verified loop pose) is EITHER also hard-anchored (default,
    // "two hard anchors spanning the whole graph" -- see
    // kGlobalBaMaxWindowKeyframes's doc comment for why this doesn't risk
    // continuous local BA's scale-collapse failure mode) OR, when
    // setGlobalBaSoftLoopAnchorEnabled() is on (DEBUGGING.md item 32, v3),
    // left FREE and pulled toward the same loop measurement by a soft
    // PosePriorCost residual instead -- matching real ORB-SLAM3's own
    // choice to never hard-pin a loop pose inside global BA itself (only
    // the origin keyframe is fixed there).
    problem.SetParameterBlockConstant(poses.front().data());
    if (m_globalBaSoftLoopAnchorEnabled) {
        problem.AddResidualBlock(
            PosePriorCost::Create(poses.back(), kGlobalBaLoopPosePriorRotWeight, kGlobalBaLoopPosePriorTransWeight),
            nullptr, poses.back().data());
    } else {
        problem.SetParameterBlockConstant(poses.back().data());
    }

    ceres::Solver::Options options;
    options.max_num_iterations = kBaMaxIterations;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1; // pinned for run-to-run reproducibility -- see kitti_ate.cpp's
    // own cv::setNumThreads(1) call for the matching OpenCV-side fix
    if (m_globalBaSchurEnabled) {
        // DEBUGGING.md item 35: Ceres' direct equivalent of real ORB-SLAM3's
        // g2o::BlockSolver_6_3 + vPoint->setMarginalized(true) -- SPARSE_SCHUR
        // with the landmark parameter blocks placed in elimination group 0
        // (Schur-complemented out first), poses in group 1. Same exact
        // optimum as SPARSE_NORMAL_CHOLESKY (Schur is a reordering of the
        // solve, not a different objective), but exploits the pose/landmark
        // block structure so it scales to the whole map instead of needing
        // the kGlobalBaMaxWindowKeyframes cap.
        auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
        for (auto &entry : landmarks)
            ordering->AddElementToGroup(entry.second.data(), 0);
        for (int i = 0; i < windowSize; ++i)
            ordering->AddElementToGroup(poses[static_cast<size_t>(i)].data(), 1);
        options.linear_solver_ordering = ordering;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::JACOBI;
        options.linear_solver_ordering.reset();
        ceres::Solve(options, &problem, &summary);
    }
    if (!summary.IsSolutionUsable()) {
        std::fprintf(stderr, "[globalba] did not converge, falling back: %s\n", summary.BriefReport().c_str());
        return false;
    }

    std::fprintf(stderr,
                  "[globalba] kf#0..kf#%d, %d landmarks, %d observations (%d loop-verified), "
                  "initial cost=%.3f final cost=%.3f\n",
                  newKfIdx, static_cast<int>(landmarks.size()), residualCount, verifiedResidualCount,
                  summary.initial_cost, summary.final_cost);

    auto poseToRT = [](const std::array<double, 6> &block, cv::Mat &R, cv::Mat &t) {
        const cv::Mat rvec = (cv::Mat_<double>(3, 1) << block[0], block[1], block[2]);
        cv::Rodrigues(rvec, R);
        t = (cv::Mat_<double>(3, 1) << block[3], block[4], block[5]);
    };

    if (m_globalBundleAdjustmentAsyncEnabled) {
        // See setGlobalBundleAdjustmentAsyncEnabled(): queue the solved
        // result instead of writing it now -- tryIntegratePendingGlobalBa()
        // applies it kGlobalBaIntegrationDelayKeyframes keyframes later.
        m_pendingGlobalBaR.assign(static_cast<size_t>(newKfIdx + 1), cv::Mat());
        m_pendingGlobalBaT.assign(static_cast<size_t>(newKfIdx + 1), cv::Mat());
        poseToRT(poses.front(), m_pendingGlobalBaR[0], m_pendingGlobalBaT[0]);
        for (int i = 1; i < newKfIdx; ++i)
            poseToRT(poses[static_cast<size_t>(i)], m_pendingGlobalBaR[static_cast<size_t>(i)],
                     m_pendingGlobalBaT[static_cast<size_t>(i)]);
        poseToRT(poses.back(), m_pendingGlobalBaR[static_cast<size_t>(newKfIdx)],
                 m_pendingGlobalBaT[static_cast<size_t>(newKfIdx)]);

        m_pendingGlobalBaLandmarks.clear();
        for (const auto &entry : landmarks) {
            m_pendingGlobalBaLandmarks[entry.first] =
                cv::Point3f(static_cast<float>(entry.second[0]), static_cast<float>(entry.second[1]),
                            static_cast<float>(entry.second[2]));
        }

        m_pendingGlobalBaTriggerKfIdx = newKfIdx;
        m_pendingGlobalBaIntegrateAtKfIdx = newKfIdx + kGlobalBaIntegrationDelayKeyframes;
        m_pendingGlobalBaValid = true;
        std::fprintf(stderr, "[globalba][async] solved at kf#%d, deferring integration to kf#%d\n", newKfIdx,
                     m_pendingGlobalBaIntegrateAtKfIdx);
        return true;
    }

    for (int i = 1; i < newKfIdx; ++i) { // 0 and newKfIdx were held constant above -- unchanged
        cv::Mat R, t;
        poseToRT(poses[static_cast<size_t>(i)], R, t);
        m_keyframeHistory[static_cast<size_t>(i)].R = R;
        m_keyframeHistory[static_cast<size_t>(i)].t = t;
    }

    std::unordered_map<long long, size_t> mapIndexById;
    mapIndexById.reserve(m_mapPointIds.size());
    for (size_t i = 0; i < m_mapPointIds.size(); ++i)
        mapIndexById[m_mapPointIds[i]] = i;

    for (const auto &entry : landmarks) {
        const cv::Point3f newPos(static_cast<float>(entry.second[0]), static_cast<float>(entry.second[1]),
                                  static_cast<float>(entry.second[2]));
        m_landmarkPositions[entry.first] = newPos;

        const auto mapIt = mapIndexById.find(entry.first);
        if (mapIt != mapIndexById.end())
            m_mapPoints[mapIt->second] = newPos;

        for (int i = 0; i <= newKfIdx; ++i) {
            Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
            const auto idIt = std::find(kf.localMapPointIds.begin(), kf.localMapPointIds.end(), entry.first);
            if (idIt != kf.localMapPointIds.end()) {
                const size_t localIdx = static_cast<size_t>(idIt - kf.localMapPointIds.begin());
                kf.localMapPoints[localIdx] = newPos;
                break; // a landmark has exactly one owning keyframe
            }
        }
    }

    // DEBUGGING.md item 33 (v4): user's hypothesis -- continuous local BA
    // re-solves the SAME trailing window many times over successive
    // keyframe insertions before any given loop closure, so by the time it
    // fires those poses are already well-converged; global BA gets exactly
    // ONE shot at the WHOLE map, including old keyframes no optimizer has
    // touched in a long time, and has to do in one solve what local BA
    // achieves through dozens of incremental re-solves. Re-running
    // literally the SAME residuals again is a no-op (Ceres LM already
    // iterates internally to convergence) -- instead, immediately follow
    // global BA with ONE call to the existing, independently-proven
    // runLocalBundleAdjustment() over the trailing window, letting it
    // "polish" the just-corrected recent keyframes with its own
    // already-validated mechanism (pose-prior regularizer, its own
    // observation density fix) rather than trusting global BA's one-shot
    // result as final for the most tracking-critical (most recent) part of
    // the map.
    if (m_globalBaPolishEnabled)
        runLocalBundleAdjustment();

    return true;
}

bool SlamWorker::runGapBundleAdjustment(int anchorKfIdx, int endKfIdx)
{
    if (endKfIdx <= anchorKfIdx)
        return false;
    if (endKfIdx - anchorKfIdx + 1 > kBaMaxWindowKeyframes)
        return false; // too large -- same runtime bound loop-BA uses

    // Same dense any-observation-in-window rule as loop/global BA (items
    // 8/10/28) -- see their own doc comments.
    std::unordered_map<long long, std::vector<std::pair<int, cv::Point2f>>> windowObservations;
    std::unordered_set<long long> processedLandmarkIds;
    for (int i = anchorKfIdx; i <= endKfIdx; ++i) {
        for (long long id : m_keyframeObservedLandmarkIds[static_cast<size_t>(i)]) {
            if (!processedLandmarkIds.insert(id).second)
                continue;
            const auto it = m_landmarkObservations.find(id);
            if (it == m_landmarkObservations.end())
                continue;
            auto &dst = windowObservations[id];
            for (const auto &obs : it->second) {
                if (obs.first >= anchorKfIdx && obs.first <= endKfIdx)
                    dst.push_back(obs);
            }
        }
    }

    const int windowSize = endKfIdx - anchorKfIdx + 1;
    std::vector<std::array<double, 6>> poses(static_cast<size_t>(windowSize));
    auto fillPose = [](std::array<double, 6> &block, const cv::Mat &R, const cv::Mat &t) {
        cv::Mat rvec;
        cv::Rodrigues(R, rvec);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(k)] = rvec.at<double>(k);
        for (int k = 0; k < 3; ++k)
            block[static_cast<size_t>(3 + k)] = t.at<double>(k);
    };
    // Warm-started from each keyframe's OWN current pose -- for the gap
    // keyframes that's whatever tryIntegratePendingGlobalBa()'s rigid-delta
    // chain propagation already put there (a starting guess, not trusted
    // as final -- that's the whole point of this function existing).
    for (int i = anchorKfIdx; i <= endKfIdx; ++i) {
        const Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
        fillPose(poses[static_cast<size_t>(i - anchorKfIdx)], kf.R, kf.t);
    }

    const CameraIntrinsics &intr = m_intrinsics;
    std::unordered_map<long long, std::array<double, 3>> landmarks;
    for (auto &entry : windowObservations) {
        if (entry.second.size() < static_cast<size_t>(kBaMinObservationsPerLandmark))
            continue;
        const cv::Point3f &p = m_landmarkPositions.at(entry.first);
        landmarks[entry.first] = {static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z)};
    }
    if (landmarks.empty())
        return false;

    ceres::Problem problem;
    for (int i = 0; i < windowSize; ++i)
        problem.AddParameterBlock(poses[static_cast<size_t>(i)].data(), 6);

    int residualCount = 0;
    for (auto &entry : landmarks) {
        const auto &obsList = windowObservations.at(entry.first);
        for (const auto &obs : obsList) {
            ceres::CostFunction *cost =
                ReprojectionCost::Create(obs.second.x, obs.second.y, intr.fx, intr.fy, intr.cx, intr.cy);
            ceres::LossFunction *loss = new ceres::HuberLoss(kBaHuberDeltaPixels);
            problem.AddResidualBlock(cost, loss, poses[static_cast<size_t>(obs.first - anchorKfIdx)].data(),
                                      entry.second.data());
            ++residualCount;
        }
    }
    if (residualCount == 0)
        return false;

    // ONLY ONE hard anchor -- anchorKfIdx, held at its just-integrated
    // corrected pose. Unlike loop-BA/global-BA's double anchor, endKfIdx is
    // a free parameter: there is no independent measurement of its "true"
    // pose here (that's the whole gap-keyframe problem), so it must be
    // solved for, not pinned.
    problem.SetParameterBlockConstant(poses.front().data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = kBaMaxIterations;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 1; // pinned for run-to-run reproducibility
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::JACOBI;
        ceres::Solve(options, &problem, &summary);
    }
    if (!summary.IsSolutionUsable()) {
        std::fprintf(stderr, "[globalba][async][gapba] did not converge, falling back: %s\n",
                     summary.BriefReport().c_str());
        return false;
    }

    std::fprintf(stderr,
                  "[globalba][async][gapba] kf#%d..kf#%d, %d landmarks, %d observations, "
                  "initial cost=%.3f final cost=%.3f\n",
                  anchorKfIdx, endKfIdx, static_cast<int>(landmarks.size()), residualCount,
                  summary.initial_cost, summary.final_cost);

    auto poseToRT = [](const std::array<double, 6> &block, cv::Mat &R, cv::Mat &t) {
        const cv::Mat rvec = (cv::Mat_<double>(3, 1) << block[0], block[1], block[2]);
        cv::Rodrigues(rvec, R);
        t = (cv::Mat_<double>(3, 1) << block[3], block[4], block[5]);
    };
    for (int i = anchorKfIdx + 1; i <= endKfIdx; ++i) { // anchorKfIdx held constant -- unchanged
        cv::Mat R, t;
        poseToRT(poses[static_cast<size_t>(i - anchorKfIdx)], R, t);
        m_keyframeHistory[static_cast<size_t>(i)].R = R;
        m_keyframeHistory[static_cast<size_t>(i)].t = t;
    }

    std::unordered_map<long long, size_t> mapIndexById;
    mapIndexById.reserve(m_mapPointIds.size());
    for (size_t i = 0; i < m_mapPointIds.size(); ++i)
        mapIndexById[m_mapPointIds[i]] = i;

    for (const auto &entry : landmarks) {
        const cv::Point3f newPos(static_cast<float>(entry.second[0]), static_cast<float>(entry.second[1]),
                                  static_cast<float>(entry.second[2]));
        m_landmarkPositions[entry.first] = newPos;
        const auto mapIt = mapIndexById.find(entry.first);
        if (mapIt != mapIndexById.end())
            m_mapPoints[mapIt->second] = newPos;
        for (int i = anchorKfIdx; i <= endKfIdx; ++i) {
            Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
            const auto idIt = std::find(kf.localMapPointIds.begin(), kf.localMapPointIds.end(), entry.first);
            if (idIt != kf.localMapPointIds.end()) {
                const size_t localIdx = static_cast<size_t>(idIt - kf.localMapPointIds.begin());
                kf.localMapPoints[localIdx] = newPos;
                break;
            }
        }
    }

    return true;
}

void SlamWorker::tryIntegratePendingGlobalBa()
{
    if (!m_pendingGlobalBaValid)
        return;
    const int currentCount = static_cast<int>(m_keyframeHistory.size());
    if (currentCount < m_pendingGlobalBaIntegrateAtKfIdx)
        return; // still "solving" -- see setGlobalBundleAdjustmentAsyncEnabled()

    const int trigger = m_pendingGlobalBaTriggerKfIdx;

    // Snapshot the anchor keyframe's CURRENT (pre-integration) camera
    // center/orientation before overwriting it -- this is the delta gap
    // keyframes (trigger, currentCount) get propagated by below, mirroring
    // ORB-SLAM3's own child-correction composition (LoopClosing.cc's
    // Tchildc = pChild->GetPose() * Twc, generalized here to a single
    // rigid delta since this codebase has no parent/child keyframe graph
    // to walk).
    const cv::Mat RcwOld = m_keyframeHistory[static_cast<size_t>(trigger)].R.t();
    const cv::Mat COld = -RcwOld * m_keyframeHistory[static_cast<size_t>(trigger)].t;

    for (int i = 0; i <= trigger; ++i) {
        m_keyframeHistory[static_cast<size_t>(i)].R = m_pendingGlobalBaR[static_cast<size_t>(i)];
        m_keyframeHistory[static_cast<size_t>(i)].t = m_pendingGlobalBaT[static_cast<size_t>(i)];
    }

    const cv::Mat RcwNew = m_pendingGlobalBaR[static_cast<size_t>(trigger)].t();
    const cv::Mat CNew = -RcwNew * m_pendingGlobalBaT[static_cast<size_t>(trigger)];
    const cv::Mat Rdelta = RcwNew * RcwOld.t();
    const cv::Mat tDeltaC = CNew - Rdelta * COld;

    int propagatedCount = 0;
    for (int i = trigger + 1; i < currentCount; ++i) {
        Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
        const cv::Mat Rcw = kf.R.t();
        const cv::Mat C = -Rcw * kf.t;
        const cv::Mat Cwarm = Rdelta * C + tDeltaC;
        const cv::Mat RcwWarm = Rdelta * Rcw;
        const cv::Mat Rwarm = RcwWarm.t();
        const cv::Mat twarm = -Rwarm * Cwarm;
        kf.R = Rwarm;
        kf.t = twarm;
        ++propagatedCount;
    }

    std::unordered_map<long long, size_t> mapIndexById;
    mapIndexById.reserve(m_mapPointIds.size());
    for (size_t i = 0; i < m_mapPointIds.size(); ++i)
        mapIndexById[m_mapPointIds[i]] = i;

    for (const auto &entry : m_pendingGlobalBaLandmarks) {
        m_landmarkPositions[entry.first] = entry.second;
        const auto mapIt = mapIndexById.find(entry.first);
        if (mapIt != mapIndexById.end())
            m_mapPoints[mapIt->second] = entry.second;
        for (int i = 0; i <= trigger; ++i) {
            Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
            const auto idIt = std::find(kf.localMapPointIds.begin(), kf.localMapPointIds.end(), entry.first);
            if (idIt != kf.localMapPointIds.end()) {
                const size_t localIdx = static_cast<size_t>(idIt - kf.localMapPointIds.begin());
                kf.localMapPoints[localIdx] = entry.second;
                break;
            }
        }
    }

    std::fprintf(stderr,
                  "[globalba][async] integrated pending solve (trigger kf#%d) at kf#%d, propagated to %d "
                  "gap keyframes, %zu landmarks\n",
                  trigger, currentCount - 1, propagatedCount, m_pendingGlobalBaLandmarks.size());

    // DEBUGGING.md item 31: the rigid-delta propagation above is only a
    // WARM START, not trusted as final (item 30 measured the rigid patch
    // alone as a net regression, 64.667m -> 136.095m) -- now actually
    // re-optimize the gap keyframes against real reprojection residuals,
    // anchored at the just-corrected trigger keyframe.
    if (m_globalBaGapRefinementEnabled && currentCount - 1 > trigger)
        runGapBundleAdjustment(trigger, currentCount - 1);

    m_pendingGlobalBaValid = false;
    m_pendingGlobalBaR.clear();
    m_pendingGlobalBaT.clear();
    m_pendingGlobalBaLandmarks.clear();
}

std::vector<pose_graph::KeyframePose> SlamWorker::keyframePoses() const
{
    std::vector<pose_graph::KeyframePose> out;
    out.reserve(m_keyframeHistory.size());
    for (const Keyframe &kf : m_keyframeHistory)
        out.push_back({kf.R.clone(), kf.t.clone(), kf.frameIndex});
    return out;
}

std::vector<pose_graph::SequentialEdgeRecord> SlamWorker::covisibilityEdgeRecords(int minSharedLandmarks) const
{
    // Same covisibility-graph construction as cullRedundantKeyframes() --
    // see that function's own doc comment -- just without the culling
    // side effects, and reading every keyframe's CURRENT pose (not an
    // independent re-measurement) once at the end instead of feeding a
    // live redundancy decision.
    const int n = static_cast<int>(m_keyframeHistory.size());
    std::unordered_map<int, std::unordered_map<int, int>> covisibility;
    for (const auto &entry : m_landmarkObservations) {
        std::unordered_set<int> distinctKfs;
        for (const auto &obs : entry.second) {
            if (obs.first >= 0 && obs.first < n)
                distinctKfs.insert(obs.first);
        }
        for (auto it1 = distinctKfs.begin(); it1 != distinctKfs.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != distinctKfs.end(); ++it2) {
                covisibility[*it1][*it2]++;
                covisibility[*it2][*it1]++;
            }
        }
    }

    std::vector<pose_graph::SequentialEdgeRecord> edges;
    for (const auto &fromEntry : covisibility) {
        const int i = fromEntry.first;
        for (const auto &toEntry : fromEntry.second) {
            const int j = toEntry.first;
            // i<j once per pair (covisibility is symmetric, iterated both
            // ways above); skip anything already covered by a real
            // sequential edge (|i-j|==1) -- no need to duplicate it here.
            if (j <= i + 1)
                continue;
            if (toEntry.second < minSharedLandmarks)
                continue;
            const Keyframe &kfI = m_keyframeHistory[static_cast<size_t>(i)];
            const Keyframe &kfJ = m_keyframeHistory[static_cast<size_t>(j)];
            const cv::Mat Rrel = kfJ.R * kfI.R.t();
            const cv::Mat trel = kfJ.t - Rrel * kfI.t;
            edges.push_back({i, j, Rrel.clone(), trel.clone()});
        }
    }
    return edges;
}

LoopEstimateSnapshot SlamWorker::buildLoopEstimateSnapshot(
    int oldKfIdx, int newKfIdx, const cv::Mat &loopR, const cv::Mat &loopT,
    const std::unordered_set<long long> &loopVerifiedIds) const
{
    LoopEstimateSnapshot snapshot;
    snapshot.keyframes.reserve(static_cast<size_t>(newKfIdx - oldKfIdx + 1));
    for (int i = oldKfIdx; i <= newKfIdx; ++i) {
        const Keyframe &kf = m_keyframeHistory[static_cast<size_t>(i)];
        LoopEstimateSnapshot::KeyframeSnapshot kfSnap;
        kfSnap.R = kf.R.clone();
        kfSnap.t = kf.t.clone();
        kfSnap.descriptors = kf.descriptors.clone();
        kfSnap.keypoints = kf.keypoints;
        kfSnap.localMapPoints = kf.localMapPoints;
        kfSnap.localMapDescriptors = kf.localMapDescriptors.clone();
        kfSnap.localMapImagePoints = kf.localMapImagePoints;
        kfSnap.localMapPointIds = kf.localMapPointIds;
        kfSnap.frameIndex = kf.frameIndex;
        snapshot.keyframes.push_back(std::move(kfSnap));
    }
    snapshot.loopR = loopR.clone();
    snapshot.loopT = loopT.clone();
    snapshot.intrinsics = m_intrinsics;
    snapshot.loopVerifiedLandmarkIds = loopVerifiedIds;
    snapshot.trajectory = m_trajectory;
    snapshot.trajectoryFrameIndex = m_trajectoryFrameIndex;
    snapshot.oldFrameIndex = m_keyframeHistory[static_cast<size_t>(oldKfIdx)].frameIndex;
    snapshot.newFrameIndex = m_keyframeHistory[static_cast<size_t>(newKfIdx)].frameIndex;

    if (!m_groundTruthT.empty()) {
        snapshot.hasGroundTruth = true;
        snapshot.groundTruth.reserve(static_cast<int>(m_groundTruthT.size()));
        for (const cv::Mat &t : m_groundTruthT)
            snapshot.groundTruth.append(QPointF(t.at<double>(0), t.at<double>(2)));
    }

    return snapshot;
}

void SlamWorker::buildCovisibilityLocalMap()
{
    m_framesSinceCovisibilityMapRebuild = 0; // see its own doc comment (SlamWorker.h) -- staleness fallback
    m_localMapPoints.clear();
    m_localMapDescriptors = cv::Mat();
    m_localMapPointIds.clear();
    if (m_keyframeHistory.empty())
        return;
    const int refIdx = static_cast<int>(m_keyframeHistory.size()) - 1;

    // Reverse index: landmark ID -> distinct observing keyframes (same
    // pattern cullRedundantKeyframes() uses).
    std::unordered_map<long long, std::unordered_set<int>> observers;
    for (const auto &entry : m_landmarkObservations) {
        for (const auto &obs : entry.second)
            observers[entry.first].insert(obs.first);
    }

    std::unordered_set<long long> refLandmarks;
    for (const auto &entry : observers) {
        if (entry.second.count(refIdx))
            refLandmarks.insert(entry.first);
    }

    std::unordered_map<int, int> sharedCount;
    for (long long id : refLandmarks) {
        for (int kf : observers.at(id)) {
            if (kf != refIdx)
                ++sharedCount[kf];
        }
    }

    std::unordered_set<int> covisibleKfs = {refIdx};
    for (const auto &entry : sharedCount) {
        if (entry.second >= kCovisibilityMinSharedLandmarks)
            covisibleKfs.insert(entry.first);
    }

    std::unordered_set<long long> seen;
    for (int kf : covisibleKfs) {
        const Keyframe &k = m_keyframeHistory[static_cast<size_t>(kf)];
        for (size_t i = 0; i < k.localMapPointIds.size(); ++i) {
            const long long id = k.localMapPointIds[i];
            if (!seen.insert(id).second)
                continue;
            const auto posIt = m_landmarkPositions.find(id);
            m_localMapPoints.push_back(posIt != m_landmarkPositions.end() ? posIt->second
                                                                            : k.localMapPoints[i]);
            m_localMapPointIds.push_back(id);
            m_localMapDescriptors.push_back(k.localMapDescriptors.row(static_cast<int>(i)));
        }
    }

    std::fprintf(stderr, "[covismap] ref=kf#%d: %zu covisible keyframes, %zu local map points\n", refIdx,
                 covisibleKfs.size(), m_localMapPoints.size());
}

void SlamWorker::cullRedundantKeyframes()
{
    const int n = static_cast<int>(m_keyframeHistory.size());
    if (n <= kCullingExclusionWindow + 1) // +1: never touch keyframe 0, the world-frame anchor
        return;

    // Covisibility graph (ORB-SLAM2's own definition: edge weight = number
    // of landmarks two keyframes jointly observe), built fresh from
    // m_landmarkObservations -- the same ground truth
    // runLoopBundleAdjustment() already relies on for "who observes what".
    // observedBy is the per-keyframe reverse index (which landmark IDs does
    // keyframe k observe -- its own triangulated points AND anything it
    // later re-observed via recordLandmarkObservations()/tryLoopClosure())
    // this graph is built from; kept around after the graph itself only to
    // drive the redundancy check below without a second full pass.
    std::vector<std::unordered_set<long long>> observedBy(static_cast<size_t>(n));
    std::unordered_map<int, std::unordered_map<int, int>> covisibility;
    for (const auto &entry : m_landmarkObservations) {
        std::unordered_set<int> distinctKfs;
        for (const auto &obs : entry.second) {
            if (obs.first >= 0 && obs.first < n)
                distinctKfs.insert(obs.first);
        }
        for (int kf : distinctKfs)
            observedBy[static_cast<size_t>(kf)].insert(entry.first);
        for (auto it1 = distinctKfs.begin(); it1 != distinctKfs.end(); ++it1) {
            for (auto it2 = std::next(it1); it2 != distinctKfs.end(); ++it2) {
                covisibility[*it1][*it2]++;
                covisibility[*it2][*it1]++;
            }
        }
    }

    const int searchLimit = n - kCullingExclusionWindow;
    int culledCount = 0;
    for (int k = 1; k < searchLimit; ++k) {
        Keyframe &kf = m_keyframeHistory[static_cast<size_t>(k)];
        if (kf.culled)
            continue;
        const std::unordered_set<long long> &myLandmarks = observedBy[static_cast<size_t>(k)];
        if (static_cast<int>(myLandmarks.size()) < kCullingMinOwnLandmarks)
            continue;

        int redundant = 0;
        for (long long id : myLandmarks) {
            const auto &obsList = m_landmarkObservations.at(id);
            std::unordered_set<int> otherObservers;
            for (const auto &obs : obsList) {
                if (obs.first != k)
                    otherObservers.insert(obs.first);
            }
            if (static_cast<int>(otherObservers.size()) >= kCullingMinObservers)
                ++redundant;
        }

        const double ratio = static_cast<double>(redundant) / static_cast<double>(myLandmarks.size());
        if (ratio >= kCullingRedundancyRatio) {
            kf.culled = true;
            ++culledCount;
            const int covisibleNeighbors =
                covisibility.count(k) ? static_cast<int>(covisibility.at(k).size()) : 0;
            std::fprintf(stderr,
                         "[cull] kf#%d (frame %d) marked redundant: %d/%zu landmarks over-observed "
                         "(ratio=%.3f), %d covisible neighbors\n",
                         k, kf.frameIndex, redundant, myLandmarks.size(), ratio, covisibleNeighbors);
        }
    }
    if (culledCount > 0)
        std::fprintf(stderr, "[cull] %d keyframe(s) newly marked redundant this pass\n", culledCount);
}

bool SlamWorker::loopHasSpatialConsensus(const std::vector<int> &qualifying, int bestIdx) const
{
    if (!m_loopSpatialConsensusEnabled)
        return true;
    for (int idx : qualifying) {
        if (idx != bestIdx && std::abs(idx - bestIdx) <= kLoopSpatialConsensusWindow)
            return true;
    }
    return false;
}

void SlamWorker::tryLoopClosure(size_t newKeyframeIndex)
{
    if (newKeyframeIndex < static_cast<size_t>(kLoopExclusionWindow))
        return; // not enough history yet for a real (non-trivial) loop candidate

    const Keyframe &newKf = m_keyframeHistory[newKeyframeIndex];
    const size_t searchLimit = newKeyframeIndex - kLoopExclusionWindow;

    int bestIdx = -1;
    // bestMatchCount/bestDbowScore/bestVladScore/bestSiftDbowScore/usedDbow/
    // usedVlad/usedSiftDbow: kept at this scope (not just inside their
    // respective branch below) because the "[loop] closure" log further
    // down reports whichever one actually drove candidate selection.
    size_t bestMatchCount = 0;
    double bestDbowScore = 0.0;
    float bestVladScore = 0.0f;
    double bestSiftDbowScore = 0.0;
    bool usedDbow = false;
    bool usedVlad = false;
    bool usedSiftDbow = false;

    // SIFT DBoW2 place-recognition scoring (see
    // setSiftDbowLoopClosureEnabled()) when available -- a second SIFT-
    // compatible candidate search, alongside VLAD below, checked FIRST
    // when both are enabled (real TF-IDF scoring over a proper vocabulary
    // tree is expected to be the more discriminative of the two, per
    // DEBUGGING.md's own DBoW2-vs-VLAD comparison -- not yet measured
    // against each other on this pipeline, so this is a priority CHOICE,
    // not a validated result). Falls through to VLAD, then the original
    // raw-match-count search, whenever this is off, no vocabulary is
    // loaded, or this particular keyframe has no siftBowVec.
    if (m_siftDbowLoopClosureEnabled && m_siftVocabulary && !newKf.siftBowVec.empty()) {
        usedSiftDbow = true;
        std::vector<int> qualifying;
        for (size_t i = 0; i <= searchLimit; ++i) {
            if (m_keyframeHistory[i].culled || m_keyframeHistory[i].siftBowVec.empty())
                continue;
            const double score = m_siftVocabulary->score(m_keyframeHistory[i].siftBowVec, newKf.siftBowVec);
            if (score >= kSiftDbowMinScore)
                qualifying.push_back(static_cast<int>(i));
            if (score > bestSiftDbowScore) {
                bestSiftDbowScore = score;
                bestIdx = static_cast<int>(i);
            }
        }
        if (bestIdx < 0 || bestSiftDbowScore < kSiftDbowMinScore)
            return;
        if (!loopHasSpatialConsensus(qualifying, bestIdx))
            return;
        std::fprintf(stderr, "[loop][siftdbow] kf#%d candidate=kf#%d score=%.4f\n",
                     static_cast<int>(newKeyframeIndex), bestIdx, bestSiftDbowScore);
    }
    // VLAD place-recognition scoring (see setVladLoopClosureEnabled()) when
    // available -- the other SIFT-compatible counterpart to the DBoW2
    // branch further below; checked after SIFT-DBoW2 (see its own comment
    // just above) but before ORB-DBoW2, purely for ordering. Falls through
    // to ORB-DBoW2, then the raw-match-count search, whenever VLAD is off,
    // no codebook is loaded, or this particular keyframe has no vladVector
    // (non-SIFT runs).
    else if (m_vladLoopClosureEnabled && m_vladVocabulary && !newKf.vladVector.empty()) {
        usedVlad = true;
        std::vector<int> qualifying;
        for (size_t i = 0; i <= searchLimit; ++i) {
            if (m_keyframeHistory[i].culled || m_keyframeHistory[i].vladVector.empty())
                continue;
            const float score = m_vladVocabulary->score(m_keyframeHistory[i].vladVector, newKf.vladVector);
            if (score >= kVladMinScore)
                qualifying.push_back(static_cast<int>(i));
            if (score > bestVladScore) {
                bestVladScore = score;
                bestIdx = static_cast<int>(i);
            }
        }
        if (bestIdx < 0 || bestVladScore < kVladMinScore)
            return;
        if (!loopHasSpatialConsensus(qualifying, bestIdx))
            return;
        std::fprintf(stderr, "[loop][vlad] kf#%d candidate=kf#%d score=%.4f\n",
                     static_cast<int>(newKeyframeIndex), bestIdx, bestVladScore);
    }
    // DBoW2 place-recognition scoring (see setDbowLoopClosureEnabled()) when
    // available, replacing just this candidate-selection step -- everything
    // below (PnP re-measurement, inlier gating, the correction itself) is
    // unchanged either way. Falls through to the original raw-match-count
    // search whenever DBoW2 is off, no vocabulary is loaded, or this
    // particular keyframe has no BowVector (non-ORB runs).
    else if (m_dbowLoopClosureEnabled && m_orbVocabulary && !newKf.bowVec.empty()) {
        usedDbow = true;
        std::vector<int> qualifying;
        for (size_t i = 0; i <= searchLimit; ++i) {
            if (m_keyframeHistory[i].culled || m_keyframeHistory[i].bowVec.empty())
                continue;
            const double score = m_orbVocabulary->score(m_keyframeHistory[i].bowVec, newKf.bowVec);
            if (score >= kDbowMinScore)
                qualifying.push_back(static_cast<int>(i));
            if (score > bestDbowScore) {
                bestDbowScore = score;
                bestIdx = static_cast<int>(i);
            }
        }
        if (bestIdx < 0 || bestDbowScore < kDbowMinScore)
            return;
        if (!loopHasSpatialConsensus(qualifying, bestIdx))
            return;
        std::fprintf(stderr, "[loop][dbow] kf#%d candidate=kf#%d score=%.4f\n",
                     static_cast<int>(newKeyframeIndex), bestIdx, bestDbowScore);
    } else {
        for (size_t i = 0; i <= searchLimit; ++i) {
            if (m_keyframeHistory[i].culled)
                continue;
            std::vector<cv::DMatch> matches;
            if (!matchDescriptors(m_keyframeHistory[i].descriptors, newKf.descriptors, matches))
                continue;
            if (matches.size() > bestMatchCount) {
                bestMatchCount = matches.size();
                bestIdx = static_cast<int>(i);
            }
        }
        if (bestIdx < 0 || bestMatchCount < static_cast<size_t>(kLoopMinMatches))
            return;
    }

    const Keyframe &oldKf = m_keyframeHistory[bestIdx];
    if (oldKf.localMapPoints.empty())
        return;

    // Snapshotted NOW, before anything below can touch it -- see
    // PoseGraphOptimizer.h's LoopClosureRecord doc comment for why this
    // closure's own relative measurement must be built from oldKf's pose
    // AT THIS INSTANT, not whatever it happens to be by the time the whole
    // run finishes (a later, unrelated closure could still move it).
    const cv::Mat oldKfRAtDetection = oldKf.R.clone();
    const cv::Mat oldKfTAtDetection = oldKf.t.clone();

    // Re-measure the new keyframe's pose via PnP against the OLD
    // keyframe's own locally-triangulated 3D points -- a measurement
    // independent of everything accumulated along the path in between.
    std::vector<cv::DMatch> pnpMatches;
    if (!matchDescriptors(oldKf.localMapDescriptors, newKf.descriptors, pnpMatches) ||
        static_cast<int>(pnpMatches.size()) < kLoopMinPnpInliers)
        return;

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    objectPoints.reserve(pnpMatches.size());
    imagePoints.reserve(pnpMatches.size());
    for (const auto &m : pnpMatches) {
        objectPoints.push_back(oldKf.localMapPoints[m.queryIdx]);
        imagePoints.push_back(newKf.keypoints[m.trainIdx].pt);
    }

    const cv::Mat K = m_intrinsics.toMat();
    cv::Mat rvec, loopT;
    std::vector<int> inliers;
    bool ok = false;
    try {
        ok = cv::solvePnPRansac(objectPoints, imagePoints, K, cv::Mat(), rvec, loopT, false, 200, 8.0f,
                                 0.99, inliers, cv::SOLVEPNP_P3P);
    } catch (const cv::Exception &) {
        ok = false;
    }
    if (!ok || static_cast<int>(inliers.size()) < kLoopMinPnpInliers)
        return;

    // These inlier correspondences are exactly the long-baseline evidence
    // that directly links the two loop endpoints -- oldKf's own landmarks,
    // confirmed visible again all the way over at newKf. Without recording
    // them here, runLoopBundleAdjustment() below would only ever see
    // short, local, adjacent-keyframe tracks (whatever recordLandmarkObservations()
    // happened to catch before the rolling map evicted them), never the
    // evidence that actually spans the loop -- confirmed empirically this
    // session: BA measurably *hurt* ATE (41.2m vs. 27.2m without it) before
    // this fix, converging fine on the wrong (locally-chained, not
    // loop-spanning) problem.
    std::unordered_set<long long> loopVerifiedIds;
    loopVerifiedIds.reserve(inliers.size());
    for (int idx : inliers) {
        const long long id = oldKf.localMapPointIds[static_cast<size_t>(pnpMatches[idx].queryIdx)];
        m_landmarkObservations[id].emplace_back(static_cast<int>(newKeyframeIndex),
                                                  newKf.keypoints[pnpMatches[idx].trainIdx].pt);
        loopVerifiedIds.insert(id);
    }

    cv::Mat loopR;
    cv::Rodrigues(rvec, loopR);

    // Real, multi-point, RANSAC-robust scale measurement -- ORB-SLAM2/3's
    // own Sim3Solver approach (Horn 1987 closed-form quaternion method +
    // RANSAC, see solveSim3Ransac()'s own doc comment; ported directly from
    // third_party/ORB_SLAM3/src/Sim3Solver.cc, not reconstructed from
    // memory), replacing the earlier single-point camera-center-distance
    // ratio (confirmed unstable this session: 0.0058-16.27 across early
    // closures on real runs). oldKf.localMapPoints (its own trusted local
    // map, small but reliable) matched via descriptor similarity against
    // m_mapPoints (the current rolling map at newKf's own insertion time --
    // this codebase's closest equivalent to ORB-SLAM3's own "current
    // keyframe's local map", since m_mapPoints was just extended with
    // newKf's own new points and covers its recent covisibility
    // neighborhood) gives the 3D-3D correspondences Sim3Solver needs.
    // Each side transformed into its OWN camera's local frame (oldKf's own
    // trusted pose for X3Dc1, newKf's own DRIFTED pose for X3Dc2) -- the
    // resulting s12 directly measures how far the drifted trajectory's own
    // scale has diverged from oldKf's trusted one, exactly the quantity
    // LoopClosureRecord::scale needs.
    double scaleMeas = 1.0;
    bool usedSim3Solver = false;
    int sim3InlierCount = 0;
    {
        std::vector<cv::DMatch> sim3Matches;
        if (matchDescriptors(oldKf.localMapDescriptors, m_mapDescriptors, sim3Matches) &&
            static_cast<int>(sim3Matches.size()) >= kSim3SolverMinCorrespondences) {
            std::vector<cv::Point3f> X3Dc1, X3Dc2;
            X3Dc1.reserve(sim3Matches.size());
            X3Dc2.reserve(sim3Matches.size());
            for (const auto &m : sim3Matches) {
                const cv::Mat P1w = (cv::Mat_<double>(3, 1) << oldKf.localMapPoints[m.queryIdx].x,
                                      oldKf.localMapPoints[m.queryIdx].y, oldKf.localMapPoints[m.queryIdx].z);
                const cv::Mat P1c = oldKfRAtDetection * P1w + oldKfTAtDetection;
                const cv::Point3f &P2w = m_mapPoints[static_cast<size_t>(m.trainIdx)];
                const cv::Mat P2wMat = (cv::Mat_<double>(3, 1) << P2w.x, P2w.y, P2w.z);
                const cv::Mat P2c = newKf.R * P2wMat + newKf.t;
                X3Dc1.emplace_back(static_cast<float>(P1c.at<double>(0)), static_cast<float>(P1c.at<double>(1)),
                                    static_cast<float>(P1c.at<double>(2)));
                X3Dc2.emplace_back(static_cast<float>(P2c.at<double>(0)), static_cast<float>(P2c.at<double>(1)),
                                    static_cast<float>(P2c.at<double>(2)));
            }
            cv::Mat sim3R, sim3T;
            double sim3S = 1.0;
            std::vector<int> sim3Inliers;
            if (solveSim3Ransac(X3Dc1, X3Dc2, sim3R, sim3T, sim3S, sim3Inliers, kSim3SolverMinInliers) &&
                sim3S > 0.0) {
                scaleMeas = std::clamp(sim3S, kScaleMeasClampMin, kScaleMeasClampMax);
                usedSim3Solver = true;
                sim3InlierCount = static_cast<int>(sim3Inliers.size());
            }
        }
    }

    const int f0 = oldKf.frameIndex;
    const int f1 = newKf.frameIndex;
    if (f1 <= f0)
        return;

    // World-frame correction (rotation + translation) mapping the drifted
    // camera-to-world pose at the new keyframe onto the loop-measured one.
    // Working in camera-to-world (Rcw, C) rather than this class's usual
    // world-to-camera (R, t) convention because a world-frame correction is
    // naturally expressed as "rotate/translate the camera's position in the
    // world", which is exactly what (Rcw, C) represents: for any
    // camera-to-world pose (Rcw, C), the corrected pose is
    // (Rdelta * Rcw, Rdelta * C + tDelta).
    const cv::Mat RcwDrifted = newKf.R.t();
    const cv::Mat CDrifted = -RcwDrifted * newKf.t;
    const cv::Mat RcwLoop = loopR.t();
    const cv::Mat CLoop = -RcwLoop * loopT;

    // Fallback scale-drift measurement, only used when solveSim3Ransac()
    // above didn't find enough 3D-3D correspondence support (e.g. too few
    // points in oldKf's own local map, or too few survived descriptor
    // matching against the current rolling map) -- the original single-
    // point camera-center-distance ratio this session used before porting
    // the real Sim3Solver. Kept as a fallback rather than removed: some
    // real correction is better than none when the richer measurement
    // isn't available (confirmed empirically this session that scaleMeas=1
    // everywhere, i.e. no measurement at all, left the Sim3 pose-graph's
    // scale DOF mathematically inert).
    if (!usedSim3Solver) {
        const cv::Mat ColdDrifted = -oldKfRAtDetection.t() * oldKfTAtDetection;
        const double distDrifted = cv::norm(CDrifted - ColdDrifted);
        const double distLoop = cv::norm(CLoop - ColdDrifted);
        if (distDrifted > kMinScaleMeasBaseline && distLoop > kMinScaleMeasBaseline)
            scaleMeas = std::clamp(distLoop / distDrifted, kScaleMeasClampMin, kScaleMeasClampMax);
    }

    const cv::Mat Rdelta = RcwLoop * RcwDrifted.t();
    const cv::Mat tDelta = CLoop - Rdelta * CDrifted;

    cv::Mat deltaRvec;
    cv::Rodrigues(Rdelta, deltaRvec);
    if (cv::norm(tDelta) < 1e-6 && cv::norm(deltaRvec) < 1e-6)
        return; // already consistent, nothing to correct

    // NOTE: an attempt at a sanity cap on correction magnitude here (both
    // an isPlausibleStep-based version and a fixed/scale-adaptive
    // threshold TUNED CLOSE TO NORMAL DRIFT MAGNITUDES) was tried and
    // reverted in an earlier session -- both made things worse. Rejecting
    // a correction doesn't undo the drift it would have fixed, so drift
    // keeps compounding unchecked, which makes every subsequent loop
    // measurement look even larger and get rejected too: a vicious cycle
    // that (confirmed empirically) drove every single closure in a run to
    // rejection and made ATE worse, not better. A single bad ~2337-unit
    // correction was observed once that session; this VLAD-enabled run
    // hit a far more extreme case -- pnpInliers=108 (well past
    // kLoopMinPnpInliers) but a 9,809,633-unit / 179.8-degree correction,
    // a degenerate PnP solve that happened to satisfy the reprojection-
    // error inlier gate anyway. That's not "large drift", it's numerically
    // pathological -- kLoopMaxCorrectionMagnitude/kLoopMaxCorrectionAngleDeg
    // below are set roughly 50-90x past the largest genuine correction
    // observed across both sessions (~2337 units, ~30 degrees) specifically
    // so they never engage for real drift, however large, and only catch
    // solves that are off by orders of magnitude -- deliberately not the
    // same kind of threshold as the reverted attempts, which were tight
    // enough to reject real corrections and trigger the vicious cycle
    // above. Requiring multi-candidate agreement (as originally floated
    // here) remains a more principled fix for the general case, just not
    // attempted yet.
    const double correctionMagnitude = cv::norm(tDelta);
    const double correctionAngleDeg = cv::norm(deltaRvec) * 180.0 / CV_PI;
    if (correctionMagnitude > kLoopMaxCorrectionMagnitude || correctionAngleDeg > kLoopMaxCorrectionAngleDeg) {
        std::fprintf(stderr,
                      "[loop] REJECTED degenerate correction: kf#%d (frame %d) <-> kf#%d (frame %d), "
                      "pnpInliers=%d, translation=%.3f world units, rotation=%.3f deg (past sanity caps "
                      "%.0f/%.0f -- treating as a degenerate PnP solve, not real drift)\n",
                      bestIdx, f0, static_cast<int>(newKeyframeIndex), f1, static_cast<int>(inliers.size()),
                      correctionMagnitude, correctionAngleDeg, kLoopMaxCorrectionMagnitude,
                      kLoopMaxCorrectionAngleDeg);
        return;
    }

    // Loop-closure QUALITY gate (item 40, setLoopQualityGateEnabled()):
    // reject an UNRELIABLE scale measurement -- an extreme scaleMeas (far
    // from 1.0) backed by too few Sim3 inliers is the garbage-loop signature
    // (e.g. the diagnosed scaleMeas=1.947 on 10 inliers that map-compressed
    // and collapsed tracking). A genuine large drift has many inliers, so it
    // passes; this only catches few-inlier extreme-scale measurements, which
    // avoids the documented reject-real-drift vicious cycle. Placed after the
    // degenerate-solve caps, before the temporal-consistency gate.
    if (m_loopQualityGateEnabled && usedSim3Solver && sim3InlierCount < kLoopQualityMinSim3Inliers &&
        (scaleMeas < kLoopQualityScaleLo || scaleMeas > kLoopQualityScaleHi)) {
        std::fprintf(stderr,
                      "[loop][quality] REJECTED unreliable closure: kf#%d<->kf#%d scaleMeas=%.4f "
                      "sim3Inliers=%d (< %d) -- extreme scale on too few inliers\n",
                      bestIdx, static_cast<int>(newKeyframeIndex), scaleMeas, sim3InlierCount,
                      kLoopQualityMinSim3Inliers);
        return;
    }

    // Temporal-consistency gate (see setLoopConsistencyGroupEnabled()'s own
    // doc comment for the full rationale and its relationship to real
    // ORB-SLAM3's mnLoopNumCoincidences mechanism). Placed here, after
    // every other verification gate above (candidate score, PnP inlier
    // count, degenerate-solve sanity caps) has already passed -- this is
    // the LAST thing standing between a real geometrically-verified
    // candidate and actually applying its correction.
    if (m_loopConsistencyGroupEnabled) {
        bool samePlace = false;
        if (m_pendingLoopOldIdx >= 0 &&
            newKeyframeIndex - m_pendingLoopNewKfIdx <= kLoopConsistencyMaxGapKeyframes) {
            // Ground "same place" in real appearance evidence (whichever
            // candidate-search backend actually found this candidate)
            // between the two OLD keyframes -- see kLoopConsistencyPlaceMinScore's
            // own doc comment for why this replaced a pure keyframe-index
            // proxy. Falls back to the index-window test only when neither
            // old keyframe has a usable place-recognition vector (raw-
            // match-count candidate search).
            const Keyframe &candidateOldKf = m_keyframeHistory[static_cast<size_t>(bestIdx)];
            const Keyframe &pendingOldKf = m_keyframeHistory[static_cast<size_t>(m_pendingLoopOldIdx)];
            if (usedSiftDbow && m_siftVocabulary && !candidateOldKf.siftBowVec.empty() &&
                !pendingOldKf.siftBowVec.empty()) {
                samePlace = m_siftVocabulary->score(candidateOldKf.siftBowVec, pendingOldKf.siftBowVec) >=
                            kLoopConsistencyPlaceMinScore;
            } else if (usedVlad && m_vladVocabulary && !candidateOldKf.vladVector.empty() &&
                       !pendingOldKf.vladVector.empty()) {
                samePlace = m_vladVocabulary->score(candidateOldKf.vladVector, pendingOldKf.vladVector) >=
                            kLoopConsistencyPlaceMinScore;
            } else if (usedDbow && m_orbVocabulary && !candidateOldKf.bowVec.empty() &&
                       !pendingOldKf.bowVec.empty()) {
                samePlace = m_orbVocabulary->score(candidateOldKf.bowVec, pendingOldKf.bowVec) >=
                            kLoopConsistencyPlaceMinScore;
            } else {
                samePlace = std::abs(bestIdx - m_pendingLoopOldIdx) <= kLoopConsistencyOldIdxWindow;
            }
        }
        if (samePlace) {
            ++m_pendingLoopStreak;
        } else {
            m_pendingLoopOldIdx = bestIdx;
            m_pendingLoopStreak = 1;
        }
        m_pendingLoopNewKfIdx = newKeyframeIndex;

        if (m_pendingLoopStreak < kLoopConsistencyRequiredCount) {
            std::fprintf(stderr,
                          "[loop][consistency] kf#%d<->kf#%d verified but only %d/%d consecutive "
                          "confirmations, waiting\n",
                          bestIdx, static_cast<int>(newKeyframeIndex), m_pendingLoopStreak,
                          kLoopConsistencyRequiredCount);
            return;
        }
        // Confirmed: reset the pending state so the NEXT closure (whenever
        // it happens) starts its own fresh streak, and fall through to
        // commit using THIS call's own (latest, freshest) measurement.
        m_pendingLoopOldIdx = -1;
        m_pendingLoopStreak = 0;
    }

    char candidateLabel[64];
    if (usedSiftDbow)
        std::snprintf(candidateLabel, sizeof(candidateLabel), "siftDbowScore=%.4f", bestSiftDbowScore);
    else if (usedVlad)
        std::snprintf(candidateLabel, sizeof(candidateLabel), "vladScore=%.4f", bestVladScore);
    else if (usedDbow)
        std::snprintf(candidateLabel, sizeof(candidateLabel), "dbowScore=%.4f", bestDbowScore);
    else
        std::snprintf(candidateLabel, sizeof(candidateLabel), "matches=%zu", bestMatchCount);
    std::fprintf(stderr,
                  "[loop] closure: kf#%d (frame %d) <-> kf#%d (frame %d), %s, pnpInliers=%d, "
                  "translation correction=%.3f world units, rotation correction=%.3f deg, scaleMeas=%.4f "
                  "(%s%s)\n",
                  bestIdx, f0, static_cast<int>(newKeyframeIndex), f1, candidateLabel,
                  static_cast<int>(inliers.size()), cv::norm(tDelta), cv::norm(deltaRvec) * 180.0 / CV_PI, scaleMeas,
                  usedSim3Solver ? "sim3solver, inliers=" : "fallback ratio",
                  usedSim3Solver ? std::to_string(sim3InlierCount).c_str() : "");

    // A loop closure just committed -- the map is about to be perturbed by
    // the loop correction/loop-BA below. Suppress pose-only BA for the next
    // few frames ONLY when this closure applied a significant SCALE
    // correction (scaleMeas far from 1.0) -- that's the case that
    // compresses/expands the map and sends freshly-tracked pose-only frames
    // into a local scale collapse (the diagnosed frames-2500-3300 failure,
    // triggered by scaleMeas 1.29/1.947 loops). A near-unit-scale closure
    // (scaleMeas ~1.0) barely perturbs the map, so leaving pose-only BA on
    // through it preserves its accuracy elsewhere -- the blunt
    // suppress-after-EVERY-loop version fixed the collapse region but
    // degraded the clean-loop regions (item 39). See
    // kPoseOnlyLoopSuppressFrames / setPoseOnlyLoopSuppressEnabled().
    if (m_poseOnlyLoopSuppressEnabled &&
        (scaleMeas < kPoseOnlyLoopSuppressScaleLo || scaleMeas > kPoseOnlyLoopSuppressScaleHi))
        m_poseOnlyLoopSuppressFrames = kPoseOnlyLoopSuppressFrames;

    // Distribute the correction across every trajectory point and keyframe
    // between the two loop endpoints, proportional to how far along the
    // loop each one is (identity at the old keyframe, full correction at
    // the new one). Interpolating a single rotation linearly in axis-angle
    // space (scaling the rotation vector by alpha, then re-exponentiating
    // via Rodrigues) is exact for a single-axis rotation and a standard
    // small-angle-safe approximation otherwise -- equivalent to slerp along
    // the shortest geodesic from identity to Rdelta.
    auto partialRotation = [](const cv::Mat &rvec, double alpha) -> cv::Mat {
        cv::Mat R;
        cv::Rodrigues(rvec * alpha, R);
        return R;
    };

    // m_mapPoints (the rolling global map) doesn't track which keyframe
    // contributed each point, so it can't be corrected with the same
    // per-point interpolated alpha as the trajectory/keyframes above.
    // Applying the FULL (alpha=1) correction to every current map point
    // instead is an approximation, but a necessary one: leaving the map
    // uncorrected while the live pose jumps by the full correction was
    // confirmed this session to be actively harmful, not just cosmetically
    // "slightly offset" as originally assumed here -- for a large
    // correction, trackFrame()'s PnP against the now wildly-inconsistent
    // (uncorrected) map produces real poses that read as implausible
    // relative to the (corrected) last-known position and get rejected
    // forever, a permanent lockup indistinguishable in the log from the
    // avgStepScale-collapse deadlock (see m_longTermStepScale). The
    // approximation is reasonable because kMaxMapPoints' rolling eviction
    // means the current map is mostly recent contributions already close
    // to f1's frameIndex, not spread evenly across [f0, f1].
    {
        const double r00 = Rdelta.at<double>(0, 0), r01 = Rdelta.at<double>(0, 1),
                     r02 = Rdelta.at<double>(0, 2);
        const double r10 = Rdelta.at<double>(1, 0), r11 = Rdelta.at<double>(1, 1),
                     r12 = Rdelta.at<double>(1, 2);
        const double r20 = Rdelta.at<double>(2, 0), r21 = Rdelta.at<double>(2, 1),
                     r22 = Rdelta.at<double>(2, 2);
        const double tdX = tDelta.at<double>(0), tdY = tDelta.at<double>(1), tdZ = tDelta.at<double>(2);
        for (auto &p : m_mapPoints) {
            const double x = p.x, y = p.y, z = p.z;
            p.x = static_cast<float>(r00 * x + r01 * y + r02 * z + tdX);
            p.y = static_cast<float>(r10 * x + r11 * y + r12 * z + tdY);
            p.z = static_cast<float>(r20 * x + r21 * y + r22 * z + tdZ);
        }
    }

    // m_trajectory only ever stored (world X, world Z) -- no Y, no
    // orientation -- so it gets a reduced, yaw-only version of the same
    // correction: the component of Rdelta that rotates about the world
    // Y/vertical axis, i.e. exactly the axis a ground vehicle's heading
    // drift lives on. Extracted by reading off Rdelta's effect on the
    // X-basis vector's X/Z components (the general 3-axis correction below
    // is the rigorous version, applied to keyframes and the live pose).
    const double yaw = std::atan2(-Rdelta.at<double>(2, 0), Rdelta.at<double>(0, 0));
    const double tdx = tDelta.at<double>(0), tdz = tDelta.at<double>(2);
    for (int i = 0; i < m_trajectory.size(); ++i) {
        const int f = m_trajectoryFrameIndex[i];
        if (f < f0 || f > f1)
            continue;
        const double alpha = static_cast<double>(f - f0) / static_cast<double>(f1 - f0);
        const double c = std::cos(alpha * yaw), s = std::sin(alpha * yaw);
        QPointF &p = m_trajectory[i];
        const double x = p.x(), z = p.y();
        p.setX(c * x + s * z + alpha * tdx);
        p.setY(-s * x + c * z + alpha * tdz);
    }

    // If enabled, a real joint bundle adjustment over this window's
    // keyframes and jointly-observed landmarks replaces the linear
    // interpolation below for keyframe poses specifically -- see
    // runLoopBundleAdjustment()'s doc comment for why (it uses actual
    // reprojection-error evidence instead of assuming the discrepancy
    // accrues uniformly along the loop). Everything else in this function
    // (trajectory, map points, live/reference pose) is unaffected either
    // way. Falls through to the existing interpolation if BA is off, or
    // declines (e.g. too few jointly-observed landmarks in this window).
    //
    // Global BA is tried FIRST when enabled (it's a strict superset of the
    // windowed version's own window, [0, newKfIdx] vs [bestIdx, newKfIdx],
    // so there's nothing extra windowed would add ON TOP of a successful
    // global solve) -- but unlike windowed BA, global is also capped by
    // kGlobalBaMaxWindowKeyframes (a real-time-cost bound, not a quality
    // one) and DECLINES outright once newKfIdx exceeds it, which happens
    // for every loop closure past the first ~400 keyframes on a long
    // sequence. The comment that used to be here treated this as "nothing
    // extra windowed would add", which was wrong for exactly that
    // declined case: confirmed empirically (2026-07-21) that enabling
    // global BA alone made LATE loop closures fall all the way through to
    // the naive linear interpolation below (worse than windowed BA would
    // have given them), regressing full-sequence ATE from 126.134m
    // (windowed) to 169.465m (global-only) even though global BA's own
    // successful early solves were fine. Falling back to windowed BA
    // whenever global declines -- instead of only when global is off
    // entirely -- fixes this: every loop closure now gets the best BA
    // that's actually feasible for its own span, never just interpolation
    // when a real solve was available.
    bool baApplied = false;
    if (m_globalBundleAdjustmentEnabled && m_globalBaPoseGraphPolishEnabled)
        baApplied = runPoseGraphThenGlobalBundleAdjustment(static_cast<int>(newKeyframeIndex), loopR, loopT,
                                                             loopVerifiedIds);
    else if (m_globalBundleAdjustmentEnabled)
        baApplied = runGlobalBundleAdjustment(static_cast<int>(newKeyframeIndex), loopR, loopT, loopVerifiedIds);
    if (!baApplied && m_loopBundleAdjustmentEnabled)
        baApplied = runLoopBundleAdjustment(bestIdx, static_cast<int>(newKeyframeIndex), loopR, loopT, loopVerifiedIds);
    if (!baApplied) {
        for (size_t i = static_cast<size_t>(bestIdx); i < m_keyframeHistory.size(); ++i) {
            Keyframe &kf = m_keyframeHistory[i];
            if (kf.frameIndex < f0 || kf.frameIndex > f1)
                continue;
            const double alpha = static_cast<double>(kf.frameIndex - f0) / static_cast<double>(f1 - f0);
            const cv::Mat RdeltaAlpha = partialRotation(deltaRvec, alpha);
            const cv::Mat RcwOrig = kf.R.t();
            const cv::Mat COrig = -RcwOrig * kf.t;
            const cv::Mat RcwCorrected = RdeltaAlpha * RcwOrig;
            const cv::Mat CCorrected = RdeltaAlpha * COrig + alpha * tDelta;
            kf.R = RcwCorrected.t();
            kf.t = -kf.R * CCorrected;
        }
    }

    // The live pose (m_currR/m_currT) and short-term reference keyframe
    // (m_refR/m_refT) are both, by construction, at frameIndex == f1 right
    // now (this just-inserted keyframe is also the current reference
    // frame) -- apply the full (alpha = 1) correction to both directly.
    for (cv::Mat *rPtr : {&m_currR, &m_refR}) {
        cv::Mat &tPtr = (rPtr == &m_currR) ? m_currT : m_refT;
        const cv::Mat RcwOrig = rPtr->t();
        const cv::Mat COrig = -RcwOrig * tPtr;
        const cv::Mat RcwCorrected = Rdelta * RcwOrig;
        const cv::Mat CCorrected = Rdelta * COrig + tDelta;
        *rPtr = RcwCorrected.t();
        tPtr = -(*rPtr) * CCorrected;
    }

    // Persisted purely for pose_graph::optimizePoseGraph() (see
    // keyframePoses()/loopClosureRecords()) -- an offline, opt-in
    // post-processing consumer; nothing here reads this back live. Stored
    // as the RELATIVE transform from oldKf (at detection time, snapshotted
    // above) to the loop-measured pose -- see PoseGraphOptimizer.h's
    // LoopClosureRecord doc comment for why this must be relative, not
    // oldKfIdx's current/final absolute pose.
    {
        cv::Mat loopRelR = loopR * oldKfRAtDetection.t();
        const cv::Mat loopRelT = loopT - loopRelR * oldKfTAtDetection;
        // Same IMU-measured-rotation substitution as the sequential edge
        // above (see its doc comment) -- particularly valuable here since a
        // loop edge often spans a much longer, more rotation-drift-prone
        // stretch than a single sequential step.
        if (m_imuEnabled && !m_oxtsNavFromBody.empty() && !m_imuToCameraCalib.R.empty()) {
            const cv::Mat imuRrel = imu_rotation::relativeCameraRotation(m_oxtsNavFromBody, m_imuToCameraCalib,
                                                                          oldKf.frameIndex, newKf.frameIndex);
            if (!imuRrel.empty())
                loopRelR = imuRrel;
        }
        m_loopClosureRecords.push_back(
            {bestIdx, static_cast<int>(newKeyframeIndex), loopRelR.clone(), loopRelT.clone(), scaleMeas});
    }

    // Fire off a background, off-thread re-estimate for this loop window --
    // see loopClosureDetected()'s doc comment. Snapshot is taken here, at
    // the very end, so it reflects the trajectory/keyframes *after*
    // whichever live correction (BA or interpolation) above just ran.
    emit loopClosureDetected(
        buildLoopEstimateSnapshot(bestIdx, static_cast<int>(newKeyframeIndex), loopR, loopT, loopVerifiedIds));
}

bool SlamWorker::recoverViaEpipolar(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors)
{
    if (m_refDescriptors.empty())
        return false;

    // Match against just the last keyframe (cheap) rather than the whole
    // map, then run the same model-selecting two-view pipeline used for
    // initialization. m_refR/m_refT are already a world pose, so the
    // triangulated points land directly in the existing map's frame.
    std::vector<cv::DMatch> matches;
    if (!matchDescriptors(m_refDescriptors, descriptors, matches) ||
        static_cast<int>(matches.size()) < kMinInitMatches) {
        std::fprintf(stderr, "[recover] match-count fail: matches=%d min=%d\n",
                     static_cast<int>(matches.size()), kMinInitMatches);
        // The reference keyframe has drifted too far out of view to match
        // reliably (mirrors the same situation -- and same fix -- as
        // initializeFromFrame()'s match-count-failure branch). Slide the
        // reference forward to the current frame's fresh visual content so
        // future attempts have a chance again, instead of matching against
        // an ever-more-stale keyframe forever. Anchor the new reference's
        // world pose at the last known good pose (m_currR/m_currT) rather
        // than identity -- approximate (real motion happened since that
        // pose was recorded), but far better than losing world-frame
        // consistency entirely. Without this, a single sustained low-
        // overlap stretch (e.g. a fast turn) permanently freezes the map:
        // confirmed this session with cv::SOLVEPNP_P3P, which is more
        // failure-prone than the default SOLVEPNP_ITERATIVE and triggers
        // this gap much earlier in the video (~frame 250-350).
        setReferenceFrame(kps, descriptors, m_currR, m_currT);
        return false;
    }

    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto &m : matches) {
        pts1.push_back(m_refKeypoints[m.queryIdx].pt);
        pts2.push_back(kps[m.trainIdx].pt);
    }

    cv::Mat Rrel, trel, mask;
    if (!estimateTwoViewPose(pts1, pts2, Rrel, trel, mask)) {
        std::fprintf(stderr, "[recover] estimateTwoViewPose fail\n");
        return false;
    }

    // recoverPose (or, in the homography branch, the manual normalization
    // in estimateTwoViewPose) normalizes trel to unit length -- rescale it to a
    // plausible real-world step before using it, based on the recent
    // average per-frame displacement observed during normal PnP tracking
    // (times however many frames have elapsed since the reference
    // keyframe, in case several failed attempts share the same reference).
    // Capped: the reference only slides forward on a match-*count*
    // failure, not on an F/pose/triangulation-quality failure, so a
    // prolonged run of the latter (e.g. a near-pure-rotation turn, which
    // is geometrically degenerate for essential-matrix recovery) can leave
    // it frozen while frame count keeps climbing -- without a cap that
    // inflates the rescaled step to an arbitrarily huge, made-up distance
    // the moment something finally clears the inlier threshold again.
    const int framesElapsed = std::min(kMaxFramesElapsedForRescale, std::max(1, m_frameCount - m_refFrameIndex));
    const double oxtsDist = oxtsDistanceBetween(m_refFrameIndex, m_frameCount);
    if (oxtsDist > 0.0) {
        // Real metric distance (OXTS speed data, when loaded) replaces the
        // m_avgStepScale heuristic here -- this is exactly the situation
        // that heuristic exists for (recoverViaEpipolar has no absolute-
        // scale reference of its own), so a ground-truth-quality one is a
        // strict improvement with no new risk. Unlike the heuristic, this
        // doesn't need kMaxFramesElapsedForRescale's safety cap: that cap
        // exists to bound how far a *guess* can run away over many stale
        // frames, which doesn't apply to an actual measurement.
        trel = trel * oxtsDist;
    } else if (m_avgStepScale > 0.0) {
        trel = trel * (m_avgStepScale * framesElapsed);
    }

    // Compose the relative pose onto the reference keyframe's known world
    // pose to get the current frame's world pose.
    const cv::Mat newR = Rrel * m_refR;
    const cv::Mat newT = Rrel * m_refT + trel;

    // isPlausibleStep()'s bound is entirely a function of m_avgStepScale --
    // a vision-only *guess* about typical per-frame motion, meant to catch
    // a bad RANSAC solve masquerading as a big jump. It was never meant to
    // second-guess an actual measurement: when oxtsDist was used above, trel
    // is already real metric distance, not a guess riding on m_avgStepScale.
    // Gating it through this heuristic anyway is a real bug, confirmed live
    // (GUI log, sequence 00): once m_avgStepScale drifts low relative to a
    // genuinely fast stretch, bound = kMaxStepMultiplier*avgStepScale*framesElapsed
    // grows strictly slower than the real per-frame distance as framesElapsed
    // climbs, so a correct OXTS-derived step is rejected every single time,
    // the reference frame never advances (only match-count failures slide
    // it), and the trajectory freezes permanently from that point on --
    // exactly the "trajectory stops somewhere" symptom. Skip the check
    // entirely when the step came from a real measurement; only apply it to
    // the vision-only heuristic path, which is what it was designed for.
    if (oxtsDist <= 0.0 && !isPlausibleStep(newR, newT, framesElapsed)) {
        const cv::Mat C = -newR.t() * newT;
        double stepDist = -1.0;
        if (!m_trajectory.isEmpty()) {
            const QPointF &prev = m_trajectory.last();
            stepDist = std::hypot(C.at<double>(0) - prev.x(), C.at<double>(2) - prev.y());
        }
        std::fprintf(stderr,
                      "[recover] isPlausibleStep fail: stepDist=%.3f avgStepScale=%.3f framesElapsed=%d "
                      "bound=%.3f\n",
                      stepDist, m_avgStepScale, framesElapsed,
                      kMaxStepMultiplier * m_avgStepScale * std::max(1, framesElapsed));
        return false;
    }

    std::vector<cv::Point2f> inPts1, inPts2;
    std::vector<int> trainIdx;
    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i)) {
            inPts1.push_back(pts1[i]);
            inPts2.push_back(pts2[i]);
            trainIdx.push_back(matches[i].trainIdx);
        }
    }
    if (inPts1.size() < static_cast<size_t>(kMinInitMapPoints)) {
        std::fprintf(stderr, "[recover] inlier-count fail: inPts1=%d min=%d\n",
                     static_cast<int>(inPts1.size()), kMinInitMapPoints);
        return false;
    }

    std::vector<uchar> valid;
    const std::vector<cv::Point3f> triangulated =
        triangulate(m_refR, m_refT, newR, newT, inPts1, inPts2, valid);

    std::vector<cv::Point3f> newPoints;
    cv::Mat newDescriptors;
    for (size_t i = 0; i < triangulated.size(); ++i) {
        if (!valid[i])
            continue;
        newPoints.push_back(triangulated[i]);
        newDescriptors.push_back(descriptors.row(trainIdx[i]));
    }
    if (newPoints.empty()) {
        const int validCount = static_cast<int>(std::count(valid.begin(), valid.end(), 1));
        const cv::Mat trelNorm = trel.clone();
        std::fprintf(stderr,
                      "[recover] triangulation-empty fail: triangulated=%d valid=%d inPts=%d "
                      "avgStepScale=%.4f framesElapsed=%d trel=(%.3f,%.3f,%.3f)\n",
                      static_cast<int>(triangulated.size()), validCount,
                      static_cast<int>(inPts1.size()), m_avgStepScale, framesElapsed,
                      trelNorm.at<double>(0), trelNorm.at<double>(1), trelNorm.at<double>(2));
        return false;
    }
    // These points have no owning Keyframe entry (recoverViaEpipolar()
    // doesn't insert one -- see the class doc comment), so they can never
    // be part of a bundle-adjustment landmark track; still need fresh IDs
    // to keep m_mapPointIds parallel to m_mapPoints through eviction.
    std::vector<long long> newIds;
    newIds.reserve(newPoints.size());
    for (size_t i = 0; i < newPoints.size(); ++i)
        newIds.push_back(m_nextLandmarkId++);
    appendToMap(std::move(newPoints), newDescriptors, newIds);

    m_currR = newR.clone();
    m_currT = newT.clone();
    // A tracking-loss recovery jump, not a real per-frame step -- see
    // m_velocityR/m_velocityT's doc comment for why a velocity spanning
    // this gap would be a worse prediction than assuming no motion.
    m_velocityR = cv::Mat::eye(3, 3, CV_64F);
    m_velocityT = cv::Mat::zeros(3, 1, CV_64F);
    // false: this step's distance was itself derived from m_avgStepScale
    // (trel was rescaled by avgStepScale * framesElapsed above), so it must
    // not be folded back into that same running average -- see the
    // pushTrajectoryPoint doc comment in SlamWorker.h.
    pushTrajectoryPoint(newR, newT, false);

    setReferenceFrame(kps, descriptors, newR, newT);
    m_framesSinceKeyframe = 0;
    return true;
}

void SlamWorker::drawGroundTruthOverlay(cv::Mat &display, int frameCount) const
{
    if (!m_groundTruthOverlayEnabled || m_groundTruthR.empty())
        return;

    // poses.txt line i == frame i+1 (see MapView::computeAlignment()'s same
    // convention).
    const int curIdx = frameCount - 1;
    if (curIdx < 0 || curIdx >= static_cast<int>(m_groundTruthR.size()))
        return;

    const cv::Mat &Ri = m_groundTruthR[curIdx];
    const cv::Mat &ti = m_groundTruthT[curIdx];
    const cv::Mat RiInv = Ri.t(); // orthonormal rotation: inverse == transpose
    const cv::Mat K = m_intrinsics.toMat();

    constexpr int kPathWindow = 150; // frames of ground-truth path to project ahead of the current one --
                                      // just this frame's own local stretch, not the whole remaining route
    constexpr double kMinDepth = 0.5; // meters; drop points behind or too close to the camera

    // Ground truth's own (R, t) at frame j, expressed in *this* frame's
    // camera coordinates -- self-consistent within ground truth alone, no
    // Umeyama alignment needed (see class doc comment above). offsetX/Y is
    // a parameter (not always m_groundTruthOverlayOffsetX/Y) so the road-
    // ahead line and the old-street dots can be nudged independently --
    // see setGroundTruthOverlayOffset()/setOldStreetOverlayOffset().
    auto project = [&](int j, int offsetX, int offsetY, cv::Point &out) -> bool {
        const cv::Mat Xcam = RiInv * (m_groundTruthT[j] - ti); // world -> current camera frame
        const double z = Xcam.at<double>(2);
        if (z < kMinDepth)
            return false; // behind the camera
        const cv::Mat p = K * Xcam;
        const double u = p.at<double>(0) / z + offsetX;
        const double v = p.at<double>(1) / z + offsetY;
        out = cv::Point(cv::saturate_cast<int>(u), cv::saturate_cast<int>(v));
        return true;
    };

    // Immediate stretch of road ahead: drawn as a connected line.
    const int last = std::min(static_cast<int>(m_groundTruthR.size()), curIdx + kPathWindow);
    std::vector<cv::Point> imgPts;
    imgPts.reserve(static_cast<size_t>(std::max(0, last - curIdx)));
    for (int j = curIdx; j < last; ++j) {
        cv::Point pt;
        if (project(j, m_groundTruthOverlayOffsetX, m_groundTruthOverlayOffsetY, pt))
            imgPts.push_back(pt); // behind the camera -- skip, don't cut the line short
    }
    if (imgPts.size() >= 2) {
        cv::polylines(display, imgPts, false, cv::Scalar(0, 255, 255), 3, cv::LINE_AA);
        cv::circle(display, imgPts.front(), 5, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
    }

    // Old, already-driven street: scan the *entire* ground-truth path (not
    // just the local window above) for any frame -- however far back --
    // whose camera center is both in view AND actually close to the car's
    // current position. The distance check (not just "is it in frame")
    // matters: without it, a street merely visible far in the background
    // would light up the same as one the car is really on, which is
    // confusing -- the dots should mean "you are on this street right
    // now", not "this street is visible from here". Drawn as separate
    // dots, not a connected polyline: an old revisit and the current
    // stretch aren't adjacent in world space just because the path in
    // between isn't in view right now, so connecting them would draw a
    // spurious line jumping across the frame.
    constexpr int kOldStreetExcludeWindow = 150; // skip the frames already covered by the "ahead" line
                                                  // above, plus a little recent history, so this only
                                                  // flags genuinely OLD revisits, not the current stretch
    constexpr double kOldStreetMaxDistance = 15.0; // meters; how close the car must actually be to an
                                                    // old ground-truth position to count as "on that street"
    const int excludeLo = curIdx - kOldStreetExcludeWindow;
    const int excludeHi = curIdx + kOldStreetExcludeWindow;
    for (int j = 0; j < static_cast<int>(m_groundTruthR.size()); ++j) {
        if (j >= excludeLo && j <= excludeHi)
            continue;
        if (cv::norm(m_groundTruthT[j] - ti) > kOldStreetMaxDistance)
            continue; // visible in the distance, maybe, but not where the car actually is
        cv::Point pt;
        if (!project(j, m_oldStreetOverlayOffsetX, m_oldStreetOverlayOffsetY, pt))
            continue;
        if (pt.x < 0 || pt.y < 0 || pt.x >= display.cols || pt.y >= display.rows)
            continue; // outside the frame -- projects fine but isn't actually visible
        cv::circle(display, pt, 3, cv::Scalar(255, 0, 255), cv::FILLED, cv::LINE_AA);
    }
}

void SlamWorker::refreshGroundTruthOverlayDisplay()
{
    if (m_lastDisplayBase.empty())
        return;
    cv::Mat display = m_lastDisplayBase.clone();
    drawGroundTruthOverlay(display, m_lastDisplayFrameCount);
    emit frameReady(matToQImage(display));
}

QImage SlamWorker::matToQImage(const cv::Mat &mat)
{
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
}

QString SlamWorker::buildOrbSlam3SettingsYaml() const
{
    const QString path = QDir(QStandardPaths::writableLocation(QStandardPaths::TempLocation))
                              .filePath(QStringLiteral("orbslam3_settings_%1.yaml")
                                            .arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return QString();

    // Written as plain text, NOT via cv::FileStorage::WRITE's `<<` node API
    // -- that API validates node names and rejects any key containing '.'
    // (cv::Exception: "Key names may only contain alphanumeric characters
    // [a-zA-Z0-9], '-', '_' and ' '"), which every key ORB-SLAM3's own
    // Settings.cc expects (Camera1.fx, ORBextractor.nFeatures, etc.) needs.
    // Confirmed as a real crash: Start (with ORB-SLAM3 mode on) threw that
    // exception out of a Qt event handler, which Qt cannot propagate through
    // (std::terminate). Reading dotted keys back via cv::FileStorage::READ
    // (what Settings.cc actually does) has no such restriction -- only the
    // WRITE side's node-construction path validates names -- so plain text
    // matching third_party/ORB_SLAM3's own Examples/Monocular/KITTI00-02.yaml
    // format exactly sidesteps this entirely.
    //
    // Distortion is always zero -- CameraIntrinsics has no distortion
    // fields, same as every other PnP/estimator path in this project.
    // Camera.RGB is 0 (BGR): frames reach TrackMonocular() exactly as
    // VideoSource produces them, matching the convention validated in
    // analyze/orbslam3_kitti_ate.cpp (see GrabImageMonocular()'s bBGR
    // branch). iniThFAST/minThFAST aren't exposed in the UI -- kept at
    // ORB-SLAM3's own KITTI defaults.
    QTextStream out(&file);
    out << "%YAML:1.0\n"
        << "File.version: \"1.0\"\n"
        << "Camera.type: \"PinHole\"\n"
        << "Camera1.fx: " << m_intrinsics.fx << "\n"
        << "Camera1.fy: " << m_intrinsics.fy << "\n"
        << "Camera1.cx: " << m_intrinsics.cx << "\n"
        << "Camera1.cy: " << m_intrinsics.cy << "\n"
        << "Camera1.k1: 0.0\n"
        << "Camera1.k2: 0.0\n"
        << "Camera1.p1: 0.0\n"
        << "Camera1.p2: 0.0\n"
        // Must match the REAL rate frames actually reach TrackMonocular() at
        // (kProcessIntervalMs = 100ms -> 10fps when throttled, see start()),
        // not an arbitrary/nominal value -- Tracking.cc sets
        // mMaxFrames = settings->fps() (the "force a new keyframe after
        // this many frames" threshold), so a wrong value here throws off
        // real-time keyframe spacing relative to what ORB-SLAM3's own tuning
        // assumes. 1000/kProcessIntervalMs when throttled; startUnthrottled()
        // (headless batch tools only, not used by this GUI) has no fixed
        // rate at all, so it isn't handled here.
        << "Camera.fps: " << (m_realtimeThrottle ? (1000 / kProcessIntervalMs) : 30) << "\n"
        << "Camera.RGB: 0\n"
        << "Camera.width: " << m_source.frameWidth() << "\n"
        << "Camera.height: " << m_source.frameHeight() << "\n"
        << "ORBextractor.nFeatures: " << m_orbSettings.nFeatures << "\n"
        << "ORBextractor.scaleFactor: " << static_cast<double>(m_orbSettings.scaleFactor) << "\n"
        << "ORBextractor.nLevels: " << m_orbSettings.nLevels << "\n"
        << "ORBextractor.iniThFAST: 20\n"
        << "ORBextractor.minThFAST: 7\n"
        // Settings::readViewer() (Settings.cc) treats every one of these as
        // a REQUIRED parameter and calls exit(-1) -- killing the whole GUI
        // process, not a catchable exception -- if any is missing, even
        // though this is a headless run that never constructs a real Viewer
        // (bUseViewer=false, see start()). Confirmed as the actual crash:
        // Start got past camera/ORB parsing fine and died right here.
        // Values copied from third_party/ORB_SLAM3's own
        // Examples/Monocular/KITTI00-02.yaml -- never read by anything since
        // no Viewer is ever constructed, just needed to satisfy the parser.
        << "Viewer.KeyFrameSize: 0.1\n"
        << "Viewer.KeyFrameLineWidth: 1.0\n"
        << "Viewer.GraphLineWidth: 1.0\n"
        << "Viewer.PointSize: 2.0\n"
        << "Viewer.CameraSize: 0.15\n"
        << "Viewer.CameraLineWidth: 2.0\n"
        << "Viewer.ViewpointX: 0.0\n"
        << "Viewer.ViewpointY: -10.0\n"
        << "Viewer.ViewpointZ: -0.1\n"
        << "Viewer.ViewpointF: 2000.0\n";
    file.close();
    return path;
}

QString SlamWorker::trackFrameOrbSlam3(const cv::Mat &frame, cv::Mat &display)
{
    const double timestamp = m_orbSlam3Clock.elapsed() / 1000.0;
    // Return value (Tcw) intentionally discarded -- the trajectory is
    // rebuilt below from the Atlas's own live (correction-aware) keyframe
    // poses instead, see that block's doc comment.
    m_orbSlam3System->TrackMonocular(frame, timestamp);

    // Matches ORB_SLAM3::Tracking::eTrackingState (Tracking.h): -1
    // SYSTEM_NOT_READY, 0 NO_IMAGES_YET, 1 NOT_INITIALIZED, 2 OK,
    // 3 RECENTLY_LOST, 4 LOST.
    const int trackingState = m_orbSlam3System->GetTrackingState();
    QString stateStr;
    switch (trackingState) {
    case 1: stateStr = QStringLiteral("Initializing"); break;
    case 2: stateStr = QStringLiteral("Tracking"); break;
    case 3: stateStr = QStringLiteral("Tracking (recently lost)"); break;
    case 4: stateStr = QStringLiteral("Lost"); break;
    default: stateStr = QStringLiteral("Not ready"); break;
    }

    // Rebuilt from scratch every frame from the Atlas's own live keyframe
    // poses, NOT incrementally accumulated from each frame's own Tcw --
    // loop closures and map merges (both confirmed happening live this
    // session, see *Loop detected/*Merge detected in the log) retroactively
    // correct EARLIER keyframes' poses, and a frozen per-frame snapshot
    // never picks that up. Confirmed as the actual cause of a persistently
    // misaligned MapView overlay even with continuous (non-frozen)
    // re-alignment: the underlying trajectory data itself was stale, not
    // just the alignment fit. GetAtlas()->GetAllKeyFrames() only returns the
    // CURRENTLY active map's keyframes (see Atlas::GetAllKeyFrames() in
    // Atlas.cc) -- exactly matching SaveKeyFrameTrajectoryTUM()'s own
    // Shutdown()-time behavior (the same one the validated CLI ATE
    // benchmark relies on), just called live every frame instead of once at
    // the end. mnFrameId is 0-based (ORB-SLAM3's own Frame numbering);
    // +1 converts to this project's 1-based m_frameCount/ground-truth-line
    // convention (see MapView.cpp's gtIdx = frameIndex - 1).
    {
        std::vector<ORB_SLAM3::KeyFrame *> keyframes = m_orbSlam3System->GetAtlas()->GetAllKeyFrames();
        std::sort(keyframes.begin(), keyframes.end(),
                  [](ORB_SLAM3::KeyFrame *a, ORB_SLAM3::KeyFrame *b) { return a->mnId < b->mnId; });
        m_trajectory.clear();
        m_trajectoryFrameIndex.clear();
        m_trajectory.reserve(static_cast<int>(keyframes.size()));
        m_trajectoryFrameIndex.reserve(static_cast<int>(keyframes.size()));
        for (ORB_SLAM3::KeyFrame *kf : keyframes) {
            if (!kf || kf->isBad())
                continue;
            const Eigen::Vector3f center = kf->GetPoseInverse().translation();
            m_trajectory.push_back(QPointF(center.x(), center.z()));
            m_trajectoryFrameIndex.push_back(static_cast<int>(kf->mnFrameId) + 1);
        }
    }

    // ORB-SLAM3 manages its own map entirely internally -- unlike the custom
    // pipeline's incrementally-appended-and-evicted m_mapPoints, this is a
    // full replace every frame from whatever System currently considers the
    // tracked local map.
    m_mapPoints.clear();
    for (ORB_SLAM3::MapPoint *mp : m_orbSlam3System->GetTrackedMapPoints()) {
        if (!mp)
            continue;
        const Eigen::Vector3f pos = mp->GetWorldPos();
        m_mapPoints.emplace_back(pos.x(), pos.y(), pos.z());
    }

    const std::vector<cv::KeyPoint> trackedKps = m_orbSlam3System->GetTrackedKeyPointsUn();
    cv::drawKeypoints(frame, trackedKps, display, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);

    return stateStr;
}

void SlamWorker::processNext()
{
    cv::Mat frame;
    if (!m_source.readFrame(frame)) {
        stop();
        emit statsUpdated(QStringLiteral("Stream ended"));
        return;
    }
    ++m_frameCount;

    if (m_orbSlam3Enabled) {
        cv::Mat display;
        const QString stateStr = trackFrameOrbSlam3(frame, display);
        if (display.empty())
            display = frame.clone();

        m_lastDisplayBase = display.clone();
        m_lastDisplayFrameCount = m_frameCount;
        drawGroundTruthOverlay(display, m_frameCount);
        emit frameReady(matToQImage(display));

        QVector<QPointF> mapXZ;
        mapXZ.reserve(static_cast<int>(m_mapPoints.size()));
        for (const auto &p : m_mapPoints)
            mapXZ.append(QPointF(p.x, p.z));
        emit mapUpdated(m_trajectory, mapXZ, m_trajectoryFrameIndex);

        emit trackingStateChanged(stateStr);
        emit statsUpdated(QStringLiteral("Frame %1 | ORB-SLAM3 | Map points: %2 | Trajectory: %3")
                               .arg(m_frameCount)
                               .arg(m_mapPoints.size())
                               .arg(m_trajectory.size()));
        return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat detectGray;
    cv::resize(gray, detectGray, cv::Size(), m_detectionScale, m_detectionScale, cv::INTER_AREA);

    std::vector<cv::KeyPoint> kps;
    cv::Mat descriptors;
    m_detector->detectAndCompute(detectGray, cv::noArray(), kps, descriptors);

    // Deterministic keypoint order: OpenCV's detectAndCompute() can run
    // keypoint detection through cv::parallel_for_ internally (across
    // pyramid octaves/layers when built with a threading backend), so the
    // SET of keypoints found is identical run-to-run but their ORDER in
    // kps/descriptors is not -- confirmed this session as the root cause of
    // ATE varying by up to ~60m across otherwise-identical full-sequence
    // runs: a different order feeds a different actual correspondence
    // subset to the fixed-seed (kRansacSeed) RANSAC below (same seed,
    // different index-to-point mapping), which can flip a threshold-
    // boundary accept/reject decision (loop-closure inlier gate, sanity
    // caps, etc.) and cascade through thousands of downstream frames.
    // Sorting into a fixed, position-based order after detection keeps
    // multi-threaded detection speed (measured >2x faster per frame than
    // forcing cv::setNumThreads(1) globally) while restoring
    // reproducibility -- ties (rare at float precision) fall back to
    // octave, which is itself deterministic per point.
    if (!kps.empty()) {
        std::vector<int> order(kps.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&kps](int a, int b) {
            if (kps[static_cast<size_t>(a)].pt.y != kps[static_cast<size_t>(b)].pt.y)
                return kps[static_cast<size_t>(a)].pt.y < kps[static_cast<size_t>(b)].pt.y;
            if (kps[static_cast<size_t>(a)].pt.x != kps[static_cast<size_t>(b)].pt.x)
                return kps[static_cast<size_t>(a)].pt.x < kps[static_cast<size_t>(b)].pt.x;
            return kps[static_cast<size_t>(a)].octave < kps[static_cast<size_t>(b)].octave;
        });
        std::vector<cv::KeyPoint> sortedKps(kps.size());
        cv::Mat sortedDescriptors(descriptors.rows, descriptors.cols, descriptors.type());
        for (size_t i = 0; i < order.size(); ++i) {
            sortedKps[i] = kps[static_cast<size_t>(order[i])];
            descriptors.row(order[static_cast<int>(i)]).copyTo(sortedDescriptors.row(static_cast<int>(i)));
        }
        kps = std::move(sortedKps);
        descriptors = std::move(sortedDescriptors);
    }

    // RootSIFT (see FeatureDetector.h's own doc comment) -- only meaningful
    // for SIFT's float descriptors, not ORB's binary ones. Matches the
    // transform third_party/ORB_SLAM3_SIFT/src/ORBextractor.cc applies, so
    // this codebase's own SIFT path uses the same, better-performing
    // descriptor space rather than raw SIFT.
    if (m_detectorType == feature_detector::DetectorType::Sift && !descriptors.empty())
        descriptors = feature_detector::toRootSift(descriptors);

    const float invDetectionScale = static_cast<float>(1.0 / m_detectionScale);
    for (auto &kp : kps) {
        kp.pt.x *= invDetectionScale;
        kp.pt.y *= invDetectionScale;
        kp.size *= invDetectionScale;
    }

    QString stateStr;
    switch (m_state) {
    case State::Idle:
        initializeFromFrame(kps, descriptors);
        m_state = State::Initializing;
        stateStr = QStringLiteral("Initializing");
        break;
    case State::Initializing:
        if (initializeFromFrame(kps, descriptors)) {
            m_state = State::Tracking;
            stateStr = QStringLiteral("Tracking");
        } else {
            stateStr = QStringLiteral("Initializing");
        }
        break;
    case State::Tracking: {
        // Never wipe the map or reinitialize on tracking failure. First try
        // PnP against the full map (most accurate, since it draws on every
        // point accumulated so far). If that fails, fall back to a cheap
        // epipolar step against just the last keyframe: it both keeps the
        // trajectory moving (rescaling the recovered unit-length baseline
        // by the recent average step size) and replenishes the map with
        // fresh points from the current viewpoint, so the *next* frame's
        // PnP attempt is more likely to succeed again. Only if both fail
        // does the streak advance -- there is no relocalization against
        // arbitrary past keyframes here, so a *permanent* loss of overlap
        // (e.g. driving somewhere the map never saw) eventually leaves the
        // trajectory frozen at its last known point for good.
        if (trackFrame(kps, descriptors)) {
            m_trackFailStreak = 0;
            stateStr = QStringLiteral("Tracking");
        } else if (recoverViaEpipolar(kps, descriptors)) {
            m_trackFailStreak = 0;
            stateStr = QStringLiteral("Tracking (recovered)");
        } else {
            ++m_trackFailStreak;
            if (m_trackFailStreak % kStepScaleResetStreak == 0) {
                // See kStepScaleResetStreak's doc comment: break the
                // avgStepScale-collapse deadlock by re-opening
                // isPlausibleStep()'s gate (returns true unconditionally
                // once m_avgStepScale <= 0), the same "unknown" state
                // resetSlamState() leaves these three in.
                std::fprintf(stderr,
                              "[track] step-scale reset after %d consecutive failed frames "
                              "(avgStepScale=%.4f, longTermStepScale=%.4f)\n",
                              m_trackFailStreak, m_avgStepScale, m_longTermStepScale);
                m_avgStepScale = -1.0;
                m_longTermStepScale = -1.0;
                m_recentStepDistances.clear();
            }
            stateStr = (m_trackFailStreak >= kLostDisplayThreshold)
                           ? QStringLiteral("Lost (searching for a fix...)")
                           : QStringLiteral("Tracking (recovering)");
        }
        break;
    }
    }

    cv::Mat display;
    cv::drawKeypoints(frame, kps, display, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
    m_lastDisplayBase = display.clone();
    m_lastDisplayFrameCount = m_frameCount;
    drawGroundTruthOverlay(display, m_frameCount);
    emit frameReady(matToQImage(display));

    QVector<QPointF> mapXZ;
    mapXZ.reserve(static_cast<int>(m_mapPoints.size()));
    for (const auto &p : m_mapPoints)
        mapXZ.append(QPointF(p.x, p.z));
    emit mapUpdated(m_trajectory, mapXZ, m_trajectoryFrameIndex);

    emit trackingStateChanged(stateStr);
    emit statsUpdated(QStringLiteral("Frame %1 | Keypoints: %2 | Map points: %3 | Trajectory: %4")
                           .arg(m_frameCount)
                           .arg(kps.size())
                           .arg(m_mapPoints.size())
                           .arg(m_trajectory.size()));
}
