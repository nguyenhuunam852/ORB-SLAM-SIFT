#pragma once

#include <QElapsedTimer>
#include <QObject>
#include <QImage>
#include <QPointF>
#include <QVector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "DBoW2/BowVector.h"
#include "DBoW2/FORB.h"
#include "DBoW2/FRootSift.h"
#include "DBoW2/TemplatedVocabulary.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "CameraIntrinsics.h"
#include "FeatureDetector.h"
#include "GroundPlaneScale.h"
#include "ImuRotation.h"
#include "LoopEstimateTypes.h"
#include "PnpSettings.h"
#include "PoseGraphOptimizer.h"
#include "SiftSettings.h"
#include "VideoSource.h"
#include "VladVocabulary.h"

class QTimer;

// Forward-declared only -- the full ORB_SLAM3/System.h (and everything it
// drags in: Sophus, the whole vendored third_party/ORB_SLAM3 header set) is
// only needed inside SlamWorker.cpp, not by every file that includes this
// header (e.g. MainWindow.cpp). m_orbSlam3System being a
// unique_ptr<incomplete type> is exactly why SlamWorker needs its own
// (out-of-line, in SlamWorker.cpp where System.h is visible) destructor
// instead of the implicit one -- see SlamWorker.cpp's ~SlamWorker().
namespace ORB_SLAM3 { class System; }

// DBoW2's own ORB-SLAM-style vocabulary type: a tree of visual words over
// ORB's 32-byte binary descriptors, scored via FORB's Hamming-distance-based
// TDescriptor operations. Only meaningful for the ORB detector (see
// SlamWorker::loadOrbVocabulary()) -- DBoW2 itself is descriptor-agnostic
// (templated), but a vocabulary trained on ORB descriptors cannot
// meaningfully score SIFT's float ones.
using OrbVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

// SIFT-compatible counterpart: a real DBoW2 vocabulary tree over RootSIFT's
// 128-dim float descriptors, scored via FRootSift's (squared-)L2-distance-
// based TDescriptor operations (third_party/DBoW2/DBoW2/FRootSift.h/.cpp,
// this session -- the vendored DBoW2 only shipped FORB.h before). Trained
// via analyze/train_sift_dbow_vocabulary.cpp, an alternative to
// vision/VladVocabulary.h's VLAD codebook for SIFT loop-closure candidate
// search -- see SlamWorker::loadSiftVocabulary().
using SiftVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FRootSift::TDescriptor, DBoW2::FRootSift>;

// Runs on a dedicated QThread. Owns the video source, the SIFT detector and
// a minimal monocular SLAM front-end:
//   1) Initialization: match two frames, then estimate BOTH a fundamental
//      matrix F and a homography H via RANSAC (estimateTwoViewPose()) and
//      pick whichever fits the correspondences better (ORB-SLAM's model-
//      selection heuristic). F/E-based recovery is ill-conditioned for
//      near-pure rotation or a near-planar scene (small/zero translation
//      baseline); H handles exactly that case, so a sharp turn no longer
//      forces a fundamental-matrix decomposition into an unstable
//      translation estimate. Whichever model wins, its pose is decomposed
//      and used to triangulate an initial 3D map.
//   2) Tracking: match new frames against the 3D map descriptors, recover
//      the camera pose via solvePnPRansac (method selectable from the UI).
//   3) Keyframe insertion: periodically triangulate new map points between
//      the last keyframe and the current frame.
//   4) Epipolar recovery: if PnP tracking against the map fails, fall back
//      to the same estimateTwoViewPose()/triangulate pipeline as
//      initialization, but against just the last keyframe (cheap) rather
//      than the whole map, composing the result onto the keyframe's
//      already-known world pose. This both keeps the trajectory moving
//      (scale is carried over from the recent average PnP step size,
//      since two-view geometry alone can't recover it) and replenishes
//      the map with fresh points from the current viewpoint, so the
//      *next* frame's PnP attempt is more likely to succeed again.
// The map and trajectory are never discarded once established: only an
// explicit Start/Reset clears them. There is no bundle adjustment, loop
// closure or true relocalization against arbitrary past keyframes, so a
// *permanent* loss of overlap (e.g. driving somewhere the map never saw)
// eventually leaves the trajectory frozen at its last known point.
class SlamWorker : public QObject
{
    Q_OBJECT

public:
    explicit SlamWorker(QObject *parent = nullptr);
    // Out-of-line (defined in SlamWorker.cpp, where ORB_SLAM3::System is a
    // complete type) so std::unique_ptr<ORB_SLAM3::System> can be destroyed
    // -- required since System is only forward-declared in this header. Also
    // Shuts down m_orbSlam3System (if constructed) before it's destroyed, see
    // that method's own doc comment for why this can't just be left to
    // System's own (nonexistent) destructor.
    ~SlamWorker() override;

    // Final (post-loop-closure-correction) estimated trajectory, paired
    // with the frame index each point belongs to -- used by the debug
    // harness to dump against ground truth for ATE evaluation. Not used
    // by the GUI, which gets the trajectory live via mapUpdated() instead.
    const QVector<QPointF> &trajectoryPoints() const { return m_trajectory; }
    const QVector<int> &trajectoryFrameIndices() const { return m_trajectoryFrameIndex; }

    // Read-only snapshots for pose_graph::optimizePoseGraph() (see
    // PoseGraphOptimizer.h) -- an offline, opt-in post-processing step, not
    // used by live tracking. keyframePoses() clones each keyframe's own
    // (R, t, frameIndex), deliberately not exposing the full private
    // Keyframe struct (descriptors/landmarks/etc.). sequentialEdgeRecords()/
    // loopClosureRecords() return every relative-pose measurement captured
    // at observation time so far -- see PoseGraphOptimizer.h's own doc
    // comments for why these must be relative, not reconstructed later.
    std::vector<pose_graph::KeyframePose> keyframePoses() const;
    const std::vector<pose_graph::SequentialEdgeRecord> &sequentialEdgeRecords() const
    {
        return m_sequentialEdgeRecords;
    }
    const std::vector<pose_graph::LoopClosureRecord> &loopClosureRecords() const { return m_loopClosureRecords; }

    // Essential-Graph-style covisibility edges (ORB-SLAM2/3's own
    // definition: an edge between any two keyframes sharing at least
    // minSharedLandmarks jointly-observed landmarks -- same covisibility
    // graph construction cullRedundantKeyframes() already builds from
    // m_landmarkObservations, just exposed here without the culling
    // side-effects) for pose_graph::optimizePoseGraph(). Unlike
    // LoopClosureRecord, these are NOT independent re-measurements (no
    // PnP solve against the older keyframe's own map) -- they read the
    // relative pose directly off both keyframes' own CURRENT (live-
    // tracked, already locally-BA-refined) poses. That's still real
    // information for the offline solver: it turns the ALREADY-trustworthy
    // local relative geometry continuous local BA maintains during live
    // tracking into an EXPLICIT graph edge, which the existing sequential-
    // only chain has no way to represent (confirmed empirically this
    // session that a sequential-edge-only Sim3 graph has zero real
    // internal constraint -- every sequential edge's chi2 reads exactly
    // 0.000 -- so the whole offline correction was riding on loop edges
    // alone; this is the fix). Returned as SequentialEdgeRecord (not a new
    // type) since PoseGraphOptimizer.cpp's Sim3/SE3 solve paths already
    // treat that struct as a generic Huber-robustified relative-pose edge,
    // not something requiring j==i+1 -- see optimizePoseGraphSim3()'s own
    // warm-start-chain loop, which explicitly filters for j==i+1 and
    // simply ignores any edge that isn't consecutive, so passing these in
    // is safe with no other code changes needed.
    std::vector<pose_graph::SequentialEdgeRecord> covisibilityEdgeRecords(int minSharedLandmarks = 100) const;

public:
    // Runs like the start() slot, but without the artificial real-time
    // throttle (timer interval 0 instead of kProcessIntervalMs) -- for
    // headless batch tools (kitti_ate, sift_vslam_debug) where waiting on
    // wall-clock frame timing to match KITTI's native capture rate is pure
    // wasted time; a plain function (not a slot) since it's not meant to be
    // wired to any UI control.
    void startUnthrottled();

    // Which two-view (init/recovery) pose estimator estimateTwoViewPose()'s
    // F/E branch uses. FivePoint (default) is the shipped Nister-style
    // direct calibrated solver; EightPointLegacy restores the original
    // 8-point-then-convert-plus-Gold-Standard-refinement path (see
    // EightPointLegacy.h) for benchmarking comparisons -- not meant to be
    // wired to any UI control, debug/benchmark tools only.
    enum class TwoViewEstimator { FivePoint, EightPointLegacy };
    void setTwoViewEstimator(TwoViewEstimator estimator) { m_twoViewEstimator = estimator; }

    // Loads real per-frame forward-velocity data from a KITTI OXTS (GPS/IMU)
    // folder (must contain timestamps.txt and data/<frame>.txt, KITTI raw
    // data format) to replace the vision-only scale heuristics
    // (m_avgStepScale, and the arbitrary initial bootstrap scale) with real
    // metric distance wherever the pipeline would otherwise have to guess
    // one. See DEBUGGING.md's "OXTS speed-based scale correction" writeup
    // for why this differs from a real independent sensor (KITTI's OXTS is
    // the same GPS/IMU system the ATE ground truth is itself derived from).
    // Returns false (leaving any previously loaded data in place) if the
    // folder can't be read; harmless/no-op if never called.
    bool loadOxtsSpeeds(const QString &oxtsDir);

    // Loads OXTS orientation (roll/pitch/yaw) plus the IMU->camera extrinsic
    // calibration chain (see ImuRotation.h), enabling estimateTwoViewPose()
    // to use real measured rotation instead of decomposing a homography for
    // the near-pure-rotation case homography branch exists to handle.
    // calibDir must be from the *same date* as oxtsDir's drive (KITTI
    // recalibrates between dates -- confirmed this session a wrong-date
    // calibration zip gives different, if similar-looking, values). Returns
    // false (leaving any previously loaded data in place) if either can't
    // be read; harmless/no-op if never called.
    bool loadImuOrientation(const QString &oxtsDir, const QString &calibDir);

    // Loads full per-frame ground-truth poses (rotation + translation, not
    // just the (x,z) MapView plots) from a KITTI poses.txt, so the current
    // frame's own ground-truth camera pose can project *other* frames'
    // ground-truth camera centers into the live video image (AR-style path
    // overlay -- see drawGroundTruthOverlay()). Deliberately does not reuse
    // the estimated trajectory or any Umeyama alignment: each frame's GT
    // pose is used to view GT points self-consistently, so the overlay is
    // exact and jitter-free regardless of how much the estimate has
    // accumulated so far. Returns false (leaving any previously loaded data
    // in place) if the file can't be parsed.
    bool loadGroundTruthPoses(const QString &posesPath);

public slots:
    void openVideoFile(const QString &path);
    void openCamera(int index);
    void start();
    void stop();
    void reset();

    // Thin wrappers around loadOxtsSpeeds()/loadImuOrientation() above,
    // callable from the GUI thread (ControlPanel's manual browse buttons --
    // see MainWindow's wiring) via a queued cross-thread connection, unlike
    // those two which are plain methods only ever called in-thread so far
    // (autoLoadKittiExtras(), itself a slot connected same-object). Both
    // emit statsUpdated() with a clear success/failure message since a
    // queued call's bool return value would otherwise be silently
    // discarded -- the user needs *some* feedback that a manual load
    // worked or didn't.
    void loadOxtsDir(const QString &oxtsDir);
    void loadImuDirs(const QString &oxtsDir, const QString &calibDir);

    void setIntrinsics(CameraIntrinsics intrinsics);
    void setSiftSettings(SiftSettings settings);
    void setPnpSettings(PnpSettings settings);

    // Switches the feature detector between SIFT and ORB (see
    // FeatureDetector.h) -- both rebuild the cached detector (rebuildDetector())
    // and, since ORB's binary descriptors need cv::NORM_HAMMING instead of
    // SIFT's cv::NORM_L2, the matcher norm matchDescriptors() uses.
    void setDetectorType(feature_detector::DetectorType type);
    void setOrbSettings(feature_detector::OrbSettings settings);

    // Default OFF: whether trackFrame()'s P3P/Iterative/EPnP/AP3P/SQPnP
    // branch refits over the RANSAC's full inlier set via
    // cv::solvePnPRefineLM() afterward, instead of using the minimal-sample
    // winner directly (solvePnPDltRansac() always does the equivalent --
    // a linear refit over its own full inlier set -- so this has no effect
    // on DLT). The theory was that this would remove P3P/Iterative's
    // systematic directional drift the same way DLT's own full-inlier
    // refit seems to avoid it. Measured on a full KITTI sequence-00 run
    // instead: 27.2m ATE RMSE off vs. 61.1m on -- refitting made it
    // noticeably *worse*, not better. Best guess: a full-inlier LM refit
    // converges more consistently onto whatever small camera-model
    // mismatch exists (unmodeled residual distortion, a calibration
    // constant slightly off), so removing per-frame noise doesn't remove
    // bias -- it just lets the same wrong optimum get hit every frame
    // instead of being partly cancelled out by scatter. Kept as a toggle
    // for comparison, not because on is recommended.
    void setPnpFullInlierRefineEnabled(bool enabled) { m_pnpFullInlierRefineEnabled = enabled; }

    // Overrides trackFrame()'s minimum-accepted-inlier-count gate (default
    // kMinTrackInliers = 10, see SlamWorker.cpp) -- a stricter (higher)
    // value rejects a per-frame pose unless it's well-constrained by many
    // inliers, at the cost of more frequent tracking loss/recovery.
    // Attacks the front-end tracking-accuracy bottleneck directly, unlike
    // this session's loop-closure/pose-graph work which can only clean up
    // drift after a pose is already accepted.
    void setMinTrackInliers(int count) { m_minTrackInliers = count; }

    // Window size (in keyframes) runLocalBundleAdjustment() optimizes over
    // -- was a fixed kLocalBaWindowKeyframes constexpr (default 8), now
    // runtime-overridable. Untested at any other value before this session
    // (see DEBUGGING.md) -- item 8's own observation-density fix means a
    // bigger window may now capture real multi-view constraint it
    // previously couldn't (landmarks were only pulled in via ownership
    // before that fix, so a bigger window mostly just added more
    // single-observation landmarks with nothing to triangulate against).
    void setLocalBaWindowKeyframes(int count) { m_localBaWindowKeyframes = count; }

    // Fraction of full resolution SIFT/ORB detection runs at (default 0.5,
    // half-res -- see m_detectionScale's own doc comment for the real-time-
    // budget reasoning behind that default). Was a fixed kDetectionScale
    // constexpr; moved to a runtime override the same way m_minTrackInliers
    // was, so an offline harness (unconstrained by live playback pacing)
    // can trade detection cost for denser per-frame keypoints without
    // touching the GUI's own real-time default.
    void setDetectionScale(double scale) { m_detectionScale = scale; }

    // Default off (behavior-preserving): requires each descriptor match to
    // survive a B->A nearest-neighbor cross-check in addition to the
    // existing one-directional A->B Lowe's-ratio-test (see
    // feature_detector::matchDescriptors()'s mutualCheck parameter). Cuts
    // ambiguous/tied matches before they ever reach RANSAC -- most useful
    // for ORB, whose coarse Hamming distances produce far more ties than
    // SIFT's continuous L2 ones.
    void setMutualMatchEnabled(bool enabled) { m_mutualMatchEnabled = enabled; }

    // Default off (new, untested against every sequence the plain
    // isPlausibleStep() heuristic has been run on): feeds OXTS/IMU into
    // trackFrame()'s own plausibility gate, the one documented gap
    // recoverViaEpipolar() already got fixed for -- see trackFrame()'s doc
    // comment at the check site for the full mechanism (OXTS-measured
    // step distance replaces the m_avgStepScale heuristic bound when
    // available; IMU-measured rotation is cross-checked against the PnP
    // solve's own rotation). Falls through to the exact original
    // isPlausibleStep() heuristic whenever OXTS/IMU data doesn't cover the
    // current step, or this is off -- zero behavior change for anyone not
    // using it, same as every other OXTS/IMU toggle in this class.
    void setOxtsImuInPnpEnabled(bool enabled) { m_oxtsImuInPnpEnabled = enabled; }

    // Runtime on/off toggles for already-loaded OXTS speed / IMU rotation
    // data -- distinct from loadOxtsSpeeds()/loadImuOrientation() (which
    // load the data once) so a UI checkbox can flip usage on/off without
    // reloading. Disabled falls back to the vision-only heuristics exactly
    // as if the data had never been loaded.
    void setOxtsEnabled(bool enabled) { m_oxtsEnabled = enabled; }
    void setImuEnabled(bool enabled) { m_imuEnabled = enabled; }

    // Vision-only fallback scale correction (VISO2-M-style, see
    // GroundPlaneScale.h) for when OXTS isn't loaded/enabled -- currently
    // only applied at initializeFromFrame()'s bootstrap scale (the highest-
    // leverage point: everything downstream inherits it), not
    // recoverViaEpipolar() (a real structural mismatch there: its
    // triangulation happens after the scale decision, unlike
    // initializeFromFrame(), so there's no clean "compute the correction
    // from this step's own points before finalizing scale" ordering the
    // way OXTS gets -- left for a future session, not attempted here).
    // Default off; uses hardcoded KITTI defaults (1.65m height, level
    // camera) unless setGroundPlaneConfig() is called first.
    void setGroundPlaneEnabled(bool enabled) { m_groundPlaneEnabled = enabled; }
    void setGroundPlaneConfig(const ground_plane_scale::GroundPlaneConfig &config) { m_groundPlaneConfig = config; }

    // Default off (new/expensive, and untested against every sequence this
    // codebase has been run on so far -- the existing interpolated
    // correction in tryLoopClosure() remains the fallback when this is
    // off). When on, a confirmed loop closure runs a real joint
    // reprojection-error bundle adjustment (Ceres) over every keyframe
    // strictly between the loop's two matched keyframes plus every
    // landmark they jointly observe, instead of interpolating a single
    // measured discrepancy across them -- see runLoopBundleAdjustment().
    void setLoopBundleAdjustmentEnabled(bool enabled) { m_loopBundleAdjustmentEnabled = enabled; }

    // Loads a pretrained DBoW2 ORB vocabulary (ORB-SLAM2/3's own
    // ORBvoc.txt format, via OrbVocabulary::loadFromTextFile()) for
    // DBoW2-based loop-closure candidate search -- see
    // setDbowLoopClosureEnabled(). Only meaningful with the ORB detector
    // active (a vocabulary trained on ORB descriptors can't score SIFT's
    // float ones -- see the OrbVocabulary typedef's doc comment above).
    // Loading is a one-time cost (the vocabulary file is tens of MB of
    // text), meant to be called once at startup. Returns false (leaving
    // any previously loaded vocabulary in place) if the file can't be
    // parsed.
    bool loadOrbVocabulary(const QString &path);

    // Default off: when on (and a vocabulary is loaded), tryLoopClosure()'s
    // candidate search scores each earlier keyframe's DBoW2 BowVector
    // (computed at insertKeyframe() time) against the new keyframe's own,
    // via the vocabulary's score(), instead of tryLoopClosure()'s original
    // raw-descriptor-match-count search. This is the same place-recognition
    // approach real ORB-SLAM2/3 uses for loop detection -- proper
    // appearance-based scoring instead of a plain match count, which is
    // prone to false positives (see DEBUGGING.md: frame 183<->1620 and
    // 402<->2452, confidently-wrong loop measurements the raw-count search
    // has no way to distinguish from real ones). Falls back to the exact
    // original raw-match-count search whenever this is off or no
    // vocabulary is loaded -- zero behavior change for anyone not using it.
    void setDbowLoopClosureEnabled(bool enabled) { m_dbowLoopClosureEnabled = enabled; }

    // Loads a VLAD codebook (see vision/VladVocabulary.h, and
    // third_party/ORB_SLAM3_SIFT/analyze/orbslam3_vlad_train.cpp for how
    // one is trained) for VLAD-based loop-closure candidate search -- see
    // setVladLoopClosureEnabled(). This is the SIFT-compatible counterpart
    // to loadOrbVocabulary()/setDbowLoopClosureEnabled() (DBoW2 only scores
    // ORB's binary descriptors): meaningful only with the SIFT detector
    // active, and only if the codebook was itself trained on the same
    // descriptor space this build produces (RootSIFT after the toRootSift()
    // normalization FeatureDetector.h's own doc comment describes -- e.g.
    // vocabulary_sift/vlad_codebook_all_rootsift.yml, NOT the raw-SIFT
    // vlad_codebook_all.yml). Returns false (leaving any previously loaded
    // codebook in place) if the file can't be parsed.
    bool loadVladVocabulary(const QString &path);

    // Default off: when on (and a codebook is loaded), tryLoopClosure()'s
    // candidate search scores each earlier keyframe's VLAD vector (computed
    // at insertKeyframe() time, see Keyframe::vladVector) against the new
    // keyframe's own, via the codebook's score(), instead of the original
    // raw-descriptor-match-count search -- same idea as
    // setDbowLoopClosureEnabled(), just for SIFT instead of ORB. Checked
    // before the DBoW2 branch in tryLoopClosure(); in practice the two
    // never both fire for the same run since DBoW2 requires the ORB
    // detector and VLAD requires SIFT. Falls back to the raw-match-count
    // search whenever this is off, no codebook is loaded, or this
    // particular keyframe has no vladVector -- zero behavior change for
    // anyone not using it.
    void setVladLoopClosureEnabled(bool enabled) { m_vladLoopClosureEnabled = enabled; }

    // Loads a real DBoW2 vocabulary trained on RootSIFT descriptors (see
    // SiftVocabulary's own doc comment and
    // analyze/train_sift_dbow_vocabulary.cpp) for DBoW2-based loop-closure
    // candidate search on SIFT -- a second, alternative SIFT-compatible
    // counterpart to loadOrbVocabulary(), alongside loadVladVocabulary()'s
    // VLAD codebook (both are valid at once; setSiftDbowLoopClosureEnabled()
    // decides which one tryLoopClosure() actually uses, see its own doc
    // comment for the priority order). Meaningful only with the SIFT
    // detector active. Returns false (leaving any previously loaded
    // vocabulary in place) if the file can't be parsed.
    bool loadSiftVocabulary(const QString &path);

    // Default off: when on (and a vocabulary is loaded), tryLoopClosure()'s
    // candidate search scores each earlier keyframe's DBoW2 BowVector
    // (computed from RootSIFT descriptors at insertKeyframe() time, see
    // Keyframe::siftBowVec) against the new keyframe's own, via the
    // vocabulary's TF-IDF score() -- same mechanism as
    // setDbowLoopClosureEnabled(), just for SIFT instead of ORB, and a
    // second option alongside setVladLoopClosureEnabled() for the same
    // detector. Checked BEFORE the VLAD branch in tryLoopClosure() when
    // both are enabled (real TF-IDF scoring over a proper vocabulary tree
    // vs. VLAD's flat aggregate is expected to be the more discriminative
    // of the two, per the DBoW2 vs VLAD comparison in DEBUGGING.md -- pure
    // ordering choice, not yet measured against each other on this
    // pipeline). Falls back to VLAD, then the original raw-match-count
    // search, whenever this is off, no vocabulary is loaded, or this
    // particular keyframe has no siftBowVec -- zero behavior change for
    // anyone not using it.
    void setSiftDbowLoopClosureEnabled(bool enabled) { m_siftDbowLoopClosureEnabled = enabled; }

    // Default off (behavior-preserving): when on, a geometrically-verified
    // loop candidate is no longer corrected on the FIRST confirmation --
    // tryLoopClosure() instead requires the SAME loop hypothesis (same
    // PLACE, see below) to be independently re-verified across
    // kLoopConsistencyRequiredCount consecutive-ish new-keyframe insertions
    // (within kLoopConsistencyMaxGapKeyframes of each other) before
    // actually applying the correction. This is real ORB-SLAM3's own
    // `mnLoopNumCoincidences >= 3` mechanism (see LoopClosing.cc,
    // DetectCommonRegionsFromBoW()) -- a simplified adaptation, not a
    // literal port: ORB-SLAM3 checks consistency across covisibility-graph
    // GROUPS of keyframes (this codebase has no such grouping structure).
    //
    // "Same place" is tested via real place-recognition similarity
    // (VLAD/SIFT-DBoW2/DBoW2 score, whichever candidate-search backend
    // found this candidate) between the two OLD keyframes being confirmed
    // against each other -- see kLoopConsistencyPlaceMinScore's own doc
    // comment. An EARLIER version of this gate (still available as a
    // fallback when no place-recognition vector exists for either old
    // keyframe) used a pure keyframe-index-proximity window instead
    // (kLoopConsistencyOldIdxWindow) and was measured to lose far more real
    // corrections than it filtered false positives (DEBUGGING.md item 15:
    // 72.550m->100.391m, committed closures dropped 66->21) -- viewing-
    // angle drift shifts the best-matching old-keyframe index faster than
    // a small window tolerates. Grounding the test in real appearance
    // evidence instead (same lesson item 19 already confirmed for
    // fuseWindowLandmarks()) is expected to fix this, not yet re-measured
    // after this redesign.
    //
    // A prior session note left at the degenerate-correction guard in
    // tryLoopClosure() ("Requiring multi-candidate agreement... remains a
    // more principled fix... just not attempted yet") anticipated this gate
    // in the first place. When a candidate is still pending confirmation,
    // tryLoopClosure() returns without applying any correction that call --
    // the pending streak is NOT reset by an intervening call that finds no
    // candidate at all (only by one that finds a candidate pointing
    // somewhere else, or by too large a gap since the last confirmation), a
    // deliberate simplification to avoid touching this function's many
    // existing early-return sites.
    void setLoopConsistencyGroupEnabled(bool enabled) { m_loopConsistencyGroupEnabled = enabled; }

    // Default off (behavior-preserving), independent of
    // setLoopConsistencyGroupEnabled() -- a DIFFERENT mechanism (DEBUGGING.md
    // item 23), not a variant of it. That gate required MULTIPLE SEPARATE
    // tryLoopClosure() calls (across different keyframe insertions) to
    // independently re-confirm the same place, which items 15/21/22
    // conclusively showed doesn't work on this pipeline's own sparse/noisy
    // re-detection cadence. This gate instead checks consensus WITHIN a
    // SINGLE call: the candidate-search step (whichever backend --
    // siftdbow/vlad/dbow -- is active) collects every candidate keyframe
    // whose place-recognition score independently clears the normal
    // acceptance threshold, not just the single best one. The top-scoring
    // candidate is only accepted if at least one OTHER qualifying candidate
    // lies within kLoopSpatialConsensusWindow keyframe indices of it --
    // i.e., multiple INDEPENDENTLY-scored keyframes from the current
    // history agree this is the same place, a single-call decision with no
    // waiting/pending state at all. Not yet measured.
    void setLoopSpatialConsensusEnabled(bool enabled) { m_loopSpatialConsensusEnabled = enabled; }

    // Default off (behavior-preserving): when on, every keyframe insertion
    // calls fuseWindowLandmarks() -- a simplified adaptation of real
    // ORB-SLAM3's LocalMapping::SearchInNeighbors()/ORBmatcher::Fuse(). v4
    // redesign (DEBUGGING.md item 19, this is Phase A -- COVERAGE
    // EXTENSION ONLY, no merging): for each of the current keyframe's own
    // just-triangulated landmarks, projects it into each OTHER keyframe in
    // a recent window and searches that keyframe's OWN actually-detected
    // keypoints for a reprojection-gated descriptor match (the same
    // standard recordLandmarkObservations() already uses safely for the
    // rolling map) -- if the matched keypoint is unclaimed
    // (Keyframe::keypointLandmarkId == -1), records a new, real observation
    // of the landmark there. An EARLIER version of this pass (items 16-18,
    // now superseded) compared landmarks by abstract 3D distance instead
    // of real detected keypoints and was measured to regress ATE three
    // times in a row; this redesign fixed that and measured a real
    // 29.3% improvement (72.550m -> 51.273m). See
    // setLandmarkFuseMergeEnabled() for the separate, NOT recommended,
    // Phase B (genuine-conflict merging) this pass can also do.
    void setLandmarkFuseEnabled(bool enabled) { m_landmarkFuseEnabled = enabled; }

    // Default off, and requires setLandmarkFuseEnabled() to also be on:
    // Phase B of fuseWindowLandmarks() (see its own doc comment) -- when a
    // projected landmark's search lands on a keypoint ALREADY linked to a
    // DIFFERENT landmark id (a genuine, unambiguous conflict: both ids
    // demonstrably explain the exact same detected keypoint), merges them
    // (richer-evidence-wins picks the nominal survivor id, same rule v1-v5
    // all use). v1 (MEASURED NEGATIVE, DEBUGGING.md item 20): 51.273m ->
    // 161.117m -- root-caused to survivor keeping its own stale, pre-merge
    // 3D position while the merged observation set implied a different
    // one. v2/v3 (items 20/25, both negative) fixed the local-map
    // synchronization gap and tightened the merge-specific reprojection
    // gate but never actually changed the survivor's POSITION, so the core
    // problem persisted. v5 (current, see triangulateMultiView()) is the
    // untried redesign those items flagged: re-triangulates the survivor's
    // position from BOTH ids' combined observation set via linear N-view
    // DLT+SVD, and REJECTS the merge outright if that combined set is
    // geometrically inconsistent (fails its own reprojection gate), rather
    // than accepting every candidate on a single-view pixel threshold and
    // hoping the position stays close enough. See DEBUGGING.md for whether
    // this measured better than Phase A alone (51.273m).
    void setLandmarkFuseMergeEnabled(bool enabled) { m_landmarkFuseMergeEnabled = enabled; }

    // Running total of landmarks merged/extended by fuseWindowLandmarks()
    // so far this run -- for end-of-run reporting (see kitti_ate.cpp).
    long long fusedLandmarkCount() const { return m_fusedLandmarkCount; }

    // Default off: periodically (every kCullingCheckIntervalKeyframes
    // insertions, see cullRedundantKeyframes()) builds a covisibility graph
    // from m_landmarkObservations (edge weight = landmarks two keyframes
    // jointly observe -- exactly ORB-SLAM2's own covisibility-graph
    // definition) and marks a keyframe Keyframe::culled if >=
    // kCullingRedundancyRatio of its own observed landmarks are already
    // seen by >= kCullingMinObservers OTHER keyframes (ORB-SLAM2's own
    // KeyFrameCulling() criterion) -- i.e. this keyframe's own viewpoint is
    // redundant, not uniquely informative. A culled keyframe is skipped as
    // a tryLoopClosure() candidate (fewer, less-redundant candidates to
    // search/score) but is NEVER erased or cleared -- see Keyframe::culled's
    // own doc comment for why a real erase isn't safe in this codebase.
    void setKeyframeCullingEnabled(bool enabled) { m_keyframeCullingEnabled = enabled; }

    // Default off: see runLocalBundleAdjustment()'s doc comment. When on,
    // insertKeyframe() calls runLocalBundleAdjustment() instead of
    // refineLocalKeyframes() -- when off (default), behavior is exactly
    // unchanged from before this session.
    void setLocalBundleAdjustmentEnabled(bool enabled) { m_localBundleAdjustmentEnabled = enabled; }

    // Default off: see runGlobalBundleAdjustment()'s doc comment. Takes
    // priority over setLoopBundleAdjustmentEnabled()'s windowed BA when
    // both are on (global BA is a strict superset of the windowed one).
    void setGlobalBundleAdjustmentEnabled(bool enabled) { m_globalBundleAdjustmentEnabled = enabled; }

    // Default off, requires setGlobalBundleAdjustmentEnabled() to also be
    // on: DEBUGGING.md item 29's deterministic stand-in for real
    // ORB-SLAM3's background-thread GBA + spanning-tree correction
    // propagation (see third_party/ORB_SLAM3/src/LoopClosing.cc's own
    // RunGlobalBundleAdjustment()). A REAL background std::thread was
    // deliberately not implemented: this pipeline has no locking
    // infrastructure around m_keyframeHistory/m_landmarkPositions (unlike
    // ORB-SLAM3's mMutexMapUpdate/RequestStop() synchronization), and
    // kitti_ate's own hard-won run-to-run determinism (see processNext()'s
    // keypoint-order fix and the ceres::num_threads=1 pin) depends on
    // staying single-threaded. Instead: runGlobalBundleAdjustment() still
    // SOLVES synchronously (same Ceres call, same cost), but when this is
    // on, defers WRITING the corrected keyframe poses/landmark positions
    // until kGlobalBaIntegrationDelayKeyframes keyframes later (simulating
    // "still solving in the background") via tryIntegratePendingGlobalBa().
    // Keyframes inserted during that simulated gap -- which never had
    // their own residual in the original optimization, exactly ORB-SLAM3's
    // own reason for needing spanning-tree propagation -- get corrected by
    // a single rigid delta derived from how the anchor keyframe's own pose
    // changed, chained forward; a simplified stand-in for ORB-SLAM3's real
    // per-keyframe parent/child tree walk (this codebase has no such
    // graph). Fully deterministic (no OS thread, no races) -- see
    // DEBUGGING.md for whether this measured any differently from the
    // immediate-write default.
    void setGlobalBundleAdjustmentAsyncEnabled(bool enabled) { m_globalBundleAdjustmentAsyncEnabled = enabled; }

    // Default off: when on, trackFrame() matches against the covisibility-
    // driven local map (see buildCovisibilityLocalMap()) instead of the
    // flat rolling m_mapPoints/m_mapDescriptors -- more relevant candidate
    // points when revisiting an old area (a loop closure/relocalization),
    // where the flat map's recency-based eviction would have already
    // discarded that area's points even though the covisibility graph
    // still knows they're relevant.
    void setCovisibilityLocalMapEnabled(bool enabled) { m_covisibilityLocalMapEnabled = enabled; }

    // Default off: when on, trackFrame() rejects any map-point match whose
    // keypoint lands farther than kGuidedSearchRadiusPixels from where a
    // constant-velocity motion prediction (see m_velocityR/m_velocityT's
    // doc comment) says it should project to -- a lighter-weight stand-in
    // for ORB-SLAM2/3's full SearchByProjection() (see
    // kGuidedSearchRadiusPixels' own doc comment for why this shape, not a
    // from-scratch guided matcher).
    void setGuidedSearchEnabled(bool enabled) { m_guidedSearchEnabled = enabled; }

    // Default off: when on, trackFrame() inserts a new keyframe once the
    // tracked-inlier ratio degrades past kKeyframeQualityRatioThreshold
    // (bounded by kKeyframeMinInterval/kKeyframeMaxInterval) instead of
    // unconditionally every kKeyframeEveryNFrames frames.
    void setQualityDrivenKeyframesEnabled(bool enabled) { m_qualityDrivenKeyframesEnabled = enabled; }

    // Runtime on/off toggle for the ground-truth AR overlay drawn onto the
    // video frame in processNext() (see loadGroundTruthPoses(),
    // drawGroundTruthOverlay()). Default on; a no-op visually if no ground
    // truth has been loaded yet. Immediately redraws the last-displayed
    // frame with the new setting via refreshGroundTruthOverlayDisplay(), so
    // toggling this is reflected right away instead of waiting for the next
    // processNext() tick (which may be far off, or may never come again if
    // tracking is stopped).
    void setGroundTruthOverlayEnabled(bool enabled);

    // Manual pixel offset added to every projected point of the "road
    // ahead" line (see drawGroundTruthOverlay()) -- lets the UI re-center
    // that line by eye if the projection sits off where it visually
    // "should" be (e.g. from KITTI's rectified-vs-raw principal point
    // subtleties, or simply operator preference), without touching the
    // camera intrinsics used everywhere else in the pipeline. (0, 0) by
    // default (no shift). Same immediate-redraw behavior as
    // setGroundTruthOverlayEnabled() above. Independent of
    // setOldStreetOverlayOffset() below -- kept separate per explicit user
    // request, since the two overlays are visually distinct concepts (an
    // upcoming-path preview vs. a revisit marker) even though they're
    // projected with the same underlying math.
    void setGroundTruthOverlayOffset(int dx, int dy);

    // Same idea as setGroundTruthOverlayOffset(), but for the old-street
    // revisit dots instead of the road-ahead line -- lets the two be
    // nudged independently rather than one dragging the other.
    void setOldStreetOverlayOffset(int dx, int dy);

    // Reads the source's first frame and emits it via frameReady() (with
    // the ground-truth overlay drawn as if it were frame 1) *without*
    // advancing the SLAM state machine or consuming that frame for real --
    // rewinds the source back to position 0 immediately after, so Start()
    // still processes frame 1 normally afterward. Meant to be called right
    // after a video is opened (see MainWindow::wireSignals()), so the
    // ground-truth overlay offset can be tuned by eye against a static
    // frame before committing to a full run. No-op if no source is open or
    // it can't produce a frame (e.g. an empty file).
    void previewFirstFrame();

    // Best-effort: detects a KITTI sequence number from the most recently
    // opened video's filename (trailing digits, e.g. "kitti_01.mp4" -> "01"
    // -- same heuristic MainWindow::tryAutoLoadGroundTruth() uses) and, if
    // this session has known local OXTS/calibration/poses paths for that
    // *specific* sequence (see the lookup table in SlamWorker.cpp), loads
    // them. Silently does nothing for any unrecognized sequence or
    // non-file source (camera) -- previously this always loaded sequence
    // 00's data unconditionally regardless of what was actually opened,
    // silently overwriting a manually-loaded different sequence's OXTS/IMU
    // the moment its video was (re)opened (see DEBUGGING.md). Meant to be
    // called right after any video is opened (see MainWindow::wireSignals()).
    void autoLoadKittiExtras();

    // Switches processNext() over to the vendored, real ORB-SLAM3 algorithm
    // (third_party/ORB_SLAM3, see trackFrameOrbSlam3()) instead of this
    // project's own custom pipeline above -- wired to ControlPanel's
    // checkbox. Just sets the flag and, when turning off, tears down any
    // live m_orbSlam3System (Shutdown() then reset the pointer) so state
    // never leaks between an ORB-SLAM3 run and a subsequent custom-pipeline
    // one. Turning it ON does NOT construct the System immediately -- that
    // happens lazily in start(), since building its settings YAML needs the
    // video source's resolution, which requires a source to already be open
    // (start() already guarantees this).
    void setOrbSlam3Enabled(bool enabled);

signals:
    void frameReady(const QImage &image);
    void mapUpdated(const QVector<QPointF> &trajectory, const QVector<QPointF> &mapPoints,
                     const QVector<int> &trajectoryFrameIndices);
    void statsUpdated(const QString &text);
    void trackingStateChanged(const QString &state);
    void sourceOpened(bool ok, const QString &message);

    // Fired whenever OXTS speed / IMU orientation data becomes available or
    // unavailable -- on a successful loadOxtsDir()/loadImuDirs() or
    // autoLoadKittiExtras() load (true), and whenever a new video/camera
    // source opens and clears any previously loaded data (false), since
    // that data would otherwise silently keep applying to a source it
    // doesn't correspond to. ControlPanel uses these to only let the
    // "Use OXTS speed correction"/"Use IMU rotation" checkboxes be enabled
    // once real, matching data is actually loaded -- per explicit request,
    // instead of letting them be checked (and silently doing nothing, or
    // worse, applying stale data from a previous source) with nothing
    // loaded underneath.
    void oxtsAvailabilityChanged(bool available);
    void imuAvailabilityChanged(bool available);

    // Fired from tryLoopClosure() whenever a loop closure is confirmed,
    // regardless of whether the live windowed BA/interpolation correction
    // ran -- carries a fully self-contained, deep-copied snapshot of the
    // loop window (see LoopEstimateTypes.h) for a background, off-thread
    // re-estimate (see computeLoopEstimate() in LoopEstimator.h) that never
    // touches live state and never blocks live tracking. Purely
    // informational: nothing connected to this can feed back into the
    // live trajectory/map by construction, since the snapshot is a copy.
    void loopClosureDetected(LoopEstimateSnapshot snapshot);

private slots:
    void processNext();

private:
    enum class State { Idle, Initializing, Tracking };

    void rebuildDetector();
    void resetSlamState();

    // Builds a temporary ORB-SLAM3 Settings YAML (cv::FileStorage, "File.version:
    // 1.0" format -- see third_party/ORB_SLAM3/src/Settings.cc) from the
    // current camera intrinsics (m_intrinsics) and ORB settings
    // (m_orbSettings), plus the open video source's resolution
    // (m_source.frameWidth()/frameHeight()). ORB_SLAM3::System's constructor
    // requires a settings file path up front and has no live-reconfigure
    // API afterward, unlike every setting in the custom pipeline above --
    // this is why setOrbSlam3Enabled() can't just construct the System
    // eagerly, only lazily in start() once a source is actually open.
    // Distortion is always zero (CameraIntrinsics has no distortion fields)
    // and Camera.RGB is 0 (BGR) since frames are passed to TrackMonocular()
    // exactly as VideoSource produces them, matching the convention
    // validated in analyze/orbslam3_kitti_ate.cpp. Returns the temp file
    // path, or an empty string if it couldn't be written.
    QString buildOrbSlam3SettingsYaml() const;

    // Runs one frame through the vendored ORB_SLAM3::System instead of the
    // custom pipeline -- called from processNext() in place of the whole
    // detect/initializeFromFrame/trackFrame block when m_orbSlam3Enabled is
    // set. Tracks via System::TrackMonocular() (timestamp from
    // m_orbSlam3Clock -- see its own doc comment for why wall-clock elapsed
    // time, not a real capture timestamp), converts the returned Tcw
    // (world-to-camera, Sophus::SE3f) into a world-frame camera center via
    // Tcw.inverse().translation() -- the same convention
    // pushTrajectoryPoint()'s `C = -R.t() * t` already uses for the custom
    // pipeline, so both modes plot on MapView identically -- and appends it
    // to m_trajectory/m_trajectoryFrameIndex. Rebuilds m_mapPoints from
    // System::GetTrackedMapPoints() (each MapPoint's GetWorldPos()) every
    // frame -- ORB-SLAM3 manages its own map entirely internally, so unlike
    // the custom pipeline's m_mapPoints this is a full replace, not an
    // incremental append -- and draws System::GetTrackedKeyPointsUn() onto
    // `display`. Returns a status string from System::GetTrackingState().
    QString trackFrameOrbSlam3(const cv::Mat &frame, cv::Mat &display);

    // Clears any loaded OXTS/IMU data and emits both availability signals
    // false -- called whenever a new video/camera source opens, so
    // previously-loaded data (which corresponds to whatever source it was
    // loaded for) never silently keeps applying to an unrelated new one.
    void clearOxtsImuData();

    // ratio < 0 (the default) picks the active detector's own tuned default
    // via feature_detector::defaultRatioFor() -- see that function's doc
    // comment for why SIFT and ORB need different ratio-test thresholds.
    bool matchDescriptors(const cv::Mat &descA, const cv::Mat &descB,
                           std::vector<cv::DMatch> &goodMatches, float ratio = -1.0f) const;

    // Nister-style 5-point algorithm wrapped in RANSAC: samples the minimal
    // 5 calibrated correspondences per iteration and solves for E directly
    // under the calibrated constraint (equal nonzero singular values) rather
    // than fitting an unconstrained F and converting via E = K^T F K --
    // strictly more information given K is known/exact (see DEBUGGING.md,
    // "Full F/E option menu" item 1; supersedes the earlier 8-point-then-
    // convert path plus its Gold Standard Sampson refinement, both removed).
    // Returns E (not F); mask/inlier scoring is done via Sampson distance in
    // pixel coordinates (E converted back through K for scoring) so it's
    // directly comparable to estimateHomographyRansac()'s inlier count for
    // model selection. Returns an empty Mat if fewer than 5 correspondences
    // are available.
    cv::Mat estimateEssentialRansac(const std::vector<cv::Point2f> &pts1,
                                     const std::vector<cv::Point2f> &pts2,
                                     cv::Mat &mask) const;

    // Normalized DLT (Hartley & Zisserman, Alg. 4.2) wrapped in RANSAC:
    // samples 4 correspondences per iteration, fits H linearly, scores by
    // symmetric transfer error, then refits over the largest inlier set.
    // Handles the case F/E cannot: a (near-)planar scene or (near-)pure
    // rotation, where the epipolar geometry is ill-conditioned. Returns an
    // empty Mat if fewer than 4 correspondences are available.
    cv::Mat estimateHomographyRansac(const std::vector<cv::Point2f> &pts1,
                                      const std::vector<cv::Point2f> &pts2,
                                      cv::Mat &mask) const;

    // Runs both estimateEssentialRansac() and estimateHomographyRansac(),
    // picks whichever model better explains the correspondences (ORB-SLAM's
    // R_H = S_H / (S_H + S_F) inlier-count ratio test, preferring H above
    // kHomographyPreferenceRatio), decomposes the winner (E via
    // cv::recoverPose, H via cv::decomposeHomographyMat with a cheirality
    // vote across the up-to-4 candidates using triangulate()), and returns
    // the pose of camera 2 relative to camera 1 (camera 1 == identity) plus
    // the winning model's inlier mask. Returns false if neither model finds
    // enough support.
    bool estimateTwoViewPose(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2,
                              cv::Mat &R, cv::Mat &t, cv::Mat &mask) const;

    std::vector<cv::Point3f> triangulate(const cv::Mat &R1, const cv::Mat &t1,
                                          const cv::Mat &R2, const cv::Mat &t2,
                                          const std::vector<cv::Point2f> &pts1,
                                          const std::vector<cv::Point2f> &pts2,
                                          std::vector<uchar> &validMask) const;

    // N-view linear triangulation (DLT generalized to >2 views: each
    // observation contributes 2 rows u*P_row3-P_row1 / v*P_row3-P_row2 to a
    // single homogeneous system, solved by SVD) plus a real geometric
    // acceptance gate -- every observation must reproject within
    // kFuseMergeMaxReprojErrorPixels of its actual detected keypoint (mean
    // over all observing keyframes) for the result to be accepted at all.
    // Used by fuseWindowLandmarks()'s Phase B merge (see
    // setLandmarkFuseMergeEnabled()) to re-triangulate a merged landmark's
    // position from the COMBINED observation set of both ids being merged,
    // instead of keeping whichever id "won" its own stale, pre-merge
    // position -- the untried redesign DEBUGGING.md's Phase B section
    // flagged as the one remaining plausible fix. `valid` is false (return
    // value unspecified) if fewer than 2 observations have positive depth
    // in front of their own keyframe or the mean reprojection error exceeds
    // the gate -- callers must reject the merge entirely in that case, not
    // fall back to the pre-merge position.
    cv::Point3f triangulateMultiView(const std::vector<std::pair<int, cv::Point2f>> &observations,
                                      bool &valid) const;

    // Custom linear DLT (Direct Linear Transform) pose solver wrapped in
    // RANSAC, offered as an alternative to OpenCV's built-in PnP methods
    // (selected via PnpSettings::method == kPnpMethodDlt). Samples 6
    // correspondences per iteration (minimal case for the 12-unknown
    // linear system), fits the pose matrix linearly in K-calibrated,
    // similarity-normalized coordinates (mirrors solveEightPoint /
    // solveHomographyDLT's normalized-DLT pattern), extracts a proper
    // rotation via the closest-orthogonal (Procrustes/SVD) correction, and
    // scores all correspondences by pixel reprojection error before
    // refitting over the largest inlier set. Returns false if fewer than 6
    // correspondences are available or no candidate produces a usable pose.
    bool solvePnPDltRansac(const std::vector<cv::Point3f> &objectPoints,
                            const std::vector<cv::Point2f> &imagePoints, cv::Mat &R, cv::Mat &t,
                            std::vector<int> &inlierIndices) const;

    bool initializeFromFrame(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors);
    bool trackFrame(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors);
    void insertKeyframe(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors,
                         const cv::Mat &R, const cv::Mat &t);
    bool recoverViaEpipolar(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors);

    // Called right after a new entry is appended to m_keyframeHistory.
    // Searches all earlier keyframes (excluding a recent window, so it
    // can't "close a loop" against the ordinary previous keyframe) for one
    // sharing enough raw descriptor matches to be the same place revisited.
    // On a hit, re-measures the new keyframe's pose via PnP against that
    // old keyframe's own locally-triangulated 3D points (a measurement
    // independent of everything accumulated along the path in between),
    // and distributes the discrepancy between that measurement and the
    // drifted odometry pose -- as a full 6-DOF (rotation + translation)
    // world-frame correction -- across every trajectory point and keyframe
    // between the two, proportionally to how far along the loop each one
    // is. m_trajectory only ever stored (world X, world Z), no orientation,
    // so it gets a reduced yaw-only projection of the same correction.
    // m_mapPoints (the rolling global map, which has no per-point frame
    // association) gets the full, uninterpolated correction applied
    // uniformly to every point -- an approximation (true bundle adjustment
    // would correct each point relative to its actual contributing
    // keyframe), but a necessary one: for a large correction, leaving the
    // map uncorrected while the live pose jumps isn't just cosmetically
    // "slightly offset", it actively breaks tracking (trackFrame()'s real
    // PnP solves against the now-inconsistent map read as implausible
    // relative to the corrected position and get rejected forever --
    // confirmed this session as a second permanent-lockup mechanism,
    // separate from the avgStepScale collapse m_longTermStepScale guards
    // against).
    void tryLoopClosure(size_t newKeyframeIndex);

    // See setLoopSpatialConsensusEnabled(). Returns true immediately
    // (no-op) when that feature is off. Otherwise, true iff `qualifying`
    // (every candidate keyframe index whose place-recognition score
    // independently cleared the acceptance threshold this same call)
    // contains some OTHER index within kLoopSpatialConsensusWindow of
    // bestIdx.
    bool loopHasSpatialConsensus(const std::vector<int> &qualifying, int bestIdx) const;

    // See setKeyframeCullingEnabled(). Rebuilds the covisibility graph from
    // scratch (a full pass over m_landmarkObservations) each call -- called
    // only every kCullingCheckIntervalKeyframes keyframe insertions from
    // insertKeyframe() to bound this cost, not every single one.
    void cullRedundantKeyframes();

    // Matches this new keyframe's own descriptors against the live global
    // map (m_mapDescriptors/m_mapPointIds) -- the same kind of match
    // trackFrame() already does every frame, just run once more here, on
    // insertKeyframe()'s already-heavier path rather than the hot per-frame
    // one -- to find already-triangulated landmarks it re-observes. Each
    // hit appends a (keyframeIndex, pixel) entry to
    // m_landmarkObservations[id], the cross-keyframe correspondence real
    // bundle adjustment needs and this codebase had no mechanism for
    // before (every other point of this class only records a landmark's
    // *creating* keyframe's own observation, in insertKeyframe() itself).
    // R/t are this keyframe's own (world-to-camera) pose, used to
    // reprojection-gate each descriptor match against the landmark's known
    // 3D position before accepting it -- see kMaxObservationReprojErrorPixels'
    // doc comment (SlamWorker.cpp) for why a plain ratio-test match isn't
    // trusted on its own here.
    // keypointLandmarkId: this keyframe's own Keyframe::keypointLandmarkId
    // (passed by reference since this is called before the Keyframe is
    // pushed into m_keyframeHistory) -- every accepted match additionally
    // records keypointLandmarkId[m.trainIdx] = id, so fuseWindowLandmarks()'s
    // v4 redesign (see DEBUGGING.md item 19) can tell which of this
    // keyframe's own detected keypoints already have a confirmed landmark
    // link (including ones assigned here, re-observations of the rolling
    // map, not just ones this keyframe itself triangulated).
    void recordLandmarkObservations(int keyframeIndex, const std::vector<cv::KeyPoint> &kps,
                                     const cv::Mat &descriptors, const cv::Mat &R, const cv::Mat &t,
                                     std::vector<long long> &keypointLandmarkId);

    // See setLandmarkFuseEnabled(). Simplified adaptation of real
    // ORB-SLAM3's LocalMapping::SearchInNeighbors()/ORBmatcher::Fuse():
    // for every landmark newKeyframeIndex just triangulated
    // (m_keyframeHistory[newKeyframeIndex].localMapPointIds), searches
    // everything actively observed within the last kFuseWindowKeyframes
    // keyframes for candidates within kFuseMaxWorldDistance (3D proximity,
    // a cheap first pass), then among THOSE candidates picks the one with
    // the smallest RootSIFT descriptor distance -- only actually merging if
    // that distance is also under kFuseMaxDescriptorDistance. This two-
    // stage design (proximity narrows candidates, descriptor decides)
    // mirrors real ORB-SLAM3's own ORBmatcher::Fuse() (radius search, then
    // "match to the most similar keypoint in the radius"); an EARLIER
    // version of this pass skipped the descriptor stage entirely and was
    // measured (DEBUGGING.md item 16) to merge mostly visually-dissimilar
    // landmarks (median descriptor distance among its merges: 0.7987) --
    // confirmed a real false-merge problem, not a coding bug, which is why
    // the descriptor gate was added. On an actual merge, the id with fewer
    // recorded observations is absorbed into the one with more: its
    // m_landmarkObservations entries are appended onto the survivor's and
    // its own entry erased, so future local/loop-BA density lookups --
    // items 8/10 -- and future fuse passes only ever see the survivor.
    // Real ORB-SLAM3 additionally does its whole search via projection into
    // covisible neighbors' own keypoint sets (needing a per-keyframe "map
    // point at this keypoint index" table this codebase doesn't have) --
    // this 3D-proximity-then-descriptor test is a coarser but much simpler
    // stand-in; see DEBUGGING.md for remaining caveats (kFuseMaxWorldDistance
    // is a fixed absolute threshold that assumes roughly self-consistent
    // local scale within the window, not validated against the real
    // monocular scale drift item 11 already established at the
    // full-sequence level).
    void fuseWindowLandmarks(int newKeyframeIndex);

    // Real joint bundle adjustment (Ceres, reprojection error) over every
    // keyframe strictly between oldKfIdx and newKfIdx plus every landmark
    // any two-or-more of them jointly observe (see m_landmarkObservations).
    // Both endpoints are held fixed -- oldKfIdx at its existing (trusted,
    // pre-loop) pose, newKfIdx at (loopR, loopT) (the loop-measured pose,
    // NOT its drifted pre-BA one, which is what actually pulls everything
    // in between into a consistent configuration) -- everything strictly
    // in between is free to move. On success, overwrites the affected
    // keyframes' R/t, the optimized landmarks' m_landmarkPositions entries
    // (and m_mapPoints, for any still live there), and re-derives
    // m_trajectory points in [oldKf.frameIndex, newKf.frameIndex] via
    // piecewise interpolation between each consecutive pair of (now
    // corrected) keyframes -- replacing tryLoopClosure()'s single-segment
    // interpolated correction with a per-segment one driven by what BA
    // actually found, rather than one measurement extrapolated across the
    // whole span. Called from tryLoopClosure() only when
    // m_loopBundleAdjustmentEnabled is set; a no-op (returns false) if the
    // window has too few free keyframes or jointly-observed landmarks to
    // be worth solving. loopVerifiedIds identifies which landmarks' newKfIdx
    // observation is the actual PnP-RANSAC-verified correspondence that
    // *proved* the loop (see tryLoopClosure()) -- those residuals get a much
    // higher weight and no robust loss instead of being just more
    // Huber-lossed observations among thousands of short local tracks (see
    // DEBUGGING.md Session 8 for why this matters: this is the one place
    // real, geometrically-verified long-baseline evidence exists, and it
    // was being diluted rather than trusted).
    bool runLoopBundleAdjustment(int oldKfIdx, int newKfIdx, const cv::Mat &loopR, const cv::Mat &loopT,
                                  const std::unordered_set<long long> &loopVerifiedIds);

    // Joint Ceres BA over the last kLocalBaWindowKeyframes, run on EVERY
    // keyframe insertion when setLocalBundleAdjustmentEnabled() is on --
    // see kLocalBaWindowKeyframes's own doc comment (SlamWorker.cpp) for
    // why this needs a pose-prior regularizer instead of the single-
    // hard-anchor gauge-fixing strategy a prior (reverted) attempt at this
    // exact feature used. Called from insertKeyframe() INSTEAD OF
    // refineLocalKeyframes() when enabled (both refine the same trailing
    // window of keyframes; joint BA supersedes the independent per-keyframe
    // refinement refineLocalKeyframes() does). Returns false (no-op) if the
    // window has too few keyframes/landmarks to be worth solving.
    bool runLocalBundleAdjustment();

    // Full global BA (see setGlobalBundleAdjustmentEnabled()): joint Ceres
    // BA over EVERY keyframe/landmark from 0 to newKfIdx (not just a loop's
    // own window, unlike runLoopBundleAdjustment()) -- called from
    // tryLoopClosure() in place of the windowed loop BA when enabled. See
    // kGlobalBaMaxWindowKeyframes's doc comment (SlamWorker.cpp) for why
    // this is safe from continuous local BA's scale-collapse failure mode:
    // it fires once per loop closure (not hundreds of times), and has two
    // real hard anchors (keyframe 0, newKfIdx at the loop-verified pose)
    // spanning the whole graph.
    bool runGlobalBundleAdjustment(int newKfIdx, const cv::Mat &loopR, const cv::Mat &loopT,
                                    const std::unordered_set<long long> &loopVerifiedIds);

    // See setCovisibilityLocalMapEnabled(). Rebuilds m_localMapPoints/
    // m_localMapDescriptors/m_localMapPointIds from the keyframes covisible
    // with the current (just-inserted) reference keyframe -- called from
    // insertKeyframe() when enabled, same cadence as every other periodic
    // maintenance step there.
    void buildCovisibilityLocalMap();

    // Deep-copies everything computeLoopEstimate() (LoopEstimator.h) needs
    // for the window [oldKfIdx, newKfIdx] -- every cv::Mat cloned, so the
    // result is safe to hand to a background thread regardless of what
    // this worker thread does to its own state afterward. Called from
    // tryLoopClosure() right after a loop closure is confirmed, emitted
    // via loopClosureDetected(). loopVerifiedIds is threaded into the
    // snapshot so computeLoopEstimate() can give those same correspondences
    // the same special trust runLoopBundleAdjustment() does.
    LoopEstimateSnapshot buildLoopEstimateSnapshot(int oldKfIdx, int newKfIdx, const cv::Mat &loopR,
                                                    const cv::Mat &loopT,
                                                    const std::unordered_set<long long> &loopVerifiedIds) const;

    // Refines the pose of each of the last kLocalRefineWindow keyframes via
    // a few Levenberg-Marquardt iterations (cv::solvePnPRefineLM) against
    // its own localMapPoints/localMapImagePoints, minimizing reprojection
    // error directly rather than relying on whatever RANSAC/DLT/P3P
    // produced at insertion time. This is a per-keyframe polish, not true
    // joint multi-keyframe bundle adjustment -- this codebase doesn't track
    // a landmark shared across multiple keyframes' observations (each
    // keyframe only keeps the points it personally triangulated), so
    // there's no shared structure to jointly optimize the way a full BA
    // would. Still a real, useful tightening: nonlinear reprojection-error
    // minimization instead of a closed-form/RANSAC-only estimate. Keeps
    // m_currR/m_currT/m_refR/m_refT in sync with the newest keyframe
    // afterward, since setReferenceFrame() was called with the
    // pre-refinement pose.
    void refineLocalKeyframes();

    // Sets m_refKeypoints/m_refDescriptors/m_refR/m_refT/m_refFrameIndex together.
    void setReferenceFrame(const std::vector<cv::KeyPoint> &kps, const cv::Mat &descriptors,
                            const cv::Mat &R, const cv::Mat &t);

    // Appends a new world-frame trajectory point. If updateAvgStepScale is
    // true, also folds its step distance into the running per-frame scale
    // estimate (m_avgStepScale) used to rescale recoverViaEpipolar's
    // unit-length recovered translation. Must be false for poses whose
    // distance was itself derived from m_avgStepScale (i.e.
    // recoverViaEpipolar's own output) -- otherwise the estimate is fed
    // its own synthetic output as if it were an independent measurement,
    // which lets a single outlier lock in and never correct back down.
    void pushTrajectoryPoint(const cv::Mat &R, const cv::Mat &t, bool updateAvgStepScale = true);

    // Appends newly triangulated world-frame points (and matching
    // descriptors) to the map, then evicts the oldest points past
    // kMaxMapPoints. newIds must be parallel to newPoints -- the landmark
    // IDs the caller already assigned (see insertKeyframe()) and recorded
    // in m_landmarkPositions/the new keyframe's localMapPointIds; kept in
    // lockstep with m_mapPoints/m_mapDescriptors through eviction so
    // runLoopBundleAdjustment() can still find a live map point by ID.
    void appendToMap(std::vector<cv::Point3f> &&newPoints, const cv::Mat &newDescriptors,
                      const std::vector<long long> &newIds);

    // Rejects a candidate pose whose implied step distance from the last
    // known position is wildly larger than recent normal motion -- a
    // degenerate/bad RANSAC solve can still satisfy the minimum inlier
    // count while producing a physically implausible jump. Always true
    // until m_avgStepScale is established (nothing yet to compare against).
    // Deliberately has no "give up and accept anyway" escape hatch: an
    // earlier version widened the bound after a long fail streak to avoid
    // a permanent lockout, but that let occasional wild, fabricated jumps
    // through right at the hardest spots (e.g. intersections). A frozen
    // trajectory during a genuinely difficult stretch is preferable to a
    // fabricated one; the reference-sliding fallback in
    // initializeFromFrame()/insertKeyframe()/recoverViaEpipolar() already
    // keeps retrying from a fresh vantage point without needing this to
    // ever relax. framesElapsed scales the allowance up for a candidate
    // that covers more than one frame's worth of motion (e.g. epipolar
    // recovery after several failed attempts against the same reference).
    bool isPlausibleStep(const cv::Mat &R, const cv::Mat &t, int framesElapsed = 1) const;

    // AR-style ground-truth path overlay, drawn directly onto `display`
    // (BGR, same size as the source frame). For frameCount's own ground-
    // truth pose (GT line curIdx = frameCount-1), transforms a window of
    // *other* frames' ground-truth camera centers into this frame's camera
    // coordinates (X_cam = R_i^T * (t_j - t_i), the inverse of poses.txt's
    // world<-camera convention) and projects them with the camera
    // intrinsics -- entirely self-consistent within ground truth's own
    // frame, so it needs no Umeyama alignment against the (possibly still
    // very sparse) live estimate the way MapView's overlay does. frameCount
    // is a parameter rather than always reading m_frameCount so
    // previewFirstFrame() can reuse this against a frame that was never
    // actually run through the SLAM pipeline (pretend frameCount=1). No-op
    // if ground truth hasn't been loaded, disabled, or frameCount has no
    // corresponding ground-truth line.
    void drawGroundTruthOverlay(cv::Mat &display, int frameCount) const;

    // Re-emits frameReady() by redrawing the ground-truth overlay onto a
    // clone of m_lastDisplayBase (the most recent frame shown, keypoints
    // already drawn, GT overlay not yet drawn -- see processNext() and
    // previewFirstFrame(), both of which set it) at m_lastDisplayFrameCount.
    // No-op if nothing has ever been displayed yet. Deliberately doesn't
    // touch m_source at all (no read/seek), unlike previewFirstFrame() --
    // that's what makes it safe to call from setGroundTruthOverlayEnabled()/
    // setGroundTruthOverlayOffset() regardless of tracking state: an earlier
    // version instead re-called previewFirstFrame() itself, gated to only
    // fire in State::Idle to avoid rewinding a mid-run video that was merely
    // paused -- but that meant offset tuning silently stopped doing anything
    // at all the moment tracking had ever been started even once (confirmed
    // by the user: "it just move one time and stop"), since m_state never
    // returns to Idle without a full Reset. Caching the last base frame
    // instead sidesteps the whole tension.
    void refreshGroundTruthOverlayDisplay();

    static QImage matToQImage(const cv::Mat &mat);

    VideoSource m_source;
    QTimer *m_timer = nullptr;

    cv::Ptr<cv::Feature2D> m_detector; // cv::SIFT or cv::ORB, both derive from cv::Feature2D -- see FeatureDetector.h
    CameraIntrinsics m_intrinsics;
    SiftSettings m_siftSettings;
    feature_detector::DetectorType m_detectorType = feature_detector::DetectorType::Sift; // default: today's behavior
    feature_detector::OrbSettings m_orbSettings;
    int m_matcherNorm = cv::NORM_L2; // kept in sync with m_detectorType by rebuildDetector(); see matchDescriptors()
    PnpSettings m_pnpSettings;

    State m_state = State::Idle;

    // Last keyframe: pose is expressed in the world frame (world == the
    // first successfully established reference frame for this run).
    std::vector<cv::KeyPoint> m_refKeypoints;
    cv::Mat m_refDescriptors;
    cv::Mat m_refR;
    cv::Mat m_refT;
    int m_refFrameIndex = 0; // m_frameCount value when m_ref* was last (re)chosen

    // Running estimate of per-frame camera displacement in world units,
    // used to rescale recoverViaEpipolar()'s unit-length recovered
    // translation (monocular epipolar recovery can't recover scale on its
    // own). Negative means "not yet known" (nothing to rescale against;
    // used as-is, same as the very first bootstrap step). Computed as the
    // median of m_recentStepDistances (below) rather than a single
    // whole-video average -- a median-of-recent-window tracks real local
    // speed changes (a car accelerating, slowing for a turn) instead of
    // blending them into one global number, and is inherently robust to a
    // single outlier the way an EMA isn't (see kMaxAvgStepUpdateMultiplier
    // for the belt-and-suspenders clamp kept on top of this).
    double m_avgStepScale = -1.0;

    // Sliding window of recent independently-measured (trackFrame-only,
    // i.e. real PnP-derived, never recoverViaEpipolar's own rescaled
    // output) step distances that m_avgStepScale is the median of.
    QVector<double> m_recentStepDistances;

    // Slow-moving (EMA, alpha ~0.02) long-term baseline of the same step
    // distances, used only as a FLOOR under m_avgStepScale (never let it
    // fall below kMinScaleFraction of this). Without a floor, a genuine
    // sustained slowdown (stopped at an intersection for many frames) can
    // correctly drag the fast median-based m_avgStepScale down to near
    // zero -- but then once real motion resumes, isPlausibleStep's bound
    // (10x avgStepScale) stays too tight to ever accept a real step again,
    // and since m_avgStepScale only updates on an *accepted* step, it can
    // never recover: a permanent deadlock, confirmed this session on a
    // full KITTI-00 run (avgStepScale decayed to 0.003 through a slow
    // stretch around frame 2000-2800, then every frame from 1579 onward --
    // note the loop-closure event at that same frame was coincidental, not
    // the cause -- was rejected forever). The floor is clamped relative to
    // itself, not to the (possibly already-deflated) m_avgStepScale, so it
    // isn't dragged down by the same collapse it exists to guard against.
    double m_longTermStepScale = -1.0;

    TwoViewEstimator m_twoViewEstimator = TwoViewEstimator::FivePoint;
    bool m_oxtsEnabled = false; // default off -- see ControlPanel::setOxtsAvailable()
    bool m_imuEnabled = false; // default off -- see ControlPanel::setImuAvailable()
    bool m_pnpFullInlierRefineEnabled = false;
    bool m_oxtsImuInPnpEnabled = false; // default off (was on -- see DEBUGGING.md for the 18.6m-vs-27.2m
                                         // measurement this trades away by defaulting off; still available
                                         // as an opt-in toggle)
    bool m_mutualMatchEnabled = false; // see setMutualMatchEnabled()

    // Default off (unlike OXTS/IMU): a vision-only heuristic, not
    // cross-validated against ground truth the way OXTS/IMU were -- see
    // setGroundPlaneEnabled()'s doc comment. Defaults match KITTI's
    // commonly-cited (not re-verified per-sequence) rig: ~1.65m camera
    // height, level mount (ground normal = camera -Y).
    bool m_groundPlaneEnabled = false;
    ground_plane_scale::GroundPlaneConfig m_groundPlaneConfig{
        (cv::Mat_<double>(3, 1) << 0.0, -1.0, 0.0), 1.65};

    // Cumulative real-world distance (meters) from frame 0 to frame i, from
    // OXTS forward-velocity data (loadOxtsSpeeds()); index 0 is always 0.0.
    // Empty if OXTS data was never loaded -- everything that reads this
    // falls back to the vision-only heuristics in that case. See
    // oxtsDistanceBetween().
    QVector<double> m_oxtsCumulativeDistance;

    // |cumulative distance at frameB - at frameA|, i.e. real-world distance
    // traveled between two (not necessarily adjacent) frame indices.
    // Returns -1.0 (a sentinel meaning "not available") if OXTS data was
    // never loaded, rather than silently falling back to some default --
    // callers must explicitly decide what to do in that case.
    double oxtsDistanceBetween(int frameA, int frameB) const;

    // Per-frame OXTS orientation (navFromBody[i], see ImuRotation.h) and the
    // fixed IMU->rectified-camera extrinsic, from loadImuOrientation(). Both
    // empty if never loaded -- estimateTwoViewPose() falls back to
    // decomposing a homography in that case, same as before this existed.
    std::vector<cv::Mat> m_oxtsNavFromBody;
    imu_rotation::ImuToCameraCalib m_imuToCameraCalib;

    // Full per-frame ground-truth poses from loadGroundTruthPoses(), one
    // (R, t) pair per poses.txt line: world<-camera_i, i.e. p_world = R*p_cam
    // + t (so t alone is already the camera center in world, matching
    // MapView's tx/tz-only reading of the same file). Both empty if never
    // loaded -- drawGroundTruthOverlay() is then a no-op. index i == frame
    // i+1 (see drawGroundTruthOverlay()'s use of m_frameCount).
    std::vector<cv::Mat> m_groundTruthR;
    std::vector<cv::Mat> m_groundTruthT;
    bool m_groundTruthOverlayEnabled = true;
    int m_groundTruthOverlayOffsetX = 0;
    int m_groundTruthOverlayOffsetY = 0;
    int m_oldStreetOverlayOffsetX = 0;
    int m_oldStreetOverlayOffsetY = 0;

    // Most recently shown frame (BGR, keypoints already drawn if any, GT
    // overlay NOT yet drawn) and the frameCount it corresponds to -- lets
    // refreshGroundTruthOverlayDisplay() redraw just the overlay on demand
    // (e.g. when the offset changes) without re-reading anything from
    // m_source. Empty until the first frame is ever shown (previewFirstFrame()
    // or processNext()).
    cv::Mat m_lastDisplayBase;
    int m_lastDisplayFrameCount = 0;

    // Sparse map: 3D points (world frame) with matching SIFT descriptors.
    // Persists for the whole run -- never cleared just because tracking
    // temporarily fails, only on an explicit Start/Reset.
    std::vector<cv::Point3f> m_mapPoints;
    cv::Mat m_mapDescriptors;

    // Covisibility-driven local map for trackFrame() -- see
    // setCovisibilityLocalMapEnabled()/buildCovisibilityLocalMap(). Same
    // parallel-array shape as m_mapPoints/m_mapDescriptors/m_mapPointIds
    // above, just a different (covisibility-selected, not recency-capped)
    // subset. Rebuilt once per keyframe insertion; empty (and unused) when
    // the toggle is off.
    std::vector<cv::Point3f> m_localMapPoints;
    cv::Mat m_localMapDescriptors;
    std::vector<long long> m_localMapPointIds;
    bool m_covisibilityLocalMapEnabled = false; // see setCovisibilityLocalMapEnabled()
    // buildCovisibilityLocalMap() only runs from insertKeyframe(), which can
    // silently stop succeeding during a degraded-tracking patch (its own
    // reference-frame matching fails) -- without this, the local map would
    // go stale and keep being used anyway, which measurably made tracking
    // WORSE during exactly the conditions it's stuck in (confirmed this
    // session: a live run stopped rebuilding it at kf#10 but kept tracking
    // against that same stale map for thousands of subsequent frames).
    // trackFrame() falls back to the flat global map once this exceeds
    // kCovisibilityMapStaleFrames instead of trusting an old local map
    // indefinitely.
    int m_framesSinceCovisibilityMapRebuild = 0;
    bool m_guidedSearchEnabled = false; // see setGuidedSearchEnabled()
    bool m_qualityDrivenKeyframesEnabled = false; // see setQualityDrivenKeyframesEnabled()

    // Parallel to m_mapPoints -- a stable ID per live map point, surviving
    // appendToMap()'s rolling eviction in lockstep with the point itself.
    // Lets runLoopBundleAdjustment() write optimized positions back into
    // m_mapPoints for whichever landmarks are still live in it (most won't
    // be, for an old loop span -- that's fine, the authoritative copy for
    // BA purposes is m_landmarkPositions/the owning Keyframe, not this).
    std::vector<long long> m_mapPointIds;
    long long m_nextLandmarkId = 0;

    // Real multi-view bundle adjustment (runLoopBundleAdjustment(), fired
    // from tryLoopClosure() when enabled) needs each landmark's position
    // plus every keyframe that observed it -- neither of which existed
    // anywhere in this codebase before, since each keyframe only ever kept
    // the points *it* personally triangulated (Keyframe::localMapPoints),
    // with no notion of a later keyframe re-observing the same physical
    // point. These two maps, both keyed by the landmark ID assigned at
    // triangulation time (see localMapPointIds above), are that missing
    // piece:
    //   - m_landmarkPositions: current best 3D position (seeded at
    //     triangulation, overwritten in place after each BA solve).
    //   - m_landmarkObservations: every (keyframe index, observed 2D pixel)
    //     pair ever recorded for that landmark -- its owning keyframe's own
    //     observation (added in insertKeyframe()) plus any later keyframe
    //     that re-matched it against the live map (added in
    //     recordLandmarkObservations()). A landmark needs >= 2 of these
    //     from keyframes inside a given BA window to constrain anything;
    //     runLoopBundleAdjustment() filters for that itself.
    // Grows for the whole run (one entry per landmark ever triangulated,
    // not per live map point), but each entry is small and this is no
    // worse memory-wise than the already-unbounded m_keyframeHistory below.
    std::unordered_map<long long, cv::Point3f> m_landmarkPositions;
    std::unordered_map<long long, std::vector<std::pair<int, cv::Point2f>>> m_landmarkObservations;

    // The triangulating keyframe's own descriptor row for this landmark,
    // seeded once at creation time (insertKeyframe()) and never updated --
    // fuseWindowLandmarks()'s own descriptor-similarity gate (see
    // setLandmarkFuseEnabled()) is the only reader; a single representative
    // descriptor is enough for that purpose (real ORB-SLAM3's own
    // ComputeDistinctiveDescriptors() picks one representative descriptor
    // per map point too, rather than averaging or keeping all of them).
    std::unordered_map<long long, cv::Mat> m_landmarkDescriptors;

    // Reverse index, parallel to m_keyframeHistory: ALL landmark IDs
    // keyframe i has ANY observation of -- both the ones it originally
    // triangulated (Keyframe::localMapPointIds) AND every later
    // re-observation recordLandmarkObservations() finds for it. Exists so
    // runLocalBundleAdjustment() (and, in principle, the other BA
    // functions) can find every landmark a window's keyframes actually
    // observe, not just the ones some in-window keyframe happened to be
    // the ORIGINAL triangulator of -- confirmed this session that the
    // ownership-only rule silently drops real, already-recorded multi-view
    // observations from local BA whenever a landmark's owning keyframe has
    // scrolled just outside the window while still being actively
    // re-observed by keyframes inside it, needlessly starving BA of
    // constraint density it already has the data for.
    std::vector<std::vector<long long>> m_keyframeObservedLandmarkIds;

    bool m_loopBundleAdjustmentEnabled = false;

    cv::Mat m_currR;
    cv::Mat m_currT;

    // Constant-velocity motion model (see setGuidedSearchEnabled()): the
    // relative transform (rotation + translation) from the previous accepted
    // pose to m_currR/m_currT, i.e. "last frame-to-frame motion". Identity/
    // zero (no-op) until the second successfully tracked step, and reset
    // back to identity/zero whenever m_currR/m_currT jumps discontinuously
    // (recoverViaEpipolar() success) rather than through a normal per-frame
    // trackFrame() step -- a velocity estimate spanning a tracking-loss gap
    // describes drift/recovery, not real per-frame motion, and predicting
    // forward with it would be worse than assuming no motion at all.
    // Composed forward one more step (predR = m_velocityR * m_currR, predT =
    // m_velocityR * m_currT + m_velocityT) to predict the pose for the frame
    // about to be tracked, exactly like ORB-SLAM2/3's own constant-velocity
    // model.
    cv::Mat m_velocityR;
    cv::Mat m_velocityT;

    QVector<QPointF> m_trajectory;
    QVector<int> m_trajectoryFrameIndex; // m_frameCount at each m_trajectory entry, in lockstep -- lets
                                          // tryLoopClosure() find which trajectory points fall inside a
                                          // detected loop's frame range to correct them

    // One entry per keyframe ever inserted (never evicted, unlike
    // m_mapPoints' rolling window) -- the history tryLoopClosure() searches
    // for a revisit. localMapPoints/localMapDescriptors are exactly the
    // points this keyframe contributed to the map at insertion time (kept
    // here independently of the global map's eviction), so a much-later
    // loop match still has something concrete of this keyframe's to
    // PnP-solve against.
    struct Keyframe
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        // Parallel to keypoints/descriptors: which landmark id (if any, -1 =
        // unassigned) this specific detected keypoint has been confirmed
        // (via a real reprojection-gated match, same standard
        // recordLandmarkObservations() already uses) to be an observation
        // of. This is the piece real ORB-SLAM3's own KeyFrame::mvpMapPoints
        // provides and this codebase lacked before fuseWindowLandmarks()'s
        // v4 redesign (see DEBUGGING.md item 19) -- without it, there was
        // no way to distinguish "this landmark is ALSO visible in this
        // OTHER keyframe, safe to just add an observation" from "two
        // DIFFERENT landmarks are both claiming to explain the exact same
        // detected feature, a real conflict" -- items 16-18's pure-3D-
        // distance version conflated the two, which item 18 confirmed was
        // actively harmful once BA had to reconcile the resulting
        // geometrically-inconsistent residuals.
        std::vector<long long> keypointLandmarkId;
        cv::Mat R, t;
        std::vector<cv::Point3f> localMapPoints;
        cv::Mat localMapDescriptors;
        std::vector<cv::Point2f> localMapImagePoints; // this keyframe's own 2D observation of each
                                                        // localMapPoints entry (parallel array) --
                                                        // refineLocalKeyframes()'s reprojection target
        std::vector<long long> localMapPointIds; // parallel to localMapPoints -- the same IDs used in
                                                  // m_mapPointIds/m_landmarkObservations, so a landmark
                                                  // this keyframe triangulated can be recognized again
                                                  // later (by any keyframe re-observing it via
                                                  // recordLandmarkObservations()) and jointly refined by
                                                  // runLoopBundleAdjustment()
        int frameIndex = 0;
        DBoW2::BowVector bowVec; // computed at insertion time (insertKeyframe()) only when an ORB
                                  // vocabulary is loaded and the ORB detector is active; empty
                                  // (default-constructed) otherwise -- tryLoopClosure()'s DBoW2 branch
                                  // skips any keyframe with an empty bowVec, so this is safe to leave
                                  // unused when setDbowLoopClosureEnabled() is off
        cv::Mat vladVector; // computed at insertion time (insertKeyframe()) only when a VLAD codebook
                             // is loaded and the SIFT detector is active; empty otherwise --
                             // tryLoopClosure()'s VLAD branch skips any keyframe with an empty
                             // vladVector, so this is safe to leave unused when
                             // setVladLoopClosureEnabled() is off. See vision/VladVocabulary.h.
        DBoW2::BowVector siftBowVec; // computed at insertion time only when a SiftVocabulary is
                                      // loaded and the SIFT detector is active; empty otherwise --
                                      // tryLoopClosure()'s SIFT-DBoW2 branch skips any keyframe with
                                      // an empty siftBowVec, so this is safe to leave unused when
                                      // setSiftDbowLoopClosureEnabled() is off. Separate field from
                                      // the ORB-only bowVec above (not reused) so the two vocabulary
                                      // types can never be cross-scored against each other by mistake.
        bool culled = false; // set by cullRedundantKeyframes() (see setKeyframeCullingEnabled()) when
                              // this keyframe is judged redundant via the covisibility graph -- ONLY
                              // skips this keyframe as a FUTURE tryLoopClosure() candidate; deliberately
                              // never physically erased or cleared. This codebase indexes keyframes by
                              // raw position in m_keyframeHistory everywhere (m_landmarkObservations,
                              // m_sequentialEdgeRecords/m_loopClosureRecords via
                              // sequentialEdgeRecords()/loopClosureRecords(), runLoopBundleAdjustment()'s
                              // window bounds) -- actually erasing an entry would shift every later
                              // index and silently corrupt all of those, so culling here is
                              // intentionally "soft" (a search-time skip only), not a real memory
                              // reclamation. A true erase-and-compact would need those systems
                              // converted to stable IDs first; left as a known limitation, not
                              // attempted here.
    };
    std::vector<Keyframe> m_keyframeHistory;

    // Every sequential/loop-closure relative-pose measurement captured so
    // far this run, at observation time -- purely for
    // pose_graph::optimizePoseGraph() (see keyframePoses()/
    // sequentialEdgeRecords()/loopClosureRecords() above); grows for the
    // whole run like m_keyframeHistory itself, never consumed/cleared by
    // live tracking.
    std::vector<pose_graph::SequentialEdgeRecord> m_sequentialEdgeRecords;
    std::vector<pose_graph::LoopClosureRecord> m_loopClosureRecords;

    // See loadOrbVocabulary()/setDbowLoopClosureEnabled(). Null until
    // loadOrbVocabulary() succeeds -- every DBoW2 call site checks this
    // (and m_dbowLoopClosureEnabled) before use, so an unloaded vocabulary
    // just falls through to the original raw-match-count loop search.
    std::unique_ptr<OrbVocabulary> m_orbVocabulary;
    bool m_dbowLoopClosureEnabled = false;

    // See loadVladVocabulary()/setVladLoopClosureEnabled(). Null until
    // loadVladVocabulary() succeeds -- every VLAD call site checks this
    // (and m_vladLoopClosureEnabled) before use, so an unloaded codebook
    // just falls through to the original raw-match-count loop search.
    std::unique_ptr<vlad::VladVocabulary> m_vladVocabulary;
    bool m_vladLoopClosureEnabled = false;

    // See loadSiftVocabulary()/setSiftDbowLoopClosureEnabled(). Null until
    // loadSiftVocabulary() succeeds -- every call site checks this (and
    // m_siftDbowLoopClosureEnabled) before use, so an unloaded vocabulary
    // just falls through to VLAD, then the raw-match-count loop search.
    std::unique_ptr<SiftVocabulary> m_siftVocabulary;
    bool m_siftDbowLoopClosureEnabled = false;

    // See setLoopConsistencyGroupEnabled(). m_pendingLoopOldIdx == -1 means
    // no candidate is currently pending confirmation.
    bool m_loopConsistencyGroupEnabled = false;
    int m_pendingLoopOldIdx = -1;
    size_t m_pendingLoopNewKfIdx = 0;
    int m_pendingLoopStreak = 0;

    // See setLoopSpatialConsensusEnabled().
    bool m_loopSpatialConsensusEnabled = false;

    // See setLandmarkFuseEnabled()/fuseWindowLandmarks(). m_fusedLandmarkCount
    // is a running total, reported at shutdown for visibility into how much
    // this pass actually does on a given run.
    bool m_landmarkFuseEnabled = false;
    bool m_landmarkFuseMergeEnabled = false;
    long long m_fusedLandmarkCount = 0;

    bool m_keyframeCullingEnabled = false; // see setKeyframeCullingEnabled()
    int m_minTrackInliers = 10; // see setMinTrackInliers(); must match kMinTrackInliers's own default in
                                 // SlamWorker.cpp so behavior is unchanged until explicitly overridden
    int m_localBaWindowKeyframes = 8; // see setLocalBaWindowKeyframes(); must match the old
                                       // kLocalBaWindowKeyframes constexpr's own default in SlamWorker.cpp
                                       // so behavior is unchanged until explicitly overridden
    double m_detectionScale = 0.5; // see setDetectionScale(); must match the old kDetectionScale
                                    // constexpr's own default in SlamWorker.cpp so behavior is
                                    // unchanged until explicitly overridden
    bool m_localBundleAdjustmentEnabled = false; // see setLocalBundleAdjustmentEnabled()
    bool m_globalBundleAdjustmentEnabled = false; // see setGlobalBundleAdjustmentEnabled()

    // See setGlobalBundleAdjustmentAsyncEnabled()/tryIntegratePendingGlobalBa().
    bool m_globalBundleAdjustmentAsyncEnabled = false;
    bool m_pendingGlobalBaValid = false;
    int m_pendingGlobalBaTriggerKfIdx = -1;
    int m_pendingGlobalBaIntegrateAtKfIdx = -1;
    std::vector<cv::Mat> m_pendingGlobalBaR; // parallel, indices [0, triggerKfIdx]
    std::vector<cv::Mat> m_pendingGlobalBaT;
    std::unordered_map<long long, cv::Point3f> m_pendingGlobalBaLandmarks;

    // Applies a pending async global BA result once enough keyframes have
    // been inserted since it was solved (see
    // setGlobalBundleAdjustmentAsyncEnabled()) -- called at the top of
    // insertKeyframe() so it's checked on every keyframe-insertion attempt.
    void tryIntegratePendingGlobalBa();

    int m_frameCount = 0;
    int m_framesSinceKeyframe = 0;
    int m_trackFailStreak = 0; // consecutive failed PnP frames; display only, never triggers a wipe

    bool m_realtimeThrottle = true; // see startUnthrottled()

    QString m_lastOpenedVideoPath; // set by openVideoFile(), empty for a camera source -- see
                                    // autoLoadKittiExtras()'s sequence-number detection

    // See setOrbSlam3Enabled()/trackFrameOrbSlam3(). m_orbSlam3System is
    // null until start() lazily constructs it (needs the video source
    // already open); torn down (Shutdown() then reset) on disable, reset(),
    // or opening a new source, same as every other piece of run state above.
    // m_orbSlam3Clock is (re)started right when the System is constructed --
    // a generic video/camera source has no real per-frame capture
    // timestamps the way KITTI's times.txt gives orbslam3_kitti_ate.cpp, so
    // wall-clock elapsed time is the best available substitute for
    // TrackMonocular()'s monotonically-increasing timestamp requirement.
    bool m_orbSlam3Enabled = false;
    std::unique_ptr<ORB_SLAM3::System> m_orbSlam3System;
    QElapsedTimer m_orbSlam3Clock;
};
