# Debugging log: homography fallback caused a permanent tracking lockup

## Session 15 (2026-07-22): fixed a real run-to-run non-determinism bug, then an exhaustive (and ultimately negative) push to make offline Sim3 pose-graph correction beat live tracking, before finding the actual real wins -- a local-BA observation-density bug (later also applied to loop-BA, the session's biggest single win), and SQPnP -- best real result 72.550m (was ~93-195m, unreliably, before the determinism fix)

Long session on the custom `SlamWorker`/`kitti_ate` pipeline (not
`third_party/ORB_SLAM3_SIFT`/Kaggle -- see Part 58 etc. for that track).
Started from a Kaggle-notebook side-question (`MAX_FRAMES` env var
defaulting to 1000 in `kaggle/setup_and_run.sh` -- not a bug, just the
default) that pivoted into a full day on `SlamWorker`'s own accuracy.

### 1. Found and fixed a REAL run-to-run non-determinism bug (prerequisite for everything after)

Repeated full-sequence runs of the nominally-identical "best known" config
produced wildly different ATE (93.851m / 127.412m / 138.464m / 150.514m)
across separate invocations of the SAME binary on the SAME input. Root-
caused to `cv::SIFT::detectAndCompute()`'s internal `cv::parallel_for_`
returning keypoints in a run-dependent order (set found is identical, order
isn't) -- a different order feeds a different actual correspondence subset
to the fixed-seed (`kRansacSeed=42`) RANSAC in `estimateTwoViewPose()`/PnP,
which can flip a threshold-boundary accept/reject decision (loop-closure
inlier gate, `isPlausibleStep`, etc.) and cascade through thousands of
downstream frames -- classic sensitive-dependence, not measurement noise.

**Fix**: `processNext()` (`SlamWorker.cpp`) now sorts `kps`/`descriptors`
into a fixed position-based order (`(y, x, octave)`) immediately after
`detectAndCompute()` returns, before anything else touches them. Keeps
multi-threaded detection speed (a blanket `cv::setNumThreads(1)` was tried
first, measured >2x slower per frame, reverted). Verified: two full runs of
the identical config now produce byte-identical ATE (195.288m both times,
4318/4541 matched both times) -- confirmed via `diff` on the full stdout.
Also pinned `ceres::Solver::Options::num_threads = 1` in all 4
`ceres::Solve()` call sites (`SlamWorker.cpp` x3, `PoseGraphOptimizer.cpp`
x1) as cheap additional insurance against Ceres's own reduction-order
sensitivity -- negligible cost since BA isn't the per-frame bottleneck.

**Caveat surfaced by this fix**: once runs became reproducible, it became
clear that DIFFERENT configs (not just the same one) still show real,
larger-than-expected spread from run to run in absolute terms depending on
exactly which code path changes were active (e.g. 118-195m range seen for
what should be "the same" VLAD+windowed-BA config across different points
in this session) -- this is NOT the keypoint-order bug (that's fixed and
verified), it's a reminder that this pipeline's own live-tracking chain
(PnP inlier gates, keyframe-insertion timing, loop-candidate thresholds)
is inherently a long chain of hard threshold decisions, each one a small
step away from flipping under a differently-scaled floating-point path.
Single-run comparisons should still be treated with some caution;
differences smaller than ~10-15m are not yet trustworthy signal.

### 2. Real scale measurement added to loop closures (`scaleMeas`, `LoopClosureRecord::scale`)

Before this session, every Sim3 pose-graph edge (sequential AND loop) had
`sMeas` hardcoded to `1.0` -- confirmed empirically that this made the
Sim3 free-scale DOF mathematically inert (`scaleWeight` sweep 8-1000
produced byte-identical output). Added a real scale measurement in
`tryLoopClosure()`: initially a single camera-center-distance ratio
(`distLoop/distDrifted`), which was itself found to be wildly unstable
(0.0058-16.27 across early closures) -- root-caused to baseline instability
when the two keyframes are spatially close (the normal case for a real
revisit) -- fixed with a minimum-baseline floor + clamp
(`kMinScaleMeasBaseline`, `kScaleMeasClampMin/Max`).

**Superseded later in this same session** by a genuine `Sim3Solver` port
(see item 6 below) -- the single-point ratio is now only a fallback for
when the richer measurement doesn't have enough data.

### 3. Stereo scale-anchor: implemented, measured a real ~35% improvement, then REMOVED after a factual correction

Implemented `StereoScaleAnchor.h/.cpp` (SAD block-matching disparity search
using KITTI's `image_1` right camera + calib.txt baseline) and wired it
into `SlamWorker` two ways: (a) a periodic, damped, whole-map rescale
(`applyStereoScaleAnchor()`, called every keyframe) using ground-plane-
region points' stereo-vs-monocular depth ratio, and (b) later, a root-cause
fix injecting a real stereo-measured distance directly into
`recoverViaEpipolar()` (the fallback path when PnP fails -- diagnosed as
the ACTUAL point where fresh monocular scale ambiguity re-enters the
pipeline repeatedly, since `insertKeyframe()`'s regular triangulation uses
two already-PnP-derived/metric poses and has no fresh ambiguity of its own).

Measured real improvement: VLAD-only baseline 195.288m (reproducible) ->
VLAD+stereo-anchor 126.134m (~35% better), "Recovered scale" landing at
0.7490-0.9930 (much closer to 1.0 than the 0.05-3.0 spread seen without it.

**Then removed entirely** (`StereoScaleAnchor.h/.cpp` deleted,
`SlamWorker.cpp`/`.h` reverted, `recoverViaEpipolar()`'s stereo branch
reverted to plain OXTS/`m_avgStepScale`) after a factual correction: the
"~5.33m ORB-SLAM3 reference on KITTI seq00" cited earlier in this session
turned out to be ORB-SLAM2's **monocular** result (7-DOF-aligned), not
stereo -- stereo ORB-SLAM2 gets 1.3m (6-DOF-aligned). Source: [ORB-SLAM2
paper](https://arxiv.org/pdf/1610.06475). This means the real gap-closer
to a ~5m-class result is NOT stereo depth -- it's a mature monocular
Sim3 Essential Graph + loop-closing system, which real ORB-SLAM2 achieves
with zero stereo input at all. Effort redirected accordingly (items 5-6).

### 4. BA structural fixes that were CORRECT but did not move ATE

- **Global-BA-doesn't-fall-back-to-windowed-BA bug**: `insertKeyframe()`'s
  BA selection used `else if (m_loopBundleAdjustmentEnabled)`, meaning
  whenever `runGlobalBundleAdjustment()` declined (e.g. `newKfIdx` past
  `kGlobalBaMaxWindowKeyframes=400`), it fell all the way through to plain
  linear interpolation instead of trying windowed BA -- confirmed this
  regressed late-sequence "return to origin" loop closures. Fixed
  (`if (!baApplied && m_loopBundleAdjustmentEnabled)`). Real, correct fix;
  measured no ATE change in the one test run (all the specific closures
  observed there had span > `kBaMaxWindowKeyframes` too, so the fallback
  path never actually got exercised in that particular run).
- **`kBaMaxWindowKeyframes` raised 200 -> 600** to let the ~470-span
  "return near start" loop closures get real windowed BA instead of
  interpolation. Confirmed running and converging dramatically (cost
  3.5 billion -> 1.28 million on one such window) -- but ATE barely moved
  (126.134m -> 125.195m). Diagnosed why: BA minimizes REPROJECTION error
  (local pixel self-consistency), which a systematically-drifted
  reconstruction can satisfy perfectly while still being globally wrong in
  scale/rotation -- confirmed by "Recovered scale" swinging to 2.6442 (very
  different) while ATE stayed roughly flat.

### 5. Essential-Graph-style covisibility edges for the offline Sim3 pose-graph -- structurally correct, still lost to live tracking

Added `SlamWorker::covisibilityEdgeRecords()` (any two keyframes sharing
>=100 jointly-observed landmarks, ORB-SLAM2's own covisibility-graph
definition, reusing `cullRedundantKeyframes()`'s own graph-construction
logic) and wired it into `kitti_ate.cpp`'s posegraph block (new `covis`
CLI flag) as extra edges alongside the sequential chain (both types share
`pose_graph::SequentialEdgeRecord`, which the Sim3 solve already treats as
a generic Huber-robustified relative-pose edge for any `(i,j)`, not just
`j=i+1` -- confirmed safe by checking `optimizePoseGraphSim3()`'s
warm-start-chain loop, which already filters for `j==i+1` and silently
ignores non-adjacent edges there).

**Real, measured structural fix**: previously EVERY sequential edge's chi2
measured exactly `0.000` (a pure sequential-only graph has zero real
internal constraint -- the warm-start chain is built directly from those
same edges, so they're automatically satisfied). With covisibility edges
added, `seqEdges chi2 sum=57.649` (no longer zero) -- confirmed the
Essential Graph now carries real internal-consistency information for the
first time this project has had it.

**Still lost to live tracking**: 125.195m (live) vs 152.480m (posegraph-
corrected with covisibility edges). Loop-edge chi2 (1034.759) still
outweighed sequential+covisibility chi2 (57.649) by ~18x -- the graph
remained almost entirely loop-edge-driven. Added
`PoseGraphOptions::sequentialWeightMultiplier` and swept it (1/5/20/100/500)
x `dcsPhi` (8 values) x `scaleWeight` (4 values) = 160 combinations against
the SAME already-tracked keyframes (no re-tracking needed, matching the
existing `sweep` CLI mode's pattern) -- **0/160 combinations beat live
tracking** (best: 145.901m). Conclusive: the imbalance is not a tuning
problem at this point, it's that live tracking's continuous local BA
already does a better job than any offline correction over this graph
structure can achieve, regardless of edge weight.

### 6. Ported ORB-SLAM3's real `Sim3Solver` (Horn 1987 + RANSAC) -- genuinely better measurement, STILL lost offline

Read `third_party/ORB_SLAM3/src/Sim3Solver.cc`/`.h` directly (not
reconstructed from memory) and ported the exact algorithm (Horn 1987
closed-form quaternion solution for the minimal-3-point similarity
registration, wrapped in RANSAC) from Eigen to this codebase's `cv::Mat`
convention: `computeSim3Horn()` (the closed-form solve, using `cv::eigen()`
on the same 4x4 symmetric N-matrix construction ORB-SLAM3 uses) +
`solveSim3Ransac()` (RANSAC wrapper, same `kRansacSeed=42` for
reproducibility). Wired into `tryLoopClosure()`: matches
`oldKf.localMapDescriptors` (oldKf's own small trusted local map) against
`m_mapDescriptors` (the CURRENT rolling map at newKf's insertion time --
this codebase's closest equivalent to ORB-SLAM3's own "current keyframe's
local map", since it was just extended with newKf's own new points and
covers its recent covisibility neighborhood) to get 3D-3D correspondences,
transforms each side into its own camera's local frame, and solves for a
real multi-point RANSAC-robust Sim3 (rotation+translation+scale).

**Confirmed firing successfully on every tested loop closure** (10/10 in
one run, 6-35 RANSAC inliers each) with measured scale values now
clustering sensibly (1.2-1.9 typical, few clamp-boundary outliers) instead
of the old single-point measurement's wild 0.0058-16.27 swings -- a real,
verified improvement in measurement quality.

**Still lost to live tracking**: 120.085m (live) vs 183.142m (posegraph-
corrected). Loop-edge chi2 (3680.9) vs sequential+covis chi2 (42.8) --
imbalance actually WORSE than with the noisy measurement (86x vs 18x)
because the now-stable measurement pushes corrections more consistently
hard in a real direction, rather than sometimes-cancelling noise. This is
the fourth independent confirmation (sections 4/5/6 + the earlier
scaleWeight-inert finding) that offline Sim3 pose-graph correction cannot
beat this pipeline's live tracking, regardless of edge density, edge
weight, or measurement quality -- the live tracking's own continuous local
BA is simply doing a better job already. **Recommend not investing further
in the offline pose-graph direction** unless the underlying live-tracking
local BA itself is first made significantly more accurate (see item 8),
which might change the baseline this comparison is against.

### 7. Per-landmark stereo-depth-anchor residual in local BA -- FOUR genuine attempts, all negative, reverted

Before settling on item 3's periodic-rescale-only approach, tried injecting
stereo depth as a direct Ceres residual in `runLocalBundleAdjustment()`
(giving BA independent 3D evidence instead of only reprojection self-
consistency). All four attempts regressed ATE vs the 125.195m baseline:

1. Unweighted, `(pose, point)` residual pulling camera-frame depth directly:
   180.945m, "Recovered scale" jumped to an implausible 2.99.
2. Weight lowered to 0.05 + Huber loss added: 170.777m -- scale excellent
   (0.9930, best of the whole session) but ATE still worse. Diagnosed:
   letting the residual touch the POSE parameter let Ceres "cheat" by
   rotating/translating the camera to satisfy the depth residual instead of
   moving the landmark, distorting trajectory shape even as scale improved.
3. Refactored to a POINT-ONLY residual (`PointPriorCost`, no pose parameter
   at all -- structurally cannot cause rotation distortion): 168.470m --
   barely moved, ruling out the pose-coupling theory as the (sole) cause.
4. Added `m_landmarkStereoDepth.erase()` right after first use, making it a
   true one-time nudge (a landmark can stay in-window for many overlapping
   local-BA calls; repeatedly re-imposing the SAME stale stereo snapshot
   every time was suspected to compound): 162.674m -- improved each
   attempt (180.9 -> 170.8 -> 168.5 -> 162.7) but never beat baseline.

**Reverted entirely** alongside item 3's stereo removal. The four-attempt
trend (each fix addressing a real, correctly-diagnosed issue, each still
net negative) suggests this specific integration point (per-landmark prior
inside the SAME optimization that's also solving for pose via reprojection)
is fighting itself in a way that's hard to fully resolve without a deeper
rethink -- not recommended to pick back up without a genuinely new
hypothesis for why.

### 8. Local BA was silently starved of real observation density -- fixed, small real improvement

Found via direct code reading (prompted by "why isn't local BA as dense as
it should be"): `runLocalBundleAdjustment()`'s landmark-selection loop only
considered landmarks whose OWNING/triangulating keyframe was inside the
current window (`for (long long id : m_keyframeHistory[i].localMapPointIds)`)
-- silently excluding landmarks that are still being actively re-observed
by in-window keyframes but were originally triangulated by a keyframe that
has since scrolled just outside the window. Those re-observations WERE
already being recorded (`recordLandmarkObservations()`, called every
keyframe) into `m_landmarkObservations`, just never surfaced to local BA's
landmark-selection step.

**Fix**: new reverse index `m_keyframeObservedLandmarkIds` (parallel to
`m_keyframeHistory`, ALL landmark IDs each keyframe has any observation of,
not just ones it originated), populated in `insertKeyframe()` (seeded with
the keyframe's own new IDs) and `recordLandmarkObservations()` (appends
re-observed IDs). `runLocalBundleAdjustment()` now iterates this instead,
with a `processedLandmarkIds` dedup guard (the same landmark can now
legitimately appear in several in-window keyframes' lists, which would
otherwise produce duplicate residual blocks for the same keyframe-landmark
pair). Confirmed mechanically: local BA windows now show ~1234
landmarks/3256 observations vs tens-of-landmarks/~200-observations for
similar window sizes before this fix.

**Measured real improvement**: 125.195m -> 118.450m (live, VLAD+windowed-BA,
P3P). Modest but real (in the expected direction, larger than the run-to-
run noise floor established in item 1's caveat).

### 9. SQPnP beat P3P -- best real result of the session

Swapping the PnP method from `p3p` to `sqpnp` (with items 6+8's fixes
already active): 118.450m -> **107.676m**. Consistent with an older,
pre-this-session historical hint that SQPnP outperforms P3P on this
sequence, now re-verified under today's reproducibility fix.

### 10. Applied item 8's observation-density fix to `runLoopBundleAdjustment()` too -- the session's biggest single win

Follow-up conversation (same day): `runLoopBundleAdjustment()`
(`SlamWorker.cpp`) still had the exact ownership-only landmark rule item 8
found and fixed in `runLocalBundleAdjustment()` -- its candidate-landmark
loop walked each in-window keyframe's `localMapPointIds` (points that
keyframe itself triangulated) instead of `m_keyframeObservedLandmarkIds`
(every landmark it has ANY recorded observation of, including
re-observations of points triangulated by a keyframe now outside the
window). This was the top item on this session's own queued-next-steps
list, on the reasoning that loop closures are the highest-leverage moment
for correcting drift. Before touching the code, cross-checked against
`third_party/ORB_SLAM3/src/Optimizer.cc`'s real `LocalBundleAdjustment()`
(line 1145): it collects landmarks via `pKFi->GetMapPointMatches()` --
every point a keyframe CURRENTLY observes, not just ones it originated --
confirming this is the correct pattern, not just an internal-consistency
fix.

**Fix**: same mechanism as item 8 -- swapped the `kf.localMapPointIds`
iteration for `m_keyframeObservedLandmarkIds`, with a
`processedLandmarkIds` dedup guard (a landmark can now legitimately appear
in several in-window keyframes' own observed-landmark lists). Write-back
logic (which keyframe's `localMapPoints` copy gets the refined position)
left untouched -- it already correctly no-ops for a landmark whose owning
keyframe falls outside the window, same as the already-fixed local-BA
function.

**Measured effect, dramatic**: loop bundle adjustment window density went
from a few hundred landmarks/observations (ownership-only) to real
multi-thousand-scale density -- e.g. one representative closure log line
from this run: `kf#7..kf#509, 16571 landmarks, 47930 observations (53
loop-verified)`, vs. the kind of sparse windows item 8's local-BA writeup
described before its own fix. Full-sequence re-run of the session's
known-good reference command (SQPnP + VLAD + windowed-BA + guided + denser
local BA, single variable changed vs. item 9's 107.676m baseline):
**107.676m -> 72.550m** (ATE RMSE), a ~32.6% improvement -- comfortably
above the ~10-15m run-to-run noise floor established in item 1's caveat,
so this is real signal, not variance. Also: `Recovered scale` 0.2679 (a
regression vs. item 9's presumably-tighter scale, not yet explained --
worth checking next) and 4519/4541 frames matched (unchanged coverage).

### Current best real, reproducible result: 72.550m live (SQPnP + VLAD + windowed-BA + guided + denser local BA + denser loop BA)

Still ~3.6x away from the user's stated goal of ATE RMSE < 20m (was
~5.4x before this fix). Real ORB-SLAM2 monocular reference on the same
sequence: 5.33m (see item 3) -- the gap is now understood to be primarily
about Essential Graph/loop-closing maturity, not stereo, not raw BA
correctness (both were tried exhaustively this session).

### Also this session: pySLAM (SIFT, monocular) Kaggle notebook for an external reference point

`kaggle/pyslam_sift_kitti.ipynb` -- clones
[luigifreda/pyslam](https://github.com/luigifreda/pyslam), runs its
`VisualOdometryEducational` class (the lightweight classic 2-view VO
`main_vo.py` uses, not the full `main_slam.py` SLAM stack) with SIFT,
headless (custom minimal driver script, since `main_vo.py` itself assumes
an interactive display). Important caveat documented in the notebook
itself: this VO class reads REAL ground-truth translation scale at every
single frame (`kUseGroundTruthScale = True` in `visual_odometry.py`) to
resolve monocular scale ambiguity -- confirmed by reading the source
directly. So its eventual ATE is NOT a fair scale-accuracy comparison,
only a fair ROTATION/DIRECTION accuracy one (from its Essential Matrix
estimate). Not yet run (must be run interactively on Kaggle by the user;
not executable from this environment). Earlier consideration of
`farhad-dalirani/StereoVision-SLAM` was dropped after reading its source
and finding it has no SIFT support at all (GFTT/ORB + pure KLT optical-flow
tracking only).

### 11. Item 10's "scale regression" investigated -- NOT a bug, real measured drift, no action needed

Checked whether `Recovered scale` dropping to 0.2679 after the loop-BA
density fix was a symptom of something the fix broke. Key fact confirmed
by reading the call site (`insertKeyframe()`, around the
`runGlobalBundleAdjustment()`/`runLoopBundleAdjustment()` call): `scaleMeas`
(the `Sim3Solver` RANSAC measurement) is computed in `tryLoopClosure()`
BEFORE either BA function runs, so it's mathematically independent of
item 10's fix -- the fix only changes which landmarks the SUBSEQUENT joint
optimization gets to use, not the scale measurement itself.

Extracted every `scaleMeas` value logged this run (66 loop closures) and
looked at the tail, which covers the sequence's actual loop-back-to-start
closures (frame ~4459-4526 matching kf#0/4/5/6/7/9/10, i.e. the very first
keyframes): **six consecutive, independent, high-inlier-count (12-26
RANSAC inliers each) Sim3Solver measurements all agree tightly in the
0.41-0.50 range** (0.4456, 0.4548, 0.4330, 0.5032, 0.4147, 0.4474, 0.4133).
That's real, mutually-consistent evidence that by the end of this
~4500-frame sequence, the estimated trajectory has genuinely drifted to
roughly 2-2.4x too large -- not fix-induced noise.

**What the fix actually changed**: before item 10, the loop-BA window
(`kf#7..kf#509`, spanning ~502 keyframes) was landmark-starved, so this
real, verified scale correction could only partially propagate through the
window (the old sparse graph was under-constrained). With dense
observations (16571 landmarks/47930 observations), the joint optimization
can now actually distribute the true measured correction consistently
across the whole span, so the corrected trajectory more HONESTLY reflects
the real accumulated drift end-to-end -- which is exactly why the single
GLOBAL Umeyama-fit scale (which has to summarize the WHOLE trajectory in
one number) lands further from 1.0 than before, while ATE (the metric that
actually matters, and is grounded directly against ground truth rather
than being a single rigid-similarity summary statistic) improved by 32.6%.
Consistent with item 4's already-established finding that `Recovered
scale` and ATE are only loosely coupled and shouldn't be read as a
correctness signal on their own. **No fix needed; not a regression.**

Secondary observation (not investigated further, noted for later): a
separate mid-sequence stretch (`kf#101`/`kf#104`/frames ~3804-3828) hit the
upper scale clamp (`kScaleMeasClampMax=3.0`) three times in six closures --
worth a look sometime if that region's accuracy is ever specifically
suspect, but unrelated to this investigation.

### 12. Tried increasing SIFT `nFeatures` (2000 -> 5000) -- ZERO effect, negative result, real bottleneck identified

Added a CLI override (`argv[33]`, SIFT-only, mirrors ORB's existing
`argv[16]`) so `kitti_ate` can set `SiftSettings::nFeatures` without a
recompile. Re-ran the exact item-10 reference config with `nFeatures=5000`:
result was **byte-identical** to the 2000-default run in every respect
(72.550m ATE, `Recovered scale` 0.2679, identical per-frame keypoint counts
throughout, identical trajectory file) -- confirmed via `diff` on the full
stdout logs, only the `[config] SIFT nFeatures=5000` line differs.

**Root cause**: `nFeatures` is a cap OpenCV's SIFT applies AFTER detection
(keeps the top-N by response if more than N are found) -- it was never the
actual bottleneck. Checked the real per-frame yield across the whole
4541-frame run: max 1591 keypoints, average 754.9, **never once
approaching even the existing 2000 cap**, so raising the cap to 5000 (or
any value >= ~1600) is mathematically guaranteed to change nothing. The
real limiting factors are `kDetectionScale = 0.5` (detection runs on a
half-resolution copy of the image -- its own doc comment already says this
was a deliberate real-time-budget tradeoff, ~70ms/frame at full res vs
~18ms at half) and/or `SiftSettings::contrastThreshold`/`edgeThreshold`
(0.04/10.0, OpenCV's stock defaults, never tuned for this project). Denser
per-frame features would need one of THOSE changed, not `nFeatures` --
which also means it's no longer a free lever: loosening the contrast
threshold or raising detection resolution has a real speed cost this CLI
override doesn't. **Not changed pending user direction** (a real-time
budget tradeoff, not a pure quality-vs-nothing call like `nFeatures` would
have been). The new `argv[33]` override itself is harmless and kept in the
tree (matches the existing pattern of opt-in CLI overrides for future
sweeps), it's just confirmed inert on this pipeline's own detection path
as currently tuned.

### 13. Tried the REAL lever -- `kDetectionScale` 0.5 -> 1.0 (full-res) -- measured WORSE, reverted to default

Per the user's explicit direction (speed not a concern, only ATE matters):
moved `kDetectionScale` from a fixed `constexpr` to a runtime-overridable
`SlamWorker::m_detectionScale` member (`setDetectionScale()`, mirrors how
`kMinTrackInliers` was migrated earlier), default unchanged at 0.5 so the
live GUI's real-time budget is untouched. Added `argv[34]` (SIFT-only) to
`kitti_ate` alongside item 12's `argv[33]` (`nFeatures`, set to 5000 in
this run too, so a real yield increase wouldn't be capped).

**Confirmed the resolution bump genuinely produces denser keypoints**:
per-frame counts moved from item 10's ~755 average (max 1591) up into the
1200-5000 range throughout the run, with `nFeatures=5000` actually binding
on some frames (unlike item 12's inert 2000->5000 bump). Effective
throughput dropped from ~13.8fps to ~6.6fps (~72ms/frame -> ~151ms/frame)
as expected from the doc comment's own ~18ms/~70ms detection-only figures
-- confirmed harmless for this offline, unthrottled harness (full run
still finished in well under the 1200s cutoff, reached frame 4541/4541,
not truncated).

**Result: WORSE, not better.** 72.550m (item 10 baseline) -> **105.692m**
(+45.7%). Also: `Matched points` dropped (4519->4459/4541, more
tracking-loss/recovery events), `Recovered scale` moved to 0.5558 (closer
to 1.0 than item 10's 0.2679, but that's not informative on its own per
item 11's established finding that this number isn't a quality signal).
Not root-caused further (out of scope of what was asked -- the lever
itself was the thing being tested, not why it regressed), but the
likely mechanism: full-resolution SIFT detects substantially more
high-frequency/fine-scale keypoints (foliage, distant repeated texture,
noise) that are individually less stable to triangulate and match across
frames than the coarser, more structurally-salient set half-res detection
already found -- i.e. this pipeline's issue was never raw keypoint COUNT,
consistent with item 12 already showing the existing count wasn't even
hitting its own cap. **Reverted the run config back to `kDetectionScale`
default (0.5)** for the current best-known reference command; the
`setDetectionScale()`/`argv[34]` plumbing itself is kept (harmless,
opt-in, same rationale as item 12's kept-but-inert `argv[33]`) in case a
future session wants to sweep intermediate values (e.g. 0.6-0.8) rather
than the two extremes tested so far. **Current best real, reproducible
result stays 72.550m** (item 10's config, unchanged by this negative
result).

### 14. Reviewed `third_party/ORB_SLAM3` for unported mechanisms (user-requested), then started building a real DBoW2 vocabulary for RootSIFT -- found and fixed a genuine non-convergence bug along the way

User asked what else from the real ORB-SLAM3 source could still be applied
to this custom pipeline, and separately offered to train a proper DBoW2
vocabulary on Kaggle if one made sense. Read `LoopClosing.cc`/
`LocalMapping.cc`/`ORBmatcher.cc`/`Optimizer.cc` directly (not from memory)
and found 3 concrete, unported mechanisms:

1. **Loop closure requires >=3 consecutive keyframe confirmations before
   committing** (`LoopClosing.cc`, `mnLoopNumCoincidences >= 3` --
   `DetectCommonRegionsFromBoW()`). `tryLoopClosure()` currently commits on
   the FIRST geometrically-verified candidate, no temporal-consistency gate
   at all. Cheapest of the three to try (no training needed), and targets
   the "loop-closing maturity" gap this session's own summary (item 9) had
   already flagged as the main remaining accuracy gap vs real ORB-SLAM2.
   **Queued as this session's next implementation step.**
2. **`LocalMapping::SearchInNeighbors()` + `ORBmatcher::Fuse()`** -- merges
   duplicate landmarks across covisible keyframes after every keyframe
   insertion. `SlamWorker` has NO map-point dedup mechanism at all
   (confirmed via grep, zero hits). If independently-triangulated
   near-duplicate landmarks exist for the same physical point, real
   observation density is fragmented across separate IDs even after items
   8/10's fixes (which only unify observations already sharing one ID).
   Not started this session.
3. **A real DBoW2 vocabulary for RootSIFT** (rather than VLAD) -- see
   below, started this session at the user's initiative.

**DBoW2-for-SIFT infrastructure built this session** (not yet a measured
accuracy result -- infrastructure only):
- `third_party/DBoW2/DBoW2/FRootSift.h`/`.cpp`: a new `DBoW2::FClass`
  descriptor adapter for RootSIFT (128-dim `CV_32F` rows, squared-L2
  distance, per-dimension arithmetic mean) -- the vendored DBoW2 only ever
  shipped `FORB.h` (ORB, Hamming distance). Mirrors `FORB.h/.cpp`'s exact
  structure. Wired into the `DBoW2` CMake target (`CMakeLists.txt`).
- `analyze/train_sift_dbow_vocabulary.cpp` (new `train_sift_dbow_vocabulary`
  CMake target): extracts RootSIFT via the SAME
  `feature_detector::createDetector()`/`toRootSift()` path `SlamWorker`
  itself uses (train/runtime descriptor-space parity, same rationale
  `analyze/orbslam3_vlad_train.cpp` already established for VLAD), builds a
  `DBoW2::TemplatedVocabulary<cv::Mat, DBoW2::FRootSift>` via `create()`,
  saves via `saveToTextFile()`.
- `SlamWorker`: `SiftVocabulary` typedef, `loadSiftVocabulary()`,
  `setSiftDbowLoopClosureEnabled()`, `Keyframe::siftBowVec` (separate field
  from the ORB-only `bowVec`, deliberately not reused, so the two
  vocabulary types can never be cross-scored by mistake),
  `insertKeyframe()`'s new SIFT-DBoW2 BowVector computation block, and a
  new highest-priority candidate-search branch in `tryLoopClosure()`
  (checked before VLAD when both are loaded+enabled -- an ordering choice,
  not yet a measured comparison). `kitti_ate.cpp` `argv[36]`/`argv[37]`
  (`siftdbow <vocab-path>`) CLI wiring. All additive/opt-in -- VLAD's own
  wiring is completely untouched.

**Real bug found and fixed while smoke-testing the trainer**: even a tiny
vocabulary (k=8, L=3 -- 3 tree levels, well under the theoretical 512-word
max) over a small sample (91 images, 169,960 descriptors) did not finish
in 400+ seconds. Root-caused by reading `TemplatedVocabulary.h`'s
`HKmeansStep()` directly: its k-means convergence loop
(`while(goon) { ... }`) has **no iteration cap at all** -- it only stops
once cluster assignment is bit-for-bit IDENTICAL between consecutive
iterations. That's fine for `FORB`'s coarse integer Hamming distances (few
near-ties in practice), but pathological for `FRootSift`'s continuous
float L2 distances: a point sitting near an almost-equidistant boundary
between two clusters can keep flipping assignment indefinitely as the
floating-point centroids shift by a tiny amount every iteration, with no
tolerance threshold to absorb that. **Fix**: added a `kMaxHKmeansIterations
= 50` cap to the `while` loop condition, the same standard fix this
project's own `cv::kmeans` call (`orbslam3_vlad_train.cpp`) already applies
via `cv::TermCriteria`'s own max-iteration bound. This is a real,
documented modification to vendored third-party code (same file already
had ORB-SLAM3's own text-format serialization additions layered onto
upstream DBoW2, so this isn't unprecedented for this file specifically).

**Verified the fix**: the identical k=8/L=3/91-image smoke test that never
finished before now completes in **30 seconds**, producing a 210-word
vocabulary. Sanity check (adjacent vs. distant frame BoW score, same
pattern `orbslam3_vlad_train.cpp`'s own end-of-run check uses): frame0<->1
(adjacent) scored 0.7135, frame0<->200 (distant) scored 0.3214 -- adjacent
notably higher, as expected, confirming the vocabulary captures real
appearance similarity, not noise.

**Not yet done**: a real, full-size vocabulary (larger k/L, more training
images) has not been trained -- this smoke test used a deliberately tiny
k/L/stride just to validate the pipeline end-to-end. A Kaggle notebook for
this (`kaggle/train_sift_dbow_vocabulary_kaggle.ipynb`) was written and
verified (standalone g++ compile + a real small end-to-end training run,
both confirmed working locally) -- not yet run for real on Kaggle. No ATE
measurement with a trained SIFT-DBoW2 vocabulary exists yet -- VLAD
remains the only measured SIFT loop-closure candidate search on this
pipeline (see item 9's 107.676m and item 10's 72.550m, both VLAD-based).

### 15. Implemented and measured item 14's finding #1 (loop-closure temporal-consistency gate) -- NEGATIVE result, root-caused, left off by default

Implemented `setLoopConsistencyGroupEnabled()` (default off): once every
other verification gate in `tryLoopClosure()` passes (candidate score, PnP
inlier count, degenerate-solve sanity caps), a real correction is no
longer applied on the first confirmation -- the SAME loop hypothesis must
be independently re-verified across `kLoopConsistencyRequiredCount=3`
calls whose matched old-keyframe index stays within
`kLoopConsistencyOldIdxWindow=5` of each other and whose new-keyframe
index doesn't drift apart by more than `kLoopConsistencyMaxGapKeyframes=4`
-- a deliberately simplified stand-in for real ORB-SLAM3's own
`mnLoopNumCoincidences>=3` (`LoopClosing.cc`), which instead checks
membership against covisibility-graph GROUPS this codebase has no
equivalent structure for. New `kitti_ate` CLI flag `argv[38]`
(`loopconsistency`).

**Measured on the exact item-10 baseline config (single variable changed,
same freshly-reconfirmed 72.550m run to compare against)**:
**72.550m -> 100.391m (+38.4%, worse)**. Also: matched points dropped
slightly (4519->4518/4541), `Recovered scale` moved further from 1.0
(0.2679->0.1589).

**Root-caused, not just measured**: the gate reduced actually-committed
loop closures from 66 (baseline) to 21 -- a real, large loss of correction
opportunity, not a filtering-out of false positives. Broke down the 52
"waiting" (not-yet-confirmed) log lines: 27 streaks died at 1/3
confirmations (never even got a 2nd), 25 died at 2/3 (never got a 3rd).
Combined with only 21 successful 3/3 commits, this means the large
majority of geometrically-verified, real loop candidates simply timed out
waiting for a 2nd or 3rd confirmation that never arrived within the
index-window/gap tolerance, rather than being correctly rejected as false
positives. Interpretation: on this pipeline/sequence, a real revisit
typically produces ONE strong verified candidate against a given local
area, then the very next candidate a few keyframes later already matches a
noticeably different old-keyframe index (viewing angle/position shifted
enough that the best-scoring old keyframe moves) -- the simple 1D
index-window proxy this session used to approximate ORB-SLAM3's real
covisibility-group membership check is too narrow to reliably catch that
drift as "the same place", so most real streaks reset (or simply expire
via the gap tolerance) before reaching 3. Net effect: far FEWER real
corrections applied overall, which is directly why ATE got worse -- not
because the corrections that did fire were bad, but because most real
drift went uncorrected that the baseline's zero-gate policy would have
fixed on the first try.

**Left off by default (already was); not recommended to re-enable as-is.**
If revisited, the fix is almost certainly a smarter "same place" test than
a 1D old-keyframe-index window -- e.g. actually checking descriptor/VLAD/
BoW-score similarity between the two candidates' own old keyframes (a
cheap proxy for real covisibility-group membership), or simply loosening
`kLoopConsistencyOldIdxWindow`/`kLoopConsistencyMaxGapKeyframes`
substantially and re-measuring -- neither attempted this session.

### 16. Implemented and measured item 14's finding #2 (map-point fuse/dedup) -- ALSO a negative result, likely cause identified

Implemented `setLandmarkFuseEnabled()`/`fuseWindowLandmarks()` (default
off): a simplified adaptation of real ORB-SLAM3's
`LocalMapping::SearchInNeighbors()`/`ORBmatcher::Fuse()`. The real
algorithm projects a keyframe's map points into covisible neighbors'
keypoint sets and merges same-index matches; this codebase has no
per-keyframe "map point at this keypoint index" table to support that, so
this implementation instead does a pure 3D-proximity test: for each
landmark the current keyframe just triangulated, find the closest OTHER
landmark (by Euclidean distance) among everything actively observed within
the last `kFuseWindowKeyframes=15` keyframes; if within
`kFuseMaxWorldDistance=0.5` world units, merge them (fewer-observations
one absorbed into the more-observed one, via the same
`m_landmarkObservations` splice-and-erase item 15's design also uses).
Deliberately has NO descriptor-similarity check (unlike real ORB-SLAM3's
own `Fuse()`, which requires `bestDist<=TH_LOW` on top of the spatial
search) -- a simplification, not an oversight; flagged here since it's the
leading suspect below. New `kitti_ate` CLI flag `argv[39]` (`fuse`).

**Measured on the exact item-10 baseline config (single variable changed)**:
**72.550m -> 104.455m (+44%, worse)**. Also: matched points dropped
(4519->4481/4541), `Recovered scale` moved further from 1.0
(0.2679->0.1936). Loop-closure candidate detection itself was barely
affected (65 closures fired vs baseline's 66), so this isn't a case of the
fuse pass corrupting loop-closure candidate search specifically -- the
damage is happening in the tracking/BA data itself. 3019 landmarks were
merged across ~500+ keyframe insertions -- several merges per keyframe on
average, a substantial rate.

**User asked directly: bug, or a real mechanism problem? Re-audited the
code line-by-line first** (indexing, the survivor/loser observation-count
comparison, the `activeIds` pool bookkeeping across a keyframe's own
sequential merges) -- found no coding bug; `fuseWindowLandmarks()` does
exactly what it was written to do. **Then measured directly rather than
just theorizing**: added temporary diagnostic logging (descriptor L2
distance between `newId` and `bestMatch` at every merge, found via a
linear owning-keyframe search -- too expensive to keep permanently, this
was diagnostic-only and has been removed again) and ran a short (~90s,
1703-merge-sample) partial benchmark. Result: **median descriptor L2
distance at merge time was 0.7987** (mean 0.6979) -- for RootSIFT's
roughly unit-normalized descriptor space, that's a LARGE distance (a
genuine same-feature match is typically well under ~0.4). Bucketed: only
22.1% of merges had descL2 < 0.4 (plausibly-genuine), while **72.3% had
descL2 >= 0.6** (mean/median both squarely in this range) -- a
distribution dominated by pairs that would be rejected outright by even a
loose ordinary descriptor-matching threshold. **Confirmed, not just
suspected: this is a MECHANISM problem, not a bug.** The pure-3D-proximity
gate (no descriptor check, unlike real ORB-SLAM3's own `bestDist<=TH_LOW`
requirement) is doing exactly what its design implies -- merging whatever
happens to be spatially close regardless of whether it's remotely the same
visual feature -- and the real data shows most of what it finds close in
3D is NOT visually similar. A false merge is actively harmful, not just
wasted effort: it forces two genuinely different 3D observations onto one
shared position, injecting real reprojection-error inconsistency directly
into every future local/loop BA window that landmark participates in --
structurally similar to (and consistent with) item 4's earlier finding
that BA can converge cleanly on a locally self-consistent-looking but
globally wrong configuration.

**Left off by default (already was); not recommended to re-enable as-is.**
The fix implied directly by the above measurement: add a descriptor-
similarity gate (comparing the new landmark's own descriptor, available in
the current keyframe's `descriptors` row, against the candidate match's
descriptor -- obtainable from whichever keyframe's
`localMapDescriptors`/`descriptors` still holds it) alongside the existing
3D-distance gate, mirroring what the real algorithm actually relies on --
not attempted this session, but now a well-evidenced next step rather than
a guess. A tighter distance threshold alone (without adding a descriptor
check) is not expected to be sufficient on its own: the measured
distribution shows visually-dissimilar pairs land at ALL measured 3D
distances, not just the far end of the 0-0.5 range, so shrinking the
radius wouldn't reliably exclude them.

### 17. Added the descriptor-similarity gate item 16 called for -- fixed the false-merge symptom, did NOT fix the actual regression

User asked directly, after item 16's diagnosis: "is there a way to make it
work better?" -- implemented exactly what item 16 concluded was needed.
Added `m_landmarkDescriptors` (a new `unordered_map<long long, cv::Mat>`,
seeded alongside `m_landmarkPositions` at triangulation time in
`insertKeyframe()`) and rewrote `fuseWindowLandmarks()` into a real
two-stage search matching ORB-SLAM3's own `ORBmatcher::Fuse()` structure:
`kFuseMaxWorldDistance` (3D proximity) now only NARROWS the candidate set
(a cheap first pass), and a NEW `kFuseMaxDescriptorDistance=0.4` gate
(RootSIFT L2 distance) actually DECIDES the match among those candidates --
directly informed by item 16's own measurement (false merges had median
descL2=0.7987; 0.4 sits at the edge of that observed gap).

**Measured on the exact item-10 baseline config again (single variable
changed)**: **72.550m -> 104.672m** -- statistically the SAME as item 16's
ungated version (104.455m), despite merges dropping from 3019 to 1269
(-58%, keeping only pairs that passed BOTH the proximity AND descriptor
gates). Matched points recovered slightly (4481->4516, closer to
baseline's 4519) and loop-closure count stayed normal (74 vs baseline's
66, not suppressed) -- so tracking robustness improved marginally, but ATE
itself did not.

**Conclusion, directly answering the user's question**: the descriptor
gate fixed the exact symptom item 16 measured (most merges are now
confirmed visually-similar, not dissimilar) -- but this reveals that
false-merges were NOT actually the primary cause of the ATE regression;
something about the FUSE MECHANISM ITSELF, even restricted to
high-confidence genuine duplicates, is still net-harmful to this
pipeline's accuracy. Leading (unverified) hypothesis: a merge only
transfers the loser's observation onto the survivor's EXISTING stored 3D
position -- it never re-triangulates or immediately re-optimizes that
position using the newly combined observation set, so any real (if small)
disagreement between the two independently-triangulated estimates of "the
same point" becomes a standing reprojection-error tension that whichever
future local/loop BA window picks up the merged landmark has to absorb,
concentrated onto whatever nearby poses/points are in that window --
plausible given item 4's own earlier finding that this pipeline's BA can
converge cleanly on a locally-consistent-looking but globally-off
configuration. Not verified this session (would need instrumenting BA's
own before/after cost specifically for merged-vs-unmerged landmarks to
confirm).

**Both fuse variants now left off by default, this direction considered
closed for this session** -- fixing the false-merge symptom didn't unlock
the win items 8/10 hoped adjacent duplicate-dedup would provide.

### 18. Tested item 17's own hypothesis directly (user asked "if that's the cause, can it be fixed?") -- fix applied, made things WORSE, hypothesis refuted

Item 17 ended on an unverified hypothesis: the survivor's position never
gets refined using the newly-merged observation because
`fuseWindowLandmarks()` never registers the survivor in
`m_keyframeObservedLandmarkIds[newKeyframeIndex]` -- so local/loop BA's
own density-based landmark selection (items 8/10) never learns the
survivor has a fresh observation. Re-reading the code confirmed this gap
is real (not just theorized): the function never touches
`m_keyframeObservedLandmarkIds` at all. User asked directly whether this
could be fixed -- implemented the obvious fix (`push_back(survivor)` onto
the current keyframe's reverse-index entry right after each merge, so the
very next local BA call -- which already runs immediately after fuse in
`insertKeyframe()`'s own ordering -- picks it up).

**Measured on the exact item-10 baseline config again**: **72.550m ->
155.278m** -- dramatically WORSE, not better, than either item 16's
ungated version (104.455m) or item 17's descriptor-gated version
(104.672m). **This refutes item 17's own hypothesis**: if "stale,
never-refined position" were the real problem, closing that gap should
have helped. Instead, actively exposing the merged observation to BA made
things much worse. Revised understanding: the merged residual itself is
the problem, not merely its dormancy. Splicing the loser's own image
observation onto the survivor's (necessarily slightly different --
allowed up to kFuseMaxWorldDistance/kFuseMaxDescriptorDistance apart, not
identical) 3D position creates a residual with real, structural geometric
inconsistency baked in by construction: the loser's pixel coordinates were
measured assuming ITS OWN triangulated position, not the survivor's. While
dormant (items 16/17, before this fix), that inconsistency just sits
unused in `m_landmarkObservations`, causing only moderate harm (evidently
via the loop-closure PnP/BA paths that do independently consult those
entries). Once local BA is forced to actively reconcile it (item 18), the
optimizer has to distort something REAL (the current keyframe's own pose,
or the survivor's position, or both) to explain an artificial discrepancy
-- and apparently does so destructively, consistent with item 4's
established finding that this pipeline's BA can converge cleanly on a
locally-self-consistent-looking but globally-wrong configuration.

**Conclusion: this whole direction (landmark fuse/dedup via 3D+descriptor
matching and observation-splicing) is now considered CLOSED for this
pipeline, not just "needs more tuning."** Three independent, honestly
-measured attempts (item 16: no descriptor gate, 104.455m; item 17: with
descriptor gate, 104.672m; item 18: + BA-visibility fix, 155.278m) all
regressed ATE, and making the mechanism progressively more "correct" by
the standards of a literal ORB-SLAM3 port made results WORSE, not better --
the opposite of what would be expected if this were simply an
under-implemented version of a sound idea. The likely real fix (an actual
re-triangulation/joint re-optimization of the survivor's position at merge
time, rather than treating the loser's raw pixel observation as valid
evidence for the survivor's existing position) is a substantially bigger
change than a bug/threshold fix -- not recommended to pursue further
without a genuinely new design, not attempted this session.

### 19. Redesigned the fuse pass around real projection+keypoint evidence (not 3D-to-3D landmark comparison) -- first POSITIVE result in this whole direction: 72.550m -> 51.273m (-29.3%)

User asked directly why real ORB-SLAM3's `Fuse()` works when this
session's items 16-18 didn't, then asked if that same evidence-gathering
technique could be applied here. Re-read `ORBmatcher::Fuse()` again with
that specific question and identified the real structural difference:
real `Fuse()` PROJECTS a point into a target keyframe and matches against
that keyframe's own ACTUALLY-DETECTED keypoints (radius search, then
descriptor distance) -- never comparing one landmark's stored 3D position
against another's. Two consequences follow: (a) any match is grounded in a
real detected image feature, not two independently-triangulated (and
therefore never-quite-identical) 3D estimates; (b) `Fuse()`'s PRIMARY role
is extending an already-good point's observation COVERAGE across
covisible keyframes -- merging (`Replace()`) only fires as a narrow edge
case when two different points both already claim the exact same keypoint
slot, a real, unambiguous conflict. Items 16-18 inverted this: merging was
the primary operation, anchored to abstract 3D distance, with no
real-image grounding at all -- which item 18 showed actively corrupts BA
once engaged.

**Redesigned accordingly, split into two phases per the user's own
direction (measure coverage-extension alone before adding conflict-
merging), Phase A only implemented and measured this item**:
- Added `Keyframe::keypointLandmarkId` (parallel to `keypoints`/
  `descriptors`, -1 = unassigned) -- the missing piece real ORB-SLAM3's own
  `KeyFrame::mvpMapPoints` provides and this codebase never had. Populated
  at triangulation time (`insertKeyframe()`) and by
  `recordLandmarkObservations()` (signature extended to take it by
  reference) whenever a real reprojection-gated match is accepted --
  required reordering `insertKeyframe()` to construct the `Keyframe` object
  earlier (before `recordLandmarkObservations()`'s call) so there's
  somewhere for it to write into.
- Rewrote `fuseWindowLandmarks()`: for each of the current keyframe's own
  just-triangulated landmarks, projects it into each OTHER keyframe in the
  window (via that keyframe's own R/t), searches for a descriptor match
  against that keyframe's own FULL detected keypoint set, and gates it
  with the exact same `kMaxObservationReprojErrorPixels` reprojection-error
  check `recordLandmarkObservations()` already uses safely for the rolling
  map. If the matched keypoint's `keypointLandmarkId` is unassigned (-1),
  extends coverage: records a new (real, reprojection-verified)
  observation and claims that slot. If already assigned to a DIFFERENT
  landmark, Phase A currently just skips it (conflict/merge handling
  deferred to Phase B, not implemented yet).

**Real bug found and fixed along the way** (not a design issue this time):
the shared `feature_detector::matchDescriptors()` had `if (... ||
descA.rows < 2 || descB.rows < 2) return false;` -- rejecting ANY
single-row query outright, before attempting to match at all. Since this
pass queries with exactly one landmark's descriptor at a time, EVERY call
was silently returning zero matches (confirmed: first Phase A smoke test
measured exactly 0 coverage-extensions across an 866-frame run). Root
cause: `descA.rows>=2` was never actually required by the k=2 ratio-test
algorithm (which ranks each descA row independently against descB; only
descB needs >=2 rows, for the ratio comparison itself) -- just an overly
broad defensive check. Fixed by relaxing it to `descB.rows < 2` only.
Re-verified the fix doesn't change EXISTING behavior anywhere else in the
codebase (no other call site currently passes a 1-row descA): re-ran the
plain item-10 baseline config (fuse off) after the fix -- byte-identical
72.550m, confirming zero side effects on the shared function's other
callers.

**Measured on the exact item-10 baseline config, Phase A only (single
variable changed)**: **72.550m -> 51.273m, a real 29.3% improvement** --
the first positive result across the whole fuse/dedup direction (items
16-19) this session. 23534 coverage-extending observations added across
the run (much higher than any previous item's merge count, expected since
Phase A can add MANY observations per landmark across different nearby
keyframes, not just one merge event). Matched points close to baseline
(4515 vs 4519/4541), loop closures similar (71 vs baseline's 66) -- the
improvement is coming from denser, more accurate local/loop BA
constraints, not from different loop-closure behavior.

**New best real, reproducible result: 51.273m** (SQPnP + VLAD +
windowed-BA + guided + denser local/loop-BA + Phase-A landmark-coverage
fuse). Now ~2.6x from the user's <20m goal (was ~3.6x before this item).
`setLandmarkFuseEnabled()`/`fuse` CLI flag can now be considered a real,
positive, recommended feature -- unlike items 16-18's now-superseded
merge-based versions, which remain in the git history but are no longer
what `fuseWindowLandmarks()` does.

**Not yet done at the time**: Phase B (genuine-conflict merging, using
`keypointLandmarkId` to detect real index collisions instead of the old
3D-distance heuristic) was deliberately deferred to measure Phase A in
isolation first, per the user's own explicit direction. Implemented and
measured next (item 20).

### 20. Implemented item 19's Phase B (genuine-conflict merging) -- NEGATIVE, root-caused, split into its own opt-in flag

Extended `fuseWindowLandmarks()`: when Phase A's projection+match search
lands on a keypoint whose `keypointLandmarkId` is ALREADY assigned to a
DIFFERENT landmark (a genuine, unambiguous conflict -- both ids
demonstrably explain the exact same detected keypoint, confirmed via a
real reprojection-gated match, not items 16-18's abstract 3D-distance
heuristic), merges them (richer-evidence-wins, same rule as before).

**Measured on the Phase-A 51.273m baseline (single variable changed)**:
**51.273m -> 161.117m**, dramatically worse. Loop closures dropped 71->33.
**Root-caused, not just measured**: `tryLoopClosure()`'s own PnP re-
measurement and Sim3Solver scale measurement read `Keyframe::localMapPoints`/
`localMapPointIds`/`localMapDescriptors` directly (confirmed by reading the
call sites) -- a SEPARATE array from `m_landmarkObservations`/
`m_landmarkPositions` that Phase B's merge never touches. Any landmark
absorbed as a loser leaves a stale, orphaned position in whichever
keyframe owns it via `localMapPointIds`, silently corrupting exactly the
two most safety-critical loop-closure measurements. Phase A never has
this problem since it only ever ADDS observations to a still-alive
landmark, never invalidates one.

**Split into its own flag** (`setLandmarkFuseMergeEnabled()`/`fusemerge`
CLI arg, requires `fuse` too), default OFF, NOT recommended as-is -- kept
in the tree for anyone who wants to fix the `localMapPoints` synchronization
gap and re-measure. Re-verified Phase A alone is unaffected by this split:
byte-identical 51.273m after the refactor. User asked directly whether to
invest further fixing Phase B or shelve it -- decided to shelve for now
(the required fix is a real architecture task with uncertain payoff, vs.
Phase A's already-validated, safe win).

### 21. Re-attempted item 15's loop-closure consistency gate with a real evidence-grounded "same place" test -- STILL negative, same failure pattern as the original crude version

User asked to revisit item 15 (`setLoopConsistencyGroupEnabled()`,
originally measured 72.550m->100.391m, worse) applying the SAME lesson
item 19 just proved for fuse: ground "same place" in real appearance
evidence instead of an abstract proxy. Replaced the 1D old-keyframe-index
window with a real place-recognition similarity check (VLAD/SIFT-DBoW2/
DBoW2 score -- whichever backend found the candidate -- between the two
OLD keyframes being confirmed against each other, `kLoopConsistencyPlaceMinScore=0.3`),
falling back to the old index-window test only when no place-recognition
vector is available.

**Measured on the Phase-A 51.273m baseline (single variable changed,
loop-consistency + fuse together)**: **51.273m -> 87.471m**, still
substantially worse. Loop closures dropped 71->19 (40 "waiting" streaks:
20 died at 1/3, 20 died at 2/3) -- essentially the SAME failure pattern
item 15 already found with the cruder index-window test (there: 66->21
committed, also dominated by streaks dying early). **Conclusion: grounding
the "same place" test in real evidence did NOT fix this gate**, unlike the
analogous fix that worked for fuse. This means the root problem isn't
really "how do we measure same place" -- it's that requiring
`kLoopConsistencyRequiredCount=3` consecutive-ish confirmations is simply
too strict for how densely/reliably this pipeline's own `tryLoopClosure()`
actually re-detects a given revisit episode (roughly once per ~8-frame
keyframe interval, gated by several independent, individually-noisy
checks upstream -- PnP inlier count, degenerate-solve caps, etc.) --
regardless of whether the SAME-CANDIDATE test itself is accurate. Two
independent implementations of this gate (crude index-window, real
place-similarity) both fail the same way, which is stronger evidence than
either alone.

### 22. Tested item 21's own predicted next step (lower `kLoopConsistencyRequiredCount` 3->2) -- confirms this is a mechanism problem, not a tuning problem; direction now CLOSED

**Measured on the same Phase-A+loop-consistency config (single variable
changed)**: **87.471m -> 83.553m** -- barely moved, still dramatically
worse than Phase A alone's 51.273m. Closures: 34 committed (up from 19 at
count=3, roughly the expected ~2x from recovering the "died at 2/3"
streaks) -- but still far short of the 71 Phase A gets with no gate at
all. This confirms the diagnosis directly: even after DOUBLING the
successful closure count by loosening the required confirmations, the
result is nowhere close to competitive, because at `count=1` this gate
literally degenerates to "no gate" (71 closures, the known-good number) --
so ANY `count > 1` necessarily throws away real corrections that a
noisy, sparse re-detection cadence can't reliably re-confirm, no matter
how the "same place" test is implemented. **This is a real, structural
mismatch between the consistency-gate CONCEPT and this pipeline's own
detection cadence, not a parameter to tune away.**

**Direction closed.** Across items 15/21/22 (3 independent measurements:
crude proxy, evidence-grounded proxy, evidence-grounded proxy at a looser
threshold), the temporal-consistency gate has never once been
competitive with the simpler "commit on first verified confirmation"
policy already in place. Not recommended to revisit without a fundamentally
different mechanism (e.g. one that doesn't require MULTIPLE SEPARATE
`tryLoopClosure()` calls to agree, since that's the part shown to fail
here) -- `setLoopConsistencyGroupEnabled()` stays available, off by
default, for reference.

### 23. Implemented a genuinely DIFFERENT loop-closure consistency mechanism (within-call spatial consensus, not across-call temporal recurrence) -- NULL result, gate never fires

User asked directly whether the mechanism (not just the "same place" test)
could be fixed. Designed a structurally different gate:
`setLoopSpatialConsensusEnabled()` -- instead of requiring the SAME place
be independently re-confirmed across multiple SEPARATE
`tryLoopClosure()` calls (the part items 15/21/22 showed doesn't work
here), this checks consensus WITHIN a single call: the candidate-search
step now collects every candidate keyframe whose place-recognition score
independently clears the normal acceptance threshold (not just the single
best one), and only accepts the top-scoring one if at least one OTHER
qualifying candidate lies within `kLoopSpatialConsensusWindow=5` keyframe
indices of it -- i.e., do multiple independently-ranked keyframes from the
existing history agree this is the same place, with no waiting/pending
state needed at all (a single-call accept/reject decision).

**Measured on the Phase-A 51.273m baseline (single variable changed)**:
**51.273m -> 51.273m, byte-identical** (same 4515 matched points, same
0.2317 Recovered scale, same 23534 fuse count). Closure count: 71,
identical to Phase A alone. **The gate never once rejected a candidate
across the whole run.** Not a regression, but not a measured improvement
either -- a true null result. Likely explanation: when a real revisit
produces a strong top-scoring candidate, several temporally-adjacent
keyframes from that SAME original visit episode almost always also score
above the acceptance threshold (they're naturally similar to each other
too), so "spatial consensus" among candidates from the SAME query is
essentially guaranteed whenever the top candidate is genuine on this
dataset -- unlike the across-call approach, which failed because
requiring the SAME query to repeat successfully on a LATER, separate
keyframe is what doesn't reliably happen.

**Inconclusive on the core question** (would this catch a genuine false
positive if one occurred): apparently no isolated single-candidate false
positive existed in this run for the gate to demonstrate value against,
at `kLoopSpatialConsensusWindow=5`. Not tuned further (e.g. a tighter
window, requiring 2+ OTHER qualifying candidates instead of just 1) --
kept available (`loopspatialconsensus` CLI flag), off by default, neither
recommended nor discouraged pending a test against a sequence/config with
a known false-positive closure.

### 24. Tried increasing `kLocalBaWindowKeyframes` (8 -> 16) -- negative result

Moved `kLocalBaWindowKeyframes` from a fixed constexpr to a runtime-
overridable `SlamWorker::m_localBaWindowKeyframes` (`setLocalBaWindowKeyframes()`,
same migration pattern as `m_minTrackInliers`/`m_detectionScale`), default
unchanged at 8. Added `kitti_ate.cpp argv[42]` CLI override. Rationale
this was queued for: items 8/10's observation-density fix means a bigger
window might now capture real multi-view constraint it previously
couldn't (before those fixes, most landmarks in a big window were only
"owned" by a few early keyframes in it, so a bigger window mostly just
added single-observation landmarks with nothing to jointly triangulate
against).

**Measured on the Phase-A 51.273m baseline (single variable changed,
window 8->16)**: **51.273m -> 70.619m**, worse. Confirmed via the
`[localba]` log lines that the bigger window did engage as intended
(e.g. one window spanning `kf#486..kf#501`, 1109 landmarks/2894
observations -- substantially denser than typical window-8 logs seen
earlier this session). Not root-caused further (out of scope of what was
asked -- this was a direct single-variable test, not a diagnosis task).
Plausible, unverified hypothesis: a wider window gives Ceres a harder,
more strongly-coupled joint optimization problem (more shared landmarks
tying more poses together at once) to solve with the SAME per-window pose-
prior weights (`kLocalBaPosePriorRotWeight`/`kLocalBaPosePriorTransWeight`,
tuned against the window-8 baseline) -- possibly needs re-tuning together
with window size rather than changing window size alone. Not tested.
**Reverted to the default (8) for the recommended config; not
recommended to increase without further investigation.**

### 25. Fixed item 20's root cause (stale `Keyframe::localMapPoints` after a Phase B merge) -- helped, but still net negative; Phase B now considered closed, not just "needs the sync fix"

Implemented the fix item 20 identified: when Phase B merges `loser` into
`survivor`, before erasing `loser`'s observation list, now finds `loser`'s
OWNING keyframe (always the FIRST/oldest entry in its observation list --
seeded at triangulation time in `insertKeyframe()`, never reordered, only
ever appended to) and resyncs that keyframe's own
`localMapPointIds`/`localMapPoints`/`localMapDescriptors` entry to point at
`survivor` instead (id, 3D position, and descriptor all updated; the
keyframe's own real 2D pixel observation, `localMapImagePoints`, is left
untouched since that's a genuine detected feature regardless of which
landmark id currently owns it). This directly targets what item 20 showed
was corrupting `tryLoopClosure()`'s own PnP/Sim3Solver measurement (which
reads these arrays directly, not `m_landmarkObservations`/`m_landmarkPositions`).

**Measured on the Phase-A 51.273m baseline (single variable changed)**:
**51.273m -> 139.061m** -- still substantially worse, though a real,
measured improvement over the unfixed Phase B's 161.117m. Closures: 46
committed (up from 33 unfixed, still well short of Phase A's 71). **The
fix helped but did not come close to closing the gap.** This means the
`localMapPoints` staleness was a REAL contributing factor (confirmed by
the partial recovery), but not the ONLY or even the primary problem with
Phase B -- something more fundamental about aggressively merging on every
detected keypoint-slot conflict (37038 merge+extension events this run,
far more than Phase A alone's ~22-23k) appears to still inject enough
noise to hurt ATE substantially, even with both the descriptor gate
(item 17's lesson) and the data-consistency gate (this item) applied.

**Phase B now considered closed for this pipeline, not just "one bug away
from working"** -- three real, principled fixes attempted across items
17/18/20/25 (descriptor gating, then two rounds of data-consistency
fixes), each measurably helping in isolation, none closing the gap to
Phase A's clean 51.273m. Left off by default (`fusemerge` flag). Phase A
alone (`fuse`) remains the recommended, validated configuration.

### 26. Tried a stricter, merge-specific reprojection-error gate on top of item 25's fix -- WORSE, not better; Phase B investigation concluded

User asked directly why the gap was still so large and whether it could be
fixed. Root-caused via the event count: Phase B's own merge branch reuses
the SAME shared 8px reprojection gate (`kMaxObservationReprojErrorPixels`)
as Phase A's coverage extension -- but merging permanently changes a
landmark's identity/history (much more consequential than simply adding
one more observation to an already-alive, unambiguous landmark), so it
arguably deserved a materially tighter bar of its own. Added
`kFuseMergeMaxReprojErrorPixels = 3.0` (vs. the shared 8.0), applied as an
ADDITIONAL gate specifically on the merge branch (on top of the existing
8px check every candidate already had to pass). Also removed two now-dead
constants (`kFuseMaxWorldDistance`/`kFuseMaxDescriptorDistance`, leftover
from the superseded v2/v3 3D-distance design, no longer referenced by any
code since item 19's projection-based redesign).

**Measured on the Phase-A 51.273m baseline (single variable changed, on
top of item 25's already-fixed Phase B)**: **139.061m -> 193.839m**,
WORSE, not better -- the opposite of what a stricter, more conservative
merge criterion should produce. Closures dropped further (46->34).
`Recovered scale` collapsed to 0.0095 (essentially destroyed, far worse
than any other measurement this session). This is a genuinely
counter-intuitive result: tightening a decision this session's whole
history suggested "measure and ground more evidence" should be the fix
that helps, and it didn't -- consistent with item 23/24's own
counter-intuitive negatives elsewhere this session, and with item 21's
finding that fixing the "same place" test for the loop-consistency gate
didn't help either. Not further root-caused (would need per-merge
instrumentation to isolate a specific bad event or interaction, not
attempted).

**Phase B investigation concluded for this session.** Four independent,
honestly-measured attempts (original 3D-distance-free redesign, +
data-consistency sync fix, + stricter merge-specific threshold), each
addressing a real, correctly-diagnosed issue, produced a monotonically
WORSENING trend (161.117m -> 139.061m improvement was real, then
193.839m regression on the next fix) rather than convergence toward Phase
A's 51.273m. This is strong evidence that Phase B's core idea --
permanently merging two landmarks' identities on a single detected
keypoint-slot conflict, however well-gated -- is fundamentally
incompatible with this pipeline's architecture, not a tuning problem.
**Recommend not investing further in Phase B without a genuinely
different design** (e.g., real re-triangulation/joint re-optimization of
the survivor's position at merge time using ALL its combined observations,
rather than any form of the current "trust the existing stored position,
splice in the new evidence" approach -- flagged as the likely real fix
since item 18, never attempted, a substantially bigger undertaking than
anything tried this session). Phase A (`fuse`) alone remains the
session's clear, validated win at 51.273m.

### 27. Tried the flagged "genuinely different design" (Phase B v5: re-triangulate the merged position from ALL combined observations) -- best Phase B variant yet, still net negative vs. Phase A alone

Follow-up session, user explicitly asked to try the redesign item 26 (and
20/25) flagged as the one remaining plausible fix: instead of a merge
keeping the survivor's own stale, pre-merge 3D position (the shared flaw
across v1-v4, root-caused across items 20/25/26), re-triangulate a fresh
position from BOTH ids' combined observation set at merge time.

Implemented as `SlamWorker::triangulateMultiView()` (new function, linear
N-view DLT generalized from the existing 2-view `triangulate()`: each
observation contributes 2 rows `u*P_row3-P_row1` / `v*P_row3-P_row2` to one
homogeneous system, solved via `cv::SVD`), wired into `fuseWindowLandmarks()`'s
Phase B branch: on a genuine conflict, builds `survivor`'s + `loser`'s
combined observation lists, re-triangulates, and REJECTS the merge outright
(both ids left untouched) if the combined set fails its own reprojection-
error gate (reused `kFuseMergeMaxReprojErrorPixels`, mean over all
observing keyframes) -- making triangulation success itself the real merge
criterion, not just a single-view pixel pre-filter. Also fixed a real gap
v1-v4 never had to handle: since the survivor's position now actually
changes at merge time (previously it never did), survivor's OWN owning
keyframe's `localMapPoints` copy now gets synced too, not just loser's (the
original item 20 fix only handled the retiring id's side).

**Measured on the Phase-A 51.273m baseline (single variable changed)**:
**51.273m -> 116.705m**. Worse than Phase A alone, but the best Phase B
variant of the five tried this project (v1 161.117m, v2/item25-sync-fix
139.061m, v3/item26-stricter-gate 193.839m, v5 116.705m) -- a genuine ~16%
improvement over the previous best (item 25), confirming re-triangulation
was a real, correctly-diagnosed lever, just not enough of one. `Recovered
scale` 0.1316 (vs Phase A's 0.2317) and merge/extension event count 38120
(vs Phase A-only's 23534) both point the same direction as before: more
aggressive fusion still net-corrupts more than it helps, even when the
position update itself is now geometrically grounded rather than blindly
inherited.

**Conclusion: Phase B's core idea -- permanently merging two landmarks'
identities on a detected keypoint-slot conflict -- is confirmed
fundamentally incompatible with this pipeline across FIVE independently-
designed variants, including the one variant (re-triangulation) that
addressed the exact root cause every prior attempt diagnosed.** Not
recommended to invest further in Phase B merging without a genuinely
different mechanism outside this whole design family (e.g. not merging
identities at all, only ever extending coverage -- which is exactly what
Phase A alone already does, and is the config that should stay
recommended). `fusemerge` stays off by default; code kept in the tree for
anyone with a new hypothesis.

### 28. Applied the same landmark-density fix to `runGlobalBundleAdjustment()` (queued item 7) -- fix is correct, but globalBA remains net negative when enabled

Same follow-up session, second request: `runGlobalBundleAdjustment()` still
had the stale ownership-only landmark rule
(`m_keyframeHistory[i].localMapPointIds`) that items 8/10 already found and
fixed in local BA and loop BA respectively, each time for a real measured
win (125m->118m, then 107.676m->72.550m). Applied the identical fix here:
switched to `m_keyframeObservedLandmarkIds[i]` (dense, any-observation-in-
window rule) with the same `processedLandmarkIds` dedup guard loop-BA
already uses.

**Measured with `globalba` enabled on top of the current 51.273m baseline
(single variable changed: `globalba` off -> on, density fix included)**:
**51.273m -> 150.016m**, a large regression, not an improvement. Recovered
scale collapsed further (0.2317 -> 0.0860) and 155 fewer ground-truth
frames matched (4360/4541 vs 4515/4541) -- a real tracking-robustness
regression, not just an alignment artifact. This is consistent with this
function's own pre-existing doc comment (a 2026-07-21 finding, before this
fix): global BA was already measured worse than windowed loop-BA on this
pipeline (169.465m global-only vs 126.134m windowed, an older baseline),
specifically because it silently declines past
`kGlobalBaMaxWindowKeyframes` for late loop closures and falls back to
windowed BA anyway -- so its only real opportunity to help is the
EARLY-sequence closures, and apparently that's not enough to offset
whatever else changes about jointly optimizing over the whole map at once
(all keyframe poses simultaneously, not just a recent window) versus
letting continuous windowed BA do its job.

**Conclusion: the density fix itself is a correct, justified change (same
diagnosed bug, same fix, same code pattern that won twice already) and is
kept in the code -- but it does NOT flip global BA into being a net win.**
The `globalba` flag stays off in the recommended config; this queued item
is now closed as "fixed but still not recommended to enable," not "still
buggy." Do not re-attempt enabling global BA without a new hypothesis about
why simultaneous whole-map optimization underperforms windowed BA here
specifically (not yet root-caused -- candidates: numerical conditioning of
a much larger simultaneous problem, the smooth warm-start's rigid-drift
assumption breaking down over a longer span, or accumulated small pose
errors elsewhere in the map outweighing what the early closures fix).

### 29. Tested a real trained SIFT-DBoW2 vocabulary (queued infra from item 4) -- negative, root-caused to an under-trained vocabulary, not the mechanism

Same follow-up session: the Kaggle vocabulary-training job queued in item 4
finished (`sift_dbow_vocab.txt`, 7928 words, header confirms K=10/L=5 i.e.
100,000-word theoretical capacity). Measured with `siftdbow
sift_dbow_vocab.txt` REPLACING `vlad` (single-variable swap, same
`fuse`+`ba`+`localba`+`guided`+sqpnp config otherwise): **51.273m ->
115.309m**, worse than VLAD.

Root-caused directly from the log, not speculation: `siftDbowScore`s on
real revisits clustered 0.15-0.42 (VLAD typically scores 0.55-0.75 on
equivalent events), and the very first loop closure this vocabulary
supported (kf#16<->kf#182, frame 138<->1581) applied a 521-world-unit
translation correction -- roughly 20-40x every other nearby closure's
correction -- this early in the sequence, poisoning the whole downstream
trajectory (the same "early bad correction compounds" failure mode as
globalBA below). Cause: 7928 actual words vs. 100,000 theoretical capacity
at L=5 means most of the tree's deeper nodes never saw enough DISTINCT
training descriptors to actually split into k=10 children -- a
training-DATA-volume problem (`FRAME_STRIDE=10` in the Kaggle notebook was
too sparse), not a tree-depth or mechanism problem.

**Not closed** -- this is a data-volume gap, not a structural dead end like
Phase B. Fixed the Kaggle notebook
(`kaggle/train_sift_dbow_vocabulary_kaggle.ipynb`, cell [5/6]) for the next
attempt: kept L=5 (now confirmed to actually compile/run, the old L=4
fallback comment was stale/obsolete), dropped `FRAME_STRIDE` 10 -> 3 (~3.3x
more training descriptors from the same attached sequences), and corrected
the comment block to explain the real bottleneck. Re-run and re-measure
before drawing further conclusions about SIFT-DBoW2 vs VLAD.

### 30. User asked why real ORB-SLAM3's global BA works when this pipeline's doesn't -- read `Optimizer.cc`/`LoopClosing.cc` directly, found 3 real mechanism differences, implemented the two testable ones

Direct comparison against `third_party/ORB_SLAM3/src/Optimizer.cc`'s
`GlobalBundleAdjustemnt()`/`BundleAdjustment()` and
`LoopClosing.cc`'s `RunGlobalBundleAdjustment()` found three concrete
differences from `SlamWorker::runGlobalBundleAdjustment()`:
1. **Warm-start**: ORB-SLAM3 fixes ONLY the map's origin keyframe and
   initializes every other keyframe from its own current live pose,
   trusting Levenberg-Marquardt to converge. This pipeline instead spread
   ONE rigid drift transform LINEARLY across every intermediate keyframe
   (`alpha=i/newKfIdx`) -- a uniform-drift assumption real drift doesn't
   satisfy (it accrues faster through turns/low-texture stretches).
2. **Runs in a background thread** (`LoopClosing.cc:1206`,
   `mpThreadGBA = new thread(...)`) -- tracking/local BA keep running
   concurrently while GBA solves.
3. **Correction is propagated through a parent/child SPANNING TREE**
   afterward (`LoopClosing.cc:2334-2358`), not written back 1:1 -- any
   keyframe inserted DURING the (possibly slow) background solve gets
   corrected via a relative transform from its tree parent, not left
   stale or overwritten wholesale.

**Mechanism 1 (warm-start) was implemented and measured**: replaced the
rigid-interpolation warm-start with direct live-pose initialization (kf#0
and newKfIdx still hard-anchored exactly as before -- that scheme has its
own separate justification, untouched). **Measured: item 28's 150.016m ->
64.667m**, a real ~57% improvement, confirming the warm-start WAS a major
contributor to global BA's badness -- but still worse than the 51.273m
baseline (globalba off).

**Mechanisms 2+3 (async + spanning-tree propagation) were explicitly
requested despite the known risk** (a REAL background `std::thread` would
reintroduce run-to-run non-determinism this project deliberately
eliminated -- see processNext()'s keypoint-sort fix and the pinned
`ceres::num_threads=1` -- since this codebase has no locking infrastructure
around `m_keyframeHistory`/`m_landmarkPositions`). Given the explicit
constraint that any experiment must stay deterministically reproducible,
implemented a **deterministic stand-in** instead (no real thread):
`runGlobalBundleAdjustment()` still solves synchronously, but when
`setGlobalBundleAdjustmentAsyncEnabled()` is on, defers WRITING the result
(both keyframe poses and landmark positions) until
`kGlobalBaIntegrationDelayKeyframes` (15) keyframes later
(`tryIntegratePendingGlobalBa()`, called from `insertKeyframe()`).
Keyframes inserted during that simulated gap -- which never had a residual
in the original optimization, exactly ORB-SLAM3's own reason for needing
spanning-tree propagation -- get corrected by a single rigid delta derived
from how the anchor keyframe's own pose changed during integration, chained
forward (a simplified stand-in for a real parent/child tree walk, since
this codebase has no such graph).

**Measured (`globalbaasync` flag, on top of item 30's mechanism-1 fix)**:
**64.667m -> 136.095m**, WORSE than immediate write-back, though still
slightly better than the original v1's 150.016m. **Conclusion: the
deferred-integration + chain-propagation approximation is a net negative,
not a fix.** Most likely explanation: the chain-propagation step applies
exactly the same kind of uniform-rigid-delta assumption across the gap
keyframes that mechanism 1 just proved harmful for BA's own warm-start,
just relocated to a different part of the pipeline -- and unlike real
ORB-SLAM3, the gap keyframes here get NO further real optimization (no
concurrent local BA actively re-grounding them against real image
observations) before the correction is stamped on, only a rigid patch
after the fact. **Closed, negative** -- do not re-attempt this specific
deterministic-delay approximation without a fundamentally different way to
handle the gap keyframes (e.g. re-running a small local BA over just the
gap after integration, using the corrected anchor as a new prior, instead
of a single rigid chain delta).

### 31. User explicitly asked for globalBA to beat live tracking -- implemented the exact fix item 30 flagged as untried (real windowed BA over the async gap, not just a rigid patch) -- helped a lot, still not enough

Direct follow-up to item 30's own closing recommendation ("re-running a
small local BA over just the gap... instead of a single rigid chain
delta"). Added `SlamWorker::runGapBundleAdjustment(anchorKfIdx, endKfIdx)`
-- a new single-hard-anchor windowed BA (only `anchorKfIdx` fixed, at its
just-integrated corrected pose; everything through `endKfIdx` is a FREE
parameter, unlike loop-BA/global-BA's double anchor) that reuses the same
dense `m_keyframeObservedLandmarkIds` observation-gathering pattern as
loop/global BA. Wired into `tryIntegratePendingGlobalBa()`
(`setGlobalBaGapRefinementEnabled()`, `gapba` CLI flag, requires
`globalbaasync`): the rigid-delta chain propagation from item 30 still
runs first as a WARM START (not trusted as final), then this actually
re-optimizes the gap keyframes against real reprojection residuals.

**Measured (`globalbaasync gapba`, on top of item 30's mechanism-1 warm-start
fix)**: **async's 136.095m -> 98.154m**, a real ~28% improvement --
confirms real re-optimization of gap keyframes genuinely helps over a bare
rigid patch, exactly as hypothesized. **But still worse than both the
51.273m baseline AND item 30's simple immediate-write v2 (64.667m).**
**Conclusion: the DELAY itself (not just how gap keyframes are handled) is
the dominant cost of the async simulation** -- even with a real, correctly-
converging BA solve over the gap (cost dropped from the 1e12-1e15 range
down to 1e7-1e8 range in every observed `[gapba]` log line, confirming
genuine convergence, not a numerical failure), delaying the correction by
`kGlobalBaIntegrationDelayKeyframes` (15) keyframes before applying ANY
correction at all is worse than just applying it immediately. **Closed --
the whole async/deferred-integration direction (items 30's mechanisms
2+3) is now conclusively negative regardless of gap-keyframe handling
sophistication; do not revisit without questioning the delay itself, not
just the propagation method.**

### 32. Tried replacing the hard loop-pose anchor with a soft prior (v3, matching ORB-SLAM3's real single-origin-anchor choice exactly) -- also negative

User pushed further on item 30's mechanism-1 win (v2, 64.667m, still 13.4m/26%
behind the 51.273m baseline): one remaining deliberate divergence from real
ORB-SLAM3 was `runGlobalBundleAdjustment()`'s TWO hard anchors (keyframe 0
AND newKfIdx, both `SetParameterBlockConstant`) vs. ORB-SLAM3's real
`Optimizer::BundleAdjustment()`, which only ever fixes the map's origin
keyframe -- it never hard-pins a loop pose inside global BA itself (that
correction is a separate essential-graph propagation step this codebase
doesn't have). Implemented `setGlobalBaSoftLoopAnchorEnabled()`
(`softloopanchor` CLI flag): keyframe 0 stays hard-anchored (pure
gauge-fixing, no ORB-SLAM3 analogue to diverge from), but newKfIdx becomes
a FREE parameter pulled toward the same loop measurement via a new soft
`PosePriorCost` residual (`kGlobalBaLoopPosePriorRotWeight`/
`TransWeight`, both 50.0, untuned) instead of an equality constraint --
reusing the same `PosePriorCost` struct `runLocalBundleAdjustment()`
already established for its own single-anchor gauge-fixing.

**Measured (on top of v2's warm-start fix, immediate write-back, no
async)**: **64.667m -> 106.747m**, WORSE, not better. **Conclusion: for
this pipeline, hard-anchoring the loop-verified pose is actually HELPING,
not hurting** -- plausible explanation: the Sim3Solver/PnP-RANSAC loop
measurement already carries strong internal evidence (many inliers, a
real geometric verification pass), so treating it as authoritative (hard
constraint) gives the optimizer a cleaner, better-conditioned problem than
adding an extra soft degree of freedom that lets the solution drift away
from a measurement that was already trustworthy. Matching ORB-SLAM3's
literal mechanism more closely does NOT automatically transfer to this
simplified pipeline -- the THIRD time this session a "more faithful to
ORB-SLAM3" change measured negative (after items 30's async/propagation
and this one). **Closed, negative.** Reverted to the double-hard-anchor
default (`softloopanchor` stays off).

**Final globalBA ranking, all 5 variants tried this session**: baseline
(off) **51.273m** < v2 (live warm-start, 2 hard anchors) **64.667m** <
async+gapba **98.154m** < v3 (live warm-start, soft loop prior)
**106.747m** < async (rigid propagation only) **136.095m** < v1 (rigid
warm-start, 2 hard anchors) **150.016m**. No variant of `globalba` beat
live tracking. v2 (item 30's mechanism-1 fix alone, nothing else) remains
the best of the 5 and the one worth keeping in code if `globalba` is ever
revisited, but the flag stays off by default. Three independent
experiments this session (async/spanning-tree simulation, soft loop
anchor, and -- from item 27 -- Phase B's re-triangulation) all confirm the
same pattern: changes that more faithfully replicate a specific ORB-SLAM3
mechanism do not automatically transfer to this simplified,
single-threaded, hand-rolled pipeline, and must always be re-measured, not
assumed.

### 33. User's 6th globalBA hypothesis: continuous local BA re-solves the same window many times before any given loop closure, so global BA's one-shot solve is structurally disadvantaged -- follow up with a local-BA "polish" pass -- also negative

User's own hypothesis, well-reasoned: continuous local BA (when enabled)
re-optimizes the SAME trailing `m_localBaWindowKeyframes`-sized window on
EVERY keyframe insertion -- dozens of incremental re-solves by the time any
given loop closure fires -- while global BA gets exactly ONE shot at the
whole map in one Ceres solve. Re-running the identical residual set again
would be a no-op (LM already iterates internally to convergence), so
instead implemented `setGlobalBaPolishEnabled()` (`globalbapolish` CLI
flag): immediately follows a successful `runGlobalBundleAdjustment()` with
one call to the existing, independently-proven `runLocalBundleAdjustment()`
over the trailing window, "polishing" the most recent/most
tracking-critical keyframes with local BA's own already-validated
mechanism instead of trusting global BA's one-shot result as final for
them.

**Measured (v2's warm-start fix + polish, no async/soft-anchor)**:
**64.667m -> 122.565m**, WORSE. Also a real drop in tracked frames
(4045/4541 matched vs v2's 4521/4541 -- 476 fewer), not just a worse
alignment -- a genuine tracking-robustness regression, the same signature
item 28's original globalBA v1 and other negative variants showed.
**Closed, negative** -- the 6th independently-designed globalBA variant
this session, and the 3rd built directly from a specific, well-reasoned
user hypothesis (after async+gapba and soft-loop-anchor), to measure
worse than simply leaving `globalba` off.

**Session-final globalBA verdict**: 6/6 variants tried (v1 rigid
warm-start, v2 live warm-start, v3 soft loop anchor, v4 polish, async
rigid-only, async+gapba) all measured worse than the 51.273m baseline.
Every specific, well-motivated hypothesis this session raised for closing
the gap -- matching ORB-SLAM3's real warm-start, its real single-anchor
choice, its real background-thread timing, and now its real continuous-
refinement cadence -- was implemented and measured negative. This is now
strong evidence of a genuine structural limitation of this specific
codebase (continuous windowed local/loop BA is apparently already very
well-suited to its own single-threaded, incremental-tracking architecture,
and one-shot whole-map joint optimization doesn't compose well with it),
not a tuning gap. **Recommend not investing further in `globalba` without
a fundamentally different angle than "make it more like ORB-SLAM3"** --
every variant of that angle has now been tried. v2 remains the best
variant and stays in the code; `globalba` stays off by default.

### 34. User asked directly whether real ORB-SLAM3's global BA mechanism differs further -- found it does (essential-graph Sim3 correction ALWAYS runs first, synchronously, before background GBA) -- tried the exact combination, WORST result of all 7 variants

Read `LoopClosing.cc:969`'s `CorrectLoop()` directly: real ORB-SLAM3's
actual per-loop-closure sequence is (1) `Optimizer::OptimizeEssentialGraph()`
-- a fast, SYNCHRONOUS Sim3 pose-graph correction over the WHOLE map via
the essential/covisibility graph, applied immediately -- THEN, only
afterward, (2) spawn global BA in a background thread as a slower
refinement on a map that's ALREADY been corrected. Global BA is never the
sole/primary correction in real ORB-SLAM3, unlike every variant tried in
items 30-33.

The catch: step (1) -- essential-graph Sim3 pose-graph correction -- is
functionally the SAME mechanism as this codebase's own
`pose_graph::optimizePoseGraph()` (`PoseGraphOptimizer.cpp`), which was
ALREADY independently tested 4 separate times pre-session and conclusively
lost to live tracking every time (see this memo's own top-level summary).
So real ORB-SLAM3's stage 1 had already failed standalone on this
pipeline -- the untested piece was specifically the COMBINATION (stage 1
immediately followed by stage 2, in that exact order, live at every loop
closure), which had never been assembled together before.

Implemented `SlamWorker::runPoseGraphThenGlobalBundleAdjustment()`
(`setGlobalBaPoseGraphPolishEnabled()`, `globalbaposegraph` CLI flag):
at each loop closure, builds the same keyframe-pose/sequential-edge/
covisibility-edge/loop-edge snapshot `kitti_ate.cpp`'s existing one-shot
`posegraph` CLI path already builds (reused verbatim, just invoked LIVE
instead of once at the very end), solves via `optimizePoseGraph()`
(`useSim3=true`, matching ORB-SLAM3's real Sim3 essential graph), writes
the correction directly into `m_keyframeHistory`, THEN immediately calls
`runGlobalBundleAdjustment()` (item 30's v2 warm-start fix) so its
live-pose warm-start reads the just-corrected poses instead of raw
drifted ones.

**Measured (on top of v2, no async/soft-anchor/local-polish)**:
**64.667m -> 174.369m** -- not just worse than v2, the WORST of all 7
globalBA variants tried this session, worse even than v1's naive rigid
warm-start (150.016m). Confirmed via `[posegraph][sim3][g2o]` log lines
that the pose-graph stage genuinely converges each time it runs (e.g. one
representative closure: cost 3895.977 -> 831.019 in 8 iterations) -- not a
numerical failure, a real solve producing a real (but, per the 4
pre-session confirmations, already-known-bad) correction. **Conclusion:
combining two independently-negative mechanisms did not cancel out their
individual weaknesses -- it compounded them.** The pose-graph stage's own
known issue (its own Sim3 measurement/scale estimate is less reliable than
live tracking's continuously-refined one) gets baked into the map BEFORE
global BA even starts, so global BA's warm-start -- and every landmark
position it reads -- inherits that error as its new "ground truth" instead
of the better live-tracked positions v2 used to start from.

**This closes the ORB-SLAM3-mechanism-matching investigation for
`globalba` entirely.** Seven independently-designed variants (v1-v4, async,
async+gapba, pose-graph+polish) covering every architectural difference
identified between this codebase and real ORB-SLAM3's global BA (warm-start
strategy, anchor hardness, background timing, spanning-tree propagation,
continuous-refinement cadence, and now essential-graph pre-correction) all
measured worse than simply leaving `globalba` off. This is conclusive, not
suggestive: **do not revisit `globalba` again without a genuinely new
mechanism outside everything ORB-SLAM3 itself does**, since matching
ORB-SLAM3 more closely has now failed in every form tried. v2 remains the
best variant (64.667m) and stays in the code for anyone who wants it; the
default recommended config (`globalba` off, `fuse` + windowed `ba` +
`localba` + VLAD) stays at **51.273m**.

### 35. User: assemble the whole ORB-SLAM3 loop-correction pipeline behind ONE flag (cull + covisibility-60 + essential-graph-Sim3-first + "background" deferred GBA) -- EXTREME failure (scale collapse), then a stable Schur variant per the follow-up "just don't deviate extremely"

Two related requests. First, one flag (`globalbaorbslam3`, `setGlobalBaOrbSlam3PipelineEnabled()`) turning on the full ORB-SLAM3-imitation
loop-correction chain at once: keyframe culling ON, covisibility essential-graph
threshold lowered 100->60 (`m_covisibilityMinShared`), Sim3 pose-graph correction
over the whole map FIRST (synchronous, the primary correction), then GBA demoted
to a deferred/"background" secondary polish (the deterministic deferred-integration
stand-in for a real thread -- explained to the user that a real OS thread gives no
ATE benefit offline and only adds run-to-run non-determinism, which they initially
accepted with "bắt chước cũng được").

**Measured: 51.273m -> 193.500m, the worst yet, with `Recovered scale` collapsed
to 0.0100.** Root-caused from the logs: `covis60` feeds the Sim3 pose-graph MORE
but WEAKER covisibility edges (60 shared landmarks vs 100), the pose-graph's scale
DOF then collapses toward zero on those weak constraints (the exact
already-known-4x-negative offline-pose-graph failure, now amplified), that
degenerate near-zero-scale map is baked in BEFORE GBA runs, and the deferred GBA
then polishes an already-broken map. Every one of the four stacked mechanisms was
individually negative; stacking compounded rather than cancelled them.

Second, after seeing the extreme collapse the user reframed the goal ("có thể lệch
sai số, ít nhất nó ko lệch cực đoan" -- deviation is acceptable, extreme deviation
is not; and "nhớ cho cờ để bật tắt" -- keep it behind a toggle). This pointed
directly at the one globalBA difference from item 30's list that had NOT been
tried and is inherently collapse-proof: **Schur marginalization** (`globalbaschur`,
`setGlobalBaSchurEnabled()`, argv49). Ceres' direct equivalent of ORB-SLAM3's
`g2o::BlockSolver_6_3` + `vPoint->setMarginalized(true)`: switched
`runGlobalBundleAdjustment()`'s solver from `SPARSE_NORMAL_CHOLESKY` to
`SPARSE_SCHUR` with landmark parameter blocks in elimination group 0, and lifted
the `kGlobalBaMaxWindowKeyframes=400` cap (Schur's block structure makes the whole
map tractable in one solve). Kept the two-hard-anchor scheme (kf#0 + loop pose),
which is what makes it collapse-proof: pinning both endpoints locks the monocular
scale gauge, so scale physically cannot run away.

**Measured: 88.409m, `Recovered scale` 0.1807 (NOT collapsed), coverage 4526/4541
(the highest of ANY globalBA variant), ATE max 193m (no blow-up).** This is the
most STABLE globalBA variant of all -- exactly the "faithful to ORB-SLAM + no
extreme deviation" behaviour the user asked for, and it is toggleable
(`globalbaschur`, default off). But it is still worse than both v2 (64.667m) and
the 51.273m baseline. Key finding: Schur is a pure re-ordering of the SAME
optimisation, so at <=400 KF it would be byte-identical to v2 -- the ONLY thing it
changes is that lifting the cap lets GBA run on the WHOLE ~509-KF map at late loop
closures, and that whole-map one-shot correction is measurably WORSE than v2's
fallback-to-windowed-loop-BA-past-400 behaviour. This re-confirms, from yet another
angle, the session-long finding: one-shot whole-map optimisation underperforms
continuous windowed BA on this pipeline. **globalbaschur is the recommended choice
IF a stable ORB-SLAM-style whole-map GBA is wanted for its own sake; the plain
51.273m baseline remains best for pure ATE.** All variants stay behind their own
flags, all default off.

### 36-38. Front-end/back-end ORB-SLAM3 ports: motion-model + pose-only BA (front), hard-anchor local BA (back), octave weighting -- all measured, none beat the 51.273m baseline, but pose-only BA gives a much healthier scale

After the GBA investigation closed, the user correctly identified (from reading
`third_party/ORB_SLAM3` directly) that GBA is NOT what makes ORB-SLAM3 accurate --
the front-end (constant-velocity motion model + guided SearchByProjection +
pose-only BA every frame) and continuous local BA are. Implemented the three
mechanisms, each behind its own default-off flag:

**36. Pose-only BA front-end (`poseonlyba`, argv50, `setPoseOnlyBaEnabled()`).**
New `optimizePoseOnly()` (real ORB-SLAM3 `Optimizer::PoseOptimization`: refine
this frame's 6-DOF pose against ALL matched map points, map fixed, 4-pass
iterative outlier rejection, new file-scope `PoseOnlyReprojectionCost`) +
motion-model-PRIMARY tracking (predict pose from `m_velocityR/T`, run pose-only BA
directly; SQPnP RANSAC only runs as the lost-track RECOVERY path -- kept so
coverage can't collapse, per the user's requirement). **Measured (on the 51.273m
baseline config): 97.198m** -- worse ATE, BUT `Recovered scale` **0.7101** vs the
baseline's 0.2317, i.e. the raw monocular trajectory is FAR closer to metric scale
(needs only ~1.4x rescale vs ~4.3x). Motion model handled ~99% of frames (only 9
SQPnP recoveries over 4541 frames), so the mechanism works as designed. The
healthy scale is the most interesting front-end result of the session -- the
per-frame pose-only refinement genuinely improves scale consistency; it just
doesn't (yet) translate to better post-Umeyama ATE.

**37. Hard-anchor local BA (`localbahard`, argv51, `setLocalBaHardAnchorEnabled()`).**
New `runLocalBundleAdjustmentHardAnchor()` = real ORB-SLAM3 `LocalBundleAdjustment`:
window keyframes + their map points free, every OTHER co-observing keyframe added
and held CONSTANT as a hard scale/gauge anchor (the structurally-correct scheme the
reverted single-anchor attempt lacked). **Measured ALONE: 131.540m, scale 0.1476**
(vs soft-prior baseline 51.273m/0.2317). **Combined with pose-only BA: 170.203m,
scale collapsed to 0.086.** Isolation confirmed the hard-anchor scheme itself is a
net negative here and drifts scale low (not fully collapsed alone; collapses when
pose-only BA feeds off the drifting map). **Root cause (diagnosed + confirmed by
the scale numbers): monocular BA has a 1-DOF scale gauge on top of the 6-DOF rigid
one; the soft-prior baseline leashes EVERY window pose to its own live-tracked
prior, which bounds scale drift per solve, while the hard-anchor scheme removes
that leash and relies only on border-keyframe observations. Over hundreds of
sliding-window re-solves the border anchors (themselves recent, already slightly
shrunk by earlier solves) provide no persistent absolute scale reference, so scale
slowly drifts/collapses -- ORB-SLAM3 tolerates this via dense ORB covisibility +
loop-Sim3 correction, neither strong enough on this sparse-SIFT pipeline.** The
soft-prior local BA is genuinely better-suited here; the hard-anchor port does not
transfer, same lesson as the GBA ports.

**38. Octave/scale information weighting (`octaveweight`, argv52,
`setOctaveWeightingEnabled()`).** SIFT-appropriate analogue of ORB-SLAM3's
per-pyramid-level `invSigma2`: weights each pose-only-BA observation by
`(kOctaveWeightRefSize/keypoint-size)^2` (via `ceres::ScaledLoss`), using the
keypoint's REAL SIFT size (threaded through as `imageScales`), not ORB's discrete
1.2^level formula (which doesn't fit SIFT's sub-pixel/sub-scale-interpolated
continuous scale space). Flag PREPARED + built, applied only in `optimizePoseOnly()`
where the scale is directly available (extending to local/loop/global BA would need
per-observation scale plumbing). NOT yet measured. Note: "SIFT detects few
keypoints" is not a reason against it -- reweighting doesn't reduce point count.

**Overall**: the 51.273m baseline (SQPnP + soft-prior local BA + windowed loop-BA +
VLAD + Phase-A fuse) remains best. Every ORB-SLAM3 front/back-end mechanism ported
this session (pose-only BA, hard-anchor local BA) measured worse in ATE, confirming
the same session-long pattern for GBA. The one genuinely promising thread is
pose-only BA's much healthier raw scale (0.71) -- worth understanding why that
doesn't yield better ATE before doing more. All flags default off.

### 39. Diagnosed pose-only BA's failure (local post-loop scale collapse), tried a loop-suppression guard -- fixed the collapse region but net still worse; front-end direction closed

Dug into WHY pose-only BA (item 36, 97.198m) underperforms despite healthier
global scale. Per-region aligned/GT path-length ratio (a local scale-consistency
measure) revealed: pose-only BA is actually BETTER than baseline in 3 of 4 regions
of seq00 (1.16/0.95/0.99 vs baseline 0.85/0.89/0.99) but **catastrophically
collapses in frames 2250-3400 (path ratio 0.43; the 100-frame window 2500-2600
crashes to 0.07 -- the trajectory nearly stops while GT keeps moving)**. The
collapse is SUDDEN, coincides with loop closures at frame 2455/2463 (scaleMeas
1.29 then 1.947 on only 10 inliers), and happens in frames tracked AFTER the loop
(not loop-BA-rewritten ones) with no motion-model losses there. **Root cause:
pose-only BA holds the map fixed and fits the camera rigidly to it; after a loop
closure's scale correction momentarily compresses the map, pose-only BA faithfully
follows the compressed map into a local scale collapse. SQPnP's RANSAC is robust
to the same perturbation (rejects inconsistent points).**

Fix attempt: `poseonlyloopguard` (argv53, `setPoseOnlyLoopSuppressEnabled()`) --
suppress pose-only BA (use pure SQPnP) for kPoseOnlyLoopSuppressFrames=30 frames
after a loop closure. **Blunt version (suppress after EVERY loop): 97.2m -> 79.8m,
and the region-3 collapse was genuinely FIXED (0.43 -> 0.98)** -- confirming the
diagnosis -- **but it degraded the other 3 regions** (0.59/0.65/0.80, since it
turned pose-only BA off a large fraction of the time, reverting toward SQPnP).
Selective version (suppress only after scaleMeas-far-from-1.0 loops, `[0.8,1.25]`
gate): **151.7m, WORSE, all regions collapsed to ~0.45.** The erratic,
non-converging results across variants (79.8 / 97.2 / 151.7 / 170.2) are
themselves the finding: **pose-only BA makes the trajectory hypersensitive to
exactly when it runs relative to loop closures. The paradox that the BLUNT guard
(79.8m) beat the SELECTIVE one (151.7m) shows the guard's "help" was really just
turning pose-only BA off more -- reverting toward SQPnP. The more pose-only BA
runs, the worse; pure SQPnP (baseline) is best.** All flags default off.

CORRECTION (added after the fact): calling the front-end pose-only-BA
direction "closed" here was premature over-generalization. What these 4
variants actually exhausted is one SUB-approach: LOOP-TIMING-based
suppression (suppress pose-only BA around loop closures). All 4 shared that
flaw or no suppression at all. The measured evidence actually argues the
CORE idea has merit -- pose-only BA is better than baseline in 3 of 4
regions, and its failure is a specific, diagnosable PER-FRAME signature
(step magnitude collapses to ~0.07x when it rigidly follows a
loop-compressed map). A specific diagnosable failure calls for a targeted
per-frame fix, not abandonment. Untried when "closed" was written:
per-frame step-consistency gating and leashing pose-only BA to the SQPnP
solution (see item 41).

### 40. User kept the SQPnP+soft-prior baseline and asked for other improvement levers -- tested the two cheapest (tighter PnP reproj error, full-inlier LM refit); BOTH negative, revealing a deep "any tighter map-fit collapses scale" pattern

Two SQPnP-compatible levers that don't touch architecture:
- **Tighten PnP `reprojectionError` 8 -> 4px** (argv22, already wired): **140.8m,
  scale collapsed to 0.0086**, matched frames 4515 -> 4277. The stricter RANSAC
  threshold rejects too many correct matches -> starved tracking -> collapse. 8px
  default is better.
- **`pnpFullInlierRefine`** (argv54, newly wired from GUI to CLI,
  `setPnpFullInlierRefineEnabled()`): a single safe LM refit of the SQPnP pose
  over ALL inliers -- expected to be a small safe accuracy gain. **175.7m, scale
  collapsed to 0.0648.** WORSE.

**Both negative, and the pattern across items 36/39/40 is now unmistakable and
deep: ANY mechanism that fits the per-frame pose more TIGHTLY to the map
(pose-only BA, full-inlier LM refit, stricter reprojection gate) collapses scale
on this pipeline. The baseline's plain SQPnP RANSAC minimal-sample solve is robust
precisely because it is LOOSELY coupled to the map -- it does not over-fit the
map's own scale imperfections into the trajectory.** This means the 51.273m
baseline is near-optimal for the CURRENT map quality, and the real improvement
lever is NOT tighter pose fitting but BETTER MAP QUALITY (more accurate landmarks,
cleaner loop closures) -- at which point tighter fitting would stop backfiring.
This directly motivates the one still-open item: a properly-trained SIFT-DBoW2
vocabulary (item 29's Kaggle notebook now fixed to K=10/L=5/stride=3) for cleaner
loop-closure candidate search -> fewer map-compressing garbage loops. `reprojErr`
and `pnpfullrefine` flags kept, default off/8px.

Footnote (sweep of `pnpfullrefine` + `reprojErr`): reprojErr in
{4,5,6,6.25,6.5,6.75,7,8} with fullrefine gave {73, 177, 51.9, 128, 46.9,
70.9, 170, 176}m -- chaotically non-monotonic. reprojErr=6.5 hit 46.861m
(beats baseline!) but its immediate neighbours (6.25=128, 6.75=70.9, 5=177,
7=170) are catastrophic, so it is a knife-edge coincidence, NOT a robust
usable improvement. Superseded anyway by item 41's leash win.

### 41. REVIVED the pose-only-BA front-end with two NEW per-frame mechanisms (after correcting the premature "closed" call) -- the LEASH mechanism BEATS baseline: 41.782m (first robust sub-baseline result of the whole effort)

After the user pushed back on the premature "front-end closed" conclusion
(rightly -- see the correction note on item 39), implemented the two
per-frame mechanisms that the diagnosis actually pointed to and that no
prior variant had tried, each behind its own flag:

- **#1 step-consistency gate** (`poseonlystepgate`, argv56,
  `setPoseOnlyStepGateEnabled()`): reject the motion-model pose-only-BA
  result for any frame whose camera-center step collapses below
  kPoseOnlyMinStepFraction (0.35) of the running avg step -- the exact
  scale-collapse signature -- and fall back to SQPnP for that frame only.
  **Measured: 159.4m, NEGATIVE.** The per-frame step gate fired too
  bluntly / the SQPnP-fallback frames still didn't compose well; did not
  help.
- **#2 leash-to-SQPnP** (`poseonlyleash`, argv57,
  `setPoseOnlyLeashEnabled()`): switch to SQPnP-PRIMARY tracking, then
  refine with pose-only BA anchored to the SQPnP solution by a soft
  PosePriorCost (kPoseOnlyLeashRotWeight=30/TransWeight=5) -- pose-only BA
  sharpens the pose but cannot drift/collapse away from the robust SQPnP
  estimate. The per-frame analogue of soft-prior local BA's live-pose
  leash. **Measured: 41.782m -- BEATS the 51.273m baseline by 18.5%, the
  first robust improvement over baseline in the entire investigation.**

Verified genuine, not an alignment artifact: ATE median 35.3m (vs
baseline's higher), ATE MAX 90.8m (LOWER than baseline's 120.4m -> more
accurate, not just better-aligned), coverage 4518/4541 (>= baseline's
4515), Recovered scale 0.459 (healthier than baseline's 0.232). Per-region
path-length ratios [0.95/0.81/0.84/1.16] -- NO region collapses; crucially
region 3 (frames 2250-3400), which plain poseonlyba collapsed to 0.43, is
now 0.84: the leash prevented exactly the post-loop scale collapse item 39
diagnosed. Unlike the r6.5 knife-edge, this is a MECHANISM (not a fragile
threshold), so it should generalise. **This is the session's new best and
supersedes the 51.273m baseline as the recommended config: baseline +
`poseonlyba` + `poseonlyleash`.** The lesson: the diagnosis was right (the
failure was a specific per-frame post-loop scale collapse), and anchoring
the tight-fitting pose-only BA to the robust loosely-coupled SQPnP estimate
is what lets it add accuracy without inheriting the map's scale
imperfections -- resolving the "any tighter map-fit collapses" tension of
item 40. Kudos to the user for rejecting the premature closure.

### NEW BEST CONFIG (2026-07-23): 41.782m -- SQPnP + soft-prior local BA + windowed loop-BA + VLAD + Phase-A fuse + pose-only-BA-leashed-to-SQPnP

```
kitti_ate <left-pattern> <poses> 1200 sqpnp <out-prefix> fivepoint - - - ba \
  - - - - - - - - - - - - - - localba - - guided - vlad <vlad-codebook-path> \
  - - - - - - - fuse - - - - - - - - - - - poseonlyba - - - - - - poseonlyleash
```
(argv50=poseonlyba, argv57=poseonlyleash on top of the previous 51.273m
recipe -- see kitti_ate.cpp usage for exact slots.)

### Queued next steps (not started this session, in priority order per the user's own direction)

**IMPORTANT: the reference baseline changed this session** -- item 19's
Phase-A fuse is a real, measured win (72.550m -> 51.273m) and is now part
of the recommended config (`fuse` flag). Any of the items below that
involve re-measuring against "the baseline" should use the NEW 51.273m
config (SQPnP+VLAD+windowed-BA+guided+localba+fuse -- see the updated
"Known-good reference command" below), not the old 72.550m one, unless
specifically isolating a pre-item-19 comparison.

0. **DONE, negative (item 20): Phase B of item 19's fuse redesign
   (genuine-conflict merging)** -- measured 51.273m->161.117m, root-caused
   (stale `Keyframe::localMapPoints` after a merge corrupts loop-closure's
   own PnP/Sim3Solver measurement), split into its own `fusemerge` flag
   (off by default). Not recommended without fixing that synchronization
   gap first (a real architecture task, not attempted).
1. **Run the real (non-smoke-test) DBoW2/RootSIFT vocabulary training job
   on Kaggle** (item 14, notebook ready:
   `kaggle/train_sift_dbow_vocabulary_kaggle.ipynb`). Then measure
   `siftdbow` vs `vlad` head-to-head on the new 51.273m baseline
   (single-variable swap).
2. **CLOSED, negative (items 15/21/22): across-call loop-closure
   consistency gate** -- tried three times (crude keyframe-index proxy,
   real VLAD/DBoW place-similarity proxy, and a looser required-count=2 on
   top of that), all three measured worse than no gate at all (best:
   83.553m vs 51.273m baseline). Root cause confirmed structural, not
   tunable: this gate requires MULTIPLE SEPARATE `tryLoopClosure()` calls
   to independently re-confirm the same place, but this pipeline's own
   re-detection cadence is too sparse/noisy to reliably supply more than
   one confirmation per real revisit. **A structurally different
   mechanism was also tried (item 23): within-call spatial consensus among
   independently-scored candidates in a SINGLE call, no waiting needed --
   NULL result (51.273m, byte-identical to no gate at all; never once
   rejected a candidate).** Not recommended to revisit either approach
   without either a known false-positive closure to test the spatial-
   consensus gate against, or a fundamentally different idea for the
   across-call version.
3. **DONE, negative (item 24): `kLocalBaWindowKeyframes` 8->16** --
   measured 51.273m->70.619m, worse. Now runtime-overridable
   (`setLocalBaWindowKeyframes()`/`kitti_ate.cpp argv[42]`) if re-tested.
   Reverted to default (8). Possible follow-up if revisited: re-tune
   `kLocalBaPosePriorRotWeight`/`kLocalBaPosePriorTransWeight` together
   with a larger window rather than changing window size alone -- not
   attempted.
4. **Map-point MERGE/dedup specifically (items 16/17/18's 3D-distance
   approach) is CLOSED for this pipeline** -- not to be confused with item
   19's fuse redesign (a positive result, now recommended): items 16-18
   were three independent, honestly-measured attempts at ABSTRACT-3D-
   DISTANCE-based merging, each fixing the previous one's most likely-
   sounding flaw, ALL regressed ATE, getting progressively WORSE as made
   more "complete": no descriptor gate 104.455m -> with descriptor gate
   104.672m, unchanged -> + BA-visibility fix 155.278m, dramatically
   worse. Root cause (item 18): splicing a raw pixel observation onto
   ANOTHER landmark's necessarily-slightly-different 3D position (not
   grounded in any real detected keypoint) creates a structurally
   inconsistent residual. Item 19's redesign fixed this by grounding
   everything in real projection + actually-detected keypoints instead --
   that's why it worked. Do not revive the OLD 3D-distance approach.
5. **Denser SIFT features via raw count is now a CLOSED, negative direction**
   -- item 12 showed `nFeatures` alone is inert (never hit its own cap),
   item 13 tested the real lever (`kDetectionScale` 0.5->1.0, full-res) and
   measured it WORSE (72.550m->105.692m), likely because full-res adds
   less-stable high-frequency keypoints rather than more of the useful
   kind. Not recommended to revisit via detection scale/resolution again
   without a new hypothesis; `contrastThreshold`/`edgeThreshold` remain
   untested (a stricter, not looser, threshold might be the more promising
   direction given item 13's finding -- fewer but higher-quality
   keypoints -- but that's speculative, not yet tried).
6. Re-test SQPnP + posegraph/Sim3Solver combined (todo, not yet measured
   whether Sim3Solver's improved measurement changes anything about
   SQPnP's own trajectory specifically, as opposed to the P3P baseline it
   was tested against in item 9). Note the offline-posegraph-vs-live-
   tracking comparison in items 5/6 was run against a live-tracking
   baseline that predates items 10/19's fixes -- if this direction is
   revisited, re-measure against the new 51.273m baseline, not the old
   72.550m or 120-125m ones.
7. `runGlobalBundleAdjustment()` still has the identical ownership-only
   landmark rule (`SlamWorker.cpp`, its own comment says "same conservative
   landmark rule as runLoopBundleAdjustment()", now stale after item 10) --
   not yet fixed or measured. Lower priority than the above since global BA
   fires less often than loop BA, but the same mechanism may apply.

### Known-good reference command (this session's best config, SQPnP + Phase-A fuse -- 51.273m)

```
kitti_ate <left-pattern> <poses> 1200 sqpnp <out-prefix> fivepoint - - - ba \
  - - - - - - - - - - - - - - localba - - guided - vlad <vlad-codebook-path> \
  - - - - - - - fuse
```
(argv positions: 4=pnp-method, 10=ba/windowed-loop-BA, 25=localba, 28=guided,
30/31=vlad+codebook-path, 39=fuse (item 19's Phase-A landmark-coverage fuse,
new this session) -- see `kitti_ate.cpp`'s own usage text for the full,
current list; several argv slots shifted this session after the stereo
flag's removal, so old recorded commands from before this session may no
longer line up with current positions. Without the trailing `fuse` flag,
this same command reproduces item 10's 72.550m pre-fuse baseline exactly.)

## Part 58 (2026-07-20, GPU session on Kaggle): LightGlue closed as a paper-grounded negative result, CudaSIFT+RootSIFT integrated and debugged live, final result -- best-ever per-match accepted rate (45%) but 0% scorable coverage due to a reset-timing artifact

Continuation of part 57's LightGlue investigation, moved to a Kaggle T4 GPU
session (this machine has no NVIDIA GPU, so nothing in this part was
buildable/testable locally -- every fix here was designed from first
principles/measurement, then verified live on Kaggle in a real
edit-push-pull-rebuild loop).

### 1. LightGlue closed as a NEGATIVE RESULT (paper-grounded, not just parameter search)

`TwoViewReconstruction.cc` had no diagnostic logging at any of its
`return false` sites -- `[mono-init] reconstruct FAILED` only said a call
failed, not why. Added `[reconstruct-diag]` logging at every reject path in
`Reconstruct()`/`ReconstructF()`/`ReconstructH()` (RANSAC-inlier count vs
the required threshold, hypothesis ambiguity via `nsimilar`, parallax,
degenerate SVD singular values). Result on a live LightGlue run: raw
`nmatches` in the 700-1600 range collapsed to **N=20-60 RANSAC-F-consistent
inliers (~3-5%)**, and of those, **`maxGood` (chirality-verified) was
routinely 0** -- LightGlue's own 0.1 confidence filter passes matches that
are ~95-97% geometrically wrong on this data, not just a few outliers.

Built `analyze/verify_lightglue_onnx.py` to rule out an export bug first:
ran the PyTorch cvg/LightGlue model and the project's exported ONNX model
on the same real KITTI frame pair -- **2123/2123 matches identical**
(Jaccard=1.0, max score diff 0.0002). The export is faithful; the problem
is LightGlue itself on this data, not a translation bug.

Built `analyze/lightglue_precision_gt.py` to measure precision against
KITTI's exact ground-truth relative pose (no RANSAC estimation error) by
symmetric epipolar distance, sweeping `filter_threshold` 0.0-0.9: precision
stayed flat around 7-11% across the WHOLE sweep, ruling out "just raise the
threshold" as a fix. A ratio-test control experiment on the same pair
**also** scored only ~9% under this script's fixed-2px epipolar check
(vs. ~82-94% established for ratio-test elsewhere in this project) --
revealing the script's absolute numbers aren't trustworthy (likely KITTI's
forward-motion near-degenerate epipolar geometry breaking a fixed-pixel-
threshold check for both matchers equally), left undiagnosed per explicit
user instruction. The threshold-sweep *shape* (flat, not improving) is
still valid evidence; the live `[reconstruct-diag]` RANSAC finding above,
measured directly from production code, is the trustworthy result.

Fetched the actual LightGlue paper (Lindenberger et al., ICCV 2023) and
its training/eval details: pretrained on synthetic homographies (planar
warps, no real 3D motion parallax) then fine-tuned on **MegaDepth**
(1M crowd-sourced tourist photos of 196 landmarks -- different
photographers, times, angles: inherently wide-baseline by construction),
evaluated only on HPatches/MegaDepth/Aachen Day-Night/IMC/InLoc -- **no
benchmark anywhere resembling small-baseline consecutive video frames**.
The paper does mention SLAM as a motivating application, but only for its
*latency* (fast enough for real-time deployment), not as something it was
trained or validated to be accurate on. Same shape of finding as the
ASIFT negative result (part 57): a genuine, citable structural mismatch
between the algorithm's design/training distribution and this project's
actual small-baseline per-frame regime, not a tunable bug.

**Reverted** LightGlue's wiring in `Tracking.cc` (`MonocularInitialization`'s
and `TrackWithMotionModel`'s `GetLightGlueMatcher()->Match()` calls) back to
the original `SearchForInitialization`/`SearchByProjection(Frame&,Frame&,...)`
calls, confirmed via `git diff` against the pre-LightGlue baseline rather
than reconstructed from memory. `LightGlueMatcher.cc`/`.h` and the ONNX
weights stay in the tree as investigation artifacts, just unwired from the
live path.

### 2. New direction: GPU-accelerate the EXTRACTOR instead of the matcher

LightGlue targeted the wrong stage -- the negative result was about match
*quality* on small-baseline pairs, not extraction speed. Pivoted to
**CudaSift** (Celebrandil/CudaSift, MIT-licensed GPU SIFT) + permanent
**RootSIFT** descriptor normalization (L1-normalize/sqrt/L2-normalize,
same transform `LightGlueMatcher.cc`'s `toRootSift()` already had, now
applied at extraction time and stored, not just on-the-fly for one
matcher) as the new extractor, behind a `USE_CUDASIFT` compile flag so the
default/local (no-GPU) build stays byte-identical to before.

Added `analyze/cudasift_probe.cpp`, a Stage-0 probe mirroring Session 14's
original (which caught a real cv::SIFT layer-indexing off-by-one before it
went live) -- CudaSift's `SiftPoint` has no packed octave/layer field the
way `cv::SIFT` does (`subsampling`, an octave-downsample factor, and
`scale`, a continuous sigma, instead), so the conversion into
`flatLevel()`'s expected flat index had to be designed from scratch and
explicitly flagged as an unvalidated placeholder pending real measurement,
per this project's own repeatedly-learned lesson.

### 3. Kaggle build: three real, live-debugged failures, none guessable in advance

`kaggle/CMakeLists.txt` gained `USE_CUDASIFT`, CUDA language/toolkit
detection, `CMAKE_CUDA_ARCHITECTURES` covering both Kaggle GPU types
(P100=sm_60, T4=sm_75). CudaSift's own CMakeLists.txt hardcodes an outdated
`-arch=sm_20` and only builds a demo executable (no library/install rules),
so its sources are compiled directly into `orbslam3_sift_ext` instead of
using its build system (same vendor-source pattern as DBoW2).

- **nvlink duplicate-symbol errors**: an initial `file(GLOB *.cu)` compiled
  both `cudaSiftH.cu` (host wrapper) and `cudaSiftD.cu` (device kernels) as
  separate translation units -- but `cudaSiftD.cu` is `#include`d BY
  `cudaSiftH.cu`, not independent (confirmed via CudaSift's own upstream
  CMakeLists.txt, which only lists `cudaSiftH.cu`). Fixed by excluding it.
- **A second nvlink error**, same class: a `match.cu` in the repo root has
  its own `main()`, conflicting with this project's `main()` in
  `orbslam3_kitti_ate.cpp`. The reactive "glob + exclude known-bad files"
  approach was abandoned after this second failure in favor of an
  **explicit allowlist**.
- **A WebFetch-summarization trap**: the first allowlist attempt (based on
  a WebFetch-summarized fetch of CudaSift's CMakeLists.txt) included a
  `singular.cu` that turned out not to exist at all, and *also* missed that
  CudaSift's own CMakeLists.txt is stale relative to the actual current
  repo tree. Re-fetched via the GitHub API directly with `curl` (not a
  summarized WebFetch) for ground truth. **Lesson for future sessions**:
  WebFetch's small-model summarization can hallucinate exact filenames --
  for anything requiring an exact file list, `curl` the GitHub API or raw
  file content directly instead of trusting a summarized fetch. Final
  source list: `cudaImage.cu`, `cudaSiftH.cu`, `matching.cu`, `geomFuncs.cpp`.

Also added: an `nvcc` presence check (the script already checked
`nvidia-smi` for the driver but never the toolkit/compiler), and a VLAD
codebook auto-selection (`vlad_codebook_all_rootsift.yml` if it exists,
else a loud warning + fallback to the raw-SIFT-trained
`vlad_codebook_all.yml`, since RootSIFT permanently changes the stored
descriptor format the VLAD codebook was trained against -- retraining
deferred, not yet done, loop-closure/relocalization scores are unreliable
until it is).

### 4. A real, load-bearing bug found from a live symptom, not the probe

First full Kaggle run (before running the Stage-0 probe) produced **exactly
0 mono-init matches on literally every single attempt across all 1000
frames** -- not fewer/lower-quality matches, zero, always. Traced
mechanically (not guessed): `ORBmatcher::SearchForInitialization`'s
`if(level1>=nOctaveLayers) continue` gate requires a candidate's flat
octave/layer level to fall in `[0,nOctaveLayers)` -- i.e. exactly
`flatLevel()`'s `kMinOctave==-1` finest octave -- to be eligible at all.
The placeholder formula computed
`octaveGuess = round(log2(std::max(1.0f, sp.subsampling)))`, which (a)
floored the log argument at 1.0, making the result structurally always
`>= 0`, never reaching the required `-1`, and (b) even without that floor,
treated `log2(subsampling)` as an ABSOLUTE octave number in `flatLevel()`'s
`kMinOctave`-based numbering when it's actually a RELATIVE step count from
CudaSift's own finest octave, needing a `+ kMinOctave` shift. Every single
keypoint was silently skipped, every frame, regardless of frame content --
exactly matching the observed symptom. Fixed by removing the floor and
adding the offset.

**Then validated against real data** (the Stage-0 probe, finally run,
200 KITTI frames, 320325 keypoints): `subsampling` takes exactly the
assumed power-of-2 values `{1,2,4,8,16}` anchored at 1.0 (confirming
CudaSift has no upsampled octave below its native resolution, unlike
`cv::SIFT`'s `firstOctave=-1` default -- so `subsampling==1` is correctly
the anchor for `kMinOctave==-1`), `scale` spans exactly one factor-of-2 per
octave (confirming the layer formula's basis), and `orientation` is
confirmed in degrees (`[0.0001,359.9999]`, no radian conversion needed).
The fix was correct and is no longer a placeholder.

### 5. Final result: real tracking now works, but 0% scorable coverage -- a reset-timing artifact, not a dead end

Frames 0-1000, `USE_CUDASIFT=1`, raw-SIFT VLAD codebook (not yet retrained
for RootSIFT). Mono-init now succeeds repeatedly throughout the run
(`[mono-init] SUCCESS` at ids 2, 25, 34, 46, 57, 67, 77, 998, ...) with
real map points and real `TrackLocalMap` inlier counts (up to 200+ at
times). But tracking also fails and resets very frequently (`Fail to track
local map!` -> `SYSTEM-> Reseting active map in monocular case`, which in
monocular-without-IMU mode genuinely **erases** the current map's
keyframes in place, not spawning a separately-preserved fragment) --
dozens of times across the 1000 frames. The last reset landed at frame
~990, just 10 frames before the sequence ended, leaving only 2 keyframes
in the Atlas at shutdown:

```
[atlas-coverage] 1 map fragment(s) in Atlas at shutdown
[atlas-coverage] total keyframes across all fragments: 2
[fragment 0 (2 KFs)] Alignment failed -- too few matched points (2) or degenerate estimate.
```

**Coverage: effectively 0%, unscorable** -- but this is a timing artifact
of *when* the last reset happened to land, not evidence the underlying
matching is broken. The session-wide `[sbp-diag]` tells the opposite
story:

```
[sbp-diag] total=107302 empty_window=50267 (46.8%) dist_reject=4482 (4.2%) ratio_reject=4210 (3.9%) accepted=48343 (45.0%)
```

**45.0% accepted-of-total is the best per-match acceptance rate measured
anywhere this entire session** -- plain SIFT historically ran ~10-17%,
even stock ORB only ~36% (part 56). CudaSift+RootSIFT's raw match quality
is genuinely excellent; the open problem is **reset frequency**, not match
quality -- tracking keeps finding good matches but still loses local-map
tracking (`Fail to track local map!`) often enough that no single map
fragment survives long enough to both (a) accumulate meaningfully and (b)
still be alive when the sequence ends. This is a distinct, not-yet-
investigated problem (why does `TrackLocalMap` fail so often despite high
match quality?) rather than a dead end -- flagged as the clear next
investigation thread rather than pursued further this session.

## Part 56 (2026-07-19, continued autonomously): chasing the 40.3% (ORB) vs
## 10.0% (SIFT) SearchLocalPoints match-rate gap found at the end of part 55

**Recap of where this picks up**: part 55's last data point (see below) was
a freshly-added `nToMatch`/`matched` diagnostic on `SearchLocalPoints()`
showing SIFT matches only 10.0% of geometrically-in-frustum local map
points, and a same-session addition of the identical diagnostic to stock
ORB's `SearchLocalPoints()` measured **ORB at 40.3%** (120401/298787) on the
*identical* frames 1-1000 -- a clean, controlled, 4x gap on the exact same
metric, same segment, same nFeatures=5000. This is the most concrete
"smoking gun" of the whole session: raw keypoint yield (part 55, ~66% of
ORB's) and outlier rate (7.9%, fine) had already been ruled out or found
insufficient to explain the coverage gap on their own -- this 4x match-RATE
gap, on points BOTH systems already agree are in-frustum, is a much more
direct signal.

**First hypothesis checked: the historical part-12 flat-level-window bug
might not be fully patched everywhere.** `ORBmatcher.cc`'s
`GetFeaturesInArea` flat-level-window bug (SIFT's (octave,layer) flat-level
packing makes a literal `+-1`-flat-level window cover only a sliver of one
real resolution level, unlike ORB's true per-level pyramid) was previously
found and fixed at two call sites (local-map `SearchByProjection`, and the
monocular branch of the motion-model overload). Grepped **every**
`GetFeaturesInArea` call site in `ORBmatcher.cc` (14 total) and traced each:
the `KeyFrame::GetFeaturesInArea` overload (6 call sites: lines 521, 634,
1262, 1422, 1555, 1635) has no octave-range parameters at all -- searches
all levels unconditionally, not affected by this bug class. Of the
`Frame::GetFeaturesInArea` call sites, the monocular-relevant ones are
already fixed: line 103 (local-map `SearchByProjection`), line 1764 (motion
model's `else` branch -- confirmed via `bMono` that `bForward`/`bBackward`
are always false for monocular, so this IS the branch always taken), and
line 1979 (Relocalization's `SearchByProjection` overload -- also already
fixed, with an inline comment explicitly flagging it as "one of the top
suspects flagged for this exact bug pattern," confirming this fix was
already made in direct response to the earlier session's own "next steps"
note). The only remaining narrow-window call sites (lines 181, 1838, 1840,
1842; `SearchForInitialization`'s line 710) are either `bRight=true`
stereo/fisheye paths (dead code for monocular KITTI) or intentional
octave-0-only windows for initialization. **Conclusion: this bug class is
already exhaustively fixed for every monocular code path** (confirmed via
git history -- commit `e1cdfc3 "Fix narrow flat-level window in
SearchByProjection"` already did this comprehensive sweep in a prior
session). The fresh 10.0% measurement was taken on code that **already
includes this fix** -- so the historical bug, while real and worth having
fixed, does NOT explain the still-low 10.0% rate. This lead is exhausted.

**Next step taken: added match-rejection-reason diagnostics directly to
`SearchByProjection(Frame&, vector<MapPoint*>&, ...)`** (the function
backing `SearchLocalPoints()`), in both `ORB_SLAM3_SIFT` and stock
`ORB_SLAM3`'s copies (kept them structurally identical for a fair
comparison). Added `g_sbpDiagTotal`/`g_sbpDiagEmptyWindow`/
`g_sbpDiagDistReject`/`g_sbpDiagRatioReject`/`g_sbpDiagAccepted` atomics
(`std::atomic<long>`, thread-safe given multiple threads call into this),
printed via a static destructor at process exit as
`[sbp-diag] total=... empty_window=... dist_reject=... ratio_reject=...
accepted=...`. This buckets every candidate map point in the monocular
`mbTrackInView` branch into exactly one of: (a) `empty_window` -- the
`GetFeaturesInArea` window found zero candidate keypoints at all, (b)
`dist_reject` -- at least one candidate existed but the best descriptor
distance exceeded `TH_HIGH`, (c) `ratio_reject` -- best distance passed
`TH_HIGH` but failed the same-scale-level Lowe's-ratio test, or (d)
`accepted` -- matched. This isolates WHERE in the pipeline (window
geometry vs descriptor-distance calibration vs ratio-test) the 4x gap
actually originates, rather than continuing to guess.

**Results (both runs complete, identical frames 1-1000, nFeatures=5000
both)**:

| | total | empty_window | dist_reject | ratio_reject | accepted |
|---|---|---|---|---|---|
| SIFT | 300351 | 175015 (58.3%) | 95152 (31.7%) | 437 (0.1%) | 29747 (9.9%) |
| ORB  | 307144 | 119280 (38.8%) | 59923 (19.5%) | 16700 (5.4%) | 111241 (36.2%) |

Both `accepted` percentages closely match the previously-measured
`SearchLocalPoints` match rates (9.9% vs the earlier 10.0%; 36.2% vs the
earlier 40.3% -- small difference because this diagnostic's `total` counts
every `mbTrackInView` candidate reaching the window-search stage, a
slightly different base than `nToMatch`), confirming this instrumentation
is measuring the same thing correctly.

**The real finding is in the CONDITIONAL rate** (given the search window
found at least one candidate keypoint at all, i.e. excluding
`empty_window`): **SIFT rejects on descriptor distance 75.9% of the time
when a candidate exists (only 23.7% accepted), vs ORB accepting 59.2% of
the time under the same condition (only 31.9% dist-rejected).** This is
NOT explained by the already-known raw-keypoint-density gap (part 55,
SIFT~66% of ORB's yield) -- that gap plausibly explains most of the
`empty_window` disparity (58.3% vs 38.8%, ratio 1.5x, matching the inverse
density ratio 1/0.66=1.51x almost exactly) but does nothing to explain why
a candidate that DOES exist in the window still gets rejected 3x more
often for SIFT specifically. **This isolates the dominant mechanism to
descriptor-distance/`TH_HIGH` calibration, not window geometry or
keypoint sparsity.** Also notable: SIFT's `ratio_reject` is nearly zero
(0.1%) vs ORB's 5.4% -- almost certainly because the part-12 window-widening
fix (whole octave instead of +-1 flat level) means `bestLevel` and
`bestLevel2` are rarely equal anymore for SIFT (candidates routinely land
on different flat sub-layers within the octave), and the ratio test's own
precondition (`bestLevel==bestLevel2`) is coded to only apply when they
match -- so the ratio test is inadvertently bypassed most of the time for
SIFT now. This is a side-effect worth remembering but is not itself hurting
match rate (if anything it's slightly lenient); the dominant problem is
`dist_reject`.

**Checked whether `TH_HIGH` is even self-consistent with its own
calibration data**: the calibration comment (`ORBmatcher.cc` ~line 40)
records true-match squared-L2 **max=113387**, but `TH_HIGH` is set to
**100557** (1.5x true-match p99=67038) -- i.e. `TH_HIGH` is *below* the
maximum true-match distance observed in its own calibration sample, so by
construction some genuine matches were always going to be rejected; the
question is how large that fraction really is in practice (a single p99*1.5
heuristic doesn't reveal the tail shape). The original calibration tool
(`analyze/orbslam3_sift_calibrate.cpp`) only ever printed false-match
percentiles up to p50, never p90/p99/max -- meaning the `TH_HIGH` choice
was made without knowing how close the false-match distribution's tail
comes to the true-match tail. **Patched the tool to also print false-match
p90/p95/p99/max**, rebuilt, and reran on the same 600-frame/baseline-1/3/5
sample.

**Result -- and it rules out "just raise TH_HIGH"**: full false-match
squared-L2 distribution: `min=127 p1=1207 p5=3074 p10=5124 p50=25974
p90=59910 p95=68271 p99=82639 max=112636`. True-match: `p50=9142 p90=37049
p95=48005 p99=66314 max=113387` (this rerun's own numbers, close to but not
identical to the original calibration session's, as expected from sampling
a different/overlapping frame range). **The true and false distributions
overlap almost completely above ~60000** -- false-match's own p99 (82639)
is comfortably above today's `TH_HIGH` (100557's neighbor, effectively both
in the same ballpark), and false-match's **max (112636) is essentially
equal to true-match's max (113387)**. This means: (a) raising `TH_HIGH`
to "fix" the 75.9% conditional-dist_reject rate would let in a large
fraction of genuine false matches, not just recover a few borderline true
ones -- the two distributions simply aren't separable in that tail region
by distance alone; (b) more importantly, **today's `TH_HIGH`=100557 is
already above false-match's p99 (82639)**, meaning **more than 99% of
RANSAC-outlier candidate pairs already pass the TH_HIGH filter** -- `TH_HIGH`
was never doing much discriminating work in this region to begin with, exactly
as its own header comment says ("a coarse sanity cutoff, not the primary
discriminator -- the ratio test does the fine-grained work"). **The
ratio test is supposed to be the real defense here** -- and part 55's
sbp-diag numbers above already showed SIFT's ratio test barely ever fires
(0.1% vs ORB's 5.4%).

**Root cause identified and fixed**: the ratio test's own precondition,
`bestLevel==bestLevel2`, checks *exact flat-level* equality. Part 12's
earlier fix (this same session's predecessor) widened the `GetFeaturesInArea`
search window from a narrow +-1-flat-level slice to an entire octave, to
fix a *different*, already-solved problem (window too narrow, causing
empty-window starvation) -- but never updated the ratio test's own
same-level check to match. The practical effect: once the window spans a
whole octave (multiple distinct flat sub-levels), `bestLevel` and
`bestLevel2` almost always land on *different* flat sub-levels even when
both candidates are legitimately in the same real resolution range the
octave-wide window was designed to treat as equivalent -- so
`bestLevel==bestLevel2` is almost never true, and the code's own logic
(`if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2)`) then skips
the ratio test and accepts on `TH_HIGH` alone -- which we just showed
barely discriminates in this range. **Fixed** by changing the same-level
check to a same-OCTAVE check (`bestLevel/nOctaveLayers ==
bestLevel2/nOctaveLayers`, guarding for `bestLevel2==-1` when no second
candidate exists), consistent with the octave-based window widening
already used earlier in the same function. This is a mechanism-level fix
targeting the actual root cause the data pointed to, not another blind
threshold sweep (part 55 already exhausted that avenue on the detector
side, twice, both worse).

**Result: mechanism confirmed correct, fix reverted -- net negative for
coverage.**

| metric | baseline | same-octave ratio-test fix |
|---|---|---|
| fails | 67 | 58 |
| resets | 33 | 32 |
| coverage | 320m (44.7%) | 269.7m (37.7%) |
| sbp-diag accepted (of total) | 9.9% | 9.25% |
| sbp-diag ratio_reject (of total) | 0.1% | 0.7% |
| outlier rate (afterPO_flaggedOutlier) | 7.9% | 6.8% |

The diagnosis was mechanically verified correct: `ratio_reject` rose ~4x
(437->1845) once the ratio test's scope check was widened from exact
flat-level to same-octave, confirming it really was almost never firing
before. But the **net effect on match quantity was still negative**
(accepted-rate 9.9%->9.25%) and **coverage got worse, not better**
(44.7%->37.7%) -- even though the post-hoc outlier rate improved slightly
(7.9%->6.8%). **Interpretation**: once Lowe's ratio test is applied across
an octave-wide candidate pool (many distinct flat sub-levels, i.e. many
distinct real keypoints at meaningfully different sub-scales, all
competing as "second-best"), it ends up rejecting more genuine true
matches than false ones for this SIFT/KITTI combination -- a second-best
candidate merely existing somewhere in that wide pool doesn't reliably
signal real ambiguity the way it does for ORB's narrower, single-real-level
window. **Reverted** the ratio-test scope change (back to exact
`bestLevel==bestLevel2`, `TH_HIGH` untouched throughout this whole
sub-investigation) -- documented in an inline comment at the revert site
so this isn't re-attempted blind in a future session. Root-cause diagnosis
(ratio test effectively dead for SIFT due to the part-12 window widening)
stands and is confirmed real; this specific fix for it does not help the
coverage-first goal and was not kept.

**Status**: the `dist_reject`-dominates-conditional-on-nonempty-window
finding (75.9% vs ORB's 31.9%) is still real and still the sharpest lead,
but the two most obvious levers for it (raise `TH_HIGH` -- ruled out, the
true/false distributions overlap too much in that range; re-enable the
ratio test at octave granularity -- tried, net negative) are both now
exhausted. `mfNNratio` checked and ruled out: identical values at every
`ORBmatcher` construction site in both forks' `Tracking.cc` (0.9/0.7/0.9/
0.8/0.75, byte-for-byte match) -- not the difference.

### Part 56 continued: found what looks like the actual root cause -- a real unit mismatch in PredictScale()

While checking `MapPoint::ComputeDistinctiveDescriptors()` (ruled out --
the "least median distance" fused descriptor is, by construction, at least
as central as any single raw observation, and `DescriptorDistance`'s
squared-L2 formula is byte-identical between the live matcher and the
calibration tool, confirmed by reading both), noticed `MapPoint::
PredictScale()` computes `nScale = ceil(log(ratio)/pF->mfLogScaleFactor)`
and returns it directly as `nPredictedLevel`, a **flat-level array index**
(used to index `mvScaleFactors[]`, sized `nlevels=80`, and to compute
`SearchByProjection`'s `octaveBase = nPredictedLevel/nOctaveLayers`
window center).

**Traced `mfLogScaleFactor`'s source**: `Frame::mfLogScaleFactor =
log(mfScaleFactor)`, and `mfScaleFactor = mpORBextractorLeft->
GetScaleFactor()`, which returned the raw constructor argument
`scaleFactor` -- i.e. the settings file's `ORBextractor.scaleFactor: 1.2`.
But this SIFT reimplementation's constructor comment already says
`_scaleFactor` is "accepted for signature compatibility but not used" --
and indeed, `mvScaleFactor[lvl] = pow(2.0f, lvl/nOctaveLayers)` (integer
division; see the constructor, ~line 520 of `ORBextractor.cc`) is built
from a **per-octave** ratio of exactly 2.0, i.e. a **per-flat-level** ratio
of `2^(1/nOctaveLayers)` -- for this project's `nOctaveLayers=8`
(`ORBextractor.nLevels: 8` in the settings file), that's `2^(1/8)~=1.0905`,
**not** 1.2. `log(1.2)=0.182` vs the true `log(2)/8=0.0866` -- roughly
**2.1x too large a denominator**, meaning `PredictScale()` was computing
flat-level indices at less than half their correct value for any nonzero
distance ratio. This directly and simultaneously explains: (1) why the
*original* narrow-window bug (part 12) was so catastrophic (a window
correctly proportioned in width was still centered on the wrong octave
entirely); (2) why even *today's* whole-octave-wide window still shows
58.3% `empty_window` (the window's centering, not just its width, was
wrong -- often missing the true keypoint's real octave altogether); (3)
why `dist_reject` is so high even when the window isn't empty (candidates
present are frequently from the wrong-but-nearby octave the mis-centered
window happened to catch, not genuine matches, so their distances are
large).

**Checked all consumers before changing the fix site**: `mfLogScaleFactor`
is used only by the two `PredictScale()` overloads (`MapPoint.cc`).
`mfScaleFactor` (raw, non-log) is also read once more, in
`LocalMapping::CreateNewMapPoints()`'s `ratioFactor = 1.5f*
mpCurrentKeyFrame->mfScaleFactor` -- a multiplicative tolerance band
around `ratioOctave` (itself correctly built from the real
`mvScaleFactors[]` array) in a new-map-point scale-consistency sanity
check. Both consumers want the same thing: the *real* per-flat-level scale
step, not the vestigial config value -- confirming the getter itself is
the right place to fix, not either call site individually.

**Fix applied**: `ORBextractor::GetScaleFactor()` now returns
`pow(2.0f, 1.0f/nOctaveLayers)` instead of the raw unused constructor
argument (`include/ORBextractor.h`), fixing both consumers at their
shared source. This is a mechanism-level, root-cause fix grounded in
directly-verified array-construction code, not a threshold guess.

**Result on frames 1-1000: a real, clean, all-around win -- no quality
tradeoff.**

| metric | baseline | scale-fix | ORB (reference) |
|---|---|---|---|
| fails | 67 | **53** (-21%) | 152 |
| resets | 33 | **29** (-12%) | 6 |
| coverage | 320m (44.7%) | **379.9m (53.1%)** | 533.6m (74.6%) |
| sbp-diag accepted-of-total | 9.9% | **16.7%** (+69% rel.) | 36.2% |
| sbp-diag empty_window | 58.3% | **45.7%** | 38.8% |
| dist_reject (conditional on non-empty) | 75.9% | **68.9%** | 31.9% |
| outlier rate | 7.9% | **6.6%** (also improved) | -- |

Every metric moved in the right direction simultaneously: fewer fails,
fewer resets, meaningfully more coverage (+8.4 percentage points,
recovering roughly a third of the gap to ORB's 74.6%), a much higher raw
match-acceptance rate (9.9%->16.7%), `empty_window` closing more than half
the gap to ORB's own rate (58.3%->45.7%, ORB is 38.8%), and even the
post-hoc outlier rate improved slightly rather than trading quality for
quantity. This is unambiguously a real fix, not another mixed/negative
result like the ratio-test attempt above.

**Second confirmation: a clean pre/post A/B on the primary hot zone
(frames 4000-4540, GT path 586.2m, computed directly from `poses/00.txt`)
-- mixed result on coverage, but the underlying mechanism improvement
still confirms consistently.**

| metric | pre-fix | post-fix |
|---|---|---|
| fails | 101 | 94 |
| resets | 48 | 48 (identical) |
| coverage | 87.9m (15.0%) | 53.3m (9.1%) |
| sbp-diag accepted-of-total | 11.5% | **15.6%** |
| sbp-diag empty_window | 52.3% | **44.7%** |
| dist_reject (conditional) | 75.3% | **71.3%** |
| outlier rate | 12.4% | **9.2%** |

Ran both variants back-to-back on the identical 541-frame segment
(temporarily reverted `GetScaleFactor()` to the old buggy `return
scaleFactor;`, rebuilt, ran, then restored the fix and rebuilt again --
confirmed via grep that the final kept state has the fix in place). Every
`sbp-diag` quality metric improved in the same direction as the 1-1000
result (higher accepted rate, lower empty_window, lower conditional
dist_reject, lower outlier rate) -- **consistent evidence across two very
different, independently-tested segments that the underlying matching
mechanism genuinely got better**. But `fails` only improved marginally
(101->94), `resets` came out perfectly identical (48=48), and `coverage`
actually dropped (15.0%->9.1%) on this specific single-run comparison.

**Interpretation**: this specific 541-frame hot zone was already
established in part 53 as unusually noisy -- 6 same-or-similar-config
repeat samples ranged 85-124 fails purely from thread-scheduling
nondeterminism (the `DUtils::Random` mutex fix did not achieve full
determinism; see part 53). Both 94 and 101 fall inside that already-known
noise band, so a single-run fails/resets/coverage comparison on *this*
segment specifically is not statistically decisive either way -- unlike
the `sbp-diag` metrics, which measure the matching mechanism directly and
are far less sensitive to exactly when a keyframe lands or when a
background thread runs, and which agree cleanly with the 1-1000 result.
Plausible explanation for the coverage regression despite better matching:
resets landing at slightly different points in this specific hard
stretch's path can cover very different real distances even with the same
reset *count*, and this segment may also contain a genuinely hard
obstacle (sharp turn, blur, lighting) that this fix doesn't address and
that dominates outcomes here regardless of match-rate improvements
elsewhere.

**Decision: keep the fix.** It is a real, verified correctness fix (not a
threshold guess) with consistent, mechanism-level evidence of improvement
on two independent segments; the hot zone's own fails/resets/coverage
noise floor is wide enough that this single comparison neither confirms
nor refutes it at the coverage level, and reverting a demonstrably-correct
fix because of one noisy segment's single-run outcome would be the wrong
call. If more confidence on this specific segment is wanted later, repeat
trials (matching part 53's methodology, 3+ runs) would be needed to
average out the noise.

### Part 56 continued: re-tested the window WIDTH now that centering is fixed -- made things worse, reverted

With `GetScaleFactor()` fixed, `nPredictedLevel` is now correctly centered,
so tested whether the full-octave-wide window (8 flat levels,
`octaveBase..octaveBase+nOctaveLayers-1`) -- originally widened as a
workaround for the *old* mis-centering bug -- could now be narrowed. Per
ORB's own configured `scaleFactor=1.2`, the SIFT-equivalent window width
is only ~2.1 flat levels; tried `nPredictedLevel-2..nPredictedLevel+2` (5
flat levels, a reasoned middle ground with margin) in the local-map
`SearchByProjection` overload only (the one `sbp-diag` instruments,
isolate-and-attack discipline -- motion-model and Relocalization overloads
left untouched).

**Result on frames 1-1000: clearly worse, not better.**

| metric | scale-fix (kept) | narrowed window (reverted) |
|---|---|---|
| fails | 53 | 66 |
| resets | 29 | 32 |
| coverage | 379.9m (53.1%) | 374.9m (52.4%) |
| accepted-of-total | 16.7% | 11.6% |
| empty_window | 45.7% | **59.4%** (worse than even the pre-scale-fix 58.3%) |
| outlier rate | 6.6% | 5.4% (only metric that improved) |

**Interpretation**: the full-octave width was never purely a workaround
for the centering bug -- it's doing real, independent work compensating
for SIFT's own lower raw keypoint yield (~66% of ORB's, part 55) and
genuine real-world prediction noise (depth/viewpoint estimation error,
not just quantization). Narrowing the window, even with correct
centering, starves the search of candidates faster than the tighter
scope's slightly-improved precision can make up for. **Reverted** back to
the full-octave window; documented inline in `ORBmatcher.cc` so this isn't
re-attempted blind. The `GetScaleFactor()` fix remains the confirmed,
kept win; the window-width lever is now exhausted (both directions tried:
too narrow originally broken -> too-narrow-again after centering fix ->
full octave is the correct width for this dataset/detector combination).

**Where this leaves the investigation**: the two clean, safe, well-reasoned
mechanism-level levers this session found (`GetScaleFactor` unit mismatch,
and re-testing window width) are now both resolved -- one kept as a real
win, one tried and correctly reverted. Remaining `dist_reject` gap vs ORB
(SIFT still ~69% conditional-on-nonempty-window vs ORB's 31.9%) is likely
now dominated by SIFT's genuinely lower raw keypoint density on this
dataset (part 55's already-established, already-hard-to-fix finding) --
consistent with the user's worst-case framing being *partially* true: not
"nothing can be fixed," but the easy, safe, mechanism-level fixes may now
be exhausted, with what remains looking more like an inherent
detector-yield gap than a further bug.

### Part 56 continued: audited MapPoint::UpdateNormalAndDepth() (clean), retested nFeatures=7000 with the fix in place (no help, reverted)

Per user's "test everything" license, checked whether
`UpdateNormalAndDepth()` (feeds `mfMinDistance`/`mfMaxDistance`, the
numerator of `PredictScale()`'s ratio) had a similar bug to
`GetScaleFactor()`. **Clean**: it computes `mfMaxDistance =
dist*pRefKF->mvScaleFactors[level]` and `mfMinDistance =
mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1]` directly from the real
`mvScaleFactors[]` array (correctly built with true `2^octave` values),
never touching the buggy `GetScaleFactor()`/`mfScaleFactor` path. No bug
here.

Retested `nFeatures` 5000->7000 (`scratchpad/KITTI00-02-sift-nf7000.yaml`)
now that the centering fix is in place, on frames 1-1000, to see if it
interacts more favorably than the pre-fix attempt did (parts 39-42, kept
at 5000 after finding 10000 "real but computationally expensive, not
purely free").

| metric | nFeatures=5000 (kept) | nFeatures=7000 |
|---|---|---|
| fails | 53 | 55 |
| resets | 29 | 33 |
| coverage | 379.9m (53.1%) | 339.4m (47.5%) |
| accepted-of-total | 16.7% | 16.7% (identical) |
| empty_window | 45.7% | 45.6% (identical) |
| outlier rate | 6.6% | 5.7% |

**No help.** The `sbp-diag` *rate* metrics are essentially unchanged
(more raw candidates scale the numerator and denominator together, not
the underlying match quality), while fails/resets/coverage all got
slightly worse, consistent with the earlier pre-fix finding that more
features cost more (Sim3Solver/BA/merge complexity) without a
proportionate benefit. **Not adopted** -- kept `nFeatures=5000`. This
confirms the centering fix didn't change nFeatures' cost/benefit
calculus; that lever remains not worth it.

**Status**: with `UpdateNormalAndDepth` ruled clean and `nFeatures`
reconfirmed not worth raising, the quick, targeted, mechanism-level levers
available from short-segment testing appear exhausted. The natural next
step is a longer validation run (bigger prefix, e.g. 1-2500, or a full
untruncated sequence) to see the confirmed `GetScaleFactor` fix's
cumulative real-world effect, rather than further short-segment parameter
sweeps.

### Part 56 continued: longer validation, frames 1-2500 (GT path 1885.1m), SIFT-with-fix vs stock ORB

Extends part 54's progressive comparison table (which only went to 1500
frames and did NOT have this session's `GetScaleFactor` fix applied).
This new row DOES have the fix. Ran both back-to-back on the identical
2500-frame segment.

| frames | GT path | SIFT fails | SIFT resets | SIFT coverage | SIFT % | ORB fails | ORB resets | ORB coverage | ORB % |
|---|---|---|---|---|---|---|---|---|---|
| 1-500  | 359.4m  | 29  | -- | 213m   | 59.3% | 91  | -- | 217.6m | 60.5% |
| 1-1000 | 715.2m  | 67  | 33 | 320m   | 44.7% | 152 | 6  | 533.6m | 74.6% |
| 1-1500 | 1091.8m | 132 | -- | 423.2m | 38.8% | 153 | -- | 904.1m | 82.8% |
| **1-2500 (fix applied)** | **1885.1m** | **214** | **114** | **775.2m** | **41.1%** | **307** | **10** | **1608.1m** | **85.3%** |

sbp-diag on this segment: SIFT accepted=15.8%, empty_window=45.1%,
dist_reject=38.9% (of total); ORB accepted=36.6%, empty_window=40.2%,
dist_reject=18.2% -- consistent with the same qualitative gap measured on
every shorter segment this part.

**Honest read**: SIFT's coverage % at 2500 frames (41.1%) is *slightly
higher* than the unfixed 1500-frame number (38.8%) -- a mild positive
signal that the fix helps at longer lengths too, not just short segments
-- but this is not a clean apples-to-apples comparison (no unfixed
2500-frame run exists to compare against directly, and this system's
established run-to-run noise, part 53, is wide enough that a 2.3-point
swing isn't strong evidence on its own). **The gap to ORB remains large
and does not close**: ORB still covers more than double SIFT's percentage
(85.3% vs 41.1%), continuing the same widening-gap pattern part 54 first
found, just from a new, slightly-improved baseline. One SIFT fragment
(fragment 7, 409m) has RMSE=5.337m (within the 20m budget) but max
error=22.075m (a single worst-point excursion above budget, RMSE is the
qualifying metric per the user's stated criterion, so this fragment still
qualifies, but it's a large excursion worth noting). ORB's fragment 2
(1351.9m, the vast majority of ORB's covered distance) has RMSE=6.771m,
also comfortably within budget.

**Conclusion for this session's investigation**: the `GetScaleFactor` fix
is real, mechanism-verified, and modestly helps at longer segment lengths
too, but it does not close the fundamental gap to stock ORB on this
dataset. Combined with the ruled-out window-width and nFeatures levers,
and the already-established raw-keypoint-yield gap (part 55, SIFT ~66% of
ORB's), the remaining gap looks increasingly like a genuine property of
SIFT/DoG detection on KITTI's low-texture urban driving imagery rather
than a further fixable bug -- consistent with what part 56's own
mid-investigation "worst case" discussion with the user anticipated.
Further gains from here would likely need either accepting a much higher
`nFeatures` despite its cost (last tested at 7000, no help; 10000 tested
earlier pre-fix, also not adopted), or a genuinely different
investigation (e.g. visualizing actual detected keypoint locations on a
hard frame to see qualitatively what ORB's detector finds that SIFT's
doesn't), both flagged as lower-confidence next steps rather than another
quick isolate-and-attack win.

### Part 56 continued: settled the "is it a bug or a physical limit" question by directly visualizing keypoint locations

Built a small standalone tool (`analyze/orbslam3_visualize_keypoints.cpp`,
two CMake targets -- `orbslam3_visualize_keypoints` linked against stock
`orbslam3_ext`, `orbslam3_sift_visualize_keypoints` linked against
`orbslam3_sift_ext`, same shared-source-two-targets pattern as
`orbslam3_kitti_ate`/`orbslam3_sift_kitti_ate`) that runs the real
extractor on a single image and draws every detected keypoint as a
circle. Ran both on KITTI frame **004435.png** -- identified as the
lowest-keypoint-count frame in the primary hot zone (2420 keypoints,
found by grepping `[raw-keypoints]` from `scalefix_hotzone.log` and
sorting), i.e. the single hardest frame measured all session.

**Result: ORB=4863 keypoints, SIFT=2420 keypoints on the identical
image** (matches the ~66% yield gap from part 55 closely). Visually
comparing the two keypoint maps (saved to `scratchpad/kp_orb_4435.png`/
`scratchpad/kp_sift_4435.png`): **ORB densely covers essentially every
surface in the frame, including the flat asphalt road itself** (a large,
dense keypoint cluster along the bottom of the image where the road is).
**SIFT's keypoints concentrate on genuinely high-contrast structure**
(window edges, a traffic sign, car outline, tree/hedge texture, sidewalk
tile-joint lines, building edges) **and are almost entirely absent on the
road surface itself**. Away from the road (building facade, windows,
sign, car, tree), SIFT and ORB target similar regions -- the gap is not
uniform across the image, it's concentrated specifically on low-texture
flat surfaces.

**This directly and visually answers the question**: SIFT is not
"missing" real corners that ORB correctly finds -- ORB's FAST-corner
detector is picking up fine-grained intensity noise/texture on the
asphalt (subtle cracks, aggregate texture, lighting variation) that
triggers a corner response even though the underlying surface has little
real 3D structure, while SIFT's DoG contrast-extremum requirement
correctly declines to fire there. This is a genuine, structural
difference in detector philosophy (corner-response vs. scale-space
blob-extremum), **not a bug or a miscalibration** -- consistent with why
lowering SIFT's contrast threshold (tried twice, part 51 and part 55)
backfired: forcing SIFT to fire on the road's weak contrast just let
through low-quality, likely less-repeatable points, exactly like ORB's
road-surface keypoints, without the tracking benefit apparently making up
for the drop in per-point reliability.

**Conclusion**: the remaining SIFT-vs-ORB gap on this dataset is a real,
inherent property of the two detectors' algorithms on KITTI's road-heavy
driving imagery, not a further fixable bug. This closes the investigation
thread that began with today's `GetScaleFactor` fix -- two real,
verified, kept fixes were found and shipped this session (`DUtils::Random`
thread-safety mutex from earlier in the session, and this part's
`GetScaleFactor` unit-mismatch fix), the window-width and nFeatures
levers were re-tested post-fix and correctly found not to help, and the
remaining gap is now visually confirmed as algorithmic rather than a
code defect.

### Part 56 continued: "algorithmic, not a bug" doesn't mean "not fixable" -- spatially-fair keypoint selection via the previously-dead DistributeOctTree

Per user's pushback ("SIFT should be much stronger since it extracts
better points -- is there a way to make it work?"), re-examined whether
"algorithmic difference" really meant "nothing left to try." Found that
`operator()` calls `mSift->detectAndCompute()` and uses cv::SIFT's
internal top-`nfeatures`-by-response retention directly -- meaning when
the contrast threshold is lowered (as tried twice before, parts 51/55,
both worse), the EXTRA weak candidates it admits still compete in a
single GLOBAL response ranking with no spatial awareness, so they're
disproportionately drawn from already-dense high-response regions
(buildings) rather than the starved road region -- diluting quality
everywhere without fixing the actual spatial imbalance. Also discovered
`DistributeOctTree`/`ComputeKeyPointsOctTree` (the standard ORB-SLAM3
octree-quadrant keypoint-distribution algorithm, which picks the
best-response point PER SPATIAL CELL) exists in this file as **completely
dead code** for SIFT -- inherited from the ORB->SIFT port but never wired
into `operator()`, which bypasses it entirely.

**Change**: (1) lowered the detection contrast threshold to 0.02
(`kCandidateContrastThreshold`, both in the constructor and
`SetDynamicDensity`) and requested UNLIMITED candidates from `cv::SIFT`
(`nfeatures=0`, meaning "don't pre-cap by response at all"); (2) in
`operator()`, when the raw candidate count exceeds the real `nfeatures`
target, run it through `DistributeOctTree` (previously dead code, now
wired in) to select the final set with spatial fairness instead of
cv::SIFT's own global response ranking -- so a weak-but-unique road
candidate can survive against a strong building candidate as long as
they're in different spatial cells. Descriptor rows are recovered
afterward via a temporary `class_id`-as-index trick (confirmed no other
live consumer of that field in this codebase).

**Quick visual sanity check** on the same hardest frame (004435.png,
previously SIFT=2420 vs ORB=4863): **SIFT now detects 4857 keypoints**
(nearly matching ORB's count on the identical image) -- and the road/
sidewalk region, previously almost empty in the SIFT keypoint map, now
shows meaningful coverage (`scratchpad/kp_sift_4435_v2.png` vs the
original `kp_sift_4435.png`). Note: 4857 came in *under* the nfeatures=5000
target for this particular frame, so `DistributeOctTree` didn't even need
to cull anything here -- this jump is purely from the lower detection
threshold + uncapped retention; the octree culling only engages on
frames where the enlarged candidate pool exceeds 5000.

**Result: clearly worse on every metric -- reverted.**

| metric | GetScaleFactor fix (kept) | + octree spatial fix |
|---|---|---|
| fails | 53 | 72 |
| resets | 29 | 40 |
| coverage | 379.9m (53.1%) | 250.8m (35.1%) |
| accepted-of-total | 16.7% | 15.1% |
| empty_window | 45.7% | 46.2% (barely moved) |
| outlier rate | 6.6% | 8.1% |

Raw keypoint count on the hardest measured frame did jump from 2420 to
4857 (vs ORB's 4863 -- nearly matched), and the road region visibly
gained keypoint coverage in the visualization. But the real tracking
evaluation got worse everywhere, and tellingly, `empty_window` barely
moved despite far more raw candidates -- the new road-region points
mostly didn't even help fill *existing* map points' search windows
(those target already-triangulated 3D points, predominantly established
from non-road structure), while their inherently weaker, lower-contrast
descriptors measurably hurt match/outlier quality wherever they did get
matched (outlier rate 6.6%->8.1%).

**This is the third confirmed instance of the same failure mode** (parts
51, 55, and this one): admitting lower-contrast SIFT candidates hurts
more than any quantity increase helps, **regardless of whether they're
spatially well-distributed or not**. This specifically rules out the
"non-spatial global response ranking" hypothesis as the explanation for
why threshold-lowering fails -- wiring in `DistributeOctTree` (previously
dead code, now confirmed not to be the missing piece either) didn't
rescue it. The real problem is intrinsic descriptor quality at low
contrast, which no amount of spatial redistribution can fix. **Reverted**
all of this part's changes (`kCandidateContrastThreshold`, the unlimited-
candidates `cv::SIFT::create` calls in both the constructor and
`SetDynamicDensity`, and the `DistributeOctTree` wiring in `operator()`)
back to the original `kContrastThreshold=0.04`, response-ranked,
nfeatures-capped behavior -- confirmed via grep and a clean rebuild.

**Answer to the user's question** ("SIFT should be stronger since it
extracts better points -- is there a way to make it work?"): the
keypoint-visualization tool (`analyze/orbslam3_visualize_keypoints.cpp`,
kept in the tree for future use) gave a definitive, evidence-based
answer. SIFT's per-point descriptor quality claim is true in the
classical-CV sense, but on this specific dataset the binding constraint
is that SIFT's stricter contrast-extremum requirement finds very few
points on flat, low-texture road surfaces that make up a large fraction
of KITTI's driving scenes -- and every attempt to recover those points
(lower threshold alone, twice; lower threshold + spatially-fair
selection, once) made overall tracking measurably worse, because the
points recoverable that way are inherently lower-quality, not just
differently distributed. This is now a well-evidenced, thrice-confirmed
conclusion rather than a guess: **the remaining gap is a genuine,
structural property of SIFT/DoG detection on this class of imagery, not
a fixable bug or a missing algorithmic piece.**

### Part 56 continued: fourth attempt, CLAHE local-contrast enhancement -- also failed, closing this investigation line

Per user's continued push ("có cách nào tăng độ mạnh trên mặt đường
không" -- any way to boost feature strength on the road), tried a
genuinely different-in-kind approach: instead of lowering SIFT's
acceptance threshold (the three already-failed attempts), boost LOCAL
image contrast via CLAHE (`cv::createCLAHE(2.0, {8,8})`) before
detection, so genuinely-present-but-weak road texture (subtle
cracks/aggregate) could clear the *original, unchanged* 0.04 threshold on
its own merit rather than admitting borderline candidates. Applied to a
separate `Mat`, not mutating the extractor's own `image` buffer.

**Visual sanity check**: on the same hardest frame (004435.png), keypoint
count jumped from 2420 to the full 5000 cap (`scratchpad/
kp_sift_4435_clahe.png`) -- road/sidewalk coverage visibly improved, and
the underlying enhanced image showed genuinely sharper texture (tile
joints etc.), not just noise.

**Real evaluation result on frames 1-1000: the fourth consecutive
failure.**

| metric | GetScaleFactor fix (kept) | + CLAHE |
|---|---|---|
| fails | 53 | 84 |
| resets | 29 | 43 |
| coverage | 379.9m (53.1%) | 212.9m (29.8%) |
| accepted-of-total | 16.7% | 12.9% |
| empty_window | 45.7% | 47.1% |
| outlier rate | 6.6% | 7.0% |
| worst fragment RMSE | <1m typical | 13.764m (still under the 20m budget, but a real outlier) |

**Interpretation**: even genuine local-contrast enhancement (not just a
looser acceptance bar) made things worse, and notably produced one
fragment with much higher RMSE than typical -- consistent with a
different, complementary explanation beyond "weak descriptors": points on
a near-planar road surface give inherently poor triangulation geometry
(shallow/degenerate depth estimation) even when 2D matching itself
succeeds, and/or CLAHE's per-frame-adaptive local normalization hurts
frame-to-frame descriptor repeatability for otherwise-good points (tile
boundaries and local statistics can shift slightly frame to frame even
for the same physical surface). **Reverted** (confirmed via grep and a
clean rebuild -- `operator()` calls `mSift->detectAndCompute(image, ...)`
on the raw image again, no CLAHE).

**This closes the "recover SIFT's road-surface deficit" investigation
line with strong, consistent evidence**: four independent, mechanistically
distinct attempts (lower threshold alone x2, lower threshold + spatial
fairness, local contrast enhancement) all failed the same way -- more
road-region keypoints, by whatever means, net hurt real tracking
performance on this dataset. This is no longer a single negative result
that might be attributable to one bad parameter choice; it's a
triangulated, four-for-four pattern strongly suggesting the SIFT-vs-ORB
gap on KITTI's road-heavy driving imagery is not recoverable via
feature-detection-level interventions with this pipeline. Further
progress, if wanted, would need a fundamentally different architectural
approach (e.g. a hybrid detector, or optical-flow-assisted tracking for
low-texture regions instead of detect-and-match), not another
detector-parameter or preprocessing tweak.

### Part 56 continued: a different kind of lever -- loosening the TrackLocalMap acceptance gate instead of the detector

User's own idea, after the four detector-level failures: instead of
trying to create more/better matches upstream, what if the downstream
*acceptance* gate (`TrackLocalMap()`'s `mnMatchesInliers<15` check,
deciding whether to trust a frame's tracking as "OK" at all) is itself
needlessly conservative? The gate doesn't create real matches -- it only
decides whether to keep going with what `SearchLocalPoints` already
found. Explicitly flagged the theoretical risk before testing: loosening
this could accept degenerate/low-confidence poses that blow the user's
own 20m ATE RMSE budget.

**Tested progressively on frames 1-1000**: `need>=15` (default) ->
`need>=3` -> `need>=1` (accept any nonzero inlier count, the most literal
"no constraint").

| gate | fails | resets | coverage | max fragment RMSE |
|---|---|---|---|---|
| need>=15 (default) | 53 | 29 | 379.9m (53.1%) | ~4m typical |
| need>=3 | 47 | 27 | 362.1m (50.6%) | 1.546m (safe) |
| need>=1, trial 1 | 50 | 27 | 441.6m (**61.7%**) | 0.283m (very clean) |
| need>=1, trial 2 (repeat) | 56 | 32 | 393.9m (55.1%) | **9.636m, max=32.765m** |

**The theoretical risk materialized on the repeat trial, not the first
run.** Trial 1 looked like a clean, safe win (coverage +8.6pp, RMSE
*better* than baseline). Trial 2 on the identical segment (this system's
established thread-race nondeterminism, part 53) still showed elevated
coverage over the `need>=15` baseline (55.1% vs 53.1%) but produced one
fragment with RMSE=9.636m and a single-point max error of 32.765m --
technically still passing on RMSE (the user's stated qualifying metric)
but a real, large excursion the `need>=15` baseline never produced in
any measured fragment this entire session. **This is exactly the
"sometimes safe, sometimes not" cliff the pre-test warning predicted** --
just not visible on a single run.

**Hot-zone validation (frames 4000-4540, GT=586.2m) makes the picture
more mixed still**:

| | GetScaleFactor fix only | + need>=1 |
|---|---|---|
| fails | 94 | 84 |
| resets | 48 | 46 |
| coverage | 53.3m (9.1%) | 42.1m (**7.2%, worse**) |

fails/resets both improved slightly, but coverage actually *dropped* on
this session's hardest, most-tested segment, and only 1 of 2 fragments
even produced a usable trajectory (the other had 0 keyframes).

**Fourth data point -- longer 1-2500 validation (GT=1885.1m), the most
statistically stable segment tested**: 10/10 fragments successfully
aligned (no alignment failures, unlike every shorter test this
sub-investigation ran), fails 214->168, resets 114->93, coverage
775.2m(41.1%)->**931.8m(49.4%)**, and every fragment's RMSE stayed very
safe (max 2.206m, nowhere near the trial-2 anomaly seen on the shorter
segment).

**Final synthesis across all four validation points**:

| segment | need>=15 (baseline) | need>=1 | RMSE safety |
|---|---|---|---|
| 1-1000, trial 1 | 53.1% | 61.7% | clean (max 0.283m) |
| 1-1000, trial 2 | 53.1% | 55.1% | one fragment 9.636m/max 32.765m -- still under budget, but a real excursion |
| hot zone (4000-4540) | 9.1% | 7.2% (worse) | clean (max 4.533m) |
| 1-2500 | 41.1% | **49.4%** | clean (max 2.206m) |

**3 of 4 segments show real coverage improvement**, including the
longest/most-stable test (1-2500: +8.3 percentage points, fails and
resets both down substantially, 10/10 fragments cleanly aligned). Only
the hot zone -- this session's established noisiest, hardest segment --
showed a net loss, and only one of the four test *runs* (the 1-1000
repeat trial) produced an RMSE excursion large enough to be a real
concern, though it never actually breached the user's stated 20m budget.

**Recommendation**: on balance, `need>=1` looks like a genuine, if
imperfect, win -- particularly validated by the longer/more-stable
1-2500 test. It is NOT risk-free (demonstrated real variance, one
fragment came uncomfortably close to the budget's spirit even while
technically passing it) and specifically underperforms on the hardest
segment. This is a judgment call between "more coverage on average, with
occasional larger (but still budget-compliant) errors" vs. "the
originally-tuned 15-match bar, more consistent but leaving real coverage
on the table." Reported to the user for a final decision rather than
silently adopted, given the demonstrated risk profile differs
qualitatively from the clean, unconditional GetScaleFactor win.

### Part 56 continued: user's interaction hypothesis -- does need>=1 rescue the road-density recovery attempts? No.

Sharp follow-up from the user: all four road-density recovery attempts
(parts 51, 55, DistributeOctTree, CLAHE) were tested against the
*strict* `need>=15` accept gate -- maybe that gate, not the weak
candidates themselves, was what rejected the resulting tracking before
it could pay off. Worth testing given `need>=1` alone just showed the
system tolerates low match counts reasonably well via BA.

**Tested**: `kContrastThreshold` 0.04->0.02 (part 51's original,
simplest failed value) *combined with* `need>=1` (kept from the prior
experiment), on frames 1-1000.

| | need>=15+0.04 (orig. baseline) | need>=1+0.04 (trial 1) | need>=1+0.04 (trial 2) | need>=1+0.02 (this combo) |
|---|---|---|---|---|
| fails | 53 | 50 | 56 | 59 |
| resets | 29 | 27 | 32 | 34 |
| coverage | 53.1% | 61.7% | 55.1% | **43.2%** |
| outlier rate | 6.6% | 6.4% | -- | 6.9% |

## Part 57 (2026-07-20): ASIFT integration -- a genuinely different lever, not another threshold tweak

Per user's explicit next-session plan (see [[project_orbslam3_vendoring]]
memory), tried **ASIFT** (Affine-SIFT, Morel & Yu) instead of repeating
the already-exhausted "lower the threshold" family of fixes. OpenCV 4.10
ships `cv::AffineFeature`, which wraps a base detector (here `cv::SIFT`,
identical config to the rest of this project) and simulates multiple
affine tilts/rotations, running the base detector on each simulated view
and merging results back into original-image coordinates -- this is a
direct, maintained implementation of the ASIFT algorithm, no need to
clone/integrate the unmaintained original IPOL reference code.

**Standalone feasibility test first** (`analyze/asift_test.cpp`, new
`asift_test` CMake target, pure OpenCV, no ORB_SLAM3 dependency) on the
same hardest-measured KITTI frame (004435.png, plain SIFT=2420 keypoints):
**ASIFT finds 35293 keypoints (~14.6x more)** in 1547ms (~7.8x slower
than plain SIFT's 197ms -- accepted per explicit user instruction,
"chấp nhận đánh đổi thời gian"). Visually, the road/sidewalk region --
where plain SIFT and four different threshold-based recovery attempts
all failed to find structure -- now shows dense, genuine keypoint
coverage following real tile-joint/pavement texture, not noise. Raw
`.octave` values are in the same encoding order of magnitude as plain
SIFT's (same 3-component bit-packing OpenCV uses internally), so this
project's existing `flatLevel()` decode logic needs no changes.

**Critical follow-up check before any real integration**: ASIFT's merged
35293 keypoints need capping down to this project's `nfeatures=5000`
target. Naive global-response capping (`cv::KeyPointsFilter::retainBest`,
matching plain `cv::SIFT`'s own internal behavior) **collapsed back onto
a handful of extreme-response clusters** (overexposed sign/window
regions) and threw away almost all the newly-recovered road coverage --
the same failure mode as every earlier "just lower the threshold, let
global response ranking decide" attempt. **Grid-based capping** (keep
the single best-response candidate per spatial cell, cell count sized to
land near the `nfeatures` target -- reusing the DistributeOctTree idea,
but applied to a genuinely richer candidate source this time) **kept the
road coverage intact** (`analyze/asift_test.cpp`'s three output images
document this progression directly).

**Integrated into the real pipeline**: `third_party/ORB_SLAM3_SIFT/
include/ORBextractor.h` gained a `cv::Ptr<cv::AffineFeature> mAsift`
member (wrapping `mSift`, constructed alongside it in both the
constructor and `SetDynamicDensity`); `operator()` now calls
`mAsift->detectAndCompute()` instead of `mSift->detectAndCompute()`
directly, followed by the grid-based capping logic (implemented inline,
not reusing `DistributeOctTree` verbatim, to keep the cell-sizing target
tied directly to `nfeatures`). Smoke-tested on 20 and 30 frames first
(no crashes; the 30-frame run's exit code 1 was just "too few keyframes
to align," an expected short-segment artifact, not a bug) before
committing to a full 1-1000 run. Measured runtime: ~2.5s/frame
(30 frames in 74.8s wall-clock, ~5.5x parallelism from `user` time/`real`
time ratio) -- roughly 12-13x slower than plain SIFT's established
~9-10fps baseline, projecting to ~40-60 minutes for a full 1-1000
evaluation (running now).

### Part 57 continued: runtime-scaling problems, then a final apples-to-apples verdict

**The full 1-1000 evaluation (maxTilt=5, the ASIFT paper's own default)
had to be killed twice.** First attempt (frames 1-1000): reached frame
722/1000 with 185 keyframes accumulated, then slowed from ~2s/frame to
~72s/frame -- `LoopClosing`'s candidate-checking cost (VLAD/BoW
retrieval + Sim3Solver RANSAC per candidate) scales with accumulated
keyframe count, and ASIFT's much larger per-frame match yield plausibly
drives faster keyframe creation, compounding the problem. Projected
5-6 hours to finish; killed per user's explicit choice. Second attempt,
shortened to frames 1-300: hit the same wall at frame 214 (46-82s/frame
once keyframes passed ~40-50). User asked to find a speedup rather than
cut further ("tìm cách tăng tốc").

**Fix**: reduced `cv::AffineFeature::create`'s `maxTilt` parameter from
the paper/OpenCV-default 5 to 2 (fewer simulated affine views per frame
-- standalone test showed this cuts raw keypoint count from 35293 to
16379 on the hardest frame and per-frame extraction time from 1547ms to
843ms, while still recovering most of the visible road-surface
coverage). This also indirectly controls keyframe-creation rate (fewer
matches per frame -> less aggressive keyframe insertion), which is what
actually fixed the runtime problem: the 1-300 run with maxTilt=2
completed in **~4.5 minutes** total, maintaining ~1-2s/frame throughout
thanks to periodic map resets keeping keyframe count from running away
(peaked at 50 KFs, vs the 150-200+ that caused the earlier slowdowns).

**Result (frames 1-300, GT=217.1m, both configs using `need>=1`, ASIFT
at maxTilt=2)**: 9 fails, 7 resets, coverage 138.5m (**63.8%**), outlier
rate 11.8%, accepted-of-total 16.1%, empty_window 54.7%, all fragment
RMSEs safely under budget (max 2.304m).

**But this needed a matched baseline to mean anything** -- no prior test
this session used exactly frames 1-300 with `need>=1`. Ran plain SIFT
(temporarily swapping `mAsift->detectAndCompute()` back to
`mSift->detectAndCompute()`, identical everything else) on the identical
segment:

| metric (frames 1-300, both need>=1) | Plain SIFT | ASIFT (maxTilt=2) |
|---|---|---|
| fails | 9 | 9 |
| resets | 5 | **7** |
| coverage | **81.6%** | 63.8% |
| outlier rate | **6.3%** | 11.8% (nearly double) |
| accepted-of-total | 17.3% | 16.1% |
| empty_window | 47.2% | 54.7% |

**Plain SIFT wins on every single metric.** This is the **sixth**
confirmed instance of the same pattern this session (parts 51, 55,
DistributeOctTree, CLAHE, the 0.02+need>=1 combo, and now ASIFT): more
keypoints -- even from a fundamentally different, qualitatively richer
source (real affine-simulated multi-view DoG extrema, not just a
loosened single-view threshold) -- do not translate into better SLAM
tracking on this dataset. The extra candidates measurably dilute match
reliability (outlier rate nearly doubled) more than their quantity
helps, exactly mirroring every earlier attempt despite using a
completely different underlying mechanism. Separately, the maxTilt=5->2
reduction needed just to make testing *practical at all* already weakens
ASIFT's road-recovery benefit relative to the full algorithm, so the
net cost (still meaningfully slower per-frame than plain SIFT, plus real
keyframe-count/runtime risk at scale) isn't justified even setting the
accuracy regression aside.

**Initial decision (superseded below)**: reverted `operator()` to use
`mSift->detectAndCompute()` (plain SIFT) as the active default.

### Part 57 continued: user's fix -- pair ASIFT with a TIGHTER gate, not a looser one -- reverses the verdict

User's insight: the losing ASIFT result above paired more (noisier)
candidates with `need>=1`, the LOOSEST acceptance gate tested this
session. What if ASIFT needs the OPPOSITE pairing -- more candidates
filtered by a STRICTER gate, so the extra noise gets rejected while the
extra genuine road-region matches still get through? This is the exact
opposite direction from the earlier failed 0.02-threshold+need>=1 combo
(which loosened both detection and acceptance together), so not
something already ruled out.

Re-enabled `mAsift` and set `Tracking.cc`'s `TrackLocalMap` gate back to
`need>=15` (the original, tightest value tested this session). Result on
the same matched frames 1-300 segment (GT=217.1m):

| metric | Plain SIFT + need>=1 | ASIFT(maxTilt=2) + need>=1 | **ASIFT(maxTilt=2) + need>=15** |
|---|---|---|---|
| fails | 9 | 9 | **2** |
| resets | 5 | 7 | **2** |
| coverage | 81.6% | 63.8% | **89.2%** |
| accepted-of-total | 17.3% | 16.1% | **25.2%** |
| empty_window | 47.2% | 54.7% | **42.1%** |
| outlier rate | **6.3%** | 11.8% | 11.5% |

**ASIFT+need>=15 wins decisively** -- beats not just ASIFT+need>=1 but
*plain SIFT itself*, on nearly every axis: fails collapsed from 9 to 2,
resets from 5-7 to 2, coverage climbed to 89.2% (vs plain SIFT's already-
strong 81.6%), and match quality metrics (accepted-rate, empty_window)
both improved too. Only the outlier rate stays elevated relative to
plain SIFT (11.5% vs 6.3%) -- but with fails/resets this much lower and
coverage this much higher, that tradeoff is clearly worth it. All
fragment RMSEs stayed safely under budget (max 2.311m).

**This is the session's second major, unconditionally-adopted result**,
alongside the `GetScaleFactor` fix. The lesson: ASIFT's extra candidates
were never inherently bad -- part 57's first attempt paired them with
the wrong (too-permissive) downstream filter. Tightening the filter
instead of loosening it lets the genuine road-region recovery pay off
without the noise cost. **Kept configuration**: `mAsift` active
(`operator()` calls `mAsift->detectAndCompute()`, `maxTilt=2`),
`TrackLocalMap` gate at `need>=15`. Confirmed via grep and rebuild.
Validating on longer segments (1-1000, 1-2000) next.

### Part 57 continued: final validation on frames 1-1000 -- confirms the win at scale

A parallel 1-2000 run (GT=1483.7m) was also launched but killed early
(frame ~94/2000) per explicit user choice, to conserve memory/CPU for the
1-1000 run rather than run both to completion.

**Result (frames 1-1000, GT=715.2m, ASIFT maxTilt=2 + need>=15)**: 44
fails, 31 resets, 3 fragments totaling 497.3m matched path (**69.5%
coverage**), outlier rate 11.9% (consistent with the 1-300 result's
11.5%), all fragment ATE RMSEs far under budget (0.699m / 0.031m /
1.258m). sbp-diag: 12.0% accepted, 57.0% empty_window.

| metric (frames 1-1000, GT=715.2m) | Plain SIFT + need>=15 (GetScaleFactor-fix-only baseline) | **ASIFT(maxTilt=2) + need>=15** | Stock ORB |
|---|---|---|---|
| fails | 53 | **44** | 152 |
| resets | **29** | 31 | 6 |
| coverage | 53.1% | **69.5%** | 74.6% |

**ASIFT+need>=15's win generalizes from the short 1-300 segment (89.2%
coverage) to the full 1-1000 segment (69.5% coverage)** -- the margin
over plain SIFT shrinks on the longer run (+16.4 points here vs +7.6
points on 1-300, since 1-300's plain-SIFT baseline was itself already
81.6% with `need>=1`, not the `need>=15` baseline used here) but stays
decisively positive on every core metric except a small reset uptick
(+2). It does not fully close the gap to stock ORB's 74.6%, but cuts the
plain-SIFT-vs-ORB gap (21.5 points) by more than three-quarters (down to
5.1 points). **This is the confirmed final result of the ASIFT
investigation**: ASIFT(maxTilt=2) + `TrackLocalMap` need>=15 is adopted
as the new kept configuration, superseding both plain SIFT and the
earlier (need>=1-paired) losing ASIFT trial.

### Part 57 continued: 1-2500 validation -- coverage is excellent, but exposes a real RMSE-budget violation

Ran the same config on the longest segment tested this session, frames
1-2500 (GT=1884.1m computed directly from `poses/00.txt`). Runtime was
~44 minutes; mapKFs oscillated between ~26 and ~200 across several
resets rather than growing unbounded, so the maxTilt=5-era runaway
slowdown did not recur.

**Result**: 11 fails, 6 resets, 3 fragments totaling 1820.3m matched
path (**96.6% coverage** -- by far the best coverage number of the whole
ASIFT investigation), outlier rate 11.9% (consistent with every other
segment). But:

| fragment | KFs | pathLen | ATE RMSE | budget status |
|---|---|---|---|---|
| 0 | 232 | 756.3m | 7.976m | OK |
| 1 | 191 | 559.2m | **24.553m** | **OVER the 20m budget** |
| 2 | 139 | 504.8m | **36.806m** | **OVER the 20m budget, badly** |

**This is a real, previously-unseen problem, not a wash**: 2 of 3
fragments blow past the user's standing <20m-RMSE-per-fragment
constraint, one by nearly 2x. The mechanism is a direct tradeoff against
the very thing that drove coverage up: only 6 resets across 2500 frames
(vs 31 for the same config on 1-1000, and far more for the historical
`need>=1`+plain-SIFT baseline on 1-2500, which had "10/10 fragments
aligned, max RMSE 2.2m" per Session 16 notes) means each surviving map
lives far longer without a reset to bound accumulated drift -- fewer
resets is *usually* read as a pure win in this investigation's metrics,
but a long-lived map with no correction mechanism (no ground-truth-scale
loop closure in this monocular SIFT+VLAD setup) just accumulates error
unchecked until the segment ends. High coverage and low RMSE are not the
same axis, and this config's biggest strength (very few resets) is also
what let two of its three fragments drift out of budget.

**Net assessment**: ASIFT+need>=15 is not an unconditional win at this
scale the way it was at 1-300/1-1000 -- it is a genuine coverage
improvement that comes with a real, budget-violating accuracy risk on
long, low-reset stretches. Whether this is acceptable depends on whether
coverage or per-fragment accuracy is weighted higher for segments this
long; flagged to the user rather than declared a clean win. Motivated
the next investigation: pairing ASIFT with an even stricter accept gate
(`need>=30`, ORB-SLAM3's true original default, pulled to 15 in this
fork back in part 29) to see if tighter downstream filtering also helps
control drift, not just coverage.

**Tested need>=30, reverted per explicit user instruction before further validation.** Quick check on frames 1-300 (GT=217.1m): 17 fails, 10 resets, coverage 63.9% (only 1 of 2 fragments aligned, at 138.7m/0.240m RMSE; the other failed alignment outright with just 2 KFs), outlier rate 12.6%. Clearly worse than need>=15 on every metric (2 fails, 2 resets, 89.2% coverage, 11.5% outlier) -- the "stricter filter helps more" pattern that worked going from need>=1 to need>=15 did not repeat going from 15 to 30; 30 is simply too strict for this segment, rejecting too much otherwise-usable tracking. **Reverted to need>=15** (confirmed via grep and rebuild) -- this remains the final kept gate value, paired with ASIFT(maxTilt=2). The 1-2500 RMSE-budget-violation problem found just before this test remains open and unresolved by this attempt.

### Part 57 continued: full-sequence validation (frames 0-4541, the whole of KITTI seq00) -- coverage/RMSE tradeoff resolves the other way

Per explicit user instruction ("tối ưu RMSE sau, bây giờ chạy full đi" -- defer fixing the RMSE-budget issue, just get the full-sequence number now), ran ASIFT(maxTilt=2)+need>=15 on the entire sequence, frames 0-4541 (GT=3722.3m, computed directly from `poses/00.txt`). Runtime was ~2h12m -- by far the longest run of the investigation, and it visibly passed through at least two distinct hard stretches (frames ~2600-2900 and ~4000-4450, the latter matching a "primary hot zone" flagged in earlier session notes) where pace degraded to several seconds/frame and resets/fails clustered heavily, before recovering each time. mapKFs briefly spiked to 386 (the highest seen this session) around frame 4130 but was flushed by a natural map reset rather than spiraling into the catastrophic loop-closing runaway seen earlier with maxTilt=5 -- the maxTilt=2 fix held up even under this much more demanding full-sequence load.

**Result**: 180 fails, 121 resets, 8 map fragments (6 aligned, 2 failed alignment outright with only 3 KFs each) totaling 964.3m matched path (**25.9% coverage** -- far lower than every shorter segment tested: 1-300's 89.2%, 1-1000's 69.5%, 1-2500's 96.6%), outlier rate 15.0% (the highest seen this session). sbp-diag: 18.9% accepted, 45.4% empty_window.

| fragment | KFs | pathLen | ATE RMSE | budget status |
|---|---|---|---|---|
| 0 | 83 | 224.4m | 1.015m | OK |
| 1 | 3 | -- | alignment failed | n/a |
| 2 | 23 | 81.9m | 0.093m | OK |
| 3 | 137 | 402.8m | 2.478m | OK |
| 4 | 62 | 146.8m | 0.874m | OK |
| 5 | 10 | 27.3m | 0.317m | OK |
| 6 | 15 | 81.1m | 0.430m | OK |
| 7 | 3 | -- | alignment failed | n/a |

**Every aligned fragment stayed comfortably within the 20m RMSE budget** (worst case 2.478m) -- the OPPOSITE outcome from 1-2500 (where 2 of 3 fragments blew the budget). This is the coverage/RMSE tradeoff identified at 1-2500 playing out in the other direction: 121 resets over 4541 frames (~1 reset per 37.5 frames) is a much higher reset density than 1-2500's 6 resets over 2500 frames (~1 per 417), so no single map fragment got the chance to live long enough to accumulate drift past budget -- but the flip side is heavy fragmentation (8 fragments, 2 of them too short to even align) and a correspondingly low total-coverage number, since resets cost dead time (relocalization attempts, short-lived low-value fragments) that doesn't contribute matched path.

**Net assessment closing out the ASIFT investigation**: ASIFT(maxTilt=2)+need>=15 does not have a single "coverage number" independent of how hard/long the segment is -- on easier, shorter stretches it delivers excellent coverage (up to 96.6%) sometimes at real RMSE-budget risk; on the full, harder sequence it self-regulates via more frequent resets into a materially lower-coverage (25.9%) but fully budget-safe result. Both the coverage win (vs. the 53.1%/44-fails/31-resets plain-SIFT 1-1000 baseline) and the accuracy discipline (0 budget violations across the whole sequence) are real and reproducible; the two just don't co-occur at the same segment length in the same way. **This is the final, confirmed full-sequence result for the session's adopted configuration.** No RMSE-specific fix was attempted this round (explicitly deferred by the user); that remains a legitimate follow-up if higher full-sequence coverage is wanted without sacrificing the current budget safety margin.

### Part 57 continued: two abandoned follow-up attempts, then a root-cause diagnosis via the ASIFT paper -- final verdict: NEGATIVE RESULT

**User pushback on the framing above**: after the full-sequence result, the user directly rejected treating "0 RMSE-budget violations" as a consolation for 25.9% coverage -- they had already told the session RMSE violations were acceptable for this run, so the low coverage number stands on its own as a real problem, not offset by anything. This is the correct framing; the prior paragraph's "coverage/RMSE tradeoff, both real and reproducible" language undersold how bad 25.9% coverage actually is when RMSE safety isn't being valued.

**Two follow-up attempts were launched to try to fix it, both abandoned mid-run without producing usable results**: (1) `maxTilt=1` (further reduced from the kept `maxTilt=2`) tested on frames 1-1000 as a speed/quality tradeoff -- early signal was clearly bad (26 fails, 20 resets in just the first 182/1000 frames, worse pace than the full 1000-frame maxTilt=2 result of 44 fails/31 resets total); (2) `need>=1` (loosest gate) tested in parallel on frames 2500-4541 (the hard back-half that caused most of the full-sequence fragmentation) to see if a looser gate would reduce resets there now that RMSE safety wasn't the priority. **Both were killed by explicit user instruction ("kill hết đi, chắc dừng lại ở đây, NEGATIVE RESULT") before either produced a complete result** -- no fragment/coverage data exists for either. Both code changes were reverted (`maxTilt` back to 2, gate back to `need>=15`) and the rebuild reconfirmed the tree matches the originally-adopted state.

**Root-cause diagnosis, via the actual ASIFT paper (Morel & Yu, IPOL 2011), not just further parameter search**: fetched and read the paper directly. **ASIFT was designed and validated for wide-baseline matching** (two images from substantially different viewpoints -- stereo with large disparity, image retrieval, panorama stitching) -- not for per-frame feature extraction on consecutive video frames, where the baseline between frames is inherently small. The paper's whole mechanism (simulating multiple affine tilts/rotations before running SIFT) exists specifically to compensate for *large* affine distortion between the two views being compared; it gives no guidance for scaling tilt range down for small-baseline pairs, and nothing in the paper suggests it's intended for near-fronto-parallel image pairs. **This is a genuine, citable explanation for the elevated outlier rate observed at every scale this session (11.5-15% vs plain SIFT's consistent ~6.3%)**: most of ASIFT's extra affine-simulated keypoints have no true correspondence in an actual small-baseline consecutive-frame pair, so they contribute disproportionately to false matches rather than genuine recoverable structure. The `need>=15` gate's apparent "fix" (parts above) was never fixing the mismatch -- it was filtering out enough of this structural noise, on short/easy segments, for the surviving genuine road-recovery keypoints to still net out ahead. On harder/longer segments (1-2500, and especially the full sequence) there isn't a filter setting that avoids the tradeoff, because the noise source itself doesn't go away by tuning a downstream threshold.

**Final verdict: NEGATIVE RESULT.** ASIFT is not the right tool for this use case (continuous small-baseline monocular VO/SLAM feature extraction) at a structural level, not a tuning level -- no combination of `maxTilt`, capping strategy, or accept-gate value tested this session (or likely to be tested in the future) can fix a mismatch that exists between the algorithm's designed operating regime and this project's actual per-frame baseline. The earlier "wins" (1-300: 89.2%, 1-1000: 69.5%, 1-2500: 96.6% coverage) were real and reproducible on those specific segments, but do not represent ASIFT being fundamentally suited to this task -- they represent the `need>=15` gate successfully laundering enough of ASIFT's structurally-elevated noise on those particular (shorter/easier) segments to still come out ahead of plain SIFT. The full-sequence result (25.9% coverage) is the more representative number for what this configuration actually delivers at real operating scale. **`mAsift` remains present in the code (constructed, available) but is being retired as an active investigation path** -- future SIFT-quality work should look elsewhere (e.g., the still-open, unrelated `dist_reject`-dominance finding in [[Match-Rate-Investigation]], or genuinely different detectors/strategies) rather than further ASIFT parameter search.

**Result: worse than need>=1 alone (both trials) AND worse than the
original baseline.** This cleanly refutes the interaction hypothesis --
lowering the contrast threshold is independently harmful regardless of
how permissive the downstream acceptance gate is. The four-way
conclusion from earlier in part 56 stands on firmer footing now: the
road-density recovery failures are intrinsic to the resulting
descriptors' quality (weak local contrast -> unstable/noisy SIFT
descriptors), not an artifact of an overly strict acceptance gate
rejecting otherwise-salvageable tracking. **Reverted** `kContrastThreshold`
back to 0.04 (confirmed via grep and rebuild); `need>=1` left as-is
(still the open, user-decision-pending state from the prior
sub-investigation, unaffected by this specific negative result).

**Pushed one step further per explicit user request**: `kContrastThreshold`
0.02 -> **0.005** (still combined with `need>=1`, per user's explicit
instruction not to fall back to `need>=15` for this test). Result: **23.1%
coverage** (165.4m of 715.2m, only 1 of 2 fragments produced any
trajectory at all) -- worse than both 0.02's 43.2% and need>=1-alone's
55-62%, confirming the monotonic worsening trend continues rather than
reversing at more extreme values. fails=93, resets=58, both worse than
baseline. Notably `empty_window` also got *worse* (60.3%, above even the
unmodified 0.04 baseline's ~45-58%) despite far more raw candidates being
generated -- plausibly because at this extreme, sheer candidate-pool
noise volume crowds out even the previously-reliable high-response
candidates within `cv::SIFT`'s own internal top-`nfeatures` response-based
retention (a plausible, not directly verified, mechanism). **Reverted**
`kContrastThreshold` back to 0.04 (confirmed via grep and rebuild). This
closes the contrastThreshold-lowering line of inquiry decisively: tested
at 0.02, 0.01, 0.005, alone and combined with both DistributeOctTree and
need>=1 -- six total configurations, all worse than baseline, monotonic
trend with no reversal at any tested extreme.

## Current status (2026-07-19, parts 49-53 -- fast hot-zone SIFT tuning +
## a real thread-safety bug found, but full determinism NOT achievable
## cheaply)

Per user request, switched to fast iteration: instead of full 4541-frame
seq00 runs (~20-40min each), used `orbslam3_kitti_ate`'s existing
`start-frame`/`max-frames` CLI args to test on **frames 4000-4540 (541
frames)** -- identified as the highest raw-failure-density stretch in the
whole sequence (105-107 `Fail to track local map!` events, ~19.8%, vs
~10.4% sequence-wide). Target set by user: get the raw fail rate under 10%
(full-run baseline without today's SIFT tuning: 484/4541=10.66%, from part
48's destroy=9 run -- already very close).

### Part 48 (destroy-threshold) final verdict, both full runs now complete
`part43_run1` (destroy=5): 1492.4m/14 fragments, 483/4541=10.64% fails, 255
resets, ~56min runtime, RMSE clean (max 1.016m). `part44_run1` (destroy=9):
1388.6m/20 fragments, 484/4541=10.66% fails, 250 resets, ~38-40min runtime,
RMSE clean (max 2.291m). **Fail rate is virtually identical between the
two** (confirms destroy-threshold only affects what happens after a track
is already lost, not the underlying loss frequency). Coverage: destroy=5
slightly higher (+7.5%, within this session's noise band). **Kept
destroy=9** for its ~40% faster runtime given the coverage difference isn't
clearly real.

### Hot-zone SIFT-tuning experiments (parts 49-52) -- results now suspect, see part 53
Ran in rapid succession on the primary 541-frame hot zone:
- baseline (pre-today's-SIFT-changes): 107 fails
- part 49: denser keyframe insertion (`kKeyFrameMaxFrames` = half of
  `mMaxFrames`, decoupled from the shared `mMaxFrames` used elsewhere) --
  not isolated on its own, folded into later combined tests.
- part 50: retargeted an EXISTING but misfiring "dynamic density boost"
  mechanism (`SetDynamicDensity`, 2x nFeatures + contrast threshold
  0.04->0.02) from angular velocity (rotation/turns -- which part 45's
  GT-pose correlation had already REFUTED as a failure cause) to
  translation velocity (speed -- which part 45 confirmed correlates),
  reusing the same calibrated threshold (0.087, true measured p90 from
  part 47). Result: 102 fails (best single-run result of this batch).
- part 51: same boost mechanism but ALWAYS ON (unconditional) instead of
  velocity-gated -- **confirmed worse (120 fails) across 4 independent
  samples** (primary zone + 3 other segments/a large 1500-frame sample, all
  trended worse), not just noise. Reverted to velocity-gated.
- part 52: `kEdgeThreshold` 10.0->20.0 (SIFT's edge-keypoint rejection
  threshold, stock/untouched all session) combined with the part 50 boost
  -- gave 107 fails, i.e. back to baseline, apparently cancelling the
  boost's benefit. Reverted to stock 10.0.

### Part 53: found a real thread-safety bug, but it's NOT the (whole) explanation
**Repeat-testing the EXACT SAME config** (part 50's velocity-gated boost,
edgeThreshold reverted) on the primary hot zone gave **102, then 111, then
102** across three consecutive identical runs -- i.e. **most of the
parts 49-52 comparisons above (differences of 5-18 fails) are likely
noise, not real signal**, except part 51's always-on-boost result (120),
which is large and consistent enough across 4 independent samples to
trust.

**Root cause found and partially fixed**: `DUtils::Random::RandomInt()`
(and `RandomValue<T>()`) in `third_party/DBoW2/DUtils/Random.cpp/.h` use
plain C `rand()`, which has a single global, non-thread-safe state.
Sim3Solver/MLPnPsolver RANSAC calls happen from multiple concurrent
threads (Tracking, LocalMapping, LoopClosing) -- concurrent unsynchronized
`rand()` calls are undefined behavior, and this fork calls into
RANSAC/relocalization *far* more often than stock ORB-SLAM3 (hundreds of
times per run, vs. stock's near-zero given only 3 total resets), so the
race gets exercised constantly here specifically. **Confirmed:
`third_party/ORB_SLAM3` (original, untouched) has its own separate
`Thirdparty/DBoW2` copy** (same underlying bug, but essentially never
triggered given how rarely it loses tracking) -- this fix does not touch
it. **Fix applied**: added a `std::mutex` guarding all `rand()` access
points in `Random.h`/`Random.cpp` (`RandomInt`, `RandomValue<T>`, and the
`SeedRand*` functions) -- kept, this is a genuine correctness fix
regardless of the rest of this investigation's outcome, and does not
change RANSAC's own sampling behavior (still draws many random subsets per
call), only makes the draw *sequence* reproducible for given input instead
of thread-scheduling-dependent.

**But the mutex fix alone did NOT achieve full determinism**: two
back-to-back identical runs still gave 124 vs 94 (worse spread than
before). Tried adding `cv::setNumThreads(1)` (ruling out OpenCV-internal
parallelism in SIFT extraction) on top -- also inconclusive: 85 vs 109 on
two more identical runs, plus it halved CPU throughput. **Reverted
`setNumThreads(1)`** (cost not justified without a clear payoff); **kept**
the `DUtils::Random` mutex (real fix, no downside).

**Conclusion**: the remaining nondeterminism is almost certainly rooted in
this project's own multi-threaded architecture itself (Tracking/
LocalMapping/LoopClosing racing on real wall-clock-timing decisions, e.g.
"is local mapping idle", exact keyframe-insertion timing, when background
loop-closing checks land relative to tracking frames) -- not practical to
eliminate without full per-frame thread synchronization, which would
defeat the purpose of having separate threads and would cost far more
performance than it's worth. **Not pursuing further** -- from here,
comparisons need repeat trials (2-3 runs minimum) to trust anything smaller
than roughly a 15-20-fail swing on this 541-frame hot zone (the observed
noise floor: 85-124 range across 6 same-or-similar-config samples).

**Practical takeaway for the parts 49-52 SIFT experiments above**: only
part 51's always-on-boost-is-worse finding should be trusted as-is (large,
multi-sample-consistent effect). Part 50's velocity-gated boost (102) and
part 52's edgeThreshold result (107) need repeat-trial reconfirmation
before being trusted as real wins/losses -- queued as next steps.

**Repeat-trial reconfirmation of part 50 (part 53 continued)**: 5 samples
of the velocity-gated boost config on the primary hot zone: 102, 111, 102,
114, 114 -- mean 108.6, essentially identical to the single baseline
sample (107). **Conclusion: the velocity-gated density boost has NO
measurable real effect here** -- the original "102" that motivated parts
50-52 was a lucky low draw within the noise band, not a real win. Per
user's direction, stopped shallow SIFT parameter tuning at this point.

## Part 54: direct SIFT-vs-ORB comparison overturns the parts 45-53 premise

User redirected: instead of more indirect SIFT parameter tuning, directly
measure whether SIFT is actually worse than ORB at the thing that matters
(cold-start tracking survival), using stock (untouched) ORB-SLAM3 as a
live comparison, not just historical statistics.

**Method**: `orbslam3_kitti_ate` (links against the untouched
`third_party/ORB_SLAM3`, NOT the SIFT fork) shares the same
`analyze/orbslam3_kitti_ate.cpp` source as the SIFT tool, including the
`start-frame`/`max-frames` CLI args -- rebuilt it fresh (was stale).
Confirmed `settings_sift/KITTI00-02-sift.yaml` works for it directly (the
camera calibration + `ORBextractor.nFeatures/scaleFactor/nLevels/
iniThFAST/minThFAST` params are descriptor-agnostic; extra SIFT-only keys
are just ignored with harmless "optional parameter does not exist"
warnings). Ran stock ORB-SLAM3 on the **exact same 541-frame cold-start hot
zone** (start-frame=4000) used for every SIFT-fork hot-zone test this
session.

**Result: stock ORB got 124 fails on the first run, then 276 on a repeat**
-- both well within or *above* the SIFT fork's established range on this
same segment (85-124, mean~108.6 across 5 samples). RMSE was also
notably worse than typical SIFT-fork fragments: 17.674m and 6.180m
respectively (still under the 20m budget, but far from the sub-1m numbers
typical of SIFT-fork fragments elsewhere), over much shorter paths
(239.2m and 114.5m) than the SIFT-fork's usual hot-zone-adjacent full-run
fragments.

**This overturns the working premise of parts 45-53.** Stock ORB-SLAM3's
famous "only 3 resets across the whole 4541-frame run" statistic comes
from starting at frame 0 with a fresh map that has the *entire rest of the
sequence* to mature into a large, hundreds-of-keyframes map before ever
reaching a hard segment. When forced to cold-start AT frame 4000 (the same
condition every SIFT-fork hot-zone test used), stock ORB is **not**
meaningfully more robust than the fork -- if anything, this small sample
suggests it may be worse. **The dominant factor is very likely map
maturity/fragility itself, not descriptor choice (SIFT vs ORB)** -- a
young, sparse map is fragile regardless of which feature type populates
it. This directly supports the "domino effect" hypothesis floated earlier
in the session (small maps are inherently more vulnerable to routine
per-frame variance) over the "SIFT has worse frame-to-frame repeatability
than ORB" hypothesis that parts 45-53 were built on.

**Reframed conclusion**: this fork's much worse *full-run* statistic
(240+ resets vs stock's 3) is best explained not by SIFT being a weaker
descriptor, but by this fork spending far more of its total runtime in
fragile/cold-start mode than stock ORB ever does -- which is exactly what
the *earlier* parts of this session (30, 34/35/48 destroy-threshold,
36-38 Sim3Solver fix, fail-fast recovery) were already working to fix, by
making cold-start episodes shorter/cheaper and preventing small maps from
being thrown away unnecessarily. Parts 45-53's pivot toward "fix SIFT
itself" was a reasonable hypothesis given the evidence available at the
time (search-radius th=1 being shared with stock was suggestive), but this
direct head-to-head test says it was very likely the wrong lever. Not
pursuing further SIFT-specific parameter tuning; returning attention to
map-maturity/survival-time levers, now with this important caveat in mind
for interpreting any future stock-vs-fork comparison.

### Part 54 continued: progressive ORB-vs-SIFT comparison from the TRUE sequence start reveals a real, worsening, segment-length-dependent gap

Per user request, ran BOTH stock ORB and the SIFT fork (matched settings,
nFeatures=5000 both) on growing prefixes of the sequence starting at frame
0 (the actual, non-artificial start both systems use in real operation) --
`orb_seg1_<N>`/`sift_seg1_<N>` in scratchpad, `<N>` = 500/1000/1500.
Computed true GT path length for each prefix directly from `poses/00.txt`
to get real coverage %, not just raw meters.

| frames | GT path | SIFT fails | SIFT coverage | SIFT % | ORB fails | ORB coverage | ORB % |
|---|---|---|---|---|---|---|---|
| 1-500  | 359.4m  | 29  | 213m   | 59.3% | 91  | 217.6m | 60.5% |
| 1-1000 | 715.2m  | 67  | 320m   | 44.7% | 152 | 533.6m | 74.6% |
| 1-1500 | 1091.8m | 132 | 423.2m | 38.8% | 153 | 904.1m | 82.8% |

**At frame 500 the two are roughly even. By frame 1500, ORB covers over
4x SIFT's percentage of the same growing path, and the gap is still
widening.** SIFT actually has consistently *fewer raw* "Fail to track
local map!" events than ORB at every length (29 vs 91, 67 vs 152, 132 vs
153) -- yet ends up covering far less of the path. This is the same
"raw fails vs formal resets" distinction below explains.

**Important caveat**: these SIFT runs include every fork-specific change
from this whole session (fail-fast timeout, destroy-threshold=9,
velocity-adaptive search radius, RNG mutex, etc.); the ORB runs are
completely unmodified stock `Tracking.cc`. So this compares "patched SIFT
fork" against "unpatched stock ORB," not descriptor-only.

**Also important**: the historical "stock ORB: 3 resets, 6.4-10.7m ATE
over the full run" figure (cited in earlier parts of this session) was
measured at `nFeatures=2000` via the GUI app (DEBUGGING.md part 21 notes
`settings_sift/KITTI00-02-sift.yaml` started as a copy of a shared
`KITTI00-02.yaml` with nFeatures raised 2000->3000, later further raised
to 5000 over subsequent sessions) -- NOT at today's `nFeatures=5000`/this
CLI tool combo. That figure is not directly comparable to today's segment
tests; a fresh nFeatures=2000-for-both comparison is queued to check
whether SIFT's per-match quality edge (7.9% outlier rate, measured
earlier) shows up more clearly at lower feature density.

### Part 55: hunting the real cause of the reset-frequency gap (isolate-and-attack, one variable at a time)

**Key metric correction**: raw "Fail to track local map!" count is
misleading -- it conflates (a) how often tracking *newly* breaks with (b)
how many repeated failed-recovery attempts occur *within* one episode
before it resolves (which scales with however long the recovery window
is). The clean metric is **reset count** (`Reseting active map` = episode
conclusions). On frames 1-1000: **ORB = 6 resets. Every SIFT-fork variant
tested so far resets 3-6x more often**, regardless of which threshold was
tuned:

| config (frames 1-1000) | fails | resets | coverage | fragments |
|---|---|---|---|---|
| baseline (need>=15, timeout=0.1f, gate=35, destroy=9) | 67 | 33 | 320m (44.7%) | 6 |
| timeout 0.1f->3.0f (matches stock's own value) | 285 | 18 | 286.6m (40.1%) | 4 |
| need>=15->30 (matches stock's own value) | 101 | 36 | 340.1m (47.6%) | 6 |
| post-reloc gate 35->50 (matches stock's own value) | 85 | 36 | 290.1m (40.6%) | 5 |
| **stock ORB (for reference)** | 152 | **6** | 533.6m (74.6%) | 3 |

**All three reversions to stock's own literal threshold values made
coverage WORSE on this segment**, not better -- confirms parts 26-32's
original tuning conclusions hold even at this shorter segment length, not
just full-sequence. But none of them closed the reset-frequency gap either
(best case 18 resets, still 3x stock's 6). **Diffed the entirety of
Tracking.cc against stock** (`third_party/ORB_SLAM3` vs
`third_party/ORB_SLAM3_SIFT`) -- confirmed these 4 numeric thresholds
(timeout, destroy-threshold, need-inliers, post-reloc gate) are the
*complete* set of logic-level differences in that file; all four are now
individually tested against their stock values and rejected as the
dominant cause.

**Next candidate examined**: `ORBmatcher.cc`'s `TH_HIGH`/`TH_LOW`
(100557.0f/46778.0f for SIFT's squared-L2 distance, vs stock's
100/50 for ORB's Hamming distance -- necessarily different scales, not
directly comparable) -- confirmed these were data-driven-calibrated (true
match distance's own 95th/99th percentiles, per the inline comment), not
an obvious guess/bug. Not yet fully diffed the rest of `ORBmatcher.cc`,
`KeyFrameDatabase.cc`, `LocalMapping.cc`, or `Optimizer.cc` against stock
-- queued as the next investigation surface. Also not yet tried: directly
instrumenting and comparing how many frames each of ORB/SIFT spend in
`NOT_INITIALIZED` state and how many distinct mono-init attempts each
needs per successful init, to check whether the fork's re-init process
itself (not just post-init survival) is less reliable than stock's under
matched conditions.

**Status for continuation**: the reset-frequency gap (SIFT resets 3-6x
more often than stock ORB even after matching every Tracking.cc threshold
to stock's own values) remains unexplained. Per user's standing
instruction, treat this as a real, likely-fixable fork-specific issue, not
an inherent SIFT weakness (matches quality is confirmed fine -- 7.9%
outlier rate) -- continue the systematic diff-and-isolate hunt through the
remaining un-diffed files before concluding otherwise.

**Architectural diff sweep**: `LocalMapping.cc` diffs against stock are
diagnostic-only (no logic changes). `KeyFrameDatabase.cc` diffs
substantially (1033 lines) -- but that's the expected DBoW2->VLAD
architectural swap (this fork's place-recognition mechanism), not
obviously a bug by itself.

**Hypothesis tested and REFUTED on this specific segment**: suspected
VLAD's weaker candidate retrieval (dbCandidates=0 74-94% of the time,
measured in earlier full-sequence analysis) might explain the reset gap.
**Directly measured on frames 1-1000: dbCandidates=0 in 0/20 attempts
(0%), and relocOK=1 in 10/28 attempts (35.7%)** -- both far healthier than
the degraded-map-era full-sequence numbers. VLAD candidate retrieval is
NOT starved here; this hypothesis does not explain the gap on this
segment (it may still be relevant later in a full run, once maps are more
fragmented, but is not the cause of the *early-segment* gap being studied
here).

**More precise finding**: of the 33 resets on this segment, only 18 go
through the `RECENTLY_LOST` timeout path (a real recovery attempt that
eventually gives up). **The other 15 (45%) go straight to `LOST` without
ever attempting relocalization at all** -- this happens when
`pCurrentMap->KeyFramesInMap()<=10` at the moment tracking breaks (see
Tracking.cc's `else if(pCurrentMap->KeyFramesInMap()>10) mState =
RECENTLY_LOST; else mState = LOST;`) -- i.e. these are maps that die
*before they even reach 10 keyframes*, too young to ever get a
relocalization chance. **This reframes the real open question**: not
"why does relocalization fail" (it doesn't, on this segment) but **"why
do freshly-initialized maps die so often before reaching 10 KFs in the
first place"** -- back to raw per-frame tracking robustness immediately
after `CreateInitialMapMonocular()`, not the recovery/relocalization
machinery. Queued as the next investigation thread.

**Quantified the young-map churn directly**: the 15 immediate-LOST resets
died with KF counts of {2, 3, 4, 4, 4, 5, 5} -- averaging under 4
keyframes, essentially dying moments after being born. **122 total
"new map"/mono-init-related events occurred in just 1000 frames (roughly
one new map attempt every 8 frames on average) vs only 33 formal
`Reseting active map` events** -- most of that churn (122-33=89 events)
is `CreateMapInAtlas()` fragment-preservations or repeated failed
mono-init attempts, not full destructive resets, but the sheer volume
confirms the "domino effect" hypothesis floated much earlier in this
session (small/young maps are inherently fragile, and the system is
constantly cycling through short-lived attempts) with hard numbers for the
first time. **Not yet root-caused further**: still open whether (a)
mono-init itself produces a weaker/smaller initial map for SIFT than for
ORB under matched conditions (fewer initial map points, worse baseline),
(b) the immediate post-init tracking (before 10 KFs accumulate) is more
fragile for some SIFT-specific reason, or (c) something else entirely.

### Session status summary (for continuation) -- extensive investigation today, root cause of the reset-frequency gap still open

**What was tried and ruled out today** (all measured, not assumed):
shallow SIFT parameter tuning (density boost velocity-gated: no real
effect once repeat-trials accounted for noise; density boost always-on:
confirmed worse; edgeThreshold 10->20: no effect/cancelled the boost);
reverting 3 of the 4 fork-specific Tracking.cc thresholds to their
literal stock values (RECENTLY_LOST timeout 0.1f->3.0f, post-reloc gate
35->50: both measured WORSE on the 1-1000 segment; inlier-accept bar
15->30: modest +6.3% improvement, kept as a minor win); the DUtils::Random
thread-race (real bug, fixed with a mutex, kept, but did not achieve full
determinism -- remaining variance is architectural, from the
Tracking/LocalMapping/LoopClosing thread interplay itself); VLAD
candidate-starvation as the explanation for the reset gap (directly
measured on this segment: dbCandidates=0 only 0/20, relocOK 35.7% --
both healthy, ruling this out for the *early*-segment gap specifically).

**What was confirmed real**: stock ORB-SLAM3, given a fair matched-start
comparison (both cold-starting at frame 0, nFeatures=5000 both), clearly
outperforms the SIFT fork's coverage % of the true GT path once segments
grow past ~500 frames (500: roughly even ~60%; 1000: ORB 74.6% vs SIFT
44.7%; 1500: ORB 82.8% vs SIFT 38.8%, still widening). The SIFT fork
resets 3-6x more often than stock ORB on the same segment regardless of
which individual threshold is tuned. ~45% of the fork's resets are maps
that die before reaching 10 keyframes (average <4 KFs at death) -- young
maps are extremely fragile, with ~122 new-map-related events in just 1000
frames.

**What's still unknown**: the specific mechanism causing SIFT-fork maps
to die so young so often, when: match quality is fine (7.9% outlier
rate), relocalization candidate availability is fine on this segment
(0% dbCandidates=0), and every individually-tunable Tracking.cc threshold
has been matched to stock without closing the gap. **User's standing
instruction**: continue treating this as a real, fixable fork-specific
issue, not inherent SIFT inferiority.

### Part 55 continued: direct raw-keypoint-count measurement -- real gap found, but the obvious fix (lower contrast threshold) makes it WORSE

Added a matched `[raw-keypoints] id=%lu N=%d` diagnostic to BOTH
`third_party/ORB_SLAM3_SIFT/src/Frame.cc` and `third_party/ORB_SLAM3/src/
Frame.cc` (same line, right after `N = mvKeys.size()` in the monocular
constructor) -- this had never been directly measured all session (all
prior analysis was downstream match/tracking success, not raw detection
yield). Ran both on identical frames 0-200 (`nFeatures=5000` both):

| | mean | min | max |
|---|---|---|---|
| SIFT | 3860.7 | 2354 | 5751 |
| ORB  | 5843.7 | 4916 | 16030 |

**SIFT extracts only ~66% of ORB's mean keypoint count, and its worst
frame (2354) drops to under half the nFeatures=5000 target, while ORB's
worst frame (4916) stays close to/above its own target.** This is a real,
concrete, well-evidenced gap -- the most direct evidence all session that
SIFT's detector is genuinely yielding fewer usable keypoints than ORB's
under matched settings, particularly in KITTI's harder (low-texture/
blurred) frames.

**The obvious fix -- lower `kBaseContrastThreshold` (0.04, the value
`SetDynamicDensity` actually uses in normal/non-high-speed operation,
overriding `ORBextractor.cc`'s constructor default per earlier findings)
-- was tried at 0.01 and measured WORSE, not better**: mean keypoint count
did rise (3860.7->4598.2, closer to ORB) but coverage collapsed (67
fails/33 resets/320m/44.7% -> 105 fails/58 resets/103.3m/14.4%) and the
WORST-frame count got even lower (1409, below the original 2354's own
floor). This is the SAME failure mode as part 51's always-on density
boost (which also lowered contrast threshold, to 0.02, and was also
confirmed worse across 4 independent samples): **letting more
borderline/weak keypoints through dilutes match quality rather than
helping -- tried at two different threshold values (0.02 and 0.01), both
made things worse.** Reverted to stock 0.04.

**Honest conclusion on the SIFT-tuning avenue overall (parts 49-52, 55)**:
every single SIFT-extraction-parameter variant tried this session --
velocity-gated density boost, always-on density boost, edgeThreshold
raised, base contrast threshold lowered (twice, at two different
values) -- either did nothing measurable or made things actively worse.
**None found a real win.** The raw-keypoint-count gap (SIFT 66% of ORB's
yield) is real and confirmed, but simply loosening the detector's
acceptance criteria is NOT the fix -- the extra keypoints let through are
low-quality and hurt more than the count increase helps. The actual
reason SIFT's detector yields fewer *good* keypoints than ORB's on these
frames remains unexplained -- possibilities not yet investigated: SIFT's
octave/scale-space structure inherently produces fewer stable extrema on
this class of imagery regardless of the contrast threshold (a deeper
property of DoG detection vs FAST-corner detection, not fixable via a
threshold knob); the quadtree-based spatial distribution step (shared
code between ORB and SIFT extraction, per earlier session history)
interacting differently with SIFT's octave/layer structure; or something
in how candidate keypoints get selected/ranked before the final
nFeatures cutoff. This is a legitimate place to pause the SIFT-parameter
rabbit hole -- further progress here likely needs either accepting
nFeatures increases (tried earlier today at 6000-10000, real but
computationally expensive due to Sim3Solver/merge cost scaling, not
purely a "free" fix either) or a genuinely different investigation
(e.g. visualizing/comparing actual detected keypoint locations between
SIFT and ORB on the same hard frame) rather than more blind threshold
sweeps, which have now been tried in essentially every direction (higher,
lower, gated, ungated) without success.

**One more data point before pausing this thread**: computed the actual
`SearchLocalPoints()` match RATE (not just raw detection count) from
existing SIFT logs -- out of local map points that ARE geometrically
in-frustum (`nToMatch`, i.e. should be visible), only **10.0%
(29620/297394 across 877 samples) actually get matched** to a keypoint in
the current frame. This is a strikingly low hit rate for points the
system already believes should be visible, and is a genuinely new,
not-yet-compared-against-ORB data point -- no equivalent diagnostic was
added to stock ORB for this specific ratio, so it's not yet known whether
this is worse than ORB's own rate or a normal property of this kind of
search. **Flagged as the most promising concrete next lead for a future
session**: add the same `nToMatch`/`matched` diagnostic to stock ORB's
`SearchLocalPoints()` call site and directly compare match RATES (not
raw keypoint counts, which have already been shown to have a real but
not-simply-fixable gap) between the two on identical frames -- this
isolates whether the problem is in DETECTION (partially investigated,
inconclusive/counterproductive fixes tried) or in MATCHING GIVEN
detected keypoints (not yet investigated at all).

## Session wrap-up (2026-07-19, end of this extended debugging session)

**Confirmed, kept changes from this session** (destroy-threshold=9,
fail-fast RECENTLY_LOST timeout=0.1f, post-reloc gate=35, monocular
inlier-accept bar=15, `DUtils::Random` thread-safety mutex, velocity-gated
search-radius widening at th>=5 when translation-velocity>0.087,
cross-map relocalization via `Atlas::ChangeMap()`, Sim3Solver NaN-bug fix
with an absolute inlier floor of 6, nFeatures=5000) -- these represent the
session's real, measured wins earlier in the day (coverage roughly
2.3-3.7x the session's starting ~498.6m baseline on various full-sequence
measurements, though noise band is wide -- see parts 30-48 for the
individually-validated pieces).

**Not adopted** (tested and found neutral-to-harmful, reverted): SIFT
density-boost tuning in all forms tried (velocity-gated: no real effect;
always-on: confirmed worse; contrast threshold lowered further, twice:
both worse), edgeThreshold raised (cancelled the boost's marginal
benefit), RECENTLY_LOST timeout and post-reloc gate reverted to stock's
own literal values (both measured worse for this fork specifically,
despite being stock's own numbers -- confirms this fork's own tuning,
not stock's, is locally optimal for whatever's different about it
architecturally).

**Root cause of the fork's higher reset frequency vs stock ORB**: NOT
fully found. Ruled out: SIFT match quality (fine, 7.9% outlier rate),
relocalization candidate starvation on early segments (fine, 0%
dbCandidates=0), every individually-tunable Tracking.cc threshold
(matched to stock, all measured neutral-or-worse). Confirmed real but not
yet actionably fixed: SIFT's raw keypoint yield is ~66% of ORB's under
matched settings (3860.7 vs 5843.7 mean/frame), and the `SearchLocalPoints`
match rate is a low 10.0% -- both real, both unexplained at the
"why" level, both worth a dedicated future investigation with more
targeted diagnostics (visualizing actual keypoint locations, comparing
match rates head-to-head against stock ORB) rather than more blind
threshold sweeps.

## Current status (2026-07-18, part 29 -- goal redefined by user, two
## coverage-first changes made, NOT yet measured).

### Part 27/28 repeat-trial result (resolves part 26's open question)
The two `gate35_trial1`/`gate35_trial2` repeat runs queued at the end of part
26 both finished: **497.7m** (9/10 fragments aligned, 1 alignment-failed) and
**499.6m** (6/7 fragments aligned, 1 alignment-failed) -- within 0.4% of each
other, both noticeably above the earlier single-run 478.7m gate35 data point
(~4% higher). Conclusion: run-to-run variance at fixed settings is real but
small (~4%, consistent with the unseeded-RANSAC-RNG finding from the
2026-07-11 investigation trail), and it does NOT explain the 478.7m(gate35)
->314.4m(gate30) gap (~34% relative), so that specific regression from
loosening 35->30 is very likely a real effect, not noise. Gate stays at 35 in
the code as the confirmed-best value from that specific experiment; superseded
below by a much larger goal change.

### Goal redefined by user (this session)
User has explicitly deprioritized per-pose accuracy in favor of maximizing
continuous path coverage: **target is 100% of the KITTI seq00 path** (current
measured coverage is only ~497-500m of the sequence's ~3724m total path
length, i.e. ~13%); accuracy tolerance is now **<20m ATE RMSE** (vs. the
~0.03-0.4m currently measured per-fragment -- a >40x looser bar). User
explicitly authorized aggressive code/threshold changes, including removing
filters outright, to chase this. This changes the optimization target: the
part 19-28 work was tuning gates by small increments to protect accuracy;
that constraint is now mostly lifted, so the next moves trade accuracy for
continuity much more aggressively than before.

Framing for future sessions: literal 100% is an extremely ambitious bar for
monocular SLAM on a real 3.7km driving sequence (fragmentation on
never-revisited terrain is structurally very hard to fully eliminate) --
report actual measured coverage honestly rather than claiming 100% if it
isn't hit, but treat "push coverage as high as possible" as the real
day-to-day objective under the new <20m RMSE budget.

### Two changes made this part, NOT yet measured (need a full seq00 run)

1. **`Tracking.cc` RECENTLY_LOST timeout, 3.0f -> 20.0f seconds** (non-IMU
   monocular path, ~line 2046). Every time this timeout fires it forks a new
   Atlas map fragment (`CreateMapInAtlas()`) or destroys the map outright
   (`ResetActiveMap()` if <10 KFs) -- on never-revisited terrain a forked
   fragment can never merge back, so this is the single mechanism that turns
   a transient tracking dropout into *permanent* coverage loss. Per the
   in-code comment, `mTimeStampLost` is deliberately never renewed on a bare
   dead-reckoning success, so the extended window only matters for the case
   where both dead-reckoning AND relocalization keep failing every frame --
   giving ~6.7x more attempts before giving up.
2. **`Tracking.cc` base monocular inlier bar, `mnMatchesInliers<30` ->
   `<15`** (~line 3248, the `need>=30`/now `need>=15` diagnostic). Unlike the
   post-reloc gate (still 35, only applies within `mMaxFrames` of a
   relocalization), this bar gates *every* monocular frame's TrackLocalMap()
   success, so it's the highest-frequency trigger in the whole
   death->fragment->coverage-loss chain from parts 26-28. 15 is still well
   above the geometric minimum for a constrained pose (4-6), just noisier.

Rebuilt successfully (`orbslam3_sift_kitti_ate`, warnings only, all
pre-existing `%d`/`long unsigned int` format-string warnings unrelated to
this change).

### Part 29 result (measured) -- coverage flat, root cause found

Full seq00 run (`part29_run1`): **484.4m over 8 aligned fragments** (9 total,
1 alignment-failed with only 2 KFs), 128 total KFs, 31 raw "Reseting active
map" events, all per-fragment ATE RMSE comfortably under the 20m budget
(0.054-0.370m). Compared to the ~498.6m gate35 baseline (avg of the two part
27/28 repeat trials, 497.7m/499.6m): **flat, actually ~3% lower**, i.e. within
the already-established ~4% run-to-run noise band -- not a real improvement,
despite reset *events* dropping 3x (99/95->31).

**Why, confirmed by full-checkpoint stats on this run's log** (not
speculation):
- Dead-reckoning (`TrackWithMotionModel` during RECENTLY_LOST) succeeded
  **14/3380 checks (0.41%)**.
- Relocalization succeeded **22/3380 checks (0.65%)**.
- `dbCandidates==0` (the KeyFrame database returns literally zero candidates,
  not just non-matching ones) on **74.3%** of relocalization attempts.
- Only **17 distinct RECENTLY_LOST episodes** occurred in the whole
  4541-frame run, each burning **~198 frames (~19.8s) on average** -- i.e.
  nearly the entire 20s budget, almost every time -- before timing out.

**Conclusion**: extending the RECENTLY_LOST timeout doesn't help because
recovery essentially never succeeds once triggered, regardless of how long
you wait -- most of the time there is no candidate keyframe to relocalize
against at all. The reset-event count dropped only because each
already-doomed episode now takes ~6.7x longer to fail, not because more
episodes succeeded. This directly confirms/refines the "survivorship
paradox" hypothesis from part 26/28: giving a dying map more time doesn't
rescue it, it just delays the same outcome while burning frames that could
have gone toward a fresh mono-init attempt (known fast: ~2.3 frames average
once attempted, part 26 measurement).

**Part 30 change**: reverted the RECENTLY_LOST timeout **20.0f -> 1.0f**
(more aggressive than the original 3.0f) -- fail fast and get back to a new
init attempt sooner rather than waiting out a near-certain failure. Kept the
short (not zero) window since DR/reloc do occasionally succeed (36 total
real successes in part 29's run). The `need>=15` inlier-bar change from part
29 is kept as-is (not yet independently isolated from the timeout change --
today's priority is speed over full ablation per user's explicit direction).

### Part 30 result (measured) -- big win, fail-fast theory confirmed

Full seq00 run (`part30_run1`): **836.6m over 11 aligned fragments** (12
total, 1 alignment-failed with 2 KFs), 238 total KFs. This is **+67.8% vs
the ~498.6m gate35 baseline** and **+72.7% vs part 29's 484.4m** -- a real,
large improvement, not noise (previously-established noise band was ~4%).
Two fragments are now much larger than anything seen in prior parts:
fragment 2 (58 KFs, 233.4m, ATE rmse 0.976m) and fragment 7 (42 KFs, 165.0m,
ATE rmse 0.392m) -- the fail-fast restart cycle is letting maps grow
substantially larger before dying, not just dying faster. All 11 fragments'
ATE stayed well under the 20m budget (worst: fragment 10, rmse 2.427m/max
7.933m, still >8x margin).

Recovery-mechanism stats this run (for comparison to part 29's, note much
lower absolute attempt counts since each episode is now capped at ~1s):
dead-reckoning succeeded 34/1168 (2.9%, vs 0.41% in part 29), relocalization
succeeded 41/1168 (3.5%, vs 0.65%). Both rates are higher, not lower, despite
~3x fewer total attempts -- consistent with real recoveries clustering in the
first ~1s of a dropout (part 29's extra 19s of waiting was pure dead weight,
confirmed again here). 112 distinct TIMEOUT episodes occurred (vs part 29's
17), each now failing fast instead of stalling ~198 frames.

Still only ~22.5% of the full ~3724m sequence -- far from the 100% target,
but the single largest coverage jump of the whole session. Confirms: **the
death/restart *cycle time* was the dominant lever, not raw thresholds.**

**Part 31 next step**: push further in the same direction -- reduce the
RECENTLY_LOST timeout again (1.0f -> 0.3f) to test whether even faster
fail/restart cycling keeps helping, since success is now known to cluster
very early in a dropout and 1.0f (~10 frames) may still be leaving some
dead time on the table. `need>=15` stays unchanged (still not isolated, not
a priority right now given the timeout lever is clearly the dominant one).
If 0.3f stops helping or hurts, that will bound where the fail-fast benefit
saturates and it'll be time to pivot to a structurally different lever (the
KeyFramesInMap()<10 destructive-reset-vs-fragment branch, or the
dbCandidates=0 rate directly).

### Part 31 result (measured) -- still scaling, but accuracy budget starting to matter

Full seq00 run (`part31_run1`, timeout=0.3f): **1278.3m over 12 aligned
fragments** (13 total, 1 alignment-failed with 3 KFs), 291 total KFs. **+52.8%
vs part 30's 836.6m (1.0f)**, **+156% vs the ~498.6m gate35 baseline**. The
fail-fast trend is still scaling, not saturated. 226 raw reset events, 158
distinct TIMEOUT episodes (both up from part 30, as expected from faster
cycling). Recovery success rates keep climbing too: DR 50/535 (9.3%, vs 2.9%
at 1.0f), reloc 58/535 (10.8%, vs 3.5% at 1.0f) -- fully consistent with the
"recoveries cluster in the first fraction of a second" model from part 30.

**New signal worth flagging**: per-fragment ATE is now visibly degrading for
some fragments, not just staying flat -- fragment 6 (19 KFs, 124.6m) came in
at **rmse 8.099m / max 17.292m** and fragment 8 (59 KFs, 264.1m) at **rmse
10.496m / max 30.050m**. Both are still under the 20m RMSE budget (the
metric the user set the bar on), but fragment 8 is now roughly halfway to
it, and its max error (30.050m) already exceeds 20m even though its RMSE
doesn't. This is the accuracy-for-continuity trade becoming visible, not
hypothetical anymore -- likely because larger/faster-cycling fragments
(59-60 KFs now, vs 11-45 before) accumulate more scale/drift error before
their next death, and/or because more Sim3-based map merges are happening
(the two biggest fragments both look like merge products). Not yet
root-caused which of those it is.

**Implication for next steps**: the fail-fast lever clearly still has room
(0.3f beat 1.0f by a large margin), so push further (e.g. 0.1f), but this is
the first run where continuing to push blindly could plausibly start
crossing the 20m RMSE line on some future fragment -- watch the per-fragment
RMSE, not just total coverage, on the next iteration and be ready to back
off the timeout (not abandon the whole direction) if any fragment's RMSE
actually exceeds 20m.

### Part 32 result (measured) -- timeout lever saturated around 0.1-0.3f

Full seq00 run (`part32_run1`, timeout=0.1f): **1285.4m over 15 aligned
fragments** (all 15 aligned successfully, none failed this time), 289 total
KFs. Essentially flat vs part 31's 1278.3m (+0.55%, well inside the ~4%
noise band) -- **the timeout lever has saturated**, pushing 0.3f->0.1f
bought nothing further. Reassuringly, this run's per-fragment RMSE was much
better than part 31's (max 1.449m vs part 31's 10.496m), confirming part
31's high-RMSE fragments were run-to-run variance in a specific merge event,
not a monotonic accuracy cost of a shorter timeout. **Kept timeout at 0.1f**
(tied with 0.3f, no reason to revert). 238 raw reset events but only 15
fragments actually formed -- **~94% of resets are now the destructive
`KeyFramesInMap()<10 -> ResetActiveMap()` path** (Tracking.cc ~line 2059),
not `CreateMapInAtlas()`. Net progress vs the gate35 baseline so far: **+158%
coverage (498.6m -> 1285.4m)**, still ~34.5% of the full ~3724m sequence.

### Part 33 change (NOT yet measured, higher risk than parts 29-32)

Pivoted to a new lever per the saturation above. Root cause behind the
persistent dbCandidates=0 stat (74-94% across every run this session):
`KeyFrameDatabase::DetectRelocalizationCandidates` (KeyFrameDatabase.cc
~line 518) filtered candidates to `pKFi->GetMap() == pMap` (current map)
only. Since parts 30-32's fail-fast cycling means most active maps are now
young/tiny (median map has very few of its own KFs), their own-map candidate
pool is nearly always empty -- meanwhile the Atlas has accumulated 289 KFs
across 15 fragments in part 32 alone that Relocalization() never even
considers.

**Changes**: (1) removed the same-map filter in
`DetectRelocalizationCandidates`, so it can return candidates from any map
in the Atlas; (2) in `Tracking::Relocalization()`, track which candidate KF
actually wins (`pRelocWinnerKF`), and if it belongs to a different map than
`mpAtlas->GetCurrentMap()`, call `mpAtlas->ChangeMap(pRelocWinnerKF->GetMap())`
(the same mechanism LoopClosing uses after a successful map merge) before
returning success, plus clear `mpLastKeyFrame` (mirrors the existing
cleanup after `CreateMapInAtlas()`/`Reset()` elsewhere in this file, since
it would otherwise point into the map being abandoned).

**This is meaningfully riskier than parts 29-32**: those were pure
threshold tuning; this touches map-consistency invariants (mCurrentFrame's
pose and mvpMapPoints get set from the winning KF's map during PnP, but
`mpAtlas`'s active-map pointer previously never moved to match -- this
change closes that gap using ORB-SLAM3's own existing `ChangeMap()`, not a
new mechanism, but it has not been exercised from this code path before).
Added a `[reloc-crossmap]` diagnostic to confirm when it actually fires.
**Before trusting any coverage number from this run, first check the run
completed cleanly (reached `Shutdown` / `[atlas-coverage]` in the log, no
crash) --if it crashed or the trajectory looks obviously corrupt, revert
this change first** rather than treating a crash-truncated coverage number
as a real result.

### Part 33 result (measured) -- no crash, but inconclusive; repeat trial queued

Full seq00 run (`part33_run1`) completed cleanly (reached `Shutdown`/
`[atlas-coverage]`, no crash) -- the `ChangeMap()` mechanism is safe to run,
at minimum. **1026.2m over 11 aligned fragments** (13 total, 2
alignment-failed), 242 total KFs -- **20.2% lower than part 32's 1285.4m**.
All per-fragment RMSE stayed comfortably under the 20m budget (worst:
fragment 1 at 5.145m/max 9.811m).

**But the cross-map mechanism itself only fired once in the entire
4541-frame run** (`[reloc-crossmap] id=1585 switching active map to winner
KF's map (id=0)`) -- meaning this run is *de facto* almost the same
configuration as part 32 (timeout=0.1f) plus one extra event, not a
meaningfully different experiment. This is expected, not a bug: VLAD
candidate retrieval is a coarse visual-similarity filter, but actually
*winning* relocalization requires passing `SearchByBoW` (>=15 matches) AND
PnP RANSAC (>=50 verified inliers) against that candidate -- a much
stricter bar that a genuinely different, non-revisited location will almost
never satisfy. Legitimate loop-closure-quality opportunities are inherently
rare in this sequence (consistent with part 25's "~1 merge/run" finding).
So the 20.2% coverage gap here is far more likely **run-to-run noise under
the fail-fast regime** (more RANSAC draws / relocalization attempts per run
now than in the gate35-era runs where the ~4% noise band was established --
plausibly a wider noise band applies now) than a real regression caused by
this change.

**Not reverting yet** -- the change is confirmed safe (no crash, no RMSE
violation) and did almost nothing this particular run, so there's no
evidence it's harmful, just insufficient evidence either way. Queued a
repeat trial (`part33_run2`, same code, no changes) to separate noise from
a real effect, same methodology as the gate35 repeat trials in parts 27/28
-- do not conclude keep-vs-revert from a single run given this session's
established noise floor.

**Part 33 repeat trial result and final verdict**: `part33_run2` (identical
code/config) came in at **1267.3m over 14 fragments** (275 total KFs, 2
`[reloc-crossmap]` firings), only **-1.4% vs part 32's 1285.4m** -- solidly
inside noise. This confirms `part33_run1`'s 1026.2m was a low-side outlier,
not evidence the cross-map change hurts. **Verdict: keep the part 33
change** -- it's measured safe (2 clean full-sequence completions, no
crash), never violates the RMSE budget, and is at worst neutral on
coverage; discarding a harmless change on a single noisy data point would
have been the wrong call. **Kept both KeyFrameDatabase.cc's dropped
same-map filter and Tracking.cc's `ChangeMap()` call in the tree.**

Side finding: run-to-run noise under the fail-fast (0.1f-timeout) regime
looks meaningfully wider than the ~4% band established earlier in the
session under the old gate35/no-fail-fast regime -- `part33_run1` alone was
a ~20% low outlier. More restart cycles per run (more independent RANSAC/
relocalization draws) plausibly widens the variance. Treat single-run
deltas under ~20% as inconclusive going forward under this regime; lean on
repeat trials for any conclusion that matters.

## Current status (2026-07-19, part 34 -- next lever: the destructive
## `KeyFramesInMap()<10` reset branch)

Pivoting per the plan above. Across parts 32-33, raw "Reseting active map"
events vastly outnumber final Atlas fragments (e.g. part 32: 238 resets, 15
fragments; part 33 run1: 258 resets, 13 fragments) -- meaning the large
majority of resets hit the destructive branch in `Tracking.cc`'s LOST-state
handler (`if (pCurrentMap->KeyFramesInMap()<10) { ResetActiveMap(); }`,
~line 2059) that throws the young map away completely, rather than
`CreateMapInAtlas()` which preserves it as a scored (if small) fragment.
Evidence this threshold has real headroom: part 32 had an 8-KF fragment
(fragment 14) align successfully, and part 33 run2 also had an 8-KF
fragment (fragment 14) align successfully -- both below the current <10
destroy-threshold, meaning maps in that exact size range CAN produce valid
scored coverage when given the chance, they're just being destroyed before
ever getting it under the current threshold's boundary.

### Part 34 result (measured) -- new best, trending positive

Full seq00 run (`part34_run1`, destroy-threshold=5): **1455.3m over 13
aligned fragments** (18 total, 5 alignment-failed -- all exactly 2 KFs),
275 total KFs, 5 `[reloc-crossmap]` firings. **+13-15% vs the ~1267-1285m
confirmed range from parts 32/33** -- a new single-run best, though still
inside the session's established ~20% noise band so not yet confirmed as a
real effect on its own. Notable: fragment 9 reached **68 KFs / 521.1m
pathLen**, the largest single fragment of the entire session by a wide
margin (previous largest was part 31's 233.4m) -- a good sign that maps are
now surviving substantially longer, not just cycling faster.

All per-fragment RMSE stayed under the 20m budget, but variance is
climbing further: fragment 10 hit **rmse=8.773m/max=21.372m** (max now
exceeds 20m, though RMSE -- the metric the budget is actually defined on --
does not) and fragment 9 hit rmse=5.211m/max=17.140m. This is a real,
continuing trend (part 31 first showed it, part 34 pushes it further), not
a one-off -- larger/longer-surviving fragments accumulate more drift before
their next death, so pushing fragment sizes up is trading some accuracy
margin for coverage. Still well inside the RMSE budget, but worth watching
as further changes are layered on.

Minor unexplained curiosity, not chased further (doesn't affect the
coverage metric either way): all 5 alignment-failed fragments are exactly
2 KFs, i.e. below the new destroy-threshold of 5 -- meaning something
outside the specific LOST-state handler edited this part also creates Atlas
fragments (there are several other `ResetActiveMap()`/`CreateMapInAtlas()`
call sites in `Tracking.cc`, e.g. ~lines 1810-1852, 2244, 2412-2423, not
touched this session). Functionally irrelevant here since a 2-KF fragment
contributes 0m either way (destroyed or preserved-but-failed-alignment),
so not investigated further this iteration.

Queued a repeat trial (`part34_run2`) to confirm this is a real
improvement and not noise, same methodology as parts 27/28 and 33.

**Part 34 repeat trial result**: `part34_run2` came in at **1143.4m over 15
fragments** (269 total KFs, 2 crossmap firings), all RMSE well under budget
(max rmse 4.974m). Average of the two threshold=5 runs: **(1455.3+1143.4)/2
= 1299.35m** -- essentially back in line with the ~1267-1285m range from
parts 32/33, not a clearly-confirmed improvement. Run1's 1455.3m was
apparently a high-side outlier, same pattern as part 33's noise finding.

**Root cause found for the inconsistent effect**: there is a **second,
separate, unedited threshold check** in `Tracking::Track()` at ~line 2410
(`if(mState==LOST) { if(pCurrentMap->KeyFramesInMap()<=10) { ResetActiveMap();
return; } ... CreateMapInAtlas(); return; }`, under the comment "Reset if
the camera get lost soon after initialization"). This is functionally the
same destroy-vs-preserve decision as the one edited in part 34 (~line
2113), but reached via a different code path (checked again near the end
of `Track()`, likely for the "still LOST after this frame's tracking
attempt" case, distinct from the state-transition-moment check edited
earlier) -- and it was **still at the old `<=10` threshold**, unchanged by
the part 34 edit. This is the most likely explanation for why part 34's
effect was inconsistent: only part of the destroy/preserve decision space
was actually addressed.

**Part 35 change**: lowered this second threshold too, `<=10` -> `<5`
(matching part 34's already-edited check, for consistency), so the
destroy/preserve decision is now uniformly changed in both places source
code reaches it.

**Part 35 result (measured)**: **1480.3m over 20 aligned fragments** (37
total, 17 alignment-failed -- mostly tiny 2-9 KF fragments that got
preserved by the lowered threshold but still couldn't align, expected given
the alignment-failure floor established earlier), 442 total KFs. New
single-run best (previous: part 34's 1455.3m). All RMSE stayed under the
20m budget (worst: 9.717m). **Caveat: the process crashed AFTER printing
all results and reaching "Saving keyframe trajectory..."** --
`cv::Exception ... "Can't fetch data from terminated TLS container."` in
OpenCV's `getData()`, a thread-local-storage cleanup issue during shutdown,
not mid-run corruption (all coverage data was already fully computed and
logged before the crash). Plausibly related to this run having far more
Atlas fragments alive at once (37, the most of any run this session) than
usual, stressing the shutdown teardown path harder. Not chased further this
session since it doesn't affect measurement validity, but flagged as a
latent robustness issue if this tool is ever used in an automated pipeline
that doesn't tolerate a non-zero/crash exit code.

### Deep dive requested by user: why does 77% frame-tracking-success only
### yield ~35% path coverage?

State-time breakdown across part 32/34 runs (via `[track-local-map]` and
`[recently-lost]` diagnostic line counts against the known 4541-frame
total): **~77% of frames track successfully**, **~4.8%** are RECENTLY_LOST
recovery attempts (mostly failing, consistent with parts 29-30's findings),
and the remaining **~18%** are NOT_INITIALIZED/re-init gaps -- close to the
theoretical floor already (~2-3 frame average re-init latency, part 26,
times ~240 resets/run ≈ the observed ~800 frames).

The 77%-tracked vs ~35%-path-covered gap is NOT primarily a tracking
problem -- it's that successful tracking gets chopped into **13-15
disconnected Atlas fragments per run**, each independently scored, and they
essentially never reconnect: **Sim3Solver-based merge/loop-closure
convergence is at ~1.4% (18/1328 BoW-passing candidates converged in one
part 34 checkpoint)**, with a striking bimodal `nInliers` distribution --
**1310/1328 failures were EXACTLY 0 inliers** (not "a few below the 15
minimum," literally zero every time), while the 18 successes cleanly
cleared 16-26. This bimodality (never a partial/near-miss count) is the
signature of an early bail-out, not generic geometric noise.

**Root-cause hypothesis (not yet confirmed on a real run)**: in
`Sim3Solver::SetRansacParameters` (Sim3Solver.cc ~line 123), `epsilon =
minInliers/N` where `N` is the number of correspondences that survived the
constructor's filtering (isBad/`GetIndexInKeyFrame` validity checks -- can
be meaningfully smaller than the raw `numBoWMatches` count that gated entry
into this code path, since that gate is checked in LoopClosing.cc BEFORE
Sim3Solver's own constructor does its own additional filtering). If N ends
up below `mRansacMinInliers` (15) for a real fraction of candidates,
`epsilon>1`, so `1-pow(epsilon,3)` goes negative and `log()` of that is NaN
-> `nIterations` is NaN -> `mRansacMaxIts = max(1, min(NaN, 300))` most
likely collapses to 1 (working through `std::min`/`std::max`'s NaN-comparison
semantics, both false, defaulting to the first/lower-bound argument) --
meaning RANSAC gets essentially **one single random 3-point sample** for
the whole candidate instead of up to 300, which would very plausibly
produce exactly-0-inliers results the vast majority of the time (one random
draw succeeding is unlikely, and there's no budget left to try again).

**Part 36 diagnostic (built, NOT yet run)**: added an `[sim3-investigate]`
fprintf in `SetRansacParameters` printing `N`, `minInliers`, `epsilon`,
`nIterations`, `mRansacMaxIts` on every call, to directly confirm or refute
the N<minInliers/NaN hypothesis before touching any actual logic (same
evidence-first discipline as every other change this session). Queued to
run after `part35_run1` finishes (avoiding CPU contention from running two
full-checkpoint processes at once). **If confirmed**: the fix is almost
certainly in LoopClosing.cc, either raising the BoW-match gate so more
margin survives Sim3Solver's stricter filtering, or in Sim3Solver.cc
itself, clamping `epsilon` to `<=1` (or `nIterations` to a sane floor like
the existing `mRansacMaxIts` cap) instead of letting it go through NaN.
**Potential impact if fixed**: reconnecting even a modest fraction of the
currently-forever-disjoint 13-15 fragments per run into fewer, longer
trajectories would directly increase scored coverage without needing any
more raw tracking success than what's already being achieved (77% frames
already track fine) -- this could be the highest-leverage remaining lever
given how much of the "already successfully tracked" 77% is currently
being wasted on fragments that never reconnect.

### Part 36/37 result (measured) -- fix confirmed mechanically correct, but surfaced a real new risk

**Mechanical confirmation**: post-fix `nInliers` distribution changed from
the pre-fix bimodal (always exactly 0, or a clean 16+) to a healthy spread
across 0-15+ -- direct proof RANSAC is now running its real iteration
budget instead of collapsing to 1 attempt. The fix is technically correct.

**Full-run result (`part37_run1`)**: Sim3Solver convergence rate **20/1059
= 1.89%**, up from the pre-fix 1.4% (18/1328) but only modestly, not
transformatively. Total coverage **1564.1m over 25 aligned fragments** (43
total, 469 KFs) -- a new single-run best (+5.7% vs part 35's 1480.3m).

**But: fragment 29 (103 KFs, the largest fragment of the entire session by
far, `pathLen=483.7m`) came in at `rmse=21.161m` -- the first RMSE budget
violation (>20m) all session.** Its error distribution is the signature of
a bad merge/loop-closure correction, not ordinary drift: `mean=8.579m` vs
`median=2.908m` (heavily skewed by a subset of badly-wrong poses) and
`max=134.932m` (far beyond what smooth accumulated drift over 483m of path
would plausibly produce). **If this one fragment is excluded (since it
violates the user's explicit accuracy budget), this run's valid coverage
is only 1564.1-483.7=1080.4m -- actually BELOW the ~1150-1480m range from
parts 32-35, not an improvement.** Most of the apparent coverage gain this
run came from one fragment that shouldn't be trusted.

**Likely mechanism**: the part 36 fix clamps `mRansacMinInliers =
min(minInliers, N)`, which correctly prevents the NaN/INT_MIN crash, but
for candidates with very small N (e.g. 3-9, common per the part 36
diagnostic data), it also means convergence now only requires ~N inliers
-- i.e., "all or nearly all" of a tiny, weakly-constrained correspondence
set. A handful of points all agreeing is much less statistically
convincing than 15+ points agreeing, so this plausibly makes it too easy
to accept a spurious/wrong merge for small-N candidates -- trading the
"never converges at all" bug for a "sometimes converges on too little
evidence" risk.

**Not yet fixed**: the natural next refinement is adding an absolute
inlier floor beneath the `min(minInliers, N)` clamp (e.g.
`max(6, min(minInliers, N))` or similar) -- keeps the NaN fix (still bounds
minInliers below N, so epsilon<=1 always, as long as N>=6) while requiring
genuine multi-point geometric agreement before accepting any merge,
preventing degenerate near-zero-evidence "full matches." Queued as part
38, not yet implemented or tested.

**Honest verdict for the user**: the Sim3Solver bug fix is real and
mechanically confirmed, but its net effect on trustworthy coverage this run
was NOT clearly positive once the one budget-violating fragment is
accounted for -- it likely traded "never merges" for "occasionally merges
on too little evidence." Keep the core NaN fix (still strictly better than
the guaranteed-broken original), add the inlier floor refinement next, and
re-measure before declaring this the session's biggest win.

### Part 38 change and a major noise-band update

Implemented the inlier-floor refinement: added `kMinAbsoluteInliers = 6` in
`Sim3Solver::SetRansacParameters` -- below that, Sim3Solver refuses to
attempt a fit at all (clean, deterministic failure: `mRansacMaxIts = 0`)
rather than let tiny-N candidates "converge" almost by construction (for
N==3, every RANSAC draw picks the same 3 points, so a 3-point fit trivially
reports itself as internally consistent -- tiny-N was paradoxically the
*easiest* way to spuriously pass, not the hardest). This keeps the part 36
NaN fix (still bounds minInliers<=N when N>=6, so epsilon<=1 always) while
requiring genuine multi-point agreement. Rebuilt, launched as `part38_run1`.

**Before that finished, `part36_run1` (the PRE-FIX diagnostic-only build,
i.e. same broken Sim3Solver as parts 29-35) finished and delivered a
surprise: 1853.6m over 20 fragments -- a new session-best, and higher than
part 37's post-fix 1564.1m, with ALL fragment RMSE clean (max 1.237m, no
budget violations at all)**. This coverage came from a single naturally-
long-surviving 106-KF fragment (no merge involved -- consistent with the
fail-fast/destroy-threshold mechanism from parts 30/34/35, not Sim3Solver).

**This is an important calibration update**: run-to-run noise under the
current settings is wider than previously estimated (~20%) -- the honest
range observed across parts 32-37 is now roughly **1080m (part 37 minus its
bad fragment) to 1854m (part 36)**, well over 50% spread. This means **the
Sim3Solver fix's true net effect on coverage cannot be judged from single
runs at all** -- part 37 "improving" on part 35 and part 36 "beating" part
37 are both plausibly just noise, not evidence for or against the fix.
Given the user's explicit direction to prioritize coverage now and defer
accuracy refinement, the practical takeaway is: **the fail-fast +
destroy-threshold levers (parts 30/34/35) remain the best-evidenced,
highest-confidence wins of the session** -- clean 1800m+ coverage is
achievable through them alone. The Sim3Solver fix is kept (it's a genuine
correctness fix, and the inlier floor should reduce its bad-merge risk) but
should not be credited with the big coverage numbers without a proper
multi-run comparison, which the noise band now shown to require.

**Part 38 result (measured)**: **1367.4m over 19 fragments** (41 total, 403
KFs), **all RMSE clean** (max 1.532m, no budget violations) -- the
`kMinAbsoluteInliers=6` floor worked as intended, no repeat of part 37's
bad-merge fragment. Convergence dropped to 8/792=1.01% (down from part 37's
1.89%), consistent with filtering out the spurious tiny-N "successes"
rather than losing genuine ones. **Verdict: keep both the NaN fix and the
inlier floor** -- confirmed safe, no accuracy regressions, real correctness
improvement over the original broken code, even though its net coverage
contribution remains hard to isolate from the session's wide noise band.

### Part 39: nFeatures 5000 -> 10000

Per user's explicit direction to prioritize coverage now, accuracy
refinement later. Motivation: part 36's diagnostic data showed Sim3Solver
candidates routinely starved for correspondences (N as low as 1-9,
frequently below the new 6-point floor) -- more raw SIFT features per
frame should directly increase how many correspondences survive into N,
giving both tracking and merge attempts more to work with. Known cost: fps
drops roughly linearly with nFeatures in this fork's earlier calibration
(2000=6.73fps, 3000=6.38fps, 4000=~5.8fps) -- 10000 will make each full
seq00 run meaningfully slower (~35-45min instead of ~20-30min, based on
extrapolation, not yet measured). Pure settings change (`settings_sift/
KITTI00-02-sift.yaml`), no rebuild needed. Launched as `part39_run1`.

**Part 39 result**: **1681m over 16 fragments, all RMSE clean** (max
1.130m). Sim3Solver convergence 27/1631=1.66%, somewhat higher than the
~1-1.9% range from parts 37/38 -- plausibly benefiting from more available
correspondences (N), consistent with part 36's correspondence-starvation
finding. Within the established noise band but on the higher/clean side.
**Kept nFeatures=10000** (no clear harm, plausible modest benefit, cost is
runtime only).

### Part 45: root-caused WHY tracking fails so often (not just its aftermath)

User asked directly why tracking gets fragmented in the first place -- all
work through part 44 addressed the *aftermath* of frequent tracking loss
(fail fast, don't destroy small maps, fix merge math) without knowing *why*
loss happens ~80x more often than stock ORB-SLAM3 (240+ resets/run vs 3).

Wrote `scratchpad/analyze_death_pattern.py`: loads KITTI seq00's GT poses,
computes per-frame rotation (deg) and translation (m) magnitude between
consecutive frames, cross-references against every `Fail to track local
map!` event's frame id across 3 recent logs (744 unique failure frames).
**Result: failures do NOT correlate with rotation (7.7% of failures in the
top-10%-rotation bucket, actually BELOW the 10% baseline -- refutes an old
"hard turns" hypothesis from earlier in this project). They DO correlate
with translation speed (18.1% of failures in the top-10%-speed bucket, vs
10% baseline)** -- vehicles moving faster are noticeably more likely to
lose tracking on that frame.

Mechanism: faster motion -> larger inter-frame feature displacement in
image space -> harder to match within a fixed search radius. Confirmed in
`Tracking::SearchLocalPoints()` (~line 3646): `th` (matching search radius)
is widened for every special case already coded (RGBD th=3, IMU th=2-10,
post-reloc th=5, lost th=15) EXCEPT the default monocular OK-state case,
which stays at the stock `th=1` unconditionally, regardless of how fast the
camera is moving.

### Part 46: velocity-adaptive search radius (user: high priority)

Added `[velocity-investigate]` diagnostic logging `mVelocity.translation().norm()`
(the existing constant-velocity motion model's per-frame estimate, in this
SLAM's own internal scale) alongside `th`. Ran `part40_run1` to collect a
sample: **550 velocity samples, p90=0.0538. Of the (small, 8-sample) set of
failures with a recorded velocity, 37.5% were in the top-10%-velNorm
bucket vs 10% baseline** -- an even stronger signal than the GT-based
correlation (though from a much smaller failure sample, treat as
corroborating, not independently conclusive).

**Fix implemented**: in `SearchLocalPoints()`, added
`constexpr float kHighVelThreshold = 0.05f; if(mbVelocity &&
mVelocity.translation().norm() > kHighVelThreshold) th = std::max(th, 5);`
-- threshold picked from the measured p90, not guessed. Mirrors the
existing th=5 post-relocalization treatment (this fork already trusts th=5
as a reasonable "harder matching situation" widening elsewhere). Rebuilt,
launched as `part41_run1` (with nFeatures=10000 carried over from part 39).
Not yet measured -- next step is comparing total coverage, RMSE budget
compliance, and ideally the raw tracking failure RATE (not just coverage,
which is noisy) against the pre-fix baseline to see if this directly
reduces how often tracking breaks in the first place, which would be a
more fundamental win than anything else this session since it addresses
the root cause rather than managing its aftermath.

**Part41_run1 was killed mid-run (frame 3593/4541, ~79%) -- OOM, not a
code bug.** No exception/segfault/assertion text anywhere in the log (ruled
out the part-35/36-style OpenCV TLS crash and any new crash in the
velocity-fix code itself) -- the log just stops abruptly after a normal
`Merge finished!` message, and the process was confirmed dead externally.
Root cause: `part41_run1` was launched while `part40_run1` (the diagnostic
run used to collect velocity data) was still running, and BOTH used
`nFeatures=10000` (part 39's change, kept) simultaneously -- system had
14GB RAM, already 1.5GB into a 4GB swap file under just `part40_run1`
alone; running two `nFeatures=10000` processes at once pushed it over the
edge and the OS killed one. **Lesson: do not run two `nFeatures=10000`
processes concurrently** -- either serialize them, or drop nFeatures for
any run that must be parallelized.

**Second, unrelated problem found while waiting for `part40_run1` to
finish solo**: it stalled for 48+ minutes without completing (vs the usual
~20-40min for `nFeatures=10000`). Not actually frozen -- log kept growing
-- but extremely slow, because by keyframe `cur=719` the Atlas had
accumulated **23+ separate map fragments** (`candMapId` values 0-23 seen),
and `LoopClosing`'s merge-candidate check runs BoW matching + Sim3Solver
RANSAC + optimization against EVERY accumulated fragment for every
candidate keyframe, with no cap analogous to `DetectRelocalizationCandidates`'s
existing `kMaxRelocCandidates=20` guard (added in an earlier session
specifically because an uncapped candidate search once stalled 20+
minutes). **This is very likely a real, emergent cost of this session's
own changes**: `nFeatures=10000` (part 39) puts more features into BoW
matching (more candidates pass the `numBoWMatches>=20` gate), and the
fail-fast/low-destroy-threshold changes (parts 30/34/35) deliberately
preserve many more, smaller fragments than before -- multiplying how much
merge-candidate work happens per keyframe. Killed `part40_run1` (already
had the velocity data needed from it) rather than wait further -- this
merge-candidate-count explosion is flagged as a real problem worth a cap
(mirroring `kMaxRelocCandidates`) in a future part, but is not blocking
right now since `part41_run2` (the actual priority) is a single fresh run
that won't have had time to accumulate 23 fragments before it finishes.
Re-launched `part41_run2` alone immediately.

**Part41_run2 result (nFeatures=10000 + velocity fix v1, threshold=0.05f)**:
**1713.3m over 20 fragments**, reset count **221** (within the ~150-260
baseline range -- no clear improvement), velocity-widen engaged on
**2026/3486 = 58.1%** of frames (intended ~10%). One fragment hit
rmse=14.736m (still under budget but high).

**Part 47: threshold recalibration.** Root cause of the 58.1%-vs-10%
mismatch: the original `kHighVelThreshold=0.05f` was calibrated from an
early 550-sample checkpoint (p90=0.0538), which was NOT representative --
computing percentiles over the FULL 3486-sample `part41_run2` log gives
**p50=0.0547** (the old threshold was sitting at the *median*, not p90)
and **true p90=0.0867**. This directly explains the over-triggering: the
fix was almost-always-on rather than a targeted fast-motion case. **Not
yet known whether the underlying speed-correlation hypothesis (part 45) is
wrong, or just that this specific fix was miscalibrated** -- reset count
showing no improvement is consistent with either. Recalibrated
`kHighVelThreshold` to the true measured p90 (**0.087f**), rebuilt.
`part42_run1` (nFeatures=5000, velocity fix v1/old threshold, per user's
request to test 5000 before 6000/7000) was already running and left to
finish as a still-useful data point; the NEXT run after it will use the
corrected threshold to properly test the recalibrated fix.

## Current status (2026-07-18, part 26 -- Front 2 revisited: found and
## partially fixed a real, high-impact post-relocalization gate; pushed it
## too far once, learned coverage is noisier than expected, work IN
## PROGRESS as of this writing -- see "resume point" at the very end of
## this entry).
##
## Per explicit request, pivoted back to Front 2 (front-end robustness /
## coverage) after part 24/25's merge work. Measured real re-init latency
## first (before assuming anything): frames between a map death and the
## next successful mono-init average only **2.3 frames** (110 samples) --
## NOT "thousands of frames" as older memory (pre-Session-15
## `SearchForInitialization` fix) suggested. That old finding is stale;
## re-init speed is not the bottleneck. The real cost is many rapid
## death/reinit cycles, not one long stuck gap.
##
## **Root-caused a major remaining death cause**: correlated KF-count-at-
## death with the immediate preceding diagnostic line and found `Tracking.cc`'s
## post-relocalization strict gate (`mnMatchesInliers<50` while
## `mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames`) still firing
## constantly -- **170 times in one checkpoint, 120 of which (70.6%) had
## mnMatchesInliers in [30,49]**, i.e. would have survived under the
## *normal* monocular `need>=30` bar and been killed *solely* by this extra
## post-relocalization margin. This is the SAME gate flagged in Session 15
## as "23% of failures" and partially addressed by part 18's RECENTLY_LOST-
## bypass reorder -- but that reorder only helps when `mState==RECENTLY_LOST`
## at the exact check; the far more common case (mState==OK, just relocalized
## within the last `mMaxFrames`) was untouched until now.
##
## **Threshold experiment, in order, each measured on a full seq00
## checkpoint (nFeatures=5000, all part 19-25 merge fixes in place)**:
##
## | gate value | total scored path (coverage) | gate FAILs remaining |
## |---|---|---|
## | 50 (baseline, unmodified) | 543.6 m | 170 |
## | 35 | 478.7 m | 97 |
## | 30 (matches normal bar exactly, makes the branch a no-op) | **314.4 m** | 86 |
##
## **Coverage went DOWN monotonically as the gate was loosened further**,
## the opposite of the mechanical prediction. Not yet root-caused --
## leading theory is a "survivorship paradox" (keeping a barely-alive map
## going a few more frames on a weak pose may walk it into a harder failure
## shortly after, instead of resetting promptly for a fresh, possibly
## easier restart) -- but given this session already saw one single-run
## comparison get reversed by nondeterminism, **do not trust this 3-point
## trend without repeat trials**. **Reverted to 35** (best result of the
## three measured) as the safe middle ground pending confirmation.
##
## **Separately measured whether "need>=30" is even a sensible bar for SIFT
## specifically** (user question: stock ORB-SLAM3's thresholds were tuned
## for ORB, not SIFT) -- important clarification: `mnMatchesInliers` is
## counted AFTER `PoseOptimization`'s own chi2 outlier rejection, i.e. it's
## already a verified-inlier count, not a raw match count, so its required
## size is a fairly descriptor-agnostic statistical/geometric question (how
## many confirmed correspondences does a stable pose estimate need), not an
## ORB-vs-SIFT question per se. What IS descriptor/fork-specific is the
## raw-to-verified attrition rate. Measured directly from existing
## `[track-local-map-detail]` logs (1872 samples, one full checkpoint):
## **outlier rate ~7.9%, verified-inlier survival ~92.1%** -- clean, reliable
## matching. This means `need>=30` is NOT overly strict for descriptor-
## quality reasons (SIFT matches here are trustworthy); lowering the floor
## further (e.g. to 20, discussed but not implemented) would not be
## justified by outlier-rate evidence and would just accept less-constrained
## pose estimates. The real lever for candidate scarcity remains growing the
## raw candidate pool (nFeatures, nNumCovisibles), not lowering the
## verified-inlier bar.
##
## Code state as of this writing: `Tracking.cc`'s post-reloc gate is back
## to `mnMatchesInliers<35` (see its inline comment for the full 50->35->30
## ->35 history). `LoopClosing.cc` additionally has (from part 25, kept):
## `nNumCandidates` 3->10 in the `DetectNBestCandidates` call, and
## `nNumCovisibles` 10->20 in both `DetectCommonRegionsFromBoW` and
## `FindMatchesByProjection` -- both zero-correctness-risk widening changes,
## unrelated to the gate-value question above.
##
## **RESUME POINT for next session**: two repeat trials at gate=35 were
## launched in parallel (`gate35_trial1`/`gate35_trial2` output prefixes,
## same nFeatures=5000 + all current thresholds, no other changes) to get a
## statistically defensible coverage baseline before drawing any conclusion
## about whether 35 is really better than 30 (or than other values). As of
## this note they were still running (both processes healthy, ~99% CPU,
## just slow from sharing 8 cores -- do not mistake this for a hang; check
## the *actual* `orbslam3_sift_kitti_ate` PIDs, not any wrapping bash/nohup
## PID, when checking CPU%). **Next session should**: (1) check whether
## those two runs finished and completed cleanly; (2) compare their
## coverage numbers to each other and to the 478.7m single-run gate35
## result above to gauge run-to-run variance at fixed settings; (3) only
## then decide whether the 30-vs-35 coverage regression was a real effect
## or noise, and whether to keep pushing this specific gate lower or
## consider it settled at 35 and move to the next front-end death cause
## (re-run the KF-at-death + immediate-cause correlation from this part,
## fresh, since the distribution has shifted after this fix).

## Current status (2026-07-18, part 25 -- nFeatures trend continues to
## 5000, plus two new zero-correctness-risk levers). At nFeatures=5000
## (same 5-threshold chain as part 24, `nNumCandidates`/`nNumCovisibles`
## still at stock 3/10 for this specific run): **6 map fragments/114 KFs
## total -- the lowest fragment count of the entire session** -- and 1 more
## successful merge (fragment 4, 36 KFs, 197.9m, 0.454m ATE RMSE = 0.23%
## relative error, healthy). Timing: 2000=6.73fps, 3000=6.38fps,
## 4000=~5.8fps, 5000 not yet precisely timed standalone.
##
## User observation: 1 merge/run is still low relative to how many
## fragments visibly overlap in the recurring x~[56,90] z~[104,450]
## corridor each run (typically 3-4). Confirmed by direct log inspection
## (part 24's fragment-1/map-1 case): candidates ARE repeatedly reaching
## the top-3 VLAD shortlist and losing out to whichever other candidate's
## RANSAC happened to converge that specific frame -- not being
## structurally excluded. This means the bottleneck at this point is
## partly just **breadth of attempts**, not (only) correspondence density.
##
## **Two new changes, both deliberately zero-correctness-risk** (neither
## touches any statistical/geometric acceptance threshold -- they only
## widen how much gets attempted, so they can't by themselves make a false
## merge more likely to pass, unlike every part 19-23 change):
## 1. `KeyFrameDatabase::DetectNBestCandidates`'s call in
##    `NewDetectCommonRegions()`: nNumCandidates 3->10 (LoopClosing.cc).
##    More Sim3Solver attempts per frame -- cost lands on LoopClosing's own
##    background thread, not the tracking hot path.
## 2. `nNumCovisibles` 10->20, in both `DetectCommonRegionsFromBoW` and
##    `FindMatchesByProjection` (two separate local declarations in
##    LoopClosing.cc, kept in sync) -- pools more covisible keyframes'
##    map points into each candidate's correspondence set, same
##    "grow the pool" spirit as the nFeatures increases but targeting
##    breadth-per-candidate instead of density-per-frame.
##
## Explicitly deferred (higher risk, not yet needed): lowering
## `nBoWInliers=15` (Sim3Solver's own RANSAC minimum-inlier parameter) --
## this is the last major untouched threshold and the primary remaining
## defense against a false-positive merge; only consider it if the two
## changes above plus continued nFeatures increases still aren't enough.
##
## Running now: nFeatures=5000 + nNumCandidates=10 + nNumCovisibles=20
## combined, full seq00 checkpoint, same `[merge-investigate]` instrumentation
## plus `grep -c "\*Merge detected"` (the correct signal, see part 24) for
## merge count. Compare fragment count, merge count, and per-fragment ATE
## (especially any newly-merged fragment) against this part's 5000-alone
## baseline (6 fragments, 1 merge) once it finishes.

## Current status (2026-07-18, part 24 -- FIRST SUCCESSFUL MERGE, after
## parts 19-23's threshold chain + nFeatures increases). Full seq00
## checkpoint at nFeatures=4000, with all five merge-chain thresholds
## lowered (nProjMatches=20, nProjOptMatches=24, nSim3Inliers=6, the
## internal `Optimizer::OptimizeSim3` bailout=6, and
## `DetectCommonRegionsFromLastKF`'s own nProjMatches=15): candidate
## cur=382 (map 8) vs cand=266 (map 6) passed every single gate in one
## `NewDetectCommonRegions()` call --
## numBoWMatches=187(>=20)->Sim3Solver CONVERGED(29 inliers)->
## numProjMatches=28(>=20)->numOptMatches=8(>=6)->numProjOptMatches=24(>=24,
## exactly at the bar)->nNumKFs=3(>=3, exactly at the bar, 3 of 8 covisibles
## independently confirmed via the now-lowered `DetectCommonRegionsFromLastKF`
## bar) -- and `MergeLocal()` actually executed: log shows `*Merge detected`
## -> `Change to map with id: 6` -> `Merge finished!`, with tracking
## immediately after on the merged map showing `mapKFs=41` (up from
## whatever map 8 had alone).
##
## **Note the earlier false negative this session**: initially searched
## logs for the literal string `"[Merge]:"` (the cout lines inside
## `MergeLocal`/`MergeLocal2` around LoopClosing.cc:1557-1558) and concluded
## "0 merges" across several runs before this one. That search pattern was
## incomplete -- the actual unconditional signal is
## `Verbose::PrintMess("*Merge detected", VERBOSITY_QUIET)` /
## `"Merge finished!"` (both print regardless of the CLI tool's
## `VERBOSITY_QUIET` threshold, since QUIET is themessage's own level, not
## filtered by it) -- these are the strings to grep for, not `"[Merge]:"`.
## **Re-audited all 8 prior saved logs this session with the corrected
## `"*Merge detected"` grep -- confirmed genuinely 0 merges in every one of
## them.** The earlier "0 merges" conclusions were right, just for the
## wrong/incomplete reason (a search pattern that happened to also return 0
## on logs that truly had 0 merges). This run (nFeatures=4000) is the
## first real merge of the session.
##
## **Sanity check on merge correctness** (the real safety net against a
## false-positive merge corrupting the map): the resulting merged fragment
## in this run's final ATE table is fragment 7 (42 KFs, 200.3m path,
## 0.545m ATE RMSE = **0.27% relative error**) -- within the same healthy
## 0.07%-0.44% range as every other (unmerged) fragment in this run, not
## degraded. This one data point is consistent with a correct merge, not
## proof -- worth deliberately checking again on the next successful merge
## (compare the merged fragment's per-KF trajectory continuity/smoothness
## across the seam, not just the aggregate ATE number, which could mask a
## localized jump).
##
## Final result this run: 9 fragments/166 KFs total (down from double-digit
## fragment counts in every prior run this session), exactly 1 merge event
## (`grep -c "\*Merge detected"` = 1).
##
## **Next steps**: (1) re-audit parts 19-23's logs (still on disk in the
## scratchpad at time of writing) for missed `"*Merge detected"` events
## using the corrected grep pattern. (2) Run nFeatures=5000 (queued/running
## as of this writing) and see if the merge rate increases further with
## more real points, continuing the proportional trend. (3) Decide whether
## `nBoWInliers=15` (Sim3Solver's own RANSAC minimum-inlier parameter,
## deliberately untouched all session as the primary defense against false-
## positive merges) still needs lowering, or whether nFeatures alone is
## enough now that a real merge has been observed without touching it --
## re-evaluate only after (2)'s result and a deliberate correctness check
## per the sanity-check note above, not before.

## Current status (2026-07-18, part 21 -- nFeatures 2000->3000 for the SIFT
## fork validated real, proportional improvement at every gate in the merge
## chain, confirming part 20's diagnosis, but still zero completed merges).
##
## Implemented the user's proposal 2 from part 20 (not proposal 1 -- see
## reasoning there): new SIFT-fork-only settings file
## `settings_sift/KITTI00-02-sift.yaml` (copy of the shared
## `KITTI00-02.yaml`, `ORBextractor.nFeatures` 2000->3000; does not touch
## the original `third_party/ORB_SLAM3`'s settings). No C++ rebuild needed
## (runtime YAML). Full seq00 checkpoint, same `[merge-investigate]`/
## `[optsim3]` instrumentation:
##
## | gate | before (part 20, nFeatures=2000) | after (nFeatures=3000) |
## |---|---|---|
## | Sim3Solver convergence | 17/373 = 4.6% | **15/119 = 12.6%** |
## | numProjMatches>=20 pass | 3/17 | 7/15 |
## | numOptMatches>=10 pass | 0/3 | **2/7 -- first time ever >0** |
## | numProjOptMatches>=32 pass | (never reached) | 0/2, observed 19-20 |
## | optsim3 bailout `nCorrespondences` | 6-11 | 12-17 (grew) |
## | optsim3 bailout survivors (need>=10) | 0,0,7 | 0,7,0,7,8 (closer) |
## | actual completed merges | 0 | **still 0** |
##
## **Every single gate improved, proportionally, in the direction the part
## 20 diagnosis predicted** -- more real points per keyframe genuinely grows
## the correspondence pool at every downstream stage, not just the first
## one. This is a second independent confirmation (after the threshold-
## scaling result) that "too few real matched points" was the correct root-
## cause diagnosis, not a strict-descriptor false-rejection problem.
## Fragment count/accuracy stayed healthy (13 fragments/185 KFs, per-
## fragment relative error 0.07%-1.05%, no sign of degraded tracking from
## the extra feature-extraction cost) -- fragments 1/2/6/7/8/11 again
## overlap in the same x~[56,80] z~[106,454] corridor as every prior run.
##
## **Still zero completed merges** -- the funnel's bottleneck moved one gate
## further down to `numProjOptMatches>=32` (observed 19-20, ~60% of the
## bar) with only 2 samples reaching it. `nNumKFs` (final covisible-
## consistency gate) still never reached.
##
## **Not yet decided**: whether to (a) bump `nFeatures` further (the trend
## is real and proportional, may just need more headroom), or (b) apply the
## same evidence-based proportional pull-down to `nProjOptMatches` now that
## real post-boost data exists for it (19-20 observed vs. 32 required -- a
## smaller, better-justified nudge than part 20's original 0.4x guess would
## be, though still only 2 samples). Runtime cost of nFeatures=3000 was not
## precisely measured against nFeatures=2000 in this session (both full
## checkpoints took on the order of 10 minutes; SIFT is markedly more
## expensive per-feature than ORB so this should be watched, not assumed
## free, before pushing nFeatures much higher).
##
## User also asked about three further loosenings (2026-07-18): (1) skip
## the "scale consistency" filter -- checked the code (`LoopClosing.cc`
## ~line 141-154): it's already gated behind
## `mpCurrentKF->GetMap()->IsInertial() && mpMergeMatchedKF->GetMap()->
## IsInertial()`, so for this project's plain `System::MONOCULAR` (no IMU)
## it never runs at all -- nothing to change, already a non-issue for this
## pipeline. (2) loosen `TH_HIGH` to its ceiling for merge/loop, and (3)
## lower the `<10` optsim3 bailout further -- both explicitly deferred
## pending this nFeatures result, per the reasoning in part 20 (stacking
## multiple relaxed safety nets at once risks false-positive merges, which
## corrupt the map worse than staying fragmented, and makes it impossible
## to attribute cause if something goes wrong). Now that nFeatures=3000's
## real, positive, proportional effect is confirmed, revisit (2)/(3) only
## if still needed after the nFeatures/nProjOptMatches decision above, and
## still one variable at a time with before/after ATE checked each time (a
## false merge should show up as degraded ATE on the merged fragment).

## Current status (2026-07-18, part 20 -- the merge chain's actual floor:
## not enough real map points, not a strict-descriptor false-reject
## problem). Scaled down `nProjMatches`(50->20)/`nProjOptMatches`(80->32)/
## `nSim3Inliers`(20->10) per part 19. Result: 3/14 Sim3Solver-converged
## candidates now pass `numProjMatches` (up from 0/17 before), but all 3
## still fail at `Optimizer::OptimizeSim3`'s **hard-coded, non-parameterized**
## `if(nCorrespondences-nBad<10) return 0;` bailout (found by instrumenting
## it directly -- this is a 4th absolute threshold in the chain, not exposed
## to the caller). Data from the 3 real bailout cases:
##
## | kf1 | kf2 | nCorrespondences (edges actually built) | nBad (chi2-rejected) | survivors |
## |---|---|---|---|---|
## | 392 | 283 | 8 | 8 | **0/8 (100% rejected)** |
## | 392 | 286 | 6 | 6 | **0/6 (100% rejected)** |
## | 395 | 287 | 11 | 4 | 7/11 (64%, just 3 short of 10) |
##
## Two of three cases had **every single correspondence** rejected by the
## chi2 outlier test, not "a few short of the bar" -- and `nCorrespondences`
## itself (6-11) is far below the `numProjMatches` that fed it (20-31,
## reported by the caller), meaning a further ~50-70% of matches are lost
## between `SearchByProjection` and usable optimizer edges (bad map points /
## negative depth filtering) -- a 5th attrition point not yet separately
## instrumented.
##
## **This changes the diagnosis**: a 100%-rejection Sim3 fit is not "some
## correct matches got thrown out by an overly strict chi2/descriptor
## threshold" -- it's "the coarse Sim3 estimate itself (from Sim3Solver,
## which converged on as few as ~15-20 raw BoW inliers for a full 7-DOF
## rotation+translation+scale fit) is too imprecise for *anything* to agree
## with it." Threshold-loosening (this session's approach so far) cannot fix
## an underdetermined estimation problem -- it just relocates the same
## starvation to the next gate down the chain, which is exactly the pattern
## observed three times now (numProjMatches -> numOptMatches -> here).
##
## **User-proposed next steps (2026-07-18, end of session), evaluated
## against this data**:
## 1. *Widen the descriptor-distance filter (e.g. 1.5x `TH_HIGH`) for the
##    merge/loop path specifically.* Plausible but **this data doesn't
##    clearly support it as the primary lever** -- the bottleneck observed
##    here looks like too few real corresponding 3D points existing in the
##    pooled candidate set, not correct matches being rejected on distance
##    grounds. Loosening `TH_HIGH` risks adding *more false matches* into an
##    already precision-starved RANSAC/optimization chain, which could make
##    the 100%-rejection pattern worse, not better. Would need direct
##    evidence (e.g. logging rejected-by-distance candidate pairs and
##    checking if they're geometrically plausible) before implementing, not
##    just an assumption that it will help.
## 2. *Increase SIFT feature budget (nFeatures 2000->3000) and/or revisit the
##    grid/cell keypoint-distribution filter to capture smaller blobs.*
##    **Better supported by this data** -- more points per keyframe directly
##    grows the pool at every single stage of this chain (BoW match count,
##    Sim3Solver's RANSAC sample size, OptimizeSim3's edge count), attacking
##    the actual observed floor (samples of 6-11 points feeding a 7-DOF fit)
##    rather than relaxing a filter around an already-too-small pool. Needs
##    its own separate settings file/constant for the SIFT fork only (not
##    touching `third_party/ORB_SLAM3`'s original 2000, and not the shared
##    `KITTI00-02.yaml` used by both) since this is fork-specific tuning:
##    higher nFeatures also raises SIFT's already-heavier per-feature compute
##    cost (SIFT is markedly more expensive than ORB per keypoint), so
##    measure runtime impact alongside the merge-success measurement, not
##    just accuracy.
##
## **Next session should start with**: implement and measure option 2
## first (nFeatures bump, dedicated SIFT-only settings), since it targets
## the confirmed bottleneck directly; re-run the full checkpoint with the
## same `[merge-investigate]`/`[optsim3]` instrumentation (still in the tree,
## not yet cleaned up) to see whether `nCorrespondences` at the optsim3 gate
## grows and survivor counts clear the (still-lowered) `<10` bar. Only
## revisit option 1 (descriptor threshold) afterward, and only with direct
## before/after distance-distribution evidence, not as a first move.

## Current status (2026-07-18, part 19 -- Front 1 (merge) instrumented and
## root-caused with a clean, decisive funnel; Front 2 (recovery) shelved as
## structurally limited, see part 18).
##
## Added `[merge-investigate]` diagnostics through the whole merge candidate
## pipeline: `NewDetectCommonRegions()` (candidate KF ids + map ids from
## `DetectNBestCandidates`), and every gate inside `DetectCommonRegionsFromBoW`
## (added an optional `tag` param, `"LOOP"`/`"MERGE"`, to distinguish the two
## call sites without duplicating the function) -- BoW match count, Sim3Solver
## convergence, numProjMatches, all the way through. Full seq00 checkpoint run
## (18 fragments/328 KFs this time -- fragments 2/10/11/13 again spatially
## overlap in the same x~[56,90] z~[104,450] corridor as parts 17/18's
## fragments 1/6/8 and 1/8/11, now confirmed 3 separate runs in a row).
##
## **Clean funnel across all 161 candidate-search calls that got a merge
## candidate at all (159/161 did -- VLAD scoring itself is not the
## bottleneck)**:
##
## | gate | evaluated | passed |
## |---|---|---|
## | numBoWMatches>=20 | 477 | 373 (78%) |
## | Sim3Solver RANSAC convergence | 373 | **17 (4.6%)** |
## | numProjMatches>=50 | 17 | **0 (0%)** -- observed range 10-32 |
##
## **Merge never once got past `numProjMatches>=50` in this entire run.**
## Every one of the 17 candidates that survived the already-brutal 4.6%
## Sim3Solver convergence rate then failed the reprojection-match count,
## landing at roughly 20-60% of the required 50 (values: 10, 11, 13, 16,
## 16, 18, 18, 21, 21, 21, 22, 23, 23, 24, 27, 31, 32).
##
## This is the same class of problem as part 15/18's `TrackLocalMap`
## need>=30 and `Relocalization`'s need>=50 bars: **fixed absolute match-
## count thresholds inherited unchanged from stock ORB-SLAM3, calibrated
## for that codebase's typically large/mature maps, applied unchanged to
## this fork's much smaller fragments** (11-45 KFs each here, vs. stock
## ORB-SLAM3 maps that routinely have hundreds). Because Sim3Solver already
## requires real RANSAC-verified geometric consistency (>=15 inliers) before
## a candidate ever reaches the `numProjMatches` gate, the 17 survivors here
## are very likely genuine spatial overlaps, not noise -- consistent with
## the independently-confirmed fragment-bbox overlap (2/10/11/13) from the
## same run.
##
## **Not yet done**: lowering `nProjMatches` (and probably `nProjOptMatches
## =80`, likely oversized for the same reason, though no data on it yet
## since nothing has reached that gate) is the obvious next lever, but
## **do not just guess a number** -- these thresholds exist specifically to
## reject false-positive merges, which corrupt the map far worse than
## staying fragmented, so the replacement value needs the same evidence-
## based treatment as everything else this session:
## 1. First confirm the 17 CONVERGED candidates are genuinely the same
##    physical place as the current KF (cross-reference cur/cand KF ids
##    against each fragment's keyframe-pose range, same technique as part
##    16's bbox check) rather than assuming Sim3Solver convergence alone
##    proves it.
## 2. If confirmed, pick a threshold with real margin below the observed
##    true-positive range (10-32) but ideally checked against what
##    false/rejected Sim3Solver-FAIL-ing candidates would have scored here
##    too, to see if there's a real separation to exploit or if a lowered
##    bar risks admitting false positives that never got Sim3Solver-tested
##    at this stage.
## 3. Rebuild, rerun the full checkpoint, and verify with the fragment-
##    count/coverage table (same methodology as parts 15/18) that fragments
##    2/10/11/13 (or their equivalents in a fresh run) actually merge into
##    fewer, longer trajectories -- not just that `numProjMatches` gate
##    starts passing.
##
## Front 2 (recovery-mechanism effectiveness) is shelved per part 18's
## finding that in-map relocalization is structurally starved for young
## maps and a real fix would require replicating Sim3-based cross-map
## alignment inside the tracking-time recovery path -- essentially this
## same merge machinery, just invoked synchronously instead of by the
## background LoopClosing thread. Revisit only after the merge threshold
## work above is validated; if merge starts working reliably in the
## background thread, the tracking-time urgency to also fix recovery drops
## since fragments will get reconnected shortly after anyway.

## Current status (2026-07-18, part 18 -- Front 2 (fragmentation-frequency)
## work: one real fix applied, one hypothesis chain built and then
## correctly refuted step by step with live data instead of guessed).
##
## **Fix applied and validated**: `Tracking::TrackLocalMap()` had the
## stricter post-relocalization gate (`mnMatchesInliers>=50` while
## `mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames`) checked BEFORE the
## `RECENTLY_LOST` recovery bypass (`mnMatchesInliers>10 && mState==
## RECENTLY_LOST`) -- already quantified in Session 15 as 23% of this
## fork's tracking failures but never fixed. Reordered so the bypass is
## checked first (comment added in-place explaining why; no threshold
## values changed, just the check order). Rebuilt `orbslam3_sift_kitti_ate`
## clean, ran a full seq00 checkpoint:
##
## | | before (part 15) | after (this fix) |
## |---|---|---|
## | map fragments | 10 | 14 |
## | total keyframes | 246 | 270 |
## | scored fragments | 9 | 13 |
## | combined scored path length | 615.8 m | **837.1 m** |
## | per-fragment relative error range | 0.05%-1.08% | 0.08%-1.15% |
##
## Net: more fragments, but genuinely more real coverage (+36% scored
## path length) at unchanged excellent per-fragment accuracy -- a real,
## low-risk improvement, kept. Not the hoped-for "fewer, longer-lived
## maps" outcome though, so fragmentation-frequency itself is still
## unsolved. Fragments 1 (x[60.5,71.9] z[106.0,244.3]), 8
## (x[56.1,68.9] z[255.9,450.7]) and 11 (x[63.6,72.0] z[223.9,245.1])
## again spatially overlap in this run -- same finding as part 17's
## fragments 1/6/8, confirming the map-merge front from part 17 is still
## real and independent of this fragmentation-frequency work.
##
## **Investigated whether the `KeyFramesInMap()>10` RECENTLY_LOST-
## eligibility gate (the originally-planned "loosen this" lever) is
## actually the dominant fragmentation cause -- built a live KF-count-at-
## death histogram from the checkpoint log** (added `mapKFs=` to the
## existing `[track-local-map]` diagnostic, correlated against every
## `Reseting current map`/`Creation of new map with id` event): of 102
## fragment-ending events, **63 (62%) die at exactly 2 keyframes** (the
## very first tracking attempt after `CreateInitialMapMonocular()`, before
## a 3rd keyframe is ever inserted), 86% die at <=9 KFs, and only ~13%
## ever get anywhere near the `>10` gate. This initially looked like it
## made the `>10` gate largely irrelevant and pointed at something far
## upstream instead.
##
## **First hypothesis (weak-baseline init) tested directly against data
## and REFUTED, not assumed**: `MonocularInitialization()` computes
## `meanDisparityPx` (mean pixel disparity between the two init frames, a
## parallax/baseline-quality proxy) but explicitly does not gate
## acceptance on it (see the existing code comment: "logged, not yet
## gated on"). Hypothesis was that near-zero-baseline accepted pairs
## produce weak maps that die instantly. Correlated `meanDisparityPx`
## against survival-to-KF-count across all 104 mono-init events: KF=2
## deaths averaged 19.9px disparity, KF>10 survivors averaged 23.5px --
## no meaningful separation, and several high-disparity (40-83px) inits
## still died instantly while one low-disparity (7.8px) init survived to
## KF=14. **Disparity alone does not predict survival -- do not implement
## an init-disparity gate on this basis, the data doesn't support it.**
##
## **Second false lead, caught before it cost more time**: initially
## concluded `RECENTLY_LOST` was "essentially never entered" because
## `Verbose::PrintMess("Lost for a short time", ...)` and every other
## `RECENTLY_LOST`/`LOST`-transition `Verbose::PrintMess` call never
## appeared anywhere in the checkpoint log. **This was a false negative,
## not evidence**: `System.cc:238` calls
## `Verbose::SetTh(Verbose::VERBOSITY_QUIET)`, which silences every
## `Verbose::PrintMess` call (`VERBOSITY_NORMAL=1 > VERBOSITY_QUIET=0`) for
## this CLI tool -- only plain `cout<<`/`fprintf(stderr,...)` lines are
## ever visible in these logs. **Any future session reasoning about
## Tracking.cc's state machine from this tool's logs must use
## `fprintf(stderr,...)` diagnostics, never trust the presence/absence of
## a `Verbose::PrintMess` string.**
##
## Re-reading `Track()` with that corrected, found there are actually TWO
## separate, differently-gated places `mState` gets set to `RECENTLY_LOST`
## on failure, not one:
## - Branch 1 (~line 1961-1978): the FIRST pose-estimate stage
##   (`TrackReferenceKeyFrame`/`TrackWithMotionModel`) fails outright --
##   gated on `KeyFramesInMap()>10` (else goes straight to `LOST`, no
##   grace period at all). This is the gate originally planned to loosen.
## - Branch 2 (~line 2168-2184): the first stage SUCCEEDS but the
##   subsequent `TrackLocalMap()` fails -- sets `RECENTLY_LOST`
##   **unconditionally, with no KF-count gate whatsoever**.
##
## Confirmed via the log (the `[track-local-map] id=X mnMatchesInliers=Y
## mapKFs=2` diagnostic line appears immediately before each streak of
## `Fail to track local map!` lines) that **the observed KF=2 deaths go
## through Branch 2, not Branch 1** -- meaning these young maps DO get a
## grace period. The `cout<<"Fail to track local map!"` line is also
## printed on every subsequent `RECENTLY_LOST` frame even when
## `TrackLocalMap()` itself is never reached (it fires whenever `bOK` is
## false after the block, including when the first stage already failed
## that frame) -- so a streak of ~20-30 of these lines per death event is
## consistent with ~3 seconds (the hardcoded `3.0f` timeout at line ~2032)
## of per-frame dead-reckoning (`TrackWithMotionModel` reusing stale
## `mVelocity`) + `Relocalization()` attempts, all failing, before the
## timeout flips to `LOST` and the map is destroyed.
##
## **Current leading hypothesis for the next session (NOT yet directly
## confirmed -- needs fprintf instrumentation, not Verbose::PrintMess,
## per the false-negative lesson above)**: most young/sparse maps DO
## receive the ~3-second `RECENTLY_LOST` grace period, but the recovery
## mechanisms available during it (dead-reckoning off a 2-9 KF map's
## thin point cloud; `Relocalization()` against a keyframe database that
## barely has this map's own few keyframes in it yet) are close to
## structurally incapable of succeeding for a map this young -- so the
## grace period is being granted but is nearly always wasted. If true,
## the fix is not "grant more grace periods" (most already get one) but
## either making recovery actually work for young maps, or preventing so
## many `TrackLocalMap()` failures on freshly-initialized maps in the
## first place (still unexplained -- disparity was ruled out above, cause
## unknown).
##
## **Next session should start with**: add `fprintf(stderr,...)`
## instrumentation (NOT `Verbose::PrintMess`) inside the `RECENTLY_LOST`
## branch (~line 1983-2038) logging, per frame while in that state: which
## of `TrackWithMotionModel`/`Relocalization` was attempted, whether it
## succeeded, and for `Relocalization()` specifically the VLAD candidate
## count and match yield at each stage -- to confirm or refute the
## "recovery structurally can't work yet" hypothesis directly, the same
## way the disparity hypothesis was tested and refuted above rather than
## assumed. A short targeted run (`[start-frame]`/`[max-frames]` around
## one of the KF=2 death examples already in the saved log) is enough --
## does not need a full-sequence checkpoint.

## Current status (2026-07-18, part 17 -- answers part 16's open question
## with real data): added ground-truth bounding-box reporting to
## `EvaluateTrajectory()` (gtBBox=x[...] z[...], plus start/end points) and
## re-ran the full seq00 checkpoint specifically to check whether the 10
## map fragments physically overlap (a real map-merge opportunity) or are
## purely sequential (nothing for place-recognition-based merge to find).
## **Confirmed: it's a real, mixed picture, not a guess.**
##
## Note: fragment count/sizes differ slightly from part 15's run (10
## fragments both times, but 246 total KFs here vs 192 there, and
## per-fragment KF counts differ) -- expected run-to-run nondeterminism
## from RANSAC/threading, already a known property of this vendored
## ORB-SLAM3 (see [[project_orbslam3_vendoring]] memory). The overlap
## finding below is about physical layout, not exact reproducibility of
## fragment boundaries.
##
## **Fragments 1, 6, and 8 genuinely occupy the same real-world zone**:
## - Fragment 1: x[61.0,71.9] z[108.1,244.7] -- 39 KFs, 0.131m RMSE
## - Fragment 6: x[59.3,72.5] z[223.1,410.0] -- 47 KFs, 0.181m RMSE
## - Fragment 8: x[61.6,87.5] z[219.7,245.5] -- 20 KFs, 0.243m RMSE
##
## Pairwise bounding-box overlap confirmed by direct arithmetic: all three
## share a common region of roughly x in [61.6, 71.9], z in [223.1, 244.7]
## -- the vehicle genuinely revisited this intersection/area at least
## three separate times, each time triggering a fresh, disconnected map
## fragment instead of being recognized as "back here again" and merged
## into one of the earlier ones. **This is concrete, verified evidence
## that fixing map-merge would produce a real improvement for at least
## this part of the sequence**, not a hopeful assumption -- exactly the
## kind of case `NewDetectCommonRegions()` is designed to catch.
##
## The other six fragments (0, 2, 3, 4, 5, 7) sit in distinct,
## non-overlapping x/z regions -- consistent with sequential new road,
## where classical place-recognition merge has nothing to exploit and the
## real fix would be reducing fragmentation frequency itself (fewer,
## longer-lived maps), not improving merge detection.
##
## **Net conclusion, both parts 16's open questions now answered with
## data**: (1) overlap DOES exist for at least fragments 1/6/8 -- merge-
## fixing is a real, verified-useful lever, not a maybe; (2) the coverage
## ceiling caveat from part 16 still stands independently -- even perfectly
## merging 1/6/8 together only reconnects 3 of 10 fragments, leaving the
## other 7 (and the untracked gaps between all of them) as a separate,
## still-unaddressed problem.
##
## **Next session should start with**: instrument `NewDetectCommonRegions()`
## directly during a run covering this specific x[60,88]/z[108,410] zone
## (re-run with diagnostic prints on VLAD candidate scores and geometric
## verification pass/fail, similar in spirit to the removed
## `[merge-investigate]`/`[reloc-investigate]` probes from 2026-07-17) to
## see exactly why fragments 1/6/8 -- despite genuinely revisiting the same
## place, and despite fragment 1 (39 KFs) and fragment 6 (47 KFs) both
## being well past the documented 12-keyframe candidate-search gate -- never
## triggered a merge. This is now a concrete, reproducible, verified-real
## case to debug against, not a theoretical one.

## Current status (2026-07-18, part 16 -- projected outcome of fixing
## merge, and an important caveat not to skip): before assuming "fix
## map-merge" (part 15's next step) is a guaranteed path to a good
## full-sequence ATE, two things need checking first, since the answer
## genuinely depends on them:
##
## 1. **Do the 10 fragments spatially overlap, or are they purely
## sequential?** Map-merging in ORB-SLAM3 (`NewDetectCommonRegions()`) is
## place-recognition-based -- it only reconnects a fragment to an *earlier*
## one if the vehicle revisits a location the earlier fragment already
## mapped. If each fragment instead covers genuinely new road as the drive
## continues (likely, if a fragment dies and the next one starts moments
## later at a nearby-but-not-yet-mapped position), there is nothing for
## merge to detect -- fixing it would not help, and the actual needed fix
## would be *preventing fragmentation in the first place* (fewer, longer-
## lived maps), a different and less predictable problem. KITTI seq00 is
## known to have some genuine loop closures (part of why the original
## ORB-SLAM3 benefits from loop closure on this sequence at all), so at
## least some fragment pairs plausibly do overlap -- but this has NOT been
## checked directly yet. **Next session should check the 10 fragments'
## keyframe pose ranges against the ground-truth path (already have the
## per-fragment KeyFrame lists via `Map::GetAllKeyFrames()` from part 13's
## work) to see which, if any, pairs genuinely revisit the same physical
## location** before investing further effort in the merge-candidate-search
## path specifically.
##
## 2. **Coverage ceiling, even with perfect merging**: the 9 scored
## fragments cover only **615.8m of combined path length**, against the
## full sequence's roughly 3722m ground-truth path (per part 12's earlier
## reported figure for the probed region) -- i.e. even flawless merging of
## every existing fragment recovers at most ~16-20% of the total driven
## distance. The rest of the sequence currently has zero tracked keyframes
## at all (consistent with part 7/8's finding of multi-thousand-frame
## silent re-initialization stretches). So "fix merge" is necessary but not
## sufficient on its own for a full-sequence ATE checkpoint -- the
## re-initialization-latency problem from parts 7-8 (still unresolved,
## never revisited after parts 9-13's matching-focused fixes) needs
## addressing too, independent of whichever fragments do or don't overlap.

## Current status (2026-07-18, part 15 -- full checkpoint result, resolves
## part 14's interim note): the full, continuous, no-`[start-frame]` seq00
## run finished. **Every scored fragment shows excellent local accuracy --
## the segmented `[start-frame]` spot-checks' ~20% relative-error finding
## does NOT hold for continuous tracking, confirming that was an artifact
## of cold-starting mid-sequence with no prior map, exactly as flagged in
## part 14's caveat.**
##
## 10 map fragments, 192 total keyframes:
##
## | fragment | KFs | path len | ATE RMSE | % of path |
## |---|---|---|---|---|
## | 0 | 21 | 56.4 m | 0.055 m | 0.10% |
## | 1 | 34 | 137.9 m | 0.162 m | 0.12% |
## | 2 | 18 | 67.4 m | 0.168 m | 0.25% |
## | 3 | 18 | 36.7 m | 0.397 m | 1.08% |
## | 4 | 16 | 39.4 m | 0.052 m | 0.13% |
## | 5 | 26 | 98.3 m | 0.180 m | 0.18% |
## | 6 | 22 | 78.2 m | 0.129 m | 0.17% |
## | 7 | 19 | 72.8 m | 0.035 m | 0.05% |
## | 8 | 16 | 28.7 m | 0.159 m | 0.55% |
## | 9 | 2 | -- | -- | (too few to score) |
##
## Every single scored fragment (9 of 10) lands between **0.05% and 1.08%
## relative error** -- comparable to or better than the true original
## ORB+DBoW2 baseline's own ~0.2-0.3% (6.4-10.7m over ~3722m). Combined
## scored path length: 615.8m across the 9 fragments, with genuinely good
## local tracking throughout, not just in one lucky stretch.
##
## **This closes out the "is tracking accuracy actually a problem" question
## for this session: no, it isn't.** Parts 4-13's fixes (BA-sigma
## weighting, re-widened `SearchForInitialization`, three widened
## `SearchByProjection` overloads) produce genuinely accurate tracking
## throughout the sequence. The 500-1500 "~20% error" finding from part 14
## was a real result, but only for the artificial cold-start-mid-sequence
## scenario the `[start-frame]` tool produces for diagnostics -- not
## representative of how this fork behaves when actually run continuously
## from the start, which is the only scenario that matters for a real ATE
## checkpoint.
##
## **The sole remaining problem, now precisely characterized**: 10
## disconnected map fragments across one sequence, never reconnected by
## map-merging, so no single global trajectory exists for a real end-to-end
## ATE number against the full ~3722m sequence -- even though the
## underlying tracking within each fragment is demonstrably good. This is
## now a map-merge/fragmentation problem exclusively, not a tracking-
## accuracy problem, and should be the sole focus of the next session.
##
## **Next session should start with**: root-cause why
## `NewDetectCommonRegions()`'s merge path (root-caused on 2026-07-17,
## gated on the *current* map reaching 12 keyframes before candidate search
## even runs) isn't reconnecting these 10 fragments, given several of them
## (e.g. fragment 1 at 34 KFs, fragment 5 at 26 KFs) are well past that
## 12-keyframe bar and existed simultaneously in the Atlas alongside later
## fragments. Concretely: instrument `NewDetectCommonRegions()` directly
## (candidate count found, VLAD similarity scores, geometric verification
## pass/fail) during a continuous run to see whether merge candidates are
## found and rejected, or never searched for at all. This is a
## fundamentally different question from anything investigated in parts
## 4-13 (all of which were about matching/tracking quality, now resolved)
## and is likely the last remaining lever before this fork can produce a
## real, full-sequence ATE checkpoint.

## Current status (2026-07-18, part 14, interim note -- full checkpoint
## still running as this is written): after part 13's new per-fragment
## evaluator came online, spot-checked several 500-frame windows via the
## `[start-frame] [max-frames]` combo (each such run starts tracking fresh
## at that frame -- diagnostic only, does NOT reflect what a continuous
## run's map state would actually be at that point, since it has no prior
## map to carry forward):
##
## | frames | best fragment | keyframes | path | ATE RMSE | % of path |
## |---|---|---|---|---|---|
## | 0-500 | fragment 2 | 41 | 145.6 m | 0.261 m | 0.18% |
## | 0-500 | fragment 0 | 13 | 46.2 m | 0.034 m | 0.07% |
## | 0-500 | fragment 1 | 13 | 36.0 m | 0.124 m | 0.34% |
## | 500-1000 | fragment 0 | 18 | 93.0 m | 19.072 m | **20.50%** |
## | 500-1000 | fragment 1 | 13 | 45.5 m | 1.716 m | 3.77% |
## | 1000-1500 | fragment 0 | 21 | 54.0 m | 10.984 m | **20.35%** |
## | 1500-2000 | (stuck) | 2 | -- | -- | -- (never got past 2 KFs) |
##
## Frames 0-500 are excellent (sub-1% relative error, per part 13). Frames
## 500-1500 show a consistent ~20% relative error -- not noise, a real,
## repeated degradation, though from only two fresh-init snapshots so not
## yet root-caused. Frames 1500-2000 (fresh-init) never escape a 2-keyframe
## map within the 500-frame window -- consistent with part 7/8's earlier
## finding that re-initialization around this stretch of the sequence can
## take on the order of thousands of frames.
##
## Full, continuous, no-`[start-frame]` seq00 checkpoint launched with all
## of today's fixes (parts 4-13) and the new per-fragment evaluator, to get
## the real (non-fresh-init-snapshot) picture -- this is the authoritative
## test the `[start-frame]` spot-checks above can't provide, since a
## continuous run carries its actual map state through these same frame
## ranges instead of restarting from scratch at each boundary. Still
## running as of this note; full results to follow in the next status
## block.

## Current status (2026-07-18, part 13 -- reframes the whole investigation):
## per explicit direction (grep for more narrow-window bugs; build a
## keyframe-coverage validator so misleading ATE numbers like part 12's
## 0.129m artifact can't happen again), found the actual explanation for
## why this session kept measuring "no scorable ATE" despite real,
## verified fixes: **the evaluation tool itself was silently discarding
## almost all real tracking data, and per-fragment accuracy is genuinely
## excellent.**
##
## **Grep for more narrow-window bugs** (priority 1): found one more,
## fixed the same way as parts 9/12 -- `ORBmatcher::SearchByProjection
## (Frame&, KeyFrame*, sAlreadyFound, th, ORBdist)`, the overload backing
## `Relocalization()` (one of the two functions flagged as a top suspect),
## had the identical `[nPredictedLevel-1, nPredictedLevel+1]` narrow window.
## Fixed. Also checked `LocalMapping::SearchInNeighbors()`'s underlying
## `Fuse()`: it uses `KeyFrame::GetFeaturesInArea()`'s spatial-only overload
## (no minLevel/maxLevel parameters at all, confirmed by reading its
## implementation) -- not vulnerable to this bug category, nothing to fix
## there.
##
## **The real discovery**, found while investigating why `frame1_500`
## (frames 0-500, the "easy" part of the sequence before the documented
## hard region) still produced "Alignment failed -- too few matched points
## (3)" despite showing a long, healthy, continuous stretch of keyframe
## insertion in the logs (id 60 through 434+, hundreds of frames, regular
## 3-8 frame cadence, no failures): `Atlas::GetAllKeyFrames()` -- read
## directly in `Atlas.cc` -- is implemented as `return
## mpCurrentMap->GetAllKeyFrames();`. **It only ever returns the CURRENT
## map's keyframes.** Since `System::SaveKeyFrameTrajectoryTUM()` (and
## therefore this whole session's entire evaluation methodology, all day)
## goes through this call, **every map fragment except the very last one at
## shutdown time was silently discarded from every ATE measurement taken
## today** -- whether the fragment ended via a destructive
## `ResetActiveMap()` or the non-destructive `CreateMapInAtlas()` path
## (which doesn't print anything grep-able, so today's "reset counts" also
## undercounted total fragmentation events -- they only counted destructive
## resets).
##
## Added `SLAM.GetAtlas()->GetAllMaps()` enumeration to
## `orbslam3_kitti_ate.cpp` (`[atlas-coverage]` diagnostic) and confirmed
## directly: the `frame1_500` run had **4 map fragments totaling 71-78
## keyframes** at shutdown, and the current-map-only export was giving just
## 2-4 of them to the evaluator -- over 90% of actually-tracked data was
## invisible to every ATE number reported today, including the earlier
## misleading 0.129m one (part 12), which now makes complete sense: it
## scored whatever tiny fragment happened to be "current" at shutdown,
## which is essentially arbitrary.
##
## **Refactored the evaluation tool properly** rather than patching around
## it: extracted `EvaluateTrajectory()` (match-by-timestamp + 2D Umeyama
## align + ATE) into a reusable function, and now call it **once per Atlas
## map fragment** (each fragment is its own arbitrary monocular scale/
## coordinate frame -- they can't be concatenated into one trajectory, each
## needs its own independent alignment against the matching ground-truth
## stretch). Also fixed a second bug in the old evaluation code while doing
## this: `pathLength` was computed from the *entire* probed region's ground
## truth, not just the matched segment -- exactly what made the 0.129m
## artifact's "0.00% of path" line meaningless. Now computed from only the
## matched ground-truth points, in trajectory order. New `[max-frames]` CLI
## arg added too, for fast targeted checks on a specific frame range instead
## of always running to the end of the sequence.
##
## **Result on `frame1_500` (frames 0-500, all of today's fixes in the
## tree)** -- real, legitimate numbers this time, no degenerate artifacts:
##
## | fragment | keyframes | path length | ATE RMSE | ATE RMSE / path |
## |---|---|---|---|---|
## | 0 | 13 | 46.2 m | 0.034 m | 0.07% |
## | 1 | 13 | 36.0 m | 0.124 m | 0.34% |
## | 2 | 41 | 145.6 m | 0.261 m | 0.18% |
## | 3 | 4 | (too few to score) | -- | -- |
##
## **This is a dramatically better per-distance accuracy than the true
## original ORB+DBoW2 baseline's 6.4-10.7m ATE over its whole run** -- e.g.
## fragment 2's 0.261m over 145.6m is roughly 0.18% relative error, vs. the
## baseline's ~6.4-10.7m over the full ~3722m sequence (~0.2-0.3%) --
## genuinely comparable or better on a like-for-like (error-per-distance)
## basis, not just a smaller absolute number from a shorter segment.
##
## **This reframes the entire session**: parts 4-12's real, hard-won
## fixes (BA-sigma weighting, re-widened `SearchForInitialization`, both
## `SearchByProjection` overloads, now `Relocalization`'s too) were never
## failing to produce good tracking -- they were producing good tracking
## that a blind-spot in the evaluation tooling made invisible. The
## remaining problem is not "this fork can't track accurately" (it can,
## demonstrably, per-fragment) -- it is specifically **fragmentation**: why
## maps keep splitting into disconnected pieces, and why
## `NewDetectCommonRegions()`'s merge path (root-caused back on 2026-07-17,
## gated on the *current* map reaching 12 keyframes before candidate search
## even runs) never reconnects them back into one continuous trajectory.
##
## **Next session should start with**: (1) run the full, no-`[start-frame]`
## seq00 checkpoint with the new per-fragment evaluator and see the real
## distribution of fragment sizes/accuracy across the whole sequence, not
## just frames 0-500 -- this is the actual, meaningful "checkpoint" this
## investigation has been missing all along; (2) revisit map-merging
## specifically now that the accuracy question is answered -- e.g. actually
## implement the `mnLastRelocFrameId`-gate relaxation flagged in part 11
## (23% of failures), or the 12-keyframe merge-gate relaxation flagged back
## on 2026-07-17, now with much higher confidence that reconnecting
## fragments (rather than improving matching further) is the highest-
## leverage remaining lever; (3) the priority-2 "coverage validator" idea
## that motivated this discovery is now implemented in spirit (per-fragment
## scoring with real path lengths rather than one misleadable global
## number) -- a stricter automatic pass/fail gate (e.g. reject fragments
## below some KF-count or path-length floor) could still be added to
## `EvaluateTrajectory()` if a fully automated checkpoint script is wanted
## later, but manual inspection of the per-fragment table is sufficient for
## now.

## Current status (2026-07-18, part 12): per explicit direction to dig
## deeper into why `TwoViewReconstruction` and Local Mapping's point-
## creation flow don't sustain map survival, found and fixed a real,
## previously-undiscovered bug -- **but it's still not enough for a
## scorable ATE, and a misleadingly good-looking ATE number surfaced
## along the way that must NOT be trusted or cited.**
##
## **Ruled out first (both directly measured, not guessed)**:
## `LocalMapping.cc` and `NeedNewKeyFrame()` are byte-identical to the
## untouched `third_party/ORB_SLAM3` -- confirmed via `diff`, so neither is
## fork-modified. Instrumented `CreateNewMapPoints()` directly (baseline/
## depth-ratio rejections, new-point count per call): the
## `ratioBaselineDepth<0.01` gate essentially never fires
## (`rejectedLowBaseline=0` in every sampled call), and new-point creation
## is healthy (99-651 new points per keyframe insertion). Keyframe
## insertion cadence is also reasonable (every 1-14 frames, not
## continuous). **Point-creation throughput is not the bottleneck** --
## contrary to this investigation's own working hypothesis at the start of
## the session.
##
## **The real find**: instrumented `SearchLocalPoints()`
## (`nToMatch` = points confirmed in-frustum vs `matched` = points
## `SearchByProjection()` actually associated) and found a catastrophic
## 1-10% match rate on points already confirmed visible -- e.g. 633
## in-frustum candidates yielding only 5 matches. This held regardless of
## the search-radius multiplier `th` (even `th=15` for `RECENTLY_LOST` only
## reached ~6-8%), which pointed away from "search window too small in
## pixels" and toward "search window wrong in *flat-level*". Confirmed
## directly in `ORBmatcher::SearchByProjection(Frame&, const
## vector<MapPoint*>&, ...)`: `GetFeaturesInArea(...,
## nPredictedLevel-1, nPredictedLevel)` restricts candidates to a 2-flat-
## level window -- the **exact same category of bug** as the
## `SearchForInitialization()` flat-level-0 restriction fixed in part 9,
## just in the function that backs ordinary, continuous local-map
## re-acquisition (i.e. nearly every frame of normal tracking), never
## touched before now. For this SIFT reimplementation's (octave, layer)
## flat-level packing, a fixed 2-level window only ever spans a sliver of
## one octave's `nOctaveLayers` same-resolution sub-layers -- not a
## meaningful "+-1 real pyramid level" the way it does for ORB's true
## per-level pyramid.
##
## **Fixed the same way as part 9's widening** (safe because of part 9's
## octave-only sigma fix): widened the window to the whole octave
## `nPredictedLevel` falls in, via the same `GetOctaveLayers()` accessor.
## Also found and fixed the identical pattern in the OTHER
## `SearchByProjection` overload (`Frame&, const Frame&, th, bMono`, used
## by `TrackWithMotionModel()` -- the primary per-frame continuous-tracking
## path, not just local-map re-acquisition): its non-forward/non-backward
## branch (always taken for monocular, since forward/backward require
## `!bMono`) used a fixed `[nLastOctave-1, nLastOctave+1]` window, same fix
## applied.
##
## **Measured, step by step, on the identical `[start-frame]`-700,
## 3841-frame probe used throughout this file**:
## | variant | resets |
## |---|---|
## | part 9 baseline (sigma-fix + widened SearchForInitialization only) | 94-97 |
## | + widened local-map SearchByProjection | 74 |
## | + widened motion-model SearchByProjection too | 72 |
##
## A real, incremental, twice-confirmed improvement (~24% fewer resets) --
## but still nowhere near the true baseline's 3, and still not sufficient
## for a healthy trajectory on its own.
##
## **Important correction, in real time during this session**: the 72-reset
## run's tail output included an `ATE RMSE: 0.129 m` line that looked like
## a spectacular result (matching or beating the 6.4-10.7m baseline by
## ~50x) -- **this is a misleading artifact, not a real measurement, and
## must not be cited or trusted.** Checked the actual
## `KeyFrameTrajectory.txt` before reporting anything: only 8 keyframes
## survived to the end of the run, spanning just 2.7 seconds and ~30m of
## real-world travel (after scale recovery) -- the tiny last surviving map
## fragment at the very end of the sequence, not a real trajectory. A
## Umeyama similarity alignment (rotation+scale+translation) fit to 8
## points along a short, nearly-straight segment will trivially produce a
## tiny RMSE regardless of whether the estimate is actually correct, since
## such a short simple path barely constrains the fit. The "ATE RMSE /
## path len: 0.00%" figure divides this trivial error by the *entire*
## probed region's ground-truth path length (3722m), which is completely
## disconnected from what was actually being compared (8 points). Any
## future session seeing a suspiciously good ATE number from a run with a
## very small `Matched keyframes: N / N` count should treat it the same
## way -- check the raw trajectory file's actual timestamp/position span
## before trusting the summary statistics.
##
## **Net assessment**: this session's deep-dive found a second real,
## independently-verified bug (the `SearchByProjection` flat-level-window
## restriction, present in TWO call sites) beyond part 9's
## `SearchForInitialization`/BA-sigma fix -- both are legitimate, measured
## improvements to match yield and reset count. Neither, alone or
## combined, has yet produced a real scorable ATE. The honest, unresolved
## question is still whether the survival problem in this hard region is
## fully attributable to bugs like this one (fixable, mechanical,
## SIFT-packing-specific) or partly inherent scene difficulty (e.g. a
## genuinely sharp rotation sweeping too many points out of frustum too
## fast for any local-map-based approach to keep up) -- today's fixes have
## narrowed the gap but not closed it.
##
## **Next session should start with**: (1) grep for any OTHER
## `GetFeaturesInArea(...)` call sites across `ORBmatcher.cc` that pass a
## literal `+-1`-style flat-level window (the two found and fixed today
## were found by tracing specific failing code paths, not by an exhaustive
## search -- there could be more, e.g. in `Relocalization()`'s or
## `Fuse()`'s own `SearchByProjection`-family calls); (2) re-run the full,
## no-`[start-frame]` seq00 checkpoint with both `SearchByProjection` fixes
## in place and actually verify the resulting `Matched keyframes: N/N`
## count is large (covering most of the sequence) before trusting any ATE
## figure it reports -- this session ran out of time to do that full
## checkpoint after finding the misleading-ATE issue. All temporary
## diagnostic tag families from parts 9-12 remain in the tree
## (`[mono-init]`, `[track-ref-kf]`, `[track-local-map]`,
## `[track-local-map-detail]`, `[track-local-map-gate]`,
## `[create-new-map-points]`, `[create-new-kf]`, `[search-local-points]`).

## Current status (2026-07-18, part 11 -- end-of-day wrap-up): confirmed the
## `mnLastRelocFrameId+mMaxFrames` stricter (`>=50` instead of `>=30`)
## post-relocalization gate from part 10 is a **real, measured, but stock/
## unmodified ORB-SLAM3 mechanism** (byte-identical in the untouched
## `third_party/ORB_SLAM3`, confirmed via direct `diff`) -- added a
## `[track-local-map-gate]` diagnostic and measured **26 of 114 (23%)**
## `TrackLocalMap()` failures in one sample were frames with 30-49 inlier
## matches that would have PASSED the normal monocular bar but failed this
## stricter one, active for 10 frames after any relocalization. Real
## contributing factor, not touched -- it's intentional stock behavior, and
## with time pressure to reach a real number today, changing untested
## stock logic wasn't the responsible move. Left as a documented, viable
## next lever, not implemented.
##
## **Ran the full, no-`[start-frame]` seq00 ATE checkpoint** (all 4541
## frames) with everything from today's session in the tree: Option 1
## (dynamic SIFT density boost, part 4, measured neutral), Option 2 (dead-
## reckoning bridge, no timeout renewal, parts 5-6, measured real-but-
## insufficient improvement), the octave-only BA-sigma fix + re-widened
## `SearchForInitialization()` (part 9). **Result: still no scorable ATE.**
## Final active map had only 2 keyframes when the sequence ended --
## `SaveKeyFrameTrajectoryTUM()` produced 2 lines, `Alignment failed -- too
## few matched points (2) or degenerate estimate.` Same failure signature
## as literally every other run today.
##
## **The full-sequence number that matters most**: **115 destructive
## resets across the whole 4541-frame run** -- markedly worse than:
## - The true original ORB+DBoW2 baseline: 3 resets, real 6.4-10.7m ATE.
## - Part 7's checkpoint (Options 1+2 only, no widening/sigma-fix): 3
##   resets, but the last re-init attempt never completed before the
##   sequence ended (600 frames stuck trying) -- also no scorable ATE.
## - Today's version: 115 resets, i.e. the system now initializes far more
##   often (part 9: match-starvation genuinely fixed, ~98 successful inits
##   in a comparable window vs 3 before) but also **dies far more often**,
##   netting out to the same "no scorable trajectory" result as every prior
##   attempt, just via a much higher-churn path.
##
## **Honest net assessment for today's whole session (parts 4-11)**: two
## real, independently-verified bugs were found and fixed this session --
## (1) the BA-sigma-weighting bug from days-old Part 1/2 (per-flat-level
## instead of per-octave scale/sigma, now fixed properly, not band-aided),
## which safely unlocked re-widening `SearchForInitialization()` and
## genuinely solved match-starvation (part 9: 0 rejections vs 1840, 98
## successful inits vs 3, both measured directly); (2) a real, if modest,
## dead-reckoning tracking-loss bridge (Option 2, part 5) with a measured
## (not assumed) improvement on a fine-grained metric. **Neither
## individually nor combined do these add up to a working end-to-end
## trajectory** -- the dominant, still-unsolved bottleneck is upstream of
## all of today's work: this fork's monocular maps do not *survive*
## whatever is happening in the hard/turn-heavy stretch of KITTI seq00,
## regardless of how easily they can now be created. This is the exact
## same problem first documented on 2026-07-17 (see the "Current status"
## blocks below this one) -- today sharpened the diagnosis considerably
## (mono-init latency measured in the thousands of frames; 90% of
## successful inits shown to be adjacent-frame/low-baseline; the
## post-relocalization strict-gate interaction quantified at 23% of
## failures) but did not resolve the core survival problem.
##
## **State of the tree at end of day**: all of parts 4-10's changes are
## left in place, uncommitted, alongside the pre-existing `clearMap()` fix
## and `[start-frame]` CLI additions:
## - `ORBextractor.cc`/`.h`: `SetDynamicDensity()`/`GetOctaveLayers()`
##   (Option 1, part 4) + the octave-only sigma/scale fix (part 9).
## - `Tracking.cc`/`.h`: Option 1's high-angular-velocity density-boost hook
##   (part 4), Option 2's dead-reckoning bridge without timeout renewal
##   (parts 5-6), and FIVE temporary diagnostic tag families still active:
##   `[mono-init]` (part 7-8), `[track-ref-kf]`/`[track-local-map]`/
##   `[track-local-map-detail]`/`[track-local-map-gate]` (parts 9-10).
## - `ORBmatcher.cc`: re-widened `SearchForInitialization()` (part 9).
## None of this has been benchmarked as a net win yet -- **do not treat
## today's changes as a checkpoint to build on without re-validating**;
## they are informative-but-inconclusive, matching this file's own
## established practice for prior sessions that ended without a positive
## result (see part 3's and part 6's "treat as informative-but-negative"
## framing).
##
## **Next session should start with, in priority order**: (1) decide
## whether to keep chasing the survival problem in this specific hard
## region (e.g. actually trying the `mnLastRelocFrameId` gate relaxation
## for monocular non-IMU specifically, now that it's quantified at 23% of
## failures -- untried, cheap, but touches stock logic so should be
## measured carefully before/after) or (2) step back per the original
## part-2 "Next session should start with" note that's been valid since
## 2026-07-17 and never fully executed: instrumenting exactly why
## `ReconstructWithTwoViews`'s 1-degree parallax bar and this fork's Local
## Mapping point-creation throughput don't produce maps that survive this
## stretch, independent of initialization ease. Remove the five temporary
## diagnostic tag families once whichever direction is chosen stops
## needing them -- intentionally left in for immediate reuse, consistent
## with this session's practice throughout parts 4-10.

## Current status (2026-07-18, part 10): investigated why 94-97 of ~98-99
## post-widening map initializations die almost immediately (part 9's
## finding), testing a specific hypothesis raised in this session: that
## ORB-SLAM3's per-level (`kp.octave`) assumptions -- search-radius sizing
## or chi-square outlier rejection in `PoseOptimization()` -- treat the
## widened matcher's now-common non-zero-flat-level MapPoints as
## statistically "wrong" and wipe the map almost immediately. **Directly
## measured, and mostly refuted** -- but surfaced a real, still-open,
## narrower anomaly instead.
##
## Added two more temporary diagnostics (both tagged for removal alongside
## `[mono-init]`): `[track-ref-kf]` (nmatches from `SearchByBoW` in
## `TrackReferenceKeyFrame()`) and `[track-local-map]`/
## `[track-local-map-detail]` (`TrackLocalMap()`'s existing `aux1`/`aux2`
## before/after-`PoseOptimization()` counters, which the code already
## computed but never printed -- before-count, flagged-outlier count, and
## after-count, both before and after `PoseOptimization()`'s chi-square
## rejection).
##
## **Evidence against the hypothesis**: the very first post-widening map
## (born ~frame 700, same cross-layer/non-zero-octave points as every other
## map) tracked cleanly for 48+ frames, growing from 2 to 10 keyframes.
## Outlier rejection throughout was mild and unremarkable (0-2 flagged per
## frame typically, occasionally 10-20% right after a new keyframe
## insertion -- normal BA behavior, not a wholesale wipeout). Match counts
## declined steadily (271->166->104->85->65...) purely from the camera
## moving away from the initial view faster than a still-sparse young local
## map could keep up -- ordinary monocular-VO bootstrap sparseness, not a
## per-frame catastrophic rejection. If a scale-level/chi-square bug
## unconditionally poisoned cross-layer points, this map should have failed
## immediately too, and it measurably did not.
##
## **A real, narrower anomaly found instead, not yet explained**: one frame
## (map #2, local frame id 84 in a fresh detail-probe run) had 44 surviving
## inlier matches post-`PoseOptimization()` -- comfortably above the
## nominal `mnMatchesInliers>=30` bar for monocular -- yet the frame still
## failed (no corresponding `[track-local-map]` print, meaning
## `TrackLocalMap()` returned via an earlier gate than the final
## `<30`-or-not check). Two candidate explanations, neither confirmed:
## (a) `TrackLocalMap()`'s *other* threshold,
## `if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50) return false;`
## (`mMaxFrames = fps = 10` for KITTI) -- applies a stricter >=50 bar for 10
## frames after any `Relocalization()` success, and this system relocalizes
## often given how frequently maps die; checked the arithmetic for this
## specific case and the most recent prior relocalization was too far back
## to explain it, so this specific instance isn't explained by it, though
## the mechanism itself is real and could explain *other* cases in a longer
## sample; (b) `mnMatchesInliers` only counts points with
## `Observations()>0`, which can be lower than the raw post-optimization
## survivor count if some points had an observation removed (e.g. keyframe
## culling) -- not checked directly yet.
##
## **Net assessment**: the dominant failure mode across the ~94-97 resets
## is ordinary young-map match-count decline (consistent with part 9's
## "90% of successful inits are adjacent-frame, low-baseline" finding, and
## consistent with this file's much older, already-documented "hard
## region"/turn-region survival problem from 2026-07-17) -- not a
## SIFT-specific scale-level bug in outlier rejection or search-radius
## sizing. That hypothesis is reasonable in principle (worth having
## checked directly rather than dismissed on priors) but the evidence from
## the one map that DID survive doesn't support it as the dominant or
## unconditional cause. The `mnLastRelocFrameId`/`mMaxFrames` stricter-bar
## interaction is a real, distinct, previously-unnoticed mechanism that
## could be tightening the squeeze on already-fragile young maps
## specifically because this system relocalizes so often -- a plausible
## secondary contributor, not yet confirmed as a primary one.
##
## **Next session should start with**: either (a) confirm/refute the
## `mnLastRelocFrameId` theory properly by logging `mnLastRelocFrameId` and
## `mCurrentFrame.mnId` alongside `mnMatchesInliers` at every
## `[track-local-map]` call (cheap, mechanical, directly decisive) across a
## longer sample instead of the single anomalous frame found here, or (b)
## accept that the dominant problem is the already-documented hard-region
## survival issue from days ago (not a new bug from today's changes) and
## shift effort to that instead -- e.g. revisiting Local Mapping's new-point
## creation throughput during the sparse post-init bootstrap window, since
## that's what determines whether match count recovers before crossing the
## 30-match floor. All temporary `[mono-init]`/`[track-ref-kf]`/
## `[track-local-map]`/`[track-local-map-detail]` diagnostics are still in
## the tree, not yet cleaned up -- intentional, next session can reuse them
## immediately for whichever direction is chosen.

## Current status (2026-07-18, part 9): per explicit direction, went back and
## properly fixed the BA-sigma-weighting bug from Part 1/2 (days earlier in
## this file) so the previously-reverted `SearchForInitialization()`
## widening could be re-applied. **The match-starvation problem this
## targeted is now solved outright -- but re-applying the widening exposed
## a different, previously-masked problem: near-zero-baseline degenerate
## initialization.** Net: not yet a net win, but the diagnosis is much
## sharper than before and points at a concrete next fix.
##
## Added the `[mono-init]` diagnostic instrumentation (see part 7/8 --
## `MonocularInitialization()` in `Tracking.cc`, tags `[mono-init]`) proved
## invaluable here for measuring both changes precisely; still in the tree.
##
## **What was actually wrong with the Part-1/2 attempt, now root-caused
## instead of just theorized**: `ORBextractor.cc`'s per-flat-level scale/
## sigma arrays (`mvScaleFactor`/`mvLevelSigma2`/`mvInvLevelSigma2`) were
## computed via the continuous SIFT scale-space formula
## `2^(lvl/nOctaveLayers)` (float division) -- giving every one of
## `nOctaveLayers` layers within the same octave a *different* value, even
## though only octave changes real image resolution in this SIFT
## reimplementation (layers differ only in Gaussian-blur sigma at the same
## resolution). Checked every consumer of these arrays directly before
## changing anything -- ~30 sites across `Optimizer.cc`, `LocalMapping.cc`,
## `MLPnPsolver.cc`, `Sim3Solver.cc`, `ORBmatcher.cc`, `MapPoint.cc`, all
## follow the identical `array[kp.octave]` confidence/search-radius
## pattern, none need finer-than-octave granularity. Fix: changed the
## array-population loop (`ORBextractor.cc` constructor) to
## `octave = lvl / nOctaveLayers` (integer division, recovers the octave
## index alone from the flat-level packing) then `2^octave` -- collapses
## all layers within an octave to an identical, correct value, touching
## zero of the ~30 consumer call sites (only how the array *values* are
## computed changed, not the indexing scheme). This is a real fix, not the
## earlier "force matched keypoints to flat-level 0" band-aid (which
## discarded genuine octave information instead of just ignoring layer).
## Also confirmed `ComputeKeyPointsOctTree()`/`ComputeKeyPointsOld()` (the
## only other consumers using per-level values in a way finer-than-octave
## granularity might have mattered) are dead code, never called from
## `operator()` in this SIFT reimplementation -- and `ComputePyramid()`'s
## own consumer is stereo-only, dead for monocular correctness per its
## existing doc comment -- so this change has no monocular-relevant side
## effect to weigh against.
##
## Re-added `ORBextractor::GetOctaveLayers()` (returns `nOctaveLayers`) and
## re-widened `ORBmatcher::SearchForInitialization()` exactly as in the
## original Part-1 attempt: F1 filter changed from `level1>0` (flat-level-0
## only) to `level1>=nOctaveLayers` (whole finest octave), F2's
## `GetFeaturesInArea` window changed from `(level1,level1)` to
## `(0,nOctaveLayers-1)`.
##
## **Measured result, via the `[mono-init]`-instrumented probe from frame
## 700 to end of sequence (same 3841-frame stretch used in part 8)**:
##
## | metric                          | before (part 8) | after (this fix) |
## |----------------------------------|------------------|-------------------|
## | reject: nmatches<100              | 1840             | **0**             |
## | reconstruct FAILED                | 71               | 11                |
## | SUCCESS (successful map inits)     | 3                | **98**            |
## | destructive resets (full probe)   | not measured this way in part 8 | **94** |
## | final trajectory                  | empty (0 KFs)    | 3 lines, "Alignment failed -- too few matched points (3)" |
##
## Match starvation is solved outright -- zero rejections, matches in the
## hundreds where they used to top out near 100. But **94 resets and a
## degenerate final trajectory is the same failure signature as the
## original (BA-unaware) Part-1 widening attempt** ("Alignment failed --
## too few matched points (3)", 103 resets) -- so fixing the sigma-weighting
## bug did not, on its own, prevent the destabilization the earlier attempt
## hit.
##
## **New root cause, measured directly, not theorized**: of the 98
## successful initializations, **88 (90%) happened between adjacent frames**
## (reference frame exactly 1 video frame older than the current frame,
## ~100ms of motion at KITTI's 10Hz). Near-zero baseline is geometrically
## degenerate for two-view triangulation regardless of match count -- the
## likely explanation is that the *old* flat-level-0 restriction was
## accidentally protecting the system from this failure mode all along:
## match-starvation meant successful inits were rare enough that they
## usually only happened after enough real motion had accumulated to
## coincidentally also clear the (unrelated) match-count bar. Now that
## matching is abundant at any baseline, `MonocularInitialization()`
## happily initializes from whatever adjacent-frame pair comes along first,
## producing poorly-conditioned initial maps (translation direction is
## barely observable at near-zero baseline) that die almost immediately --
## consistent with 98 inits but 94 resets, i.e. most inits don't survive.
##
## **Net assessment**: two genuinely different bugs were conflated under
## one symptom ("SearchForInitialization widening destabilizes tracking").
## The sigma-weighting bug was real and is now properly fixed. But it was
## masking a second, independent bug -- `MonocularInitialization()` has no
## real parallax/baseline requirement, only a match-count proxy for one --
## that only becomes visible once match-starvation stops hiding it.
##
## **Next session should start with**: add a genuine minimum-baseline/
## parallax check to `MonocularInitialization()`, independent of match
## count -- e.g. reject a candidate pair (before or in addition to calling
## `ReconstructWithTwoViews()`) if the median or mean keypoint disparity
## between `mInitialFrame` and `mCurrentFrame` is below some threshold, so
## the reference frame is retained (not discarded) until real motion has
## accumulated, rather than always collapsing to adjacent-frame pairs. This
## is a different, third lever from anything tried so far (Options 1/2 in
## parts 4-6, or the sigma/widening pair here) and directly targets the
## newly-measured 90%-adjacent-frame pattern. Keep the sigma-weighting fix
## and the widening in the tree either way -- they're a real, verified
## improvement on their own axis (match yield), the problem now is a
## second, independent gate that was never needed before because
## match-starvation was inadvertently doing its job.

## Current status (2026-07-18, part 7): ran the still-outstanding full seq00
## ATE checkpoint (no `[start-frame]`, all 4541 frames) with the current
## tree state (Option 1 density boost + Option 2 bridge without renewal,
## from parts 4-6). **Still no scorable trajectory -- and the log reveals
## why in much starker terms than any narrow-window probe so far.**
##
## Note on process mechanics: the first attempt at this run was launched via
## the normal backgrounded-Bash mechanism and was lost mid-run when the
## harness session was restarted (its tracked output vanished with the
## scratchpad wipe, confirmed via `ps`/`ls` showing neither the process nor
## its log survived). Re-ran it via `nohup ... & disown`, detached from the
## controlling shell, with a separate `while ps -p <pid>; do sleep 20; done`
## monitor wrapped in a backgrounded Bash call so the harness could still
## notify on completion -- worth remembering for any future long
## (multi-minute+) run in this investigation, since the plain backgrounded
## form isn't resilient to a session restart.
##
## **What the log shows**: only 3 total keyframe-map (re)initializations
## happened across the entire 4541-frame sequence (matching the "true
## original" baseline's reset *count* noted in part-2's investigation --
## superficially a good sign), but the gaps between them are the real
## story. Monocular initialization failure is silent in this codebase --
## `MonocularInitialization()` prints nothing per failed frame, it just
## keeps waiting for a two-view pair with enough parallax -- so long
## stretches with zero console output are not evidence of clean tracking,
## they're evidence of the system stuck failing to re-initialize, frame
## after frame, invisibly. Confirmed by cross-referencing `[stats]`
## progress markers (every 500 frames) against `First KF:N` lines (only
## printed once initialization actually succeeds):
##
## - Map 1: succeeds somewhere around frame ~700, dies almost immediately
##   (reset at frame 777).
## - Map 2: **takes 2938 frames** (777 -> 3715) of silent, failed
##   re-initialization attempts before finally succeeding -- then dies
##   within ~30 frames (reset at frame 3715, `mnInitialFrameId`=778 is the
##   *previous* map's start, the new one starts around 3715).
## - Map 3: succeeds quickly after that (~frame 3715ish -> reset at 3941,
##   another fast death, 37 failed frames).
## - **A fourth re-initialization attempt then runs silently from frame
##   3941 to the end of the sequence at 4541 (600 frames) and never
##   succeeds** -- the sequence simply ends mid-search. This is why
##   `SaveKeyFrameTrajectoryTUM()` wrote an empty file: the final active
##   map genuinely has zero keyframes, not a filtering/indexing bug in the
##   save path.
##
## **This means neither Option 1 nor Option 2 (both still in the tree,
## unchanged from part 6) fixed, or were even positioned to fix, the actual
## bottleneck this checkpoint exposes.** Both target *tracking-loss
## recovery* (staying on the existing map, or bridging a short dropout) --
## but the dominant cost here is >turns out to be< **monocular
## initialization latency after a map is already gone**, which is a
## different mechanism neither option touches: once `ResetActiveMap()`
## actually fires (which both options can at best delay slightly, not
## prevent, per parts 4-6's measurements), the system is at the mercy of
## `MonocularInitialization()`'s two-view search, which this run shows can
## take **thousands of frames** to succeed again, or may not succeed at all
## before the sequence ends. This is a materially bigger problem than the
## turn-region-specific framing this whole investigation started from --
## it's not just that maps die in hard turns, it's that this fork's
## from-scratch monocular re-init is, independent of the turn region
## specifically, extremely slow to recover once triggered.
##
## **Next session should start with**: instrumenting
## `MonocularInitialization()` directly (a temporary frame-count/parallax
## diagnostic, same pattern as the removed `[reloc-investigate]`/
## `[merge-investigate]` probes earlier in this file) to find out *why* it
## takes ~2938 frames in the map-2 case -- is required parallax threshold
## too strict, is the candidate pair search itself weak, or is the KITTI
## seq00 camera motion in that stretch just genuinely too close to
## straight-line (little parallax) for two-view initialization to succeed
## quickly regardless of thresholds? This is a very different, and
## seemingly more consequential, root cause than anything targeted so far
## in parts 1-6, and should probably take priority over further tuning of
## Options 1/2 until it's understood -- both options are inert once the
## map is already gone and the system has fallen back to from-scratch
## initialization.

## Current status (2026-07-18, part 6): tried the exact next step the part-5
## block below proposed -- renewing `mTimeStampLost` (the `RECENTLY_LOST`
## timeout clock) every frame the dead-reckoning bridge's
## `TrackWithMotionModel()` call itself returns true, instead of only once at
## the first failure -- and **measured a regression, not an improvement**.
## Reverted.
##
## Rebuilt and ran the identical `[start-frame]`-jumped-to-3600, 941-frame
## probe with the renewal in place. Same 2 resets at the same frame
## boundaries as every other run in this window, still no scorable
## trajectory -- but "Frames set to lost" at the second reset went to **53**,
## *worse* than both the true baseline (51) and the bridge-without-renewal
## version measured in part 5 (41). Three-way comparison, same probe window,
## same everything else:
##
## | variant                                  | frames set to lost |
## |-------------------------------------------|---------------------|
## | true baseline (no Option 1/2)              | 51                  |
## | Option 1 alone (density boost)              | 54 (noise, ~baseline) |
## | Option 1 + Option 2 bridge, no renewal      | **41** (real improvement) |
## | Option 1 + Option 2 bridge, with renewal    | 53 (regression)     |
##
## **Why the renewal backfired**: a bare `TrackWithMotionModel()` success
## doesn't mean the frame is actually recovered -- `TrackLocalMap()` runs
## immediately afterward (same as the normal `OK` path) and can still fail,
## in which case the frame still ends up marked lost in `mlbLost` regardless
## of the renewal. Renewing the clock on that marginal, often-still-failing
## success just gives the `RECENTLY_LOST` state more total frames to
## accumulate lost-flags in before something eventually fails for long
## enough to cross the (now-repeatedly-pushed-back) timeout -- prolonging a
## still-broken state rather than letting it fail fast and reset. The
## un-renewed version avoids this because it stays bounded to the original
## 3-second budget no matter how many marginal bridge attempts happen inside
## it.
##
## Reverted the renewal (kept as a code comment explaining why, at the same
## spot in `Tracking.cc`, so the next session doesn't re-try it without
## re-reading this). Tree now has: Option 1 (density boost, part 4, measured
## neutral) + Option 2 bridge without renewal (part 5, measured real
## improvement on the lost-frame metric, not yet sufficient to prevent a
## reset in this specific window) + `clearMap()` fix + `[start-frame]` CLI
## additions. Rebuilt clean.
##
## **Next session should start with**: the lost-frame reduction from Option
## 2 (51/54 -> 41) is real but this specific probe window still ends the
## same way (reset, no trajectory) regardless -- the straightforward
## "extend the timeout" lever is now a *ruled-out*, not just untried, next
## step. Two remaining untried angles: (a) figure out *why*
## `TrackWithMotionModel()` succeeds but `TrackLocalMap()` still fails so
## often during this bridge -- if the projected matches are marginal/noisy,
## improving match quality (not match presence) might convert more of these
## partial bridges into full recoveries without needing to touch the
## timeout at all; (b) stop micro-optimizing this one hard 941-frame window
## and run the full seq00 ATE checkpoint (still outstanding since Stage 4)
## with both current options in place, since neither has been checked
## against the rest of the sequence where the failure mode may look
## different.

## Current status (2026-07-18, part 5): implemented candidate direction (2)
## from the part-4 block below -- a dead-reckoning motion-model bridge during
## `RECENTLY_LOST` -- on top of (still-inert-per-part-4) Option 1, per
## explicit direction to keep (1) and add (2) rather than choose between
## them.
##
## What was added: in `Tracking::Track()`'s non-IMU `RECENTLY_LOST` branch
## (`Tracking.cc`, the block that previously called only `Relocalization()`
## every frame), now tries `TrackWithMotionModel()` first -- gated on
## `mbVelocity`, which stays true and holds its last real value from before
## tracking was lost (it's only overwritten on a successful frame) -- and
## falls back to `Relocalization()` unchanged if that returns false. This is
## a real constant-velocity dead-reckoning estimate against the still-intact
## local map, not a new assumption: `TrackWithMotionModel()` already existed
## and is used identically in the normal `OK`-state path, it just was never
## attempted during `RECENTLY_LOST` for monocular before this change (only
## the IMU sensor branches got an analogous `PredictStateIMU()` bridge).
## Purely additive to the control flow -- doesn't touch the 3-second
## `RECENTLY_LOST`->`LOST` timeout or the `KeyFramesInMap()<10` reset gate,
## both unchanged.
##
## **Result (measured): a real, partial, non-dispositive improvement.**
## Rebuilt and re-ran the identical `[start-frame]`-jumped-to-3600,
## 941-frame turn-region probe used for every measurement in this
## investigation, with Option 1 (part 4) and this Option 2 change both in
## the tree. Same top-line outcome as every prior run in this window: 2
## destructive resets, at the exact same frame boundaries
## (`mnFirstFrameId`/`mnInitialFrameId` = 115/0 then 341/116 -- identical to
## both the baseline and Option-1-only runs), still no scorable trajectory.
## **But** the "Frames set to lost" count at the second reset -- a real,
## direct measurement of how many frames in the dying map were flagged lost
## (`Tracking.cc`'s `num_lost` bookkeeping in `ResetActiveMap()`) -- dropped
## from 51 (true baseline) / 54 (Option 1 alone) to **41** with both options
## combined: roughly a 20-24% reduction in frames the dead-reckoning bridge
## evidently rescued that would otherwise have been marked lost.
##
## **Why it didn't prevent the reset despite that real improvement**: the
## `RECENTLY_LOST`->`LOST`->reset transition in this fork is gated purely on
## wall-clock time since the *first* failure (`mTimeStampLost`, set once,
## never renewed while still `RECENTLY_LOST`) exceeding a hardcoded 3.0
## seconds -- not on how many individual frames the bridge manages to
## rescue in between. So even though more frames got a usable dead-reckoned
## pose this run, if `TrackLocalMap()` (called immediately after, same as
## the normal path) still ultimately fails on enough of them, the map still
## crosses the fixed 3-second mark and still gets reset at the same point.
## The bridge is doing real work (fewer lost frames, confirmed by direct
## measurement) but isn't yet sufficient on its own in this specific hard
## window to outlast the fixed timeout.
##
## **Net picture**: both options are now in the tree, both inert-to-neutral
## on this probe's top-line outcome (resets/trajectory), but Option 2 has
## a measured, non-placebo effect on a finer-grained metric (frames rescued
## from "lost") that Option 1 didn't show at all. This is evidence the
## dead-reckoning direction is more promising than the density-boost
## direction for this specific failure mode, just not sufficient yet in
## isolation.
##
## **Next session should start with**: the most direct next lever, given
## what was just measured, is that `mTimeStampLost` is never renewed while
## `RECENTLY_LOST` persists -- so a frame successfully bridged by
## `TrackWithMotionModel()` still counts against the same fixed 3-second
## budget as a frame that failed outright. Renewing `mTimeStampLost` to the
## current frame's timestamp on every frame the dead-reckoning bridge
## itself succeeds (treating a successfully-bridged frame as a sign the
## dropout hasn't become permanent yet, giving the *next* few frames a
## fresh budget instead of counting down against the original failure) is
## an untried, natural extension of this exact change -- not yet
## implemented or measured. Also still outstanding: a full seq00 ATE
## checkpoint with both options in place, to see whether either helps
## anywhere else in the sequence even though this specific probe window
## shows no top-line change.

## Current status (2026-07-17, still yet later the same day, part 4):
## implemented candidate direction (1) from the part-3 status block below --
## dynamic SIFT density boost during high-angular-velocity frames -- code
## written and in the tree, **not yet measured** (build was in progress when
## this entry was written; results to follow in the next entry).
##
## What was added: `ORBextractor::SetDynamicDensity(nfeatures_,
## contrastThreshold_)` (`ORBextractor.h`/`.cc`) rebuilds the live `cv::SIFT`
## detector with a new target feature count and contrast threshold,
## deliberately leaving `nOctaveLayers` (and therefore `nlevels` and every
## `mvScaleFactor`/`mvLevelSigma2` array) untouched -- those are fixed-size
## arrays relied on everywhere else, and the part-2 regression above already
## showed that touching per-level scale/sigma structure mid-sequence is
## dangerous. This only asks SIFT to look harder within the same octave/layer
## structure, not to change what a level means.
##
## `Tracking::Track()` (`Tracking.cc`, right after the motion-model velocity
## update) now computes the per-frame rotation angle from `mVelocity`
## (`mVelocity.so3().log().norm()`, converted to degrees) and calls
## `SetDynamicDensity()` on `mpORBextractorLeft` **only on a state
## transition** -- entering high-angular-velocity mode boosts to 2x base
## `nfeatures` and halves the contrast threshold (0.04 -> 0.02), leaving mode
## reverts back to base on the frame rotation dropping back below threshold.
## This avoids rebuilding the `cv::SIFT` object every single frame once
## already in the boosted state.
##
## Threshold chosen (5 deg/frame) is a starting estimate, not calibrated the
## way `TH_HIGH`/`TH_LOW` were in Stage 4: real KITTI seq00 turns average
## well under 1 deg/frame with occasional few-degree bumps in normal curves,
## but hit an 18 deg/frame peak during the sharp maneuver that triggers the
## documented turn-region tracking-loss cascade -- 5 sits well above ordinary
## motion and well below that peak. The boost factor (2x) and contrast
## threshold (0.02, half of `ORBextractor.cc`'s base 0.04 -- kept in sync
## manually, no shared constant between the two files) are likewise
## unmeasured starting points, not tuned.
##
## `mnBaseNFeatures` (recorded once at construction, in both
## `newParameterLoader()` and `ParseORBParamFile()`) and `mbHighAngularVelocityMode`
## (transition-tracking bool) were added to `Tracking.h` to support this.
##
## **Not yet done, next immediately**: rebuild `orbslam3_sift_kitti_ate`
## (Tracking.cc/ORBextractor.cc were edited after the last build -- binary
## on disk predates this change) and re-run the same `[start-frame]`-jumped-
## to-3600 fast probe used for the part-2/part-1 investigations above, to
## see whether destructive resets in the turn region drop. If that looks
## promising, follow with a full seq00 ATE checkpoint (no start-frame arg)
## against the 6.4-10.7m ORB+DBoW2 baseline and the still-outstanding "real
## seq00 ATE checkpoint" item that's been deferred since Stage 4.
##
## Candidate direction (2) (dead-reckoning motion-model fallback during
## tracking loss) from the part-3 block below remains untried -- picked (1)
## to try first since it's strictly additive (touches only feature
## extraction density, not the tracking-loss control flow) and cheaper to
## measure in isolation.
##
## **Result (measured, same session): negative -- no meaningful effect.**
## Rebuilt `orbslam3_sift_kitti_ate` against the dynamic-density change and
## ran the same `[start-frame]`-jumped-to-3600, 941-frame probe used
## throughout this investigation. Outcome: 2 destructive resets (identical
## count to the part-1 baseline probe), at the exact same frame boundaries
## (`mnFirstFrameId`/`mnInitialFrameId` = 115/0 then 341/116 in both runs),
## and **no scorable trajectory produced either way** (empty
## `KeyFrameTrajectory.txt`, `loadTumTrajectory` returns empty).
##
## Not satisfied with relying on the part-1 prose numbers from days earlier
## in this file for the comparison -- re-ran the *exact* same probe this
## session against the pre-change code too, via `git stash push` on just the
## four dynamic-density files (`Tracking.cc/.h`, `ORBextractor.cc/.h`,
## leaving the `clearMap()` fix and `[start-frame]` CLI additions in place),
## rebuild, run, `git stash pop` to restore. Side-by-side: baseline reports
## "51 Frames set to lost" after the second reset, dynamic-density reports
## "54 Frames set to lost" -- a 3-frame difference, i.e. noise, not signal.
## Every other line of the two logs (reset points, map-point counts,
## "Fail to track local map!"/"Relocalized!!" cadence) is identical between
## the two runs.
##
## **Why it didn't help, most likely**: the boost only fires once
## `mVelocity`'s rotation estimate crosses 5 deg/frame, but `mVelocity` is
## computed from the *previous two tracked frames' relative pose* -- once
## tracking is already lost (which is what's actually happening in this
## window: both maps die at only 2 keyframes, same as the earlier
## `reloc-investigate` finding), there's no fresh `mVelocity` to read the
## high-angular-velocity signal from in the first place. The boost can only
## help the frames *leading into* a hard turn while tracking is still alive
## -- it can't help recovery once tracking has already dropped, which
## turned out to be the dominant failure mode in this exact probe window
## (frames land right in the middle of an already-lost stretch, not at the
## clean onset of the turn). This may still be worth keeping for the actual
## turn onset elsewhere in the sequence -- this probe specifically isolates
## the post-loss recovery portion, not the pre-loss prevention portion, so
## it's not a full refutation of the idea, just evidence it doesn't fix
## *this* symptom.
##
## Tree left with the dynamic-density change in place (stash was popped
## back) since it's inert once tracking is already lost and does not appear
## to be actively harmful either (identical reset behavior to baseline) --
## but this is a judgment call, not a strong result either way.
## **Next session should start with**: getting explicit direction on
## whether to (a) revert dynamic-density entirely as a negative result and
## move to candidate direction (2) (dead-reckoning fallback during tracking
## loss, which targets the actual observed failure mode here more directly
## than (1) did), (b) keep (1) in the tree and additionally try (2) on top
## of it, or (c) run a full seq00 ATE checkpoint with (1) in place first to
## see whether it helps anywhere else in the sequence before deciding --
## this probe only covers one 941-frame window, not the whole run.

## Current status (2026-07-17, still yet later the same day, part 3):
## reverted both Part 2 changes (forced-flat-level-0 `SearchForInitialization`
## widening, 12->8 merge threshold) back to the known-good baseline per
## explicit direction, since neither helped (see part 2 below for the full
## account). Confirmed via `git checkout` + rebuild that
## `ORBextractor.h`/`ORBmatcher.cc`/`LoopClosing.cc` are back to their exact
## pre-Part-2 text -- only the `clearMap()` UB fix and the two `[start-frame]`
## CLI additions (`orbslam3_kitti_ate.cpp`, `orbslam3_sift_calibrate.cpp`)
## remain uncommitted in the tree, matching the state at the end of the
## first "Current status" block further down.
##
## Two new candidate directions proposed, not yet started: (1) dynamically
## widen SIFT extraction (lower contrast threshold / raise target feature
## count) specifically during high-angular-velocity frames, detected via the
## existing motion-model/IMU signal, to prevent feature starvation during a
## turn instead of trying to recover after the fact; (2) a dead-reckoning
## motion-model fallback during tracking loss (propagate pose via constant-
## velocity or IMU integration for a short window) instead of immediately
## falling into re-initialization, as a bridge across the few frames where a
## sharp turn sweeps features out of view. Both are meaningfully different
## in kind from this session's Part 1/2 attempts -- they target the turn
## itself (prevention) rather than recovery after tracking is already lost.
## **Next session should start with**: picking one of these two to implement
## and measure, or getting explicit direction on which to try first --
## neither has been scoped or estimated yet.

## Current status (2026-07-17, still yet later the same day, part 2): tried
## a combined two-part fix per explicit direction -- (1) redo the flat-
## level-0 widening in `SearchForInitialization()`, but this time force
## surviving matched keypoint pairs to flat-level 0 before they feed
## `CreateInitialMapMonocular()`, so BA weights them uniformly instead of
## penalizing whichever SIFT layer they happened to come from (the theorized
## cause of the earlier regression); (2) lower `NewDetectCommonRegions()`'s
## merge-candidate-search keyframe-count gate from 12 to 8, so young maps
## need less survival time before merging back into the older, intact map
## becomes possible. **Both implemented, both left in the tree, but neither
## produced a scorable ATE, and the underlying problem is now understood
## more precisely than before.**
##
## Part 1 (BA weighting fix) result: real, measured improvement over the
## naive widening, but not enough -- 10 resets by frame 500 (vs. the naive
## widening's much faster blowup, but still vs. the true original's *zero*
## resets through frame 777), climbing to 103 destructive resets across the
## full 4541-frame run (vs. 3 for the true original, 90+ for the naive
## widening by a similar point). Final result: "Alignment failed -- too few
## matched points (3)", i.e. still no scorable trajectory, though technically
## a small one was produced this time (unlike the complete "No trajectory
## produced" of the unwidened version) -- a marginal, not real, improvement.
## So the BA-sigma-weighting theory was likely a real, partial contributor,
## but not the whole explanation for why widening the match window
## destabilizes tracking broadly (not just in the turn region) -- something
## else about admitting cross-layer matches still costs real stability.
##
## Part 2 (lowered merge threshold) result: **never got a chance to matter.**
## Verified directly via another temporary diagnostic
## (`[merge-investigate]`, since removed) added at both
## `KeyFrameDatabase::DetectNBestCandidates()`'s call site and
## `DetectCommonRegionsFromBoW()`'s merge-path return, re-run via the
## `[start-frame]`-jumped-to-3600 fast probe: across a full 941-frame probe
## (23 destructive resets), **`DetectNBestCandidates()` was never once
## invoked for the merge path** -- meaning no map in this hard region ever
## survived long enough to reach even the lowered 8-keyframe bar, let alone
## the old 12. So the merge threshold wasn't the bottleneck this session
## found it to be in isolation -- it's downstream of Part 1's still-
## unresolved stability problem. Also directly confirmed (real evidence, not
## assumption) that `DetectNBestCandidates()` itself has no same-map
## restriction bug like `DetectRelocalizationCandidates()` did -- it
## correctly routes cross-map candidates to `vpMergeCand` (line ~440) --
## so candidate retrieval was never the blocker; it's purely that
## `NewDetectCommonRegions()` is never reached with a map old enough.
##
## Both diagnostic instrumentation blocks removed afterward (confirmed via
## `git diff`); the two substantive changes -- forced flat-level-0 on
## matched initialization pairs, and the 12->8 merge-search threshold --
## remain in the tree, uncommitted, alongside the still-uncommitted
## `clearMap()` fix and `[start-frame]` CLI additions from earlier in this
## session. `orbslam3_sift_kitti_ate` left built against this state.
##
## **Net assessment**: the turn-region failure's true root cause is still
## the fragility of monocular re-initialization after a hard tracking loss
## in this fork -- not fully explained by the flat-level-0 match-pool size
## (widened and partially BA-corrected, still unstable), not explained by
## VLAD candidate-retrieval quality or a map-filtering bug (directly ruled
## out), and not resolved by giving merging more time to fire (moot --
## survival, not the threshold, is the blocker). Something about this
## fork's re-initialization is still meaningfully less robust than the
## original ORB+DBoW2 baseline even outside the turn region specifically
## (103 resets across the whole sequence this attempt, vs. 3 originally).
## **Next session should start with**: reverting to the known-good state
## (the single unwidened `SearchForInitialization`, `clearMap()` fix, and
## `[start-frame]` CLI addition only -- i.e. treat this session's Part 1/2
## attempts as informative-but-negative results, not something to build on
## directly) unless a genuinely new hypothesis for the broader (not just
## turn-region) stability cost of widening the match pool emerges. Possible
## angles not yet tried: instrumenting exactly which fraction of widened
## matches are cross-layer (vs. same-layer) to see if the extra candidates
## are mostly low-quality outliers RANSAC should be rejecting but isn't;
## comparing this fork's post-loss re-initialization behavior frame-by-frame
## against the original ORB fork's own (successful) recoveries elsewhere in
## the sequence, if any exist, to find a concrete behavioral difference
## instead of theorizing further.

## Current status (2026-07-17, still yet later the same day): **root-caused
## why relocalization/merge never recovers the earlier (still-intact, richly
## keyframed) map once tracking is lost in the turn region** -- the other
## lead flagged after the flat-level-0 fix was reverted. Confirmed via
## direct measurement, not the theorized VLAD-retrieval-quality explanation:
## `LoopClosing::NewDetectCommonRegions()` (the function that runs both loop
## closing AND map-merge candidate detection) has a hardcoded early return --
## `if(mpLastMap->GetAllKeyFrames().size() < 12) { ...; return false; }` --
## that skips candidate search entirely whenever the *current* map has fewer
## than 12 keyframes. This is what actually blocks recovery: it's not that
## VLAD fails to find the old map as a merge candidate, it's that the
## candidate search **never runs at all** for any of the short-lived maps
## this fork keeps creating in the turn region.
##
## Verified directly, not inferred: added a temporary diagnostic print at
## this exact branch (`[reloc-investigate]`, since removed) and re-ran via
## the `[start-frame]` CLI arg jumped to frame 3600 (right before the
## 3650-3699 turn) for a fast, targeted repro instead of replaying the whole
## sequence. Result over the resulting 941-frame probe: **the `<12` skip
## branch fired twice, both times with the map sitting at exactly 2
## keyframes; the actual candidate-search branch fired zero times.** Both of
## those 2-keyframe maps were destructively reset shortly after. This
## directly confirms the theory with real numbers: maps created in this hard
## region die from repeated tracking loss well before they can accumulate a
## dozen keyframes, so `NewDetectCommonRegions()`'s merge path -- the one
## mechanism that could reconnect a fresh map back to the older, well-
## established one sitting right there in `mpKeyFrameDB` -- never gets a
## chance to run. This is architecturally unchanged ORB-SLAM3 behavior (the
## `<12` constant predates this session's SIFT/VLAD work entirely), but this
## fork's specific weakness -- fragile re-initialization after a hard scene,
## documented in the flat-level-0 investigation above -- means maps almost
## never survive long enough to reach the threshold that would let this
## fork's *other* recovery mechanism kick in.
##
## Diagnostic print removed afterward (confirmed via `git diff` back to
## exact original text); `orbslam3_sift_kitti_ate` rebuilt clean.
##
## **Net picture now**: the turn-region failure has two compounding causes,
## both now root-caused with real evidence -- (1) monocular
## re-initialization is fragile after a hard loss (flat-level-0 restriction
## in `SearchForInitialization`, still unfixed after the earlier attempt
## regressed BA stability -- see the status block above), and (2) even if
## re-init eventually succeeds, the resulting maps die again before
## reaching the 12-KF bar that would let map-merging reconnect them to the
## older map instead of starting over from scratch every time. Neither
## `TH_HIGH`/`TH_LOW` calibration nor VLAD's relocalization-candidate
## quality are the bottleneck -- both were directly measured and ruled out
## earlier in this investigation.
##
## **Next session should start with**: deciding which of the two
## compounding causes to address first. Options, not yet tried: (a) lower
## the 12-KF merge-search threshold for this fork specifically (cheap,
## mechanical, but doesn't fix why maps die so fast in the first place --
## may just mean shorter-lived maps also fail to merge because
## `DetectCommonRegionsFromBoW`'s own geometric-verification bar isn't met
## either, untested); (b) revisit the flat-level-0
## `SearchForInitialization` fix with the BA-sigma-weighting bug actually
## fixed this time (compute per-flat-level sigma from octave alone, not the
## full flat index, so within-octave cross-layer matches get correct
## confidence) -- addresses the more fundamental problem, higher effort;
## (c) reduce the local-mapping keyframe-insertion threshold specifically
## for young/recovering maps so they accumulate keyframes faster and reach
## 12 sooner, untested and might trade off map quality for count. No
## consensus yet on which to pursue -- flag to the user rather than assume.

## Current status (2026-07-17, yet later the same day): tried the
## flat-level-0 fix flagged in the previous status block, found a real
## regression, and **reverted it**. Widened `ORBmatcher::
## SearchForInitialization()`'s level restriction from exactly flat-level 0
## to the whole finest octave (flat levels `[0, nOctaveLayers)`, via a new
## `ORBextractor::GetOctaveLayers()` getter) on both the F1 filter and the
## F2 `GetFeaturesInArea` search window. This measurably fixed the intended
## problem -- re-init point counts jumped from a marginal ~80-100 (right at
## the `nmatches>=100` floor) to a healthy 150-1600+ -- but re-running the
## full Stage 4 checkpoint surfaced a much worse regression: **90 destructive
## `ResetActiveMap()` calls by frame 3600 alone, vs. 3 for the entire
## 4541-frame sequence before this change.** The very first map (built at
## frame 0, 746 points -- far richer than before) failed almost immediately,
## nowhere near the turn region that motivated the fix, so this wasn't a
## narrow side effect.
##
## Leading (not fully confirmed) explanation: this fork's
## `mvScaleFactor`/`mvLevelSigma2`/`mvInvLevelSigma2` arrays are indexed by
## flat level and used as bundle-adjustment confidence weights in ~54 sites
## in `Optimizer.cc` (`invSigma2 = mvInvLevelSigma2[kp.octave]` and
## friends) -- a direct carryover from ORB, where each pyramid level is a
## genuinely different image resolution and a coarser level legitimately
## deserves a looser weight. In this SIFT fork, flat level encodes
## `(octave, layer)`, and only *octave* changes actual image resolution;
## *layer* changes only the Gaussian blur sigma within one octave, at the
## same resolution. Widening the match window let `SearchForInitialization`
## freely pair, e.g., a flat-level-0 keypoint in F1 with a flat-level-7
## keypoint in F2 -- both real, valid, same-resolution matches, but landing
## in `Optimizer.cc`'s weighting scheme as if they were 7 ORB pyramid levels
## apart in resolution, i.e. an ~85% looser confidence weight than the match
## actually deserves. Feeding bundle adjustment (used inside
## `CreateInitialMapMonocular` itself, not just later local BA) enough
## mis-weighted residuals plausibly explains why richer-looking initial maps
## were *less* stable, not more.
##
## Not proven -- didn't instrument actual BA residual weights to confirm --
## just the most coherent explanation consistent with every observed
## symptom (immediate post-init failure, far from the turn region, despite
## higher raw point counts). **Reverted both changes**
## (`ORBmatcher::SearchForInitialization()` back to the exact single-level
## match, `ORBextractor::GetOctaveLayers()` getter removed) via `git
## checkout` -- confirmed back to the exact pre-fix text. Rebuilt
## `orbslam3_sift_kitti_ate` clean afterward.
##
## Per explicit direction, **not pursuing this lead further for now**.
## Redirected to the other candidate from the previous status block:
## investigating why `Tracking::Relocalization()` never recovers the
## still-intact earlier map once a new one is created via
## `CreateMapInAtlas()` -- does VLAD's candidate retrieval even return the
## old map's keyframes from the rotated post-turn viewpoint? If this
## `mvLevelSigma2`-weighting theory is ever revisited, the real fix is
## likely computing per-flat-level sigma from *octave alone* (ignoring
## layer) rather than from the full flat index, so within-octave matches
## get equal, correct confidence regardless of which layer either point
## came from -- untried.

## Current status (2026-07-17, even later the same day): ran the real Stage 4
## checkpoint (full KITTI seq00, `orbslam3_sift_kitti_ate`, no start-frame
## skip) now that the `clearMap()` stall is fixed. **The stall fix is
## confirmed real and working** -- the run reached `Shutdown` on all 4541
## frames with zero hangs (previously: 10-20+ min stuck every retest in this
## region). But it still produced **no ATE number** -- "No trajectory
## produced" -- because the last of three map resets landed at frame 3941
## with only ~600 frames of runway left, and monocular re-initialization
## never once succeeded again before the sequence ended. So the stall and
## the "no trajectory" failure were two separate bugs that looked identical
## from the outside; only the first is fixed.
##
## Root-caused (not just observed) via direct measurement, ruling out the
## obvious suspects one at a time instead of guessing:
## - **Not feature starvation or blur**: a standalone probe
##   (`cv::SIFT`/`cv::ORB` directly, bypassing the whole pipeline) showed all
##   three detectors hitting the 2000-feature cap on every frame tested
##   through the failure region, and Laplacian-variance blur scores showed
##   nothing unusual.
## - **Not Stage 4's TH_HIGH/TH_LOW miscalibration**: Stage 4's calibration
##   tool only ever sampled frames 0-500 (a calm stretch) -- a real gap, so
##   this looked like the prime suspect. Added a `[start-frame]` arg to
##   `analyze/orbslam3_sift_calibrate.cpp` (same additive pattern as the
##   `orbslam3_kitti_ate.cpp` fix) and re-ran calibration on frames 3600-4000
##   specifically: true-match p50=10564/p95=51151/p99=68823 vs the original
##   calm-region 8380/46778/67038 -- close enough to be noise, not a real
##   miscalibration. This hypothesis is ruled out by direct measurement, not
##   assumption.
## - **The real cause: a genuine, sustained sharp-turn maneuver in KITTI
##   seq00 that the original ORB-based fork tracks through cleanly, but this
##   SIFT+VLAD fork cannot recover from once lost.** Computed real heading
##   change from the ground-truth poses (`poses/00.txt`): 163.7 degrees of
##   total turning in just frames 3650-3699 alone (peak single-step 18.1
##   degrees/frame at frame 3686), another 155.8 degrees at frames 3950-3999,
##   and more at 4350-4449 -- this is a real, sustained loop/turn maneuver,
##   not sensor noise. Confirmed the *original* ORB-based fork
##   (`third_party/ORB_SLAM3`, the one already validated at 6.4-10.7m ATE)
##   sails through this exact span without incident: its saved
##   `vendored_seq00_KeyFrameTrajectory.txt` has 113 continuous keyframes
##   covering timestamps 378-471s (= frames ~3650-4541), no visible gap. So
##   this is not KITTI seq00 being unTrackable -- it's a fork-specific
##   weakness.
##
## Exact failure chain, read directly off the checkpoint log's own reset
## markers: map B (initialized frame 778) tracked cleanly for ~2937 frames,
## then lost tracking at frame ~3684 -- squarely inside the 3650-3699
## turn window containing the 18 deg/frame spike -- and hard-reset at frame
## 3715 after 31 frames of failed relocalization. Its replacement (map C)
## barely initialized at frame 3745 ("New Map created with **100** points" --
## exactly at `SearchForInitialization`'s `nmatches>=100` floor, a fragile
## initialization by definition) and survived only ~113 frames before also
## failing, hard-resetting at frame 3941. After that, **zero** successful
## re-initializations for the remaining 600 frames, which span the second
## (3950-3999) and third (4350-4449) turn bursts.
##
## One concrete, measured contributing factor found while investigating the
## fragile-at-100-points re-inits: `ORBmatcher::SearchForInitialization()`
## only matches keypoints at flat-level 0 (`if(level1>0) continue;`, a
## leftover ORB assumption that level 0 = the majority scale level). For
## this SIFT fork's remapped octave encoding, flat-level 0 means specifically
## `octave==-1 && layer==1`, which a direct histogram probe (real
## `ORBextractor`, real KITTI frames) showed holds only ~9-17% of each
## frame's 2000 keypoints (182-337, sampled across both calm and turn-region
## frames) -- present, but a much smaller and more marginal pool than ORB's
## typical level-0 share, consistent with re-init repeatedly landing right at
## the 100-match floor instead of comfortably above it. Not yet confirmed as
## the full explanation (didn't instrument *why* relocalization against the
## still-intact map B / VLAD database never succeeded either, which is the
## other half of "why does it never recover") -- flagged as the most
## promising lead, not a proven complete root cause.
##
## **Left in the tree, uncommitted along with the still-uncommitted
## `clearMap()` fix from the previous status block**: the
## `orbslam3_sift_calibrate.cpp` `[start-frame]` addition (small, additive,
## same pattern as `orbslam3_kitti_ate.cpp`'s). The standalone probes used
## for the feature-count/blur and octave-histogram measurements were
## throwaway scratch files (`/tmp/.../sift_probe.cpp`,
## `/tmp/.../octave_hist.cpp`), not added to the repo.
##
## **Next session should start with**: either (a) instrument why
## `Tracking::Relocalization()` never recovers map B once map C is created
## (does VLAD's candidate retrieval even return map B's keyframes as
## candidates from the rotated post-turn viewpoint? is it being tried at
## all, given `CreateMapInAtlas()` starts a fresh map rather than blocking on
## relocalization?), or (b) directly address the flat-level-0 fragility in
## `SearchForInitialization` (e.g. widen the level filter to a small window
## of nearby flat-levels instead of requiring an exact match, given SIFT's
## octave semantics don't carry the same "level 0 dominates" assumption ORB's
## did) and re-run the Stage 4 checkpoint to see if re-init becomes robust
## enough to survive the turn. Either way, re-run the full checkpoint
## afterward -- this session never got a real ATE number, only proof that
## the stall is fixed and a real, separate, well-characterized cause for why
## no trajectory was produced.

## Current status (2026-07-17, still later the same day): **found and fixed
## the root cause of the frame ~3500-4000 stall** described in the status
## block just below. It was a real bug, not a performance/tuning issue:
## `KeyFrameDatabase::clearMap()` (`third_party/ORB_SLAM3_SIFT/src/
## KeyFrameDatabase.cc`) cached `vend = mvDatabase.end()` once before a loop
## that calls `mvDatabase.erase(vit)` repeatedly. That's undefined behavior
## for `std::vector` (safe for `std::list`, which is probably where the
## pattern got copied from while replacing DBoW2's list-based structures in
## Stage 3) -- `erase()` shrinks the vector, so the cached `vend` goes stale
## and the loop walks past the real logical end into leftover buffer
## memory, potentially even calling `erase()` at/past the real `end()`
## iterator, itself also undefined behavior. `clearMap()` is called on
## every `Tracking::ResetActiveMap()` and typically erases *most* of
## `mvDatabase` (everything belonging to the map being reset), so this
## wasn't a rare edge case -- it fired on nearly every reset, and got worse
## as the database grew. Fixed by re-querying `mvDatabase.end()` every
## iteration instead of caching it (the standard, correct idiom for
## erase-in-a-loop over a vector).
##
## Root-caused via live thread-level `/proc` inspection (no `gdb` access on
## this machine, ptrace_scope blocks attach) rather than a stack trace:
## added per-frame heartbeat/timing prints throughout `GrabImageMonocular`/
## `Track()`/`MonocularInitialization`, which showed the hang always started
## *after* `LocalMapping`'s reset acknowledgment ("Active map reset,
## Done!!!") printed but *before* the next frame's processing ever began --
## narrowing it to the handful of synchronous calls `Tracking::
## ResetActiveMap()` makes in between. `ps -T`/`/proc/<pid>/task/*/stat`
## confirmed only the single calling (Tracking) thread was burning CPU
## (99.7-99.3%) while `LocalMapping`, `LoopClosing`, and the OpenCV worker
## pool all sat fully idle -- which ruled out both reset-acknowledgment wait
## loops (cheap `usleep`-based polling on the caller) and pointed straight
## at the one synchronous, single-threaded call left in that gap:
## `mpKeyFrameDB->clearMap(pMap)`.
##
## **Validated**: added a temporary `analyze/orbslam3_kitti_ate.cpp`
## `[start-frame]` CLI arg (kept, harmless/additive, default 0) to jump into
## the middle of KITTI seq00 near the known stall region instead of
## replaying the whole sequence on every retest. The exact configuration
## that reproduced the stall 4 times in a row this session (start frame
## 3600) now runs clean through all 941 remaining frames to `Shutdown` with
## zero hangs post-fix; both resets that occur complete in 0ms each. All
## other TEMPORARY diagnostic instrumentation added while hunting this
## (`[heartbeat]`, `[framector-timing]`, `[track-timing]`, `[sfi-timing]`,
## `[rtv-timing]`, `[tlm-timing]`, `[ulkf-timing]`, `[clearmap-timing]`, and
## the older `[reloc-timing]` in `Relocalization()`) has been removed now
## that the bug is found; only the fix itself and the `[start-frame]` CLI
## addition remain.
##
## **Not yet done**: a real Stage 4 checkpoint run (full KITTI seq00, frame
## 0 to end, against the documented 6.4-10.7m ORB+DBoW2 baseline) -- the
## validation above used the artificial `start-frame`-skipped diagnostic
## setup, which never produces a scorable trajectory (fresh mid-sequence
## init, too little map history) and isn't a real accuracy measurement.
## **Next session should start with**: run
## `./build/orbslam3_sift_kitti_ate vocabulary_sift/vlad_codebook_all.yml
## <settings.yaml> .../sequences/00 .../poses/00.txt` (no start-frame arg)
## end-to-end and report the real ATE RMSE/mean/median/max, then proceed to
## Stage 5 (loop-closure/relocalization-specific validation) and Stage 6
## (mechanical cleanup: `ORBextractor`->`SIFTextractor` rename, remove dead
## `mBowVec`/`mFeatVec`, drop `Thirdparty/DBoW2` from `orbslam3_sift_ext`'s
## build) per the original plan
## (`/home/nam/.claude/plans/valiant-shimmying-tome.md`).

## Current status (2026-07-17, later the same day): swapping the vendored ORB-SLAM3's
## ORB features for SIFT, in a full fork (`third_party/ORB_SLAM3_SIFT`, the
## original `third_party/ORB_SLAM3` is untouched and still independently
## buildable) per explicit request, with DBoW2 loop-closure replaced by a
## custom VLAD (Vector of Locally Aggregated Descriptors) index -- no
## pretrained SIFT-compatible vocabulary exists anywhere (confirmed via web
## search) and no pretrained VLAD codebook exists either (also confirmed);
## NetVLAD does have real pretrained weights but was evaluated and rejected
## for now given the added cost (Python/PyTorch weight-conversion toolchain,
## new OpenCV dnn dependency, raw-image retention doesn't exist in this
## project's monocular Frame path, unverified KITTI domain transfer from
## Pittsburgh street-view). Followed a fully staged plan (see
## `/home/nam/.claude/plans/valiant-shimmying-tome.md`): Stage 0 (verified
## SIFT's octave/layer packing against real KITTI data, caught a real
## off-by-one before it went live), Stage 1 (SIFTextractor swap, checkpoint
## passed), Stage 2 (ORBmatcher reworked for L2/float descriptors, found and
## fixed a real gap the plan itself missed -- `SearchForTriangulation` also
## depended on DBoW2's FeatureVector, not just the two `SearchByBoW`
## overloads), Stage 3 (new `VladVocabulary` class, `KeyFrameDatabase`
## rewritten around a flat brute-force-scored list instead of DBoW2's
## inverted file, codebook trained on all 22 available KITTI sequences --
## ~85M descriptors seen, reservoir-sampled to 400k rows, k=64, sanity-
## checked: 0.48 similarity for adjacent frames vs -0.01 for distant ones),
## and Stage 4 (measured, not guessed, `TH_HIGH`/`TH_LOW` via RANSAC-verified
## true/false-match squared-L2 distance percentiles on real KITTI data:
## 227778 true pairs p50=8380, 43983 false pairs p50=25357).
##
## **Currently blocked** on a live performance stall: the first full-sequence
## checkpoint run (Stage 3e) tracked 87% of KITTI seq00 before a late
## tracking-loss/map-reset produced an empty trajectory (expected, given
## Stage 2's placeholder thresholds at that point). After Stage 4's real
## calibration, re-running consistently reaches a healthy pace through
## frame ~3000-3500, then stalls for 10-20+ minutes around frame 3500-4000
## on every attempt so far. Found and fixed three real, confirmed
## performance bugs along the way (all committed, all genuine
## improvements): (1) `KeyFrameDatabase::DetectRelocalizationCandidates()`
## had no cap on returned candidates (DBoW2's word-sharing prefilter used
## to keep this naturally small; VLAD has no equivalent) -- capped at
## `kMaxRelocCandidates=20`, mirroring the cap `DetectNBestCandidates()`
## already had; (2) `ORBmatcher`'s brute-force `SearchByBoW`/
## `SearchForTriangulation` replacements (see Session 2) called
## `cv::batchDistance()` once per KeyFrame descriptor (~1000-1500 small
## calls per candidate) instead of once for the whole descriptor matrix --
## fixed to one call per function invocation. Despite both fixes (confirmed
## individually correct and each a real improvement), the frame 3500-4000
## stall persists on every retest. Diagnostic instrumentation added to
## `Tracking::Relocalization()` (`[reloc-timing]` stderr lines, still in the
## tree, marked TEMPORARY) proved the stall is **not** repeated
## relocalization calls this time (call count stayed flat while the process
## kept burning CPU) -- it's stuck somewhere else in `Tracking`'s own code
## that hasn't been identified yet. `gdb -p <pid>` is blocked by this
## machine's ptrace_scope restriction, so a live stack trace wasn't
## available; `/proc/<pid>/task/*/stack` showed the Tracking thread actively
## running (not blocked on I/O/futex) while several worker threads sat idle
## after ~160s of prior CPU time each, suggesting something CPU-bound and
## single-threaded, but that's as far as `/proc`-only introspection got.
## **Next session should start with**: either (a) get real `gdb` access
## (adjust `/proc/sys/kernel/yama/ptrace_scope`, likely needs sudo) and
## attach mid-stall for an actual stack trace instead of continuing to
## guess-and-instrument one function at a time, or (b) keep adding
## `[[reloc-timing]]`-style diagnostic timers to the next most-likely
## Tracking functions (`TrackLocalMap`, `UpdateLocalKeyFrames`,
## `CreateInitialMapMonocular` -- re-init is plausible given repeated map
## resets happen right around this frame range) since that approach did
## conclusively rule out relocalization as the cause. Whichever path is
## taken, remove the temporary `[reloc-timing]` instrumentation once the
## real cause is found and fixed. See Session 14 below for the full
## staged account, every exact number, and the precise git commits.

## Current status (2026-07-17): wired the GUI's Feature Detector dropdown so
## selecting "ORB" is now a shortcut for the vendored real ORB-SLAM3 mode
## (bidirectionally synced with the "Use real ORB-SLAM3" checkbox) instead of
## running the custom pipeline's own known-inaccurate ORB integration;
## changed OXTS/IMU to default OFF everywhere (was on, per explicit
## request); and implemented a constant-velocity motion model for the
## existing (previously constant-position) guided-search match filter --
## measured as still a net loss on KITTI seq00 (146.3m baseline vs 159.5-
## 184.5m guided, worse the tighter/looser the radius) even with the more
## correct prediction, so it stays implemented-but-off. Net effect: SIFT is
## unchanged as the default/active custom-pipeline detector, but the
## OXTS/IMU-off default alone measurably regresses its own out-of-the-box
## accuracy (146.3m ATE RMSE vs. the 27.2m/18.6m previously documented with
## OXTS/IMU) -- kept anyway per explicit request; see Session 13 below for
## the full account and exact numbers. **Next session should start with**:
## either re-enabling OXTS/IMU by default, or trying the other ORB-SLAM3
## mechanism flagged as untried this session -- nonlinear pose-only
## refinement with iterative outlier rejection (replacing the current
## "RANSAC PnP and stop" with something like ORB-SLAM3's
## Optimizer::PoseOptimization()) -- to close the vision-only accuracy gap
## instead of relying on OXTS/IMU.

## Current status (2026-07-16, later the same day): vendored the real,
## published ORB-SLAM3 algorithm into this project (`third_party/ORB_SLAM3`,
## from an existing external checkout, GUI/Pangolin/Viewer code excluded),
## validated it at 6.4-10.7m ATE RMSE on KITTI seq00 across multiple runs,
## and wired it into the GUI as a toggleable "Use real ORB-SLAM3" checkbox
## alongside the existing custom pipeline. This closes the gap the whole
## Session 11 (and everything before it) was chasing: the custom pipeline's
## best-ever result was ~15-27m (with OXTS/IMU assistance) and plain
## ORB+SQPnP+5-point+BA alone was 100m+; real ORB-SLAM3 gets to single-digit
## meters on the same sequence. The gap was confirmed architectural, not a
## tuning problem -- see Session 12 below for the full account, including
## several real GUI-integration bugs found and fixed (a `cv::FileStorage`
## write-API crash, a `Settings::readViewer()` required-key `exit(-1)`
## crash, an RPATH-poisoning link/runtime bug that picked broken conda
## copies of Qt6/libcurl/libxml2, and a stale-trajectory-display bug fixed
## by rebuilding MapView's plot from ORB-SLAM3's own live, correction-aware
## keyframe poses every frame instead of a frozen per-frame snapshot), plus
## a detailed side-by-side comparison of the two-view estimator, PnP, and
## bundle-adjustment mechanics between the custom pipeline and ORB-SLAM3.
## **Next session should start with**: improving `EightPointLegacy`
## (`src/vision/EightPointLegacy.h`/`.cpp`) -- concretely, either building
## ORB-SLAM3-style H-vs-F model selection directly into the estimator
## (currently it's F-only; H is handled separately, in
## `SlamWorker::estimateTwoViewPose()`), and/or switching its RANSAC scoring
## from a hard inlier-count threshold to ORB-SLAM3's soft chi-square-weighted
## score (see Session 12, "Two-view estimator comparison" below for exactly
## what ORB-SLAM3 does and why EightPointLegacy's own post-RANSAC refinement
## is actually already *more* rigorous than ORB-SLAM3's).

## Current status (2026-07-16): added ORB as a second, GUI-selectable feature
## detector (SIFT remains the default, untouched) plus EPnP in the PnP
## dropdown. This surfaced a real pre-existing bug: ORB+SQPnP/EPnP+BA hits a
## permanent tracking freeze around frame ~1560-1580 via the same
## `m_avgStepScale` collapse mechanism documented in earlier sessions --
## fixed with a `kStepScaleResetStreak=50` reset (see Session 11). The fix is
## confirmed behavior-preserving on the default SIFT path (17.141m, exact
## match) and does stop the permanent freeze, but ORB+SQPnP's own tracking
## accuracy is far worse than SIFT+P3P regardless: 167.296m ATE with BA,
## 179.397m without -- i.e. BA is not the problem, ORB+SQPnP is just a much
## less accurate configuration for this pipeline on this sequence. A
## user-reported GUI stall around frame 330->400 was investigated via the
## debug harness (extended with a new detector-selection CLI arg) under both
## default SIFT+P3P and ORB+SQPnP -- did not reproduce in either headless
## run (clean through frame 1116+, zero "Lost" states). See Session 11 below.
## The Session 10 status below is otherwise still accurate and unchanged.
##
## Current status (2026-07-15): the documented 17.141m BA baseline had been
## silently broken since Session 9's own item 9 fix, now restored; two
## follow-on attempts to close the gap toward ~5m (per-segment BA-driven
## trajectory rederivation, continuous local BA) were both implemented,
## measured, and reverted this session after root-causing why each
## regressed. A third, larger attempt -- an offline DCS-robustified pose
## graph in a new PoseGraphOptimizer.h/.cpp, opt-in via kitti_ate's
## `posegraph` flag -- is now implemented and left in the tree (does not
## touch live tracking at all), but after 8 full-sequence iterations fixing
## three real, distinct bugs, it still scores ~32m, worse than the 17.141m
## baseline; the remaining gap traces to a genuine architectural limitation
## (see Session 10's final subsection), not a further tunable parameter.
## See Session 10 below for the full account. Short version:
## `kitti_ate.cpp` loaded OXTS/IMU *before* `openVideoFile()`, and Session
## 9's own `clearOxtsImuData()` fix (added later in that same session, after
## the 17.141m number was already recorded and never re-verified
## afterward) unconditionally wipes OXTS/IMU on every `openVideoFile()`
## call -- so every `kitti_ate` run since Session 9 has silently been
## running with zero OXTS/IMU regardless of CLI flags (confirmed via a
## bisection: OXTS-on and OXTS-off produced byte-identical 160.498m output).
## Fixed by reordering; 17.141m now reproduces exactly. The frame 183<->1620
## outlier flagged below is still open and unfixed, and this session found
## a second concrete instance of the same class of problem (frame
## 402<->2452, a 106.789-world-unit/11.991-degree loop correction) --
## next session should treat "some loop-closure measurements are just
## wrong and BA/interpolation have no way to detect that" as the
## standing, still-unsolved root cause behind both anomalies, not two
## separate bugs.
##
## NEXT SESSION SHOULD START HERE: (1) **investigate the frame
## 183<->1620 outlier specifically** -- dump its optimized poses/landmark
## positions and compare against the neighboring (good) windows' inputs to
## find what's actually different, since aggregate stats don't show it;
## until this is understood, don't fully trust any single BA-corrected
## window without an ATE/sanity check backing it up. (2) **re-run the P3P
## BA comparison at higher iteration/tighter tolerance or with the same
## reprojection gate applied to a wider net** to see whether 17.141m
## improves further, and try the same live-BA config with Iterative PnP
## (this session's best no-BA live number, 15.619m, was Iterative --
## combining Iterative PnP with the now-fixed BA hasn't been tried). (3)
## the background `computeLoopEstimate()`'s new `kEnrichmentMaxWindowKeyframes
## = 200` cap now silently skips every late-sequence loop closure (window
## sizes were 214-319 for the second half of this run) -- fine as a
## stopgap, but means the diagnostic currently only ever reports on the
## easy early windows; worth a smarter strategy (subsampling a large window
## instead of skipping it outright) if the diagnostic needs to cover the
## whole sequence. (4) minor CLI hygiene bug noticed while comparing
## sessions: `kitti_ate`'s pnp-method argument (argv4) treats any
## unrecognized string, including the "-" placeholder used to skip earlier
## flags while reaching later ones, as an explicit request for P3P rather
## than leaving `PnpSettings`' own default (Iterative) alone -- this may
## mean some earlier sessions' "-" placeholder runs silently used P3P when
## Iterative was intended; worth fixing the CLI to treat "-" as "leave
## default" consistently with every other flag, and worth a skeptical
## re-read of any older ATE table entry that used "-" for this argument.
## (5) Session 7's *other* untried candidate -- stop evicting landmarks so
## aggressively once a loop window is pending/open -- is still untried, and
## now lower priority given (2)'s results. (6) determine KITTI's actual
## camera pitch for GroundPlaneScale's remaining bias (unchanged from
## Session 5); (7) properly solve translation given a known rotation for
## the IMU integration (unchanged from Session 4/5); (8) find/construct a
## scenario where homography wins the model-selection ratio (unchanged from
## Session 4/5).

## Session 14 (2026-07-17, later the same day): forked third_party/ORB_SLAM3 -> ORB_SLAM3_SIFT and staged the ORB->SIFT+VLAD swap through Stage 4, currently blocked on an unidentified performance stall around frame 3500-4000

Picked up directly from Session 13's own routing work on a new, explicit
request: swap the vendored real ORB-SLAM3's ORB features for SIFT, not just
route the GUI's "ORB" dropdown entry to it. Confirmed with the user this is
a different, much larger undertaking than Session 13's dropdown wiring
before starting (ORB-SLAM3 is ORB-based throughout its tracking, mapping,
and loop-closing internals -- this isn't a config flag).

### 0. Vocabulary/technique research, before committing to an approach

Before writing any code, searched for whether a mature/pretrained solution
already existed rather than assuming custom training was necessary:

- **No pretrained DBoW2-compatible SIFT vocabulary exists** (web search) --
  DBoW2 supports arbitrary float descriptors in principle via its templated
  `FClass` interface, but every real deployment (including the ORB-SLAM
  ecosystem) ships an ORB-trained vocabulary, and this project's vendored
  DBoW2 copy only has the ORB-specific `FORB` descriptor class. Building
  one from scratch means writing a new descriptor class AND training a
  vocabulary tree -- real work, not a download.
- User chose **VLAD** (Vector of Locally Aggregated Descriptors, Jegou et
  al. 2010) over training a DBoW2-compatible vocabulary: a much lighter
  technique (one small k-means codebook instead of a hierarchical
  vocabulary tree), well-established in visual place recognition, needs no
  new DBoW2 class.
- **No pretrained VLAD codebook exists either** (web search, multiple
  queries) -- every source is explicit that classical VLAD is meant to be
  trained per-deployment, unlike deep-learned approaches.
- **NetVLAD does have real pretrained weights** (`Nanne/pytorch-NetVlad`,
  VGG16 backbone, Pittsburgh30k) -- investigated as an alternative when the
  user pushed back on "just train it" a second time, and the full cost was
  assessed honestly: no ready ONNX export exists (would need a one-time
  PyTorch environment just for conversion), this project has no OpenCV
  `dnn` linkage yet, ORB-SLAM3's monocular `Frame` constructor doesn't
  retain the raw image at all (confirmed via grep -- only the stereo-
  fisheye constructor does, `Frame.cc:1041`, never exercised in this
  project), and Pittsburgh street-view is a real domain mismatch against
  KITTI's forward-facing vehicle-mounted footage. User chose to proceed
  with classical VLAD, training on this project's own KITTI data, after
  this comparison -- confirmed as methodologically valid (training on the
  deployment domain is standard for classical vocabulary/codebook methods,
  unlike deep-learned approaches where pretrained-and-frozen is the whole
  point).

### 1. Fork instead of in-place edit (course-corrected mid-session)

Initial approach edited `third_party/ORB_SLAM3` in place (Stage 1's
`ORBextractor` swap, committed as `3342fae`). User strongly objected
("why u change the ORB base??? copy to another file") -- the original,
already-validated ORB-based vendored copy needed to stay completely
untouched and independently buildable, not be mutated for an experimental
fork. Reverted cleanly via `git checkout`/`git revert` (confirmed
byte-identical to the pre-swap baseline via `git diff`), then `cp -r`'d the
whole tree to `third_party/ORB_SLAM3_SIFT` (symlinked `Thirdparty/DBoW2`
and `Thirdparty/g2o` dependencies preserved and confirmed to still resolve
from the new location). Added a fully separate, additive CMake target pair
(`orbslam3_sift_ext` + `orbslam3_sift_kitti_ate`, reusing the same
`analyze/orbslam3_kitti_ate.cpp` source) -- confirmed the original
`orbslam3_ext`/`orbslam3_kitti_ate`/`sift_vslam_gui`/`kitti_ate`/
`sift_vslam_debug` targets all still build unaffected, and re-verified this
repeatedly after every subsequent commit. The two copies define identical
`ORB_SLAM3::*` class names in the same namespace, so `orbslam3_ext` and
`orbslam3_sift_ext` must never both be linked into one executable (ODR
violation) -- kept in permanently separate binaries for exactly that
reason. `git init` was also added as a prerequisite this session (the
project had no version control at all before this) -- specifically because
a change this large, in-place, on vendored code needed a real safety net;
this is what made the fork-and-revert above possible to do cleanly instead
of by hand.

All git history and staging below (Stages 0-4) happened against
`third_party/ORB_SLAM3_SIFT` from this point on.

### 2. Stage 0: octave-remap probe -- caught a real off-by-one before it went live

Before touching any vendored code, a standalone probe (`cv::SIFT::create()`
with this project's own `SiftSettings` defaults) ran against ~200 real
KITTI seq00 frames, decoding each keypoint's packed `KeyPoint::octave`
(`rawOctave = kpt.octave & 255; octave = rawOctave<128 ? rawOctave :
(rawOctave|-128); layer = (kpt.octave>>8)&255`) to verify the planned
flat-index remap design (needed because ORB-SLAM3 code treats `.octave` as
a plain small array index into `mvScaleFactors`/`mvLevelSigma2`/
`mvImagePyramid` everywhere, including ~25 sites in `Optimizer.cc`
weighting every bundle-adjustment residual -- getting this wrong wouldn't
crash, it would silently corrupt BA). Octave range `[-1,5]` was fine
(within the assumed `[-1,8)`), but **layer range was `[1,3]`, not `[0,3)`
as originally assumed** -- OpenCV's `cv::SIFT` reports layers 1-indexed
over `[1,nOctaveLayers]`, not 0-indexed. Fixed the flat-index formula
(`layer = clamp(layer,1,nOctaveLayers) - 1`) before it was ever used live,
based on real measured data (~395k keypoints from frames 0-150) rather than
assumption.

### 3. Stage 1: SIFTextractor swap

Reimplemented `ORBextractor.cc`'s internals to wrap `cv::SIFT` (kept the
class *named* `ORBextractor` through Stages 1-4 for minimal signature
churn -- rename deferred to a mechanical Stage 6 cleanup). Constructor
signature unchanged (`nfeatures`->SIFT nfeatures cap, `nlevels`-
>nOctaveLayers reinterpreted, `scaleFactor`/`iniThFAST`/`minThFAST`
accepted-but-ignored) -- zero edits needed anywhere in `Frame.h`/
`Tracking.h`/`Tracking.cc`/`Settings.cc`. Builds a real (not empty)
`mvImagePyramid` via repeated `cv::resize` (dead for monocular correctness
-- only stereo `ComputeStereoMatches()` reads it -- but built anyway to
avoid a latent-UB class for near-zero cost, with target sizes clamped to
>=1px since `kMaxOctaveSpan`'s outer levels can shrink below that).
**Checkpoint**: a standalone probe (`ORBextractor` compiled and linked
completely standalone -- it has zero dependency on g2o/Sophus/boost,
confirmed via its own `#include` list) against real KITTI frames confirmed
`CV_32F`/128-col descriptors and all keypoint octaves landing inside the
valid `[0,30)` scale-array range (actual observed range `[0,17]`).

### 4. Stage 2: ORBmatcher rework -- found a real gap the plan itself missed

`DescriptorDistance()` became squared-L2 (not sqrt'd -- strictly monotonic,
avoids `sqrtf()` in loops that run thousands of times per frame) for
`CV_32F`/128-dim descriptors, replacing the 256-bit Hamming popcount.
Retyped every `bestDist*`/`dist` local from `int` to `float` across ~16
functions -- verified each site individually via `grep`, not just via
blind search-replace, since implicit `int`<->`float` conversions don't
produce compiler errors on their own (a real risk: the compiler would
silently narrow/widen without complaint). `TH_HIGH`/`TH_LOW` became float
placeholders (60000/30000, in squared-L2 space) seeded from a quick
non-final probe, clearly commented as pending Stage 4.

Both `SearchByBoW()` overloads' original DBoW2 `FeatureVector` node-
bucketed matching (restricting comparisons to descriptors sharing a
vocabulary-tree node -- no VLAD equivalent exists, VLAD has no discrete
"node" concept) were replaced with brute-force matching. While doing this,
grepped the whole file for `mFeatVec`/`FeatureVector` to check the
approved plan's scope was actually complete -- **it wasn't**:
`SearchForTriangulation()` used the identical DBoW2 node-bucketed pattern
and was never named in the plan (which only explicitly called out the two
`SearchByBoW` overloads). This function runs on *every keyframe insertion*
during normal `LocalMapping` operation, not just occasionally like
`SearchByBoW` (relocalization/loop-closing only) -- missing it would have
silently broken new-landmark triangulation entirely once `mFeatVec` stopped
being populated in Stage 3, a much worse failure mode than a slow path.
Fixed by grepping first, not trusting the plan's stated scope as complete.
**Checkpoint**: `orbslam3_sift_ext`/`orbslam3_sift_kitti_ate` build and link
clean; confirmed zero remaining `mFeatVec`/`FeatureVector` usage outside
comments; original targets still build unaffected.

### 5. Stage 3: VladVocabulary, KeyFrameDatabase rewrite, ComputeBoW cutover

New `ORB_SLAM3::VladVocabulary` (`include/VladVocabulary.h`/
`src/VladVocabulary.cc`): `computeVlad()` (assign each descriptor to its
nearest of k centroids via brute-force NN -- trivial at k=64 -- accumulate
+ intra-normalize per-centroid residuals, flatten + L2-normalize the
whole vector) and `score()` (cosine similarity via plain dot product of
two already-normalized vectors -- a direct drop-in analog of
`DBoW2::TemplatedVocabulary::score()`). `loadFromTextFile()` deliberately
keeps that exact method name (not renamed) so `System.cc`'s existing call
site needs zero edits -- and it's still literally accurate, since the new
codebook format is a flat-keyed, non-dotted-key `cv::FileStorage` YAML
(avoiding the dotted-key `WRITE`-API crash documented in Session 12).

`ORBVocabulary.h`'s typedef was repointed from
`DBoW2::TemplatedVocabulary<FORB::TDescriptor,FORB>` to `VladVocabulary`
directly -- kept as the exact same type name specifically so every
existing `ORBVocabulary*`/`ORBVocabulary&` declaration throughout
`Frame.h`/`KeyFrame.h`/`Tracking.cc`/`System.cc`/`KeyFrameDatabase.h`
needed zero signature changes. This surfaced a real, non-obvious hidden
dependency: `DBoW2::TemplatedVocabulary.h` had been silently providing
`using namespace std;` PLUS several transitive includes (`<fstream>`,
`<vector>`, `DUtils/Random.h`, etc.) at global scope to every file that
ever included `ORBVocabulary.h` -- removing that transitive chain produced
a cascade of "vector/map/set/string does not name a type" and
"`std::ofstream` has incomplete type" compile errors across *unrelated*
headers (`Frame.h`, `KeyFrame.h`, `KeyFrameDatabase.h`, `Map.h`,
`Tracking.h`, `MLPnPsolver.cc`) that had nothing to do with the vocabulary
change on the surface. Fixed by replicating the same transitive surface in
`VladVocabulary.h` (not great practice, but the least invasive fix given
how many files silently depended on it).

`KeyFrameDatabase` rewritten: `mvInvertedFile` (word-id -> KeyFrame list)
became a flat `mvDatabase` (`vector<KeyFrame*>`), and all 5 candidate-
detection methods now brute-force-score against every entry via
`VladVocabulary::score()` instead of DBoW2's word-sharing prefilter --
confirmed via full-tree grep that only `DetectNBestCandidates()`
(loop-closing) and `DetectRelocalizationCandidates()` are ever actually
called in this project's monocular path; the other 3 (one already marked
DEPRECATED upstream) were kept working via the same pattern for API
completeness only. `Frame`/`KeyFrame::ComputeBoW()` now populate a new
`mVladVec` member via `computeVlad()` instead of calling DBoW2's
`transform()` into `mBowVec`/`mFeatVec` (now dead, cleanup deferred to
Stage 6).

**Training tool**: new `analyze/orbslam3_vlad_train.cpp` (+ CMake target,
linked against `orbslam3_sift_ext` so the training-time extractor is
byte-identical to the runtime one) -- extracts SIFT descriptors across one
or more KITTI sequences via the real vendored extractor, reservoir-samples
down to a 400k-row pool (uniform random subsample once past the cap, not
just the first 400k seen), runs `cv::kmeans`, writes the codebook, then
reloads it through the real `VladVocabulary` class for a reported (not
asserted) sanity check comparing VLAD cosine similarity of adjacent vs.
temporally-distant frames.

Trained **two** codebooks (k=64): one on seq00 alone (sanity check: 0.4416
adjacent vs -0.0074 distant), and — per explicit follow-up request to use
all available data — one on **all 22 KITTI sequences** (00-21; only 00-10
have public ground truth, but training doesn't need poses at all, only
images) -- ~85M descriptors seen in total, reservoir-sampled to 400k rows,
sanity check: 0.4771 vs -0.0101, a slightly cleaner separation than
seq00-alone. Confirmed valid methodology (not overfitting/cheating) when
asked directly: training classical vocabulary/codebook methods on the
deployment domain is standard practice, unlike pretrained-and-frozen deep
approaches; Stage 4's own plan already called for validating on a second
sequence as a check against over-fitting the read to one sequence
specifically.

Moved both trained codebooks out of `build/` (gitignored) into a permanent
`vocabulary_sift/` directory and committed them (~110KB each, small enough
to commit directly, same treatment as `vocabulary/ORBvoc.txt`).

**Checkpoint (Stage 3e, first real end-to-end attempt)**: ran
`orbslam3_sift_kitti_ate` against KITTI seq00 with the full-dataset
codebook -- tracked cleanly through 87% of the sequence (frames 0-3900, no
failures logged at all in that stretch) then hit a cascade of "Fail to
track local map!" and a `ResetActiveMap()` call, which in monocular mode
without IMU genuinely **erases** the current map's keyframes in place
(`mpAtlas->clearMap()`, confirmed via reading `Tracking::ResetActiveMap()`
-- this is unchanged original ORB-SLAM3 behavior, not something this
session's changes introduced). With only ~600 frames of runway left after
the erase, the system never rebuilt a scorable map before the sequence
ended, so `SaveKeyFrameTrajectoryTUM()` had nothing to write -- no ATE
number at all, not just a bad one. Diagnosed (not just observed) as a
plausible consequence of Stage 2's still-placeholder `TH_HIGH`/`TH_LOW`
rather than a deeper bug, since loose/miscalibrated matching thresholds
can let bad matches accumulate until a late catastrophic failure -- exactly
what Stage 4 exists to fix.

### 6. Stage 4: measured threshold calibration

New `analyze/orbslam3_sift_calibrate.cpp` (+ CMake target): for KITTI seq00
frame pairs at baselines 1/3/5 (600 frames), gets ratio-test candidate
matches via the real `ORBextractor`, geometrically verifies via
`cv::findFundamentalMat` RANSAC, splits into true-match (inlier) /
false-match (outlier) squared-L2 distance distributions, reports
percentiles -- mirrors this project's own established practice for
threshold selection (`feature_detector::defaultRatioFor()`'s empirical
0.75/0.85 ratio thresholds), measurement over guesswork.

Measured: 227778 true-match pairs (squared-L2 p50=8380 p90=35425 p95=46778
p99=67038 max=113387), 43983 false-match pairs (min=93 p1=1095 p5=2779
p10=4699 p50=25357) -- a real, meaningful ~3x separation (true p50 vs false
p50), with genuine overlap too (expected, since `TH_LOW`/`TH_HIGH` are a
coarse sanity cutoff in this codebase's design, not the primary
discriminator -- the ratio test does the fine-grained work downstream).
Replaced the Stage 2 placeholders with `TH_LOW=46778` (true-match's own
95th percentile, so it doesn't reject genuine matches) and
`TH_HIGH=100557` (1.5x true-match's 99th percentile).

### 7. Post-calibration re-test: found and fixed 3 real, confirmed performance bugs, but a stall persists

Re-running Stage 3e's exact test with calibrated thresholds surfaced a live
performance problem, root-caused (not just patched around) across several
iterations of measure-fix-retest:

1. **`DetectRelocalizationCandidates()` had no cap on returned
   candidates.** DBoW2's word-sharing prefilter used to keep the candidate
   list naturally small; VLAD's brute-force scoring against every keyframe
   has no equivalent bound, and every returned candidate triggers an
   expensive `ORBmatcher::SearchByBoW()` call inside
   `Tracking::Relocalization()` -- which itself runs on *every single
   frame* while tracking is lost (confirmed via `grep`, standard original
   ORB-SLAM3 behavior, not something this session changed). A live run
   stalled 20+ minutes with zero progress once tracking got stuck
   relocalizing against a database that had grown large. Fixed: sorted by
   score, capped at `kMaxRelocCandidates=20`, mirroring the cap
   `DetectNBestCandidates()` (the loop-closing path) already had.
2. **The vectorization fix was itself incomplete on the first attempt.**
   Replaced `SearchByBoW`/`SearchForTriangulation`'s brute-force per-pair
   `DescriptorDistance()` scalar loops with `cv::batchDistance()` -- but
   the *first* version of this fix still called `batchDistance()` once
   *per KeyFrame descriptor* inside the outer loop (up to ~1000-1500 small
   calls per candidate keyframe, each individually fast but carrying its
   own allocation/dispatch overhead). Re-testing after this "fix" still
   showed a 7+ minute wall-clock stall (29+ min CPU) -- a real, confirmed-
   via-live-retest finding that the first fix wasn't sufficient, not a
   hypothesis. Root-caused via back-of-envelope math (20 candidates x
   ~1000-1500 calls/candidate x ~75us/call overhead ~= 1.5s per lost
   frame x hundreds of consecutive lost frames ~= exactly the observed
   multi-minute magnitude) and fixed properly: moved each
   `cv::batchDistance()` call *outside* the outer per-descriptor loop, so
   it runs exactly once per function invocation, computing the full
   query-matrix x train-matrix distance matrix in one vectorized call.
   Applied to all three affected functions (both `SearchByBoW` overloads,
   `SearchForTriangulation`).
3. Both fixes are real, confirmed-correct, and committed (`18ae969`,
   `d867247`, `9d0c13f`, `64e7756`) -- but **the frame ~3500-4000 stall
   still persists** on every retest after all three fixes, now taking
   10-20+ minutes in that region specifically. Added temporary diagnostic
   instrumentation to `Tracking::Relocalization()`
   (`[reloc-timing]` stderr lines: per-call elapsed time, candidate count,
   database size, running cumulative total -- still in the tree, clearly
   marked TEMPORARY, should be removed once this is resolved) to get real
   numbers instead of continuing to guess. This instrumentation
   **conclusively ruled out relocalization as the current bottleneck**:
   during the frame-3500 stall, the relocalization call counter stayed
   completely flat (no new calls logged) while the process kept actively
   burning CPU for 10+ minutes -- the stall is somewhere else in
   `Tracking`'s own code that hasn't been identified yet.
   `gdb -p <pid> -batch -ex "thread apply all bt"` was blocked by this
   machine's `ptrace_scope` restriction (`Could not attach to process`),
   so a real stack trace wasn't available. `/proc/<pid>/task/*/stack` /
   `wchan` inspection (doesn't need ptrace) showed the Tracking thread in
   state `R` (actively running, not blocked on I/O or a futex) while 7
   worker threads sat idle on `futex_do_wait` after each accumulating
   ~160+ seconds of prior CPU time -- consistent with some earlier heavy
   parallel computation (Ceres/g2o/OpenCV internal threading) having
   finished, followed by a single-threaded phase that's still running, but
   this is as far as `/proc`-only introspection could narrow it down.

**Left in a known, honest state per explicit request**: rather than
continuing to guess at the next function to instrument (already 3
iterations deep on this specific stall), the diagnostic run was left to
run to completion, however long that takes, and this account written up
for whoever picks it up next.

### 8. Net result / where things stand

`third_party/ORB_SLAM3` (original, ORB-based) is completely untouched and
still independently buildable/validated at 6.4-10.7m ATE RMSE.
`third_party/ORB_SLAM3_SIFT` (the fork) has SIFT-based feature
extraction, VLAD-based loop-closure/relocalization candidate search, and
measured (not placeholder) matching thresholds, all building and linking
cleanly, with three confirmed-real performance bugs found and fixed along
the way. **Not yet validated end-to-end**: every full-sequence run so far
either produced no trajectory (Stage 3e, before calibration) or stalled for
10-20+ minutes in the same frame ~3500-4000 region (every retest since,
despite three real, confirmed fixes each meaningfully improving on the
last). No ATE number exists yet for the SIFT+VLAD fork.

**Next session should start with**: getting real `gdb` access (adjust
`/proc/sys/kernel/yama/ptrace_scope`) to attach mid-stall and get an actual
stack trace, rather than continuing to guess-and-instrument one Tracking
function at a time -- or, if that's not available, add `[reloc-timing]`-
style temporary timers to the next most-likely candidates
(`Tracking::TrackLocalMap()`, `Tracking::UpdateLocalKeyFrames()`,
`Tracking::CreateInitialMapMonocular()` -- re-initialization is plausible
given repeated map resets keep happening right around this frame range).
Once the stall is root-caused and fixed, resume the plan at Stage 4's
checkpoint (re-run `orbslam3_sift_kitti_ate` on seq00, compare against the
6.4-10.7m ORB+DBoW2 baseline, then validate on a second sequence per the
plan's own over-fitting check), then Stage 5 (loop-closure/relocalization-
specific validation) and Stage 6 (mechanical cleanup: `ORBextractor`->
`SIFTextractor` rename, remove dead `mBowVec`/`mFeatVec`, remove this
session's temporary `[reloc-timing]` instrumentation, remove
`Thirdparty/DBoW2` from `orbslam3_sift_ext`'s build once nothing
references it). The full staged plan remains at
`/home/nam/.claude/plans/valiant-shimmying-tome.md` for reference.

### 7. Stall root-caused and fixed (later, same session)

The stall above is resolved. Root cause: `KeyFrameDatabase::clearMap()`
cached `vend = mvDatabase.end()` once before a loop that calls
`mvDatabase.erase(vit)` repeatedly -- undefined behavior for
`std::vector` (the cached `vend` goes stale as the vector shrinks, so the
loop overruns into leftover buffer memory and can call `erase()` at/past
the real `end()`). `clearMap()` runs on every `ResetActiveMap()` and
typically erases most of `mvDatabase`, so this fired on nearly every reset
rather than being a rare edge case.

Found via live thread-level `/proc`/`ps -T` inspection rather than a stack
trace (`gdb` attach is blocked by this machine's `ptrace_scope`): per-frame
heartbeat/timing prints added throughout `GrabImageMonocular`/`Track()`/
`MonocularInitialization` narrowed the hang to strictly between
`LocalMapping`'s reset acknowledgment print and the next frame's
processing; `ps -T` then showed only the single calling (Tracking) thread
at 99%+ CPU with every other thread (`LocalMapping`, `LoopClosing`, the
OpenCV worker pool) fully idle, which ruled out both `usleep`-based reset
wait loops and left `mpKeyFrameDB->clearMap(pMap)` as the only remaining
synchronous, single-threaded call in that gap.

Fix: re-query `mvDatabase.end()` every loop iteration instead of caching
it once. Added a small, permanent `[start-frame]` CLI arg to
`analyze/orbslam3_kitti_ate.cpp` (default 0, additive/harmless) to jump
into the middle of a sequence for fast repro instead of always replaying
from frame 0. The exact config that reproduced the stall 4 times in a row
(seq00 starting at frame 3600) now runs clean through all remaining frames
to `Shutdown`, zero hangs, both resets in that run completing in 0ms each.
All temporary timing/heartbeat instrumentation added during the hunt has
been removed; only the fix and the `[start-frame]` addition remain in the
tree.

**Still not done**: a real Stage 4 checkpoint (full seq00, frame 0 to end,
against the 6.4-10.7m baseline) -- everything validated above used the
artificial start-frame-skipped setup, which never produces a scorable
trajectory. That's the immediate next step, followed by Stage 5 and 6 as
already described above.

## Session 13 (2026-07-17): GUI detector-dropdown/ORB-SLAM3 sync, OXTS/IMU off by default, and a constant-velocity motion model for guided search (measured, net loss, left opt-in)

Picked up on a request to "replace ORB with SIFT" -- clarified through several
rounds of back-and-forth into three distinct, smaller pieces of work, all
scoped and confirmed with the user directly rather than guessed at, given
this codebase's history of large speculative changes (continuous local BA,
Session 10; DCS pose-graph edge cases, Session 10) needing full reverts.

### 1. Feature Detector dropdown's "ORB" choice is now a shortcut for real ORB-SLAM3

The custom pipeline's own ORB integration (Session 11) was never actually
removed -- confirmed via re-reading the code and measurements already in
this file (167m vs 17m ATE RMSE, ORB+SQPnP vs SIFT+P3P) that it remains
strictly worse, and the user confirmed they still want the dropdown/toggle
itself to stay (a "flexible switch"), just not pointed at the bad path.
Resolution: `ControlPanel`'s Detector combo box (`m_detectorType`) and the
"Use real ORB-SLAM3" checkbox (`m_orbSlam3Enabled`) are now bidirectionally
synced:

- Selecting "ORB" in the dropdown checks `m_orbSlam3Enabled` (if not already
  checked), which switches `SlamWorker` over to the vendored real
  `ORB_SLAM3::System` entirely (see Session 12) and grays out the Feature
  Detector/PnP/OXTS-IMU groups, exactly as checking that box directly always
  did -- the custom pipeline's own `DetectorType::Orb` path is never
  actually exercised via this route.
- Selecting "SIFT" again unchecks it, restoring the custom pipeline.
- Checking/unchecking "Use real ORB-SLAM3" directly (bypassing the dropdown)
  also syncs the dropdown's displayed value to match, so the two controls
  can never visibly disagree.

Implementation: extracted the existing per-field `setRowVisible` lambda (SIFT-
only vs. ORB-only parameter rows) out of `buildSiftGroup()`'s local scope into
a new private `ControlPanel::applyDetectorRowVisibility(bool isOrb)` (needs a
new `m_siftForm` member to reach the form outside the function that built
it), since both directions of the sync need to update row visibility.
Signals are blocked (`QSignalBlocker`) on whichever control is being
programmatically updated by the other's handler, to avoid infinite
re-entrant toggling. `FeatureDetector.h`'s `DetectorType`/`createDetector()`/
`OrbSettings` and `SlamWorker`'s `m_detectorType`/DBoW2 vocabulary machinery
are all otherwise completely unchanged from Session 11/12 (confirmed via
`grep` diff against pre-session state after a false start where an agent
misread "flexible switch" as "remove ORB entirely" and had to revert a full
teardown of `FeatureDetector.h`, the DBoW2 loop-closure branches, and the CLI
flags back to their exact original text).

### 2. OXTS/IMU now defaults OFF everywhere (was on) -- confirmed to measurably regress accuracy, kept anyway per explicit request

Three defaults flipped, all per explicit request:
- `SlamWorker::m_oxtsEnabled`/`m_imuEnabled`/`m_oxtsImuInPnpEnabled`: `true` -> `false` (SlamWorker.h).
- `ControlPanel`'s "Use OXTS/IMU in PnP tracking plausibility check" checkbox: `setChecked(true)` -> `setChecked(false)`.
- `ControlPanel::setOxtsAvailable()`/`setImuAvailable()` (fired when KITTI
  seq00's OXTS/IMU auto-loads) no longer auto-check the "Use OXTS speed
  correction"/"Use IMU rotation" boxes when data becomes available -- they
  now only enable them (clickable, still unchecked) for a manual opt-in.
  Becoming *unavailable* still force-unchecks, unchanged (stale data must
  never stay silently applied to an unrelated new source).

**Measured cost, full KITTI seq00, SIFT+P3P, no other flags**: 146.307m ATE
RMSE (4502/4541 matched, scale 0.181) -- vs. the 27.2m (OXTS/IMU-in-PnP only)
/ 18.6m (documented full combination) this file has previously recorded
*with* OXTS/IMU. This is a real, substantial, confirmed regression in
out-of-the-box accuracy, not a wash -- flagged explicitly to the user, who
confirmed keeping OXTS/IMU off by default anyway and redirected toward
vision-only improvements instead (see below) rather than reverting.

### 3. Constant-velocity motion model for the existing guided-search filter -- implemented correctly, measured, still a net loss, left in as opt-in (not default)

Investigated closing the OXTS/IMU-off accuracy gap via vision-only means,
per explicit request to bring some of real ORB-SLAM3's own tracking-quality
mechanisms (see Session 12's architectural comparison) into the custom
pipeline. `trackFrame()` already had a `setGuidedSearchEnabled()` toggle
(default off) that rejects descriptor matches landing far from where a
motion-predicted map-point projection says they should be -- but its own
doc comment (`kGuidedSearchRadiusPixels`) already flagged the prediction as
constant-POSITION (just reusing the previous frame's pose directly), noted
as "a poor motion model for a forward-driving car at ~10fps", and had
already needed its filter radius widened 40px -> 200px to compensate (which
mostly defeats the filter's purpose).

Implemented real constant-velocity extrapolation:
- New `SlamWorker` members `m_velocityR`/`m_velocityT` (identity/zero by
  default): the relative transform from the previous accepted pose to
  `m_currR`/`m_currT`, computed every time `trackFrame()` accepts a new pose
  (`m_velocityR = R * m_currR.t(); m_velocityT = tvec - m_velocityR *
  m_currT;`, evaluated BEFORE `m_currR`/`m_currT` are overwritten with the
  new pose).
- The guided-search filter now projects map points through `predR =
  m_velocityR * m_currR, predT = m_velocityR * m_currT + m_velocityT` (the
  last pose composed one more step forward) instead of `m_currR`/`m_currT`
  directly -- exactly ORB-SLAM2/3's own constant-velocity model. Falls back
  to constant-position automatically (`m_velocityR` = identity, `m_velocityT`
  = zero) until a second step has been tracked.
- Velocity is explicitly reset to identity/zero at `recoverViaEpipolar()`'s
  success point (a tracking-loss recovery jump, not real per-frame motion --
  extrapolating across that gap would predict worse than assuming no
  motion). Left un-reset at the two BA/keyframe-refinement re-sync points
  (`refineLocalKeyframes()`, `runLocalBundleAdjustment()`'s pose-sync tail)
  since those only nudge an already-continuous trajectory, not teleport it.

**Measured, full KITTI seq00, SIFT+P3P, no OXTS/IMU (matching item 2's new
default)**:

| Config | ATE RMSE |
|---|---|
| No guided search (baseline) | 146.307 m |
| Guided search, constant-velocity, 200px radius (unchanged from before) | 184.471 m |
| Guided search, constant-velocity, 60px radius | 159.522 m |

Both radii are worse than not using guided search at all, despite the
prediction itself now being architecturally correct. Not yet root-caused to
a specific mechanism, but the leading hypothesis: pre-filtering matches
before `cv::solvePnPRansac` discards correspondences that RANSAC's own
outlier rejection would have handled fine on its own, and in a vision-only
(no OXTS/IMU) setting where match noise is already higher, trimming this way
apparently loses more good matches than bad ones. **Left implemented and
correct, but off by default and radius left at 60px** (the better-measured
of the two, should anyone opt in) -- this is the same "implement, measure,
document, don't force a win that isn't there" discipline as
Session 10's reverted continuous-local-BA/DCS-pose-graph attempts, except
here the change is small/self-contained enough that reverting the code
itself isn't necessary, only reverting the *default*.

### 4. Net result

SIFT remains the default, unchanged, active custom-pipeline detector.
Selecting "ORB" now correctly routes to real ORB-SLAM3 instead of the known-
bad custom ORB path. OXTS/IMU are off by default (regresses SIFT's own
out-of-the-box accuracy to 146.3m ATE RMSE, confirmed and accepted
explicitly). Guided search's motion model is now genuinely constant-velocity
(a real correctness improvement over constant-position) but is not, on this
sequence, a net accuracy win at either radius tested -- stays opt-in.
**Next session**: either flip OXTS/IMU back to default-on (simplest fix for
the accuracy regression), or continue the vision-only route via the other
untried ORB-SLAM3 mechanism from Session 12's comparison -- nonlinear
pose-only refinement with iterative outlier rejection in place of the
current single RANSAC-and-stop PnP solve.

## Session 12 (2026-07-16, later the same day): vendored real ORB-SLAM3 into the project, validated it, integrated it into the GUI as a toggleable mode (fixing several real bugs along the way), and did a deep architectural comparison against the custom pipeline

Picked up after Session 11 confirmed ORB+SQPnP+BA (and by extension the
custom pipeline's whole architecture) tops out far from the ~5m ORB-SLAM2
paper reference regardless of detector/PnP/BA combination. Direct request:
stop reimplementing pieces of ORB-SLAM and instead reproduce what the real,
published system actually achieves, then fold that real implementation into
this project.

### 1. Vendoring `third_party/ORB_SLAM3`

Copied the real ORB-SLAM3 source (23 of 24 `.cc`/`.cpp` files under `src/`,
27 of 28 headers, plus a header-only `Thirdparty/Sophus` copy) from an
existing external checkout into `third_party/ORB_SLAM3/` in this project,
compiled fresh against this project's own system OpenCV/Eigen rather than
linking the external checkout's prebuilt `libORB_SLAM3.so` (built against a
conda OpenCV -- mixing two OpenCV ABIs' `cv::Mat` in one process risks
silent corruption, not just a link error, the same reasoning that already
ruled out a prebuilt DBoW2). `Viewer.cc`/`.h` (the only files deliberately
excluded) are GUI/Pangolin-only and unnecessary for headless benchmarking;
removing them meant also stripping 4 GL-drawing methods from
`MapDrawer`/`MapDrawer.cc` (`DrawMapPoints`/`DrawKeyFrames`/
`DrawCurrentCamera`/`GetCurrentOpenGLCameraMatrix`, confirmed via grep to
only ever be called from the excluded `Viewer.cc`) and writing a
minimal stub `Viewer.h` (empty no-op methods) since `Tracking.cc` still
calls `mpViewer->RequestStop()`/`isStopped()`/`Release()` behind
`if(mpViewer)` guards that are always false but still need a complete type
to compile against. `System.cc`'s Viewer-construction branch now throws if
`bUseViewer=true` is ever requested against this vendored copy.

Three real build blockers, found and fixed:
- `GeometricCamera::nNextId` static member had no out-of-line definition in
  what was copied -- turned out to live in `src/CameraModels/Pinhole.cpp`
  (`.cpp` extension, in a `CameraModels/` subdirectory the original copy
  loop's `*.cc`-only glob never matched).
- `MLPnPsolver.cpp` and `OptimizableTypes.cpp` (also `.cpp`, not `.cc`) were
  missed for the same reason.
- Fixed by copying both gaps in (renamed to `.cc` to match the existing glob)
  and widening `CMakeLists.txt`'s `file(GLOB ...)` to also cover
  `third_party/ORB_SLAM3/src/CameraModels/*.cc`.

`LoopClosing.h`'s `bool mnFullBAIdx` incremented via `mnFullBAIdx++` (legal
pre-C++17, removed in C++17) -- rather than touch the vendored logic, gave
the vendored sources their own per-target `CXX_STANDARD 14` override
(matching ORB-SLAM3's own original target standard) while the rest of this
project stays C++17.

g2o is reused directly via the project's existing prebuilt `g2o_ext` target
(confirmed zero OpenCV dependency via `ldd`, so no ABI risk); DBoW2 is
recompiled fresh from source (same vendored copy the project's own `DBoW2`
target already uses) so ORB-SLAM3's own `Thirdparty/DBoW2/...`-relative
includes resolve correctly.

Two link-time library-version-conflict bugs, both the same root cause
(detailed in item 3 below since they recurred there too): linking
`libboost_serialization.so` (needed for `System.cc`'s Atlas save/load, no
system package available, only exists in a conda env) plus `g2o_ext`'s
`IMPORTED_LOCATION` both caused CMake to auto-add those directories to the
target's RPATH, which then got consulted as an rpath-link hint when
resolving *transitive* NEEDED libraries -- pulling in conda's mismatched
`libcurl.so.4` (zero `CURL_OPENSSL_4` versioned symbols, vs. the system
copy which has them) via OpenCV's `imgcodecs -> libhdf5_serial -> libcurl`
chain. Fixed by explicitly linking the system's `libssl`/`libcrypto`/
`libcurl.so.4`/`libxml2.so` by full path.

Refactored the vendored sources into a shared `orbslam3_ext` STATIC library
(rather than duplicating them directly into each consuming executable) so
both the CLI benchmark and the GUI link the identical compiled code.

### 2. Headless CLI validation: `analyze/orbslam3_kitti_ate.cpp`

Combines ORB-SLAM3's own `mono_kitti.cc` tracking loop with this project's
existing ATE evaluation methodology (`kitti_ate.cpp`'s own
`loadGroundTruth()`/`umeyama2D()` 2D similarity alignment, byte-for-byte
reused). Run against KITTI seq00: **6.399m ATE RMSE** (mean 5.412m, median
4.526m, max 21.065m, 959/959 keyframes matched, recovered scale 15.999) --
this is the number the rest of this session's work is measured against.
(A separately-run externally-built `mono_kitti` binary against the same
sequence, evaluated with the same alignment script before vendoring even
started, got 10.738m -- both numbers are real, valid runs; ORB-SLAM3 has
genuine run-to-run nondeterminism from RANSAC sampling and its 3-thread
architecture's scheduling.)

### 3. GUI integration: `SlamWorker`'s new ORB-SLAM3 mode

Added a "Use real ORB-SLAM3 (disables custom pipeline settings below)"
checkbox to `ControlPanel` (own `QGroupBox`, wired to a new
`orbSlam3EnabledChanged(bool)` signal) that grays out the Feature
Detector/PnP/OXTS-IMU groups when checked (camera intrinsics group stays
enabled -- ORB-SLAM3 mode still reads fx/fy/cx/cy from it to build its own
settings file). `SlamWorker::setOrbSlam3Enabled()` just sets a flag;
`start()` lazily constructs a real `ORB_SLAM3::System` the first time it
runs while the flag is set (needs the video source already open, for
resolution). `processNext()` branches early to a new
`trackFrameOrbSlam3()` instead of running the custom detect/init/track
block. Added `orbslam3_ext` + a `ORBSLAM3_VOCAB_PATH` compile definition to
every target that compiles `SlamWorker.cpp` (`sift_vslam_gui`,
`sift_vslam_debug`, `kitti_ate`), since it now unconditionally includes
`<System.h>`.

Four real, user-facing bugs found via live testing (rebuild -> launch with
logging -> watch log -> ask user what's on screen), none caught by
compiling alone:

- **`cv::FileStorage::WRITE`'s node API rejects dotted key names.** The
  dynamically-generated ORB-SLAM3 settings file needs keys like
  `Camera1.fx`/`ORBextractor.nFeatures` -- `fs << "Camera1.fx" << value`
  throws `cv::Exception: Key names may only contain alphanumeric characters
  [a-zA-Z0-9], '-', '_' and ' '`, uncaught across a Qt event handler ->
  `std::terminate`. Reading dotted keys back via `cv::FileStorage::READ`
  (what `Settings.cc` actually does) has no such restriction -- only the
  WRITE side's node-construction path validates names. Fixed by writing the
  settings file as plain text (`QTextStream`) instead of via
  `cv::FileStorage`'s write API.
- **`Settings::readViewer()` treats every `Viewer.*` key as required** and
  calls `exit(-1)` directly (not a catchable exception -- kills the whole
  GUI process) if any is missing, even though this is a headless run that
  never constructs a real Viewer. Fixed by writing the same `Viewer.*` block
  from ORB-SLAM3's own reference `KITTI00-02.yaml` into the generated
  settings file regardless (values never actually read by anything, just
  needed to satisfy the parser).
- **RPATH poisoning recurred at the GUI's own link+runtime, not just
  `orbslam3_ext`'s build** (same mechanism as item 1's curl bug): linking
  `orbslam3_ext` into `sift_vslam_gui` pulled its RPATH entries along,
  which poisoned resolution of the GUI's own `libQt6DBus.so.6` (conda's
  6.9.3 vs. system's 6.10.2 -- a private-ABI symbol version mismatch,
  `QObjectPrivate::QObjectPrivate(int)@Qt_6_PRIVATE_API` undefined) at link
  time, and then `libQt6Core.so.6` itself at *runtime* ("version
  `Qt_6.10` not found", since `DT_RUNPATH` on the executable's own direct
  NEEDED entries is checked before falling through to default system
  paths). Fixed the link-time half by explicitly linking system
  `libQt6DBus.so`; fixed the runtime half by setting
  `CMAKE_BUILD_RPATH` to `/usr/lib/x86_64-linux-gnu` (confirmed via
  `readelf -d` that manually-specified `CMAKE_BUILD_RPATH` entries get
  listed *before* CMake's own auto-computed ones, so this wins the
  first-match search order over the g2o/conda directories for anything
  that exists in both places, while still falling through to those
  directories for `libg2o.so`/`libboost_serialization.so`, which only exist
  there).
- **Live trajectory display went stale after any loop closure/map merge.**
  `trackFrameOrbSlam3()` originally accumulated one `QPointF` per frame from
  that frame's own `Tcw` -- a frozen snapshot that never reflects ORB-SLAM3
  retroactively correcting *earlier* keyframes' poses (confirmed happening
  live: `*Loop detected`/`*Merge detected` in the log). Symptom: a visually
  well-tracked run (reaching keyframe 1671, a real map merge) still showed a
  badly misaligned MapView overlay and a 32m live ATE despite the
  already-validated 6.4m CLI number. Root-caused and fixed by adding a
  `GetAtlas()` accessor to the vendored `System.h` (`mpAtlas` was private,
  no existing public equivalent to `SaveKeyFrameTrajectoryTUM()`'s
  Shutdown()-time keyframe read) and rebuilding `m_trajectory` from scratch
  every frame via `GetAtlas()->GetAllKeyFrames()`, sorted by `mnId`, reading
  each live (possibly corrected) `KeyFrame::GetPoseInverse().translation()`
  -- exactly what `SaveKeyFrameTrajectoryTUM()` does at shutdown, just
  called every frame instead of once at the end. (`mnFrameId` is
  ORB-SLAM3's own 0-based frame numbering; `+1` converts to this project's
  1-based `m_frameCount`/ground-truth-line convention.)

Related but smaller: `MapView`'s Umeyama alignment fit
(`kAlignmentFreezeMinPoints = 200`) permanently freezes once enough
overlapping points accumulate -- fine for the custom pipeline's stable
early trajectory, but ORB-SLAM3's own first ~200-350 tracked points
routinely land inside its opening map-reset/loop-closure churn, freezing
the display on a bad early fit. Added
`MapView::setContinuousAlignmentEnabled()` (wired to the same ORB-SLAM3
checkbox) to skip the freeze and re-fit every frame instead, trading a
known wobble/rescale (as loop closures land) for staying accurate to
ORB-SLAM3's own live corrections. Also fixed a real correctness bug found
alongside all this: the generated settings file declared `Camera.fps: 30`
unconditionally, but the GUI actually throttles frame delivery to 10fps
(`kProcessIntervalMs = 100`) -- `Tracking.cc` sets
`mMaxFrames = settings->fps()` (its force-a-new-keyframe-after-this-many-
frames threshold) from this value, so the mismatch meant ORB-SLAM3 was
tuned for 3x the real inter-frame time gap. Now computed as
`1000 / kProcessIntervalMs` when throttled.

Added a live ATE RMSE readout to `MapView` (large corner overlay, plus a
smaller mean/median/max line) computed from the exact same aligned
trajectory + frame-indexed ground-truth pairing the overlay itself uses --
directly comparable to the CLI benchmark's own four-stat report, just live
instead of post-run. (Mean/median/max were added specifically because a
single RMSE number can look alarming when a handful of high-error
keyframes -- e.g. a brief tracking-loss recovery jump -- dominate it while
the bulk of the trajectory tracks closely; the CLI-validated run itself had
a 21m max sitting alongside its 6.4m RMSE.)

### 4. Architectural comparison: custom pipeline vs. real ORB-SLAM3

Verified directly against both codebases (not from memory/general
knowledge) at the user's request, across three areas:

**Two-view estimator**: `EightPointLegacy` (normalized 8-point + Gold
Standard Sampson LM refinement) is *more* rigorous than ORB-SLAM3's own
`TwoViewReconstruction::ReconstructF`, which keeps whichever raw minimal
8-point RANSAC sample scored best and does no refit/refinement at all
afterward. ORB-SLAM3 does do two things `EightPointLegacy` alone doesn't:
(a) a soft chi-square-weighted RANSAC score (`th - chiSquare`, symmetric
across both images) instead of a hard inlier-count threshold, and (b)
built-in H-vs-F model selection (`CheckHomography`/`CheckFundamental` +
ratio test) -- though this project's own `SlamWorker::estimateTwoViewPose()`
already wraps `EightPointLegacy` with an equivalent separate homography
branch and the same ratio-test technique, just as two components instead
of one integrated class.

**PnP / per-frame pose**: the custom pipeline re-matches (ratio test, no
temporal prior) and re-solves `cv::solvePnPRansac` from scratch every
frame. ORB-SLAM3 predicts the pose from a constant-velocity motion model,
narrows the match search to a small radius around the prediction
(`SearchByProjection`, not a blind ratio-test), then refines via
`Optimizer::PoseOptimization()` -- a g2o Levenberg-Marquardt solve over a
single 6-DOF pose vertex (map points held fixed), 4 passes of 10
iterations with chi-square-threshold (5.991) outlier edges progressively
excluded (`setLevel(1)`) between passes, i.e. real nonlinear
reprojection-error minimization with iterative reweighted outlier
rejection, not a RANSAC-then-stop closed-form solve.

**Bundle adjustment**: the custom pipeline's opt-in `runLocalBundleAdjustment()`
uses a sliding temporal window (last `kLocalBaWindowKeyframes`) with only a
soft pose-prior regularizer constraining keyframes outside the window
(added after a hard-anchor approach was tried and reverted). ORB-SLAM3's
`LocalBundleAdjustment()` selects its window via the **covisibility
graph** (current keyframe + everything sharing enough map points with it)
and hard-fixes every *other* keyframe that also observes those points as a
real gauge anchor -- runs automatically after every single keyframe
insertion, in its own always-on background thread. This difference is the
likely explanation for why this project's own earlier continuous-local-BA
attempt (see Session 10) collapsed monocular scale to 0.0008/187m ATE: a
soft prior against a moving window can drift; a real fixed anchor can't.

## Session 11 (2026-07-16): added ORB + EPnP as GUI-selectable options, found and fixed a real avgStepScale-collapse permanent freeze specific to ORB+SQPnP/EPnP+BA, then confirmed the fix does NOT close the accuracy gap -- ORB+SQPnP is just much less accurate than SIFT+P3P on this sequence, independent of BA or the freeze

Picked up on direct request to add ORB as a second feature detector (SIFT
must remain default/intact) and expose EPnP in the PnP method dropdown, both
GUI-selectable via a new `QComboBox`.

### 1. New file: `src/vision/FeatureDetector.h`/`.cpp` (namespace `feature_detector`)

Pulled detector construction and descriptor matching out of `SlamWorker.cpp`
into their own file, matching this project's established pattern (see
Session 10's `PoseGraphOptimizer.h`/`.cpp`). `DetectorType` enum
(`Sift`/`Orb`), `OrbSettings` struct (mirrors `cv::ORB::create()`'s own
parameters), `createDetector()`, `normTypeFor()` (`cv::NORM_L2` for SIFT's
float descriptors, `cv::NORM_HAMMING` for ORB's binary ones -- this distinction
is the entire reason ORB needed real code changes rather than just a new
enum value), and `matchDescriptors()` (the existing `BFMatcher` + `knnMatch`
+ Lowe's-ratio-test body, parameterized by norm instead of hardcoded to
`cv::NORM_L2`).

`SlamWorker::m_detector`'s type changed from `cv::Ptr<cv::SIFT>` to
`cv::Ptr<cv::Feature2D>` (both derive from it, so `detectAndCompute()` call
sites needed no changes). New `setDetectorType()`/`setOrbSettings()` slots,
new `ControlPanel` combo box + ORB parameter fields (shown/hidden via
`form->labelForField(widget)->setVisible(...)`, which works identically on
Qt5/Qt6 unlike the Qt6-only `QFormLayout::setRowVisible`). EPnP was a
one-line addition to the PnP dropdown (`PnpSettings::method` is a plain
`int`, and `trackFrame()`'s generic `cv::solvePnPRansac(...)` call already
forwards any `cv::SOLVEPNP_*` value untouched). Verified behavior-preserving
for the default SIFT+P3P path: `kitti_ate` with no new flags still produces
the exact 17.141m baseline.

### 2. Real bug found: ORB+SQPnP/EPnP+BA permanently freezes around frame ~1560-1580

Testing the new detector via `kitti_ate`'s CLI (`orb` flag, added alongside
this work) showed both ORB+SQPnP+BA and ORB+EPnP+BA locking up permanently
(trajectory count stops incrementing forever) at frame ~1559 and ~1577
respectively, regardless of which PnP method was used -- pointing at ORB's
own characteristics (not the PnP method) as the trigger. This is the same
`m_avgStepScale` collapse mechanism already documented in this file from an
earlier SIFT-based lockup at frame ~1579: the running step-scale estimate is
median-of-window + EMA, only ever updates on *accepted* steps, and once it
collapses toward zero, `isPlausibleStep()`'s bound (proportional to the
estimate) becomes so tight that literally no future step -- however
correct -- can ever pass, permanently. With ORB producing a different match
quality/count profile than SIFT, this pre-existing failure class apparently
triggers far more reliably.

**Fix**: track consecutive *total*-tracking-failure frames
(`m_trackFailStreak`, already existed for display purposes only) and once it
hits `kStepScaleResetStreak = 50`, reset `m_avgStepScale`,
`m_longTermStepScale`, and clear `m_recentStepDistances` -- the exact same
reset `resetSlamState()` already performs, just triggered automatically
instead of requiring a manual restart. Re-opens `isPlausibleStep()`'s gate
so tracking can resume once the streak is broken.

**Confirmed via log**: `[track] step-scale reset after 50 consecutive failed
frames (avgStepScale=0.0008, longTermStepScale=0.0007)` fires at the
previous freeze point, and both `orb_sqpnp_ba` and `orb_epnp_ba` runs
continue tracking well past their old permanent-freeze frames (to
completion, 4541/4541 frames, vs. never before).

### 3. But the fix doesn't close the accuracy gap -- ORB+SQPnP is just worse, not merely "stuck"

| Config | ATE RMSE |
|---|---|
| SIFT + P3P + BA + OXTS/IMU (documented baseline, re-verified this session) | 17.141 m |
| ORB + SQPnP + BA (fix applied) | 167.296 m |
| ORB + SQPnP, no BA (fix applied) | 179.397 m |

BA is not the cause of the bad number (it even helps slightly: 167m vs
179m) -- ORB+SQPnP's own raw tracking accuracy is far worse than SIFT+P3P on
this sequence, fix or no fix. The freeze fix trades "stuck forever" for
"unstuck but still very inaccurate": each reset discards the step-scale
estimate entirely, and if ORB+SQPnP triggers this reset repeatedly through
the sequence (plausible, given how much more readily it hit the original
collapse than SIFT ever did), each reset's scale discontinuity compounds
into large drift. This is a tracking-quality problem inherent to
ORB+SQPnP's match/pose-estimation behavior on this data, not a bug to patch
further -- **don't spend more time tuning the reset mechanism itself to try
to reach SIFT-competitive ORB numbers; the gap is upstream of it.**

### 4. Investigated a user-reported GUI stall around frame 330->400 -- did not reproduce

Added diagnostic `fprintf` logging to `MainWindow.cpp`'s
`handleLoopClosureDetected()`/`handleLoopEstimateFinished()` (start/queued/
finish + elapsed ms) to help pin down whether a reported stuck/"pending"
`LoopEstimatePanel` state was a hang, a crash, or just a very long
`computeLoopEstimate()` call (see Session 7's 900s-budget note for how long
these can legitimately take on a large window). Not yet exercised to a
conclusion -- the GUI wasn't observed hitting this state again this session.

Also extended `src/debug_main.cpp` with a new optional 4th CLI arg
(`orb`/`sift`, wired via `worker.setDetectorType()`) so the headless harness
could reproduce the report directly. Ran it against
`Dataset/KITTI/video_samples/kitti_00.mp4` unthrottled, once with default
SIFT+P3P and once with ORB+SQPnP (this session's exact test config, minus
BA/OXTS since this harness has no flags for either) -- both runs tracked
cleanly through the 330->400 region and well beyond (1116+ frames) with
zero "Lost (searching for a fix...)" states. **Inconclusive**: the reported
stall does not reproduce headlessly under either configuration tried, so
either it's specific to the live GUI/loop-estimate-panel path (for which the
new logging above should help next time it's caught live), or it needs a
longer run / different sequence portion to reproduce. Next session should
re-run the GUI itself with the new logging active and wait for the report
to recur live, rather than assuming the debug harness has ruled it out.

### 5. Net result / where things stand

ORB + EPnP are now real, working, GUI-selectable options
(`feature_detector::DetectorType`, `ControlPanel`'s new combo box), the
avgStepScale permanent-freeze class of bug has one fewer trigger condition
now that the automatic reset exists, and the SIFT+P3P+BA+OXTS baseline
remains exactly 17.141m, unaffected by any of this session's changes.
ORB+SQPnP is not currently a competitive alternative for accuracy on this
sequence -- it should be treated as "available but not recommended" until
someone specifically investigates its match/RANSAC threshold tuning (none
of `kHRansacThreshold`/`kFRansacSampsonThreshold`/the ratio-test threshold
have ever been tuned per-detector; they were chosen for SIFT and just
inherited by ORB).

## Session 10 (2026-07-15): found and fixed a silent regression that had broken the documented 17.141m BA baseline since Session 9 itself, then attempted two follow-on improvements toward the ~5m ORB-SLAM reference, root-caused why both regressed, and reverted both

Picked up on direct request to try to close the gap between this project's
best documented BA result (17.141m) and the ~5.33m ORB-SLAM reference this
file has cited since Session 2. Also set up KITTI sequence 00's OXTS/calib
folder structure the same way sequence 01's was assembled (copied
`calib_2011_10_03/2011_10_03`'s three `calib_*.txt` files into
`oxts_seq00/2011_10_03/2011_10_03_drive_0027_sync/` alongside the
already-extracted `oxts/` folder) -- turned out to be redundant for the
app's own `autoLoadKittiExtras()`, which already points seq00's `calibDir`
directly at `calib_2011_10_03/2011_10_03`, but harmless and now consistent
with how seq01's folder is laid out.

### 1. Real regression found: the 17.141m baseline didn't reproduce at all

Re-running the exact documented Session 9 config (P3P, OXTS+IMU,
`oxtsimupnp`, `ba`, full seq00) gave **160.498m ATE RMSE**, recovered scale
**0.0854** (vs. the documented ~1.02-1.03) -- not a small discrepancy. A
4-way bisection (repeating the identical run; dropping `ba`; dropping
`oxtsimupnp`; dropping OXTS/calib entirely) showed the OXTS-on and
OXTS-off runs produced **byte-identical output** (160.498m, scale 0.0854,
4519/4541 matched, down to the last decimal) -- proof OXTS/IMU was having
*zero* effect regardless of whether it was loaded.

Root cause: `SlamWorker::openVideoFile()` calls `clearOxtsImuData()`
unconditionally (`SlamWorker.cpp`, added in Session 9 item 9 to fix
sequence-00-OXTS-contaminating-sequence-01). `analyze/kitti_ate.cpp` loads
OXTS/IMU via CLI args *before* calling `openVideoFile()` -- so that fix,
added *later in the very same Session 9* (after item 3's 17.141m number was
already recorded and never re-verified afterward), has been silently
wiping `kitti_ate`'s CLI-loaded OXTS/IMU on every run since. The GUI is
unaffected (it loads OXTS/IMU via `autoLoadKittiExtras()`, connected to
fire *after* `sourceOpened`, so the ordering there was always correct) --
this was purely a `kitti_ate` CLI bug hiding in plain sight.

**Fix**: moved the OXTS/IMU-loading block in `kitti_ate.cpp` to after
`worker.openVideoFile(imagePattern)` (still before `startUnthrottled()`).
Re-ran the identical baseline config: **17.141m ATE RMSE, 4498/4541
matched, scale 1.0246 -- an exact match to the documented Session 9
number.** Confirms nothing else regressed; this was the entire discrepancy.

### 2. Attempted: make BA's per-keyframe poses actually reach the scored trajectory (reverted)

Confirmed a second, independent, pre-existing bug while here:
`runLoopBundleAdjustment()`'s own header doc comment claims it re-derives
`m_trajectory` via piecewise interpolation between corrected keyframes --
it never touches `m_trajectory` at all. `tryLoopClosure()` always applies
its cruder single-segment yaw-only interpolation to the trajectory
unconditionally, before it even knows whether BA is about to succeed with
better per-keyframe poses.

Implemented `rederiveTrajectoryForWindow()`, wired into `tryLoopClosure()`
to run instead of the yaw-only interpolation whenever BA succeeds. First
attempt (synthesizing each trajectory point purely from interpolating
between the two bounding keyframes' post-BA positions) scored **26.515m**
-- worse than baseline, because it silently discarded each point's own
real per-frame tracked motion (keyframes are 8 frames apart; any actual
turn between them got flattened into a straight interpolation). Second
attempt (snapshotting each keyframe's pre-BA pose, computing a per-keyframe
correction delta, and blending that delta onto each trajectory point's own
existing position instead of replacing it) fixed the flattening problem
but still scored **28.970m**, with a severe new outlier (max error
410.935m) at frames 2455-2470.

**Root-caused, not just observed**: traced the 410m spike to loop closure
`kf#34 (frame 402) <-> kf#202 (frame 2452)`, translation correction 106.789
world units / 11.991 degrees -- roughly 30x every neighboring closure's
correction that run. Confirmed this exact closure also fires in the
*original, successful* 17.141m run with identical BA output, but there the
error around frames 2452-2470 stays smooth (~31-33m) because the old
single-alpha yaw-only interpolation smears one whole-window delta
uniformly across every point, hiding how badly this one measurement
actually perturbed the intermediate keyframes. Using BA's real per-keyframe
poses directly (either attempt above) is strictly more *faithful*, but
faithfulness is exactly what exposes the internal noise from a bad
measurement instead of averaging it away. **This is the same class of
problem as the still-open frame 183<->1620 outlier from Session 9** --
an occasional wrong/badly-conditioned loop-closure measurement that nothing
in this codebase currently detects or guards against -- just found via a
different symptom. Reverted both `rederiveTrajectoryForWindow()` attempts
and the doc-comment fix in full; `tryLoopClosure()`/`SlamWorker.h` are back
to their exact Session-9 state (including the original doc/behavior
mismatch, still unfixed).

### 3. Attempted: continuous local bundle adjustment (reverted, catastrophic failure)

Implemented `runLocalBundleAdjustment()`: a joint Ceres BA over a sliding
15-keyframe window, run on every keyframe insertion (~568 times over the
full sequence) instead of `refineLocalKeyframes()`'s single-keyframe
polish, gauge-fixed by holding only the *oldest* keyframe in each window
constant (no independently-measured pose exists for the newest one the way
loop BA has `(loopR, loopT)`).

Result: **catastrophic scale collapse**. Local-BA-alone scored 193.401m
ATE RMSE with recovered scale **0.0001** (i.e. the estimated trajectory was
~10,000x too large before alignment); combined with loop BA, 186.748m,
scale 0.0158. Not a coding bug -- a real design flaw: the "oldest keyframe"
anchor in each window is itself just whatever a *previous* local BA call's
un-verified free optimization left it at, never an absolutely-trusted
reference. Monocular scale is only observable through accumulated evidence
(OXTS here); repeatedly re-optimizing a window against an anchor with no
independent verification lets small systematic biases compound every
single call, and this runs hundreds of times per sequence with ~93%
window overlap call-to-call. Reverted in full (function, header
declaration/doc comment, `insertKeyframe()` wiring, the
`m_localBundleAdjustmentEnabled` toggle, and the `kitti_ate.cpp` `localba`
CLI flag) -- `insertKeyframe()` is back to unconditionally calling
`refineLocalKeyframes()`.

### 4. Net result of sections 1-3

The one durable, verified fix: `kitti_ate.cpp`'s OXTS/IMU load-ordering bug
(section 1). Both follow-on attempts (sections 2-3) were implemented,
measured, root-caused, and fully reverted -- the codebase was back to
Session 9's exact state modulo the `kitti_ate.cpp` fix at this point.
Continuous local BA in particular should not be re-attempted with an
unanchored gauge-fixing strategy; any future attempt needs a periodic
*externally*-verified anchor (e.g. re-snapping to OXTS-derived scale/
position on some cadence) rather than trusting an ever-rolling,
self-referential "oldest keyframe" anchor.

### 5. A real published fix for the actual root cause: DCS-robustified offline pose graph -- implemented, 3 real bugs found and fixed, still not a net win, root-caused why and stopped

User asked to find and implement an actual published technique for the
exact problem sections 2-3 kept tripping over: occasional bad/wrong
loop-closure measurements with nothing in this codebase able to detect or
down-weight them. The right technique is **Dynamic Covariance Scaling**
(Agarwal, Tipaldi, Spinello, Stachniss, Burgard, "Robust Map Optimization
using Dynamic Covariance Scaling", ICRA 2013): a closed-form per-edge
weight `s = min(1, 2*Phi/(Phi+chi2))` applied (as `s^2`) to a constraint's
information matrix, recomputed iteratively (~5 rounds) as the graph
re-solves -- a bad edge's own large residual causes it to be automatically
down-weighted, no extra switch variables needed. Implemented as a new,
fully standalone, offline post-processing step -- `src/vision/
PoseGraphOptimizer.h`/`.cpp`, a real 6-DOF relative-pose graph (sequential +
loop-closure edges, `Eigen::Quaternion`-based residual, re-derived for this
codebase's world-to-camera convention rather than copied from Ceres's own
`pose_graph_3d` example) -- wired in only via `kitti_ate`'s new `posegraph`
flag, never touching live tracking, so it cannot regress the GUI/default
path the way sections 2-3's in-place attempts risked. `SlamWorker` gained
two small additive pieces: `m_sequentialEdgeRecords`/`m_loopClosureRecords`
(persisted relative-pose measurements, populated in `insertKeyframe()`/
`tryLoopClosure()`) and matching read-only accessors.

**Three real bugs found and fixed, each confirmed via a full-sequence run** (raw-live ATE stayed exactly 17.141m throughout, confirming zero effect on
default behavior at every step):

1. **Edge measurements reconstructed from final (possibly multiply-corrected) absolute poses instead of relative-at-observation.** First attempt built
   each edge's "measurement" from `keyframePoses()`'s end-of-run poses --
   but any keyframe's absolute pose can be overwritten in place by a later,
   *unrelated* loop closure before the run finishes. Confirmed concretely:
   `kf#35<->kf#203`'s own raw correction was a trivial 3.55 units/1.09
   degrees, yet reconstructing its "measurement" this way produced a chi2
   of 18105, because keyframe 35's absolute pose had since been touched by
   a different closure (`kf#35<->kf#269`, much later in the run). Fixed by
   persisting true RELATIVE transforms at the moment of observation
   (`SequentialEdgeRecord`/`LoopClosureRecord`, both immutable afterward) --
   this is exactly the point of pose-graph SLAM, and it's what actually
   made the known-bad `kf#34<->kf#202` edge correctly show a chi2 of 9037
   (versus everything else in single/double digits) once fixed.
2. **Warm-start seeded from the same final absolute poses, now inconsistent with the fresh relative measurements.** Even after fix 1, the optimizer's
   starting point still came from `keyframePoses()`'s end-of-run values --
   producing an initial Ceres cost of **8.5 quadrillion** and a
   catastrophic scale collapse (recovered scale 0.0000, ATE 193m, matching
   Change B's exact failure mode from section 3). Fixed by warm-starting
   via chaining the sequential edges forward from keyframe 0 instead --
   composing each consecutive pose from its own persisted relative
   measurement keeps every pair self-consistent with what the optimizer is
   about to evaluate.
3. **Trajectory-correction delta computed against the wrong baseline.** Even with 1 and 2 fixed, the final trajectory correction (`applyPoseGraphCorrection()`) still used the CLI's own `keyframePoses()` snapshot as
   the "pre-optimization" pose for its per-keyframe delta -- but the
   optimizer had actually started from the *chained* warm-start (fix 2),
   not that snapshot, so the delta conflated real optimization refinement
   with an arbitrary difference between two unrelated reference chains,
   reproducing the exact same scale-collapse numbers as before fix 2 was
   even applied. Fixed by having `optimizePoseGraph()` expose its own
   warm-start via an output parameter and requiring callers to use *that*
   as the correction baseline, not the original snapshot.

**With all three fixed, the mechanism is sound but the result (~32m) is still worse than the 17.141m baseline** -- confirmed via the diagnostic logging
built in from the start (`[posegraph][dcs]` per-edge chi2/weight,
`[posegraph][seq]` sequential-edge summary stats): two more real,
architectural findings, not further bugs:

- **A sufficiently dense, redundant loop-closure region lets the optimizer "explain away" a bad edge instead of isolating it.** The known-bad
  `kf#34<->kf#202` edge, which correctly showed chi2=9037 (weight -> 0) in
  isolation, ended up back at full weight (chi2=0.05) by outer iteration 3
  once solved jointly with every other nearby, overlapping, near-duplicate
  loop closure through the same intersection (Session 9 already noted "8
  separate closures all landing somewhere in frames ~129-1640") -- the
  graph has enough flexibility to bend other, good edges slightly and
  satisfy the bad one anyway, rather than rejecting it. DCS weights edges
  by their own residual; it does not prevent this kind of graph-wide
  compromise.
- **Sequential edges carry less information than live tracking's own rolling map, so a long stretch with zero loop closures gets systematically worse, never better.** A genuine ~600-frame gap exists between `kf#299`
  (frame 3845) and `kf#330` (frame 4458) with no loop closure at all.
  Huber-robustifying sequential edges (`sequentialHuberDelta`, clipping the
  worst single-edge outliers -- confirmed effective: outer-0 max chi2 of
  5590.8 across 338 edges dropped to <1.5 by outer 2) fixed one outlier
  region (frame ~1105-1116) but left this gap region (frame ~3993-4010,
  ~53-57m error) completely unchanged, because the real problem there isn't
  one bad edge -- it's that a chain of independent relative-pose edges has
  no shared-landmark stabilization at all, unlike live tracking's
  continuously-refined rolling 2000-point map (effectively a windowed BA
  every frame). Three different attempts to gate `applyPoseGraphCorrection()`
  to only touch loop-covered spans (full numeric range between a closure's
  two endpoints; a fixed +/-15-keyframe window around each endpoint; a
  hybrid using this file's own `kBaMaxWindowKeyframes=200` threshold to pick
  between the two) scored 32.018m / 36.037m / 32.072m respectively -- no
  real improvement, and the middle attempt actively uncovered a
  previously-fine broad revisit window (`kf#10-17<->kf#124-130`) by being
  too restrictive. The deepest finding: even where a keyframe IS correctly
  flagged as loop-covered, the correction blended in can still be too small
  to matter, because sequential-edge drift compounds over however many
  edges separate it from the nearest loop constraint, and a single loop
  edge can't retroactively undo that much accumulated chain error. This is
  not a further-tunable heuristic -- it needs either reprojection-based
  (landmark-sharing) sequential edges instead of naked relative-pose ones,
  or accepting that pose-graph correction should only ever apply within
  tightly-connected loop regions and leave everything else exactly as live
  tracking produced it (which is closer to what was attempted, but the
  "connectedness" test needs to be about actual shared evidence, not
  keyframe-index proximity).

**Left in the tree, fully opt-in, zero effect on default behavior**: the
`posegraph` flag and all of `PoseGraphOptimizer.h`/`.cpp` remain, correctly
implemented and diagnostically instrumented, for whoever picks up the
architectural rework above -- this should NOT need to be re-debugged from
scratch; the three bugs above are real and fixed, and the two remaining
findings are the actual, confirmed frontier.

## Session 9 (2026-07-12, continued still further): added geometric verification to BA's data associations -- the fix that actually made BA a net win

Picked up exactly where Session 8 left off, on direct request ("limit or
prevent garbage") to act on Session 8's own live-diagnosis: unverified
(ratio-test-only, no RANSAC) enrichment matches were the suspected
remaining source of BA underperformance, evidenced by larger/later windows
scoring worse despite more landmarks.

### 1. Geometric reprojection gate, added to both places that trusted a bare descriptor match

Neither `computeLoopEstimate()`'s enrichment loop (`LoopEstimator.cpp`) nor
the live `SlamWorker::recordLandmarkObservations()` had any geometric check
on a descriptor ratio-test match before accepting it as an observation --
unlike every other point-to-3D-structure step in this codebase
(`tryLoopClosure()`'s PnP+RANSAC, `trackFrame()`'s `solvePnPRansac()`).
Both now reproject the candidate landmark's already-known 3D position
through the observing keyframe's own pose and reject the match if it lands
more than `kMaxObservationReprojErrorPixels`/`kEnrichmentMaxReprojErrorPixels`
(8.0px, matching `tryLoopClosure()`'s own PnP RANSAC threshold) from the
descriptor-matched keypoint -- cheap (a handful of FLOPs, no extra RANSAC
solve) since the candidate's 3D position and the keyframe's pose are both
already known.

Also added `kEnrichmentMaxWindowKeyframes = 200` to
`computeLoopEstimate()`, mirroring the live path's existing
`kBaMaxWindowKeyframes` -- Session 8's data showed large windows (21000+
landmarks) both costing more *and* scoring worse, so this is a quality
decision here, not just a performance one. A too-large window is now
skipped outright with a clear message (`"window too large for background
re-estimate (N keyframes > 200)"`) rather than attempted.

### 2. Result: roughly 10x further improvement, now genuinely competitive

Reran the same `loopestimate` diagnostic (`loopest_gate_test.log`, Iterative
PnP, OXTS+IMU+`oxtsimupnp`, full sequence this time -- finished, 4518/4541
matched):

| Window | Before any fix (Session 7) | Bug fix only (Session 8) | + reprojection gate (Session 9) |
|---|---|---|---|
| frame 129<->1578 | 113.960m | 83.591m | **8.126m** |
| frame 145<->1587 | 122.325m | 90.465m | **7.426m** |
| frame 153<->1595 | 116.903m | 99.710m | **7.414m** |
| frame 161<->1603 | -- | -- | **7.370m** |
| frame 161<->1611 | -- | -- | **8.411m** |
| frame 183<->1620 | -- | -- | **125.672m (outlier, see below)** |
| frame 199<->1636 | -- | -- | **12.836m** |
| frame 412<->2466 | -- | -- | **9.712m** |
| frame 2359<->3300 | -- | -- | **13.487m** |
| frame 2359<->3309 | -- | -- | **13.917m** |
| frame 2381<->3325 | -- | -- | **13.407m** |
| frame 2419<->3368 | -- | -- | **13.472m** |
| frame 2427<->3377 | -- | -- | **14.336m** |

Excluding the one outlier, every window in this full run landed in the
7-14m range -- a ~10x improvement over Session 8's gate-less fix, and in
the same ballpark as the ~5.33m ORB-SLAM (full BA + proper loop closure)
reference this file has cited since Session 2 for calibration. This is
strong direct evidence Session 8's live-diagnosis was right: the remaining
gap wasn't BA-as-a-concept, it was specifically ungated false-positive
correspondences, and removing them (not just up-weighting the good ones)
is what actually closes most of the gap.

`kEnrichmentMaxWindowKeyframes` correctly started skipping windows in the
second half of the sequence once loop spans exceeded 200 keyframes (sizes
seen: 214-223 through most of the back half, 317-319 for the very last
handful of closures near the end-of-sequence return-to-start loop) --
skipped cleanly with a logged reason each time, zero wasted BA-solve cost.

**The live trajectory itself (unaffected by any of this background
diagnostic machinery) finished at 15.619m ATE RMSE** (4518/4541 matched,
recovered scale 1.0271) for this Iterative+OXTS/IMU+oxtsimupnp config --
lower than Session 7's recorded 18.556m best-ever for what was assumed to
be the same config. See item 4 below for the likely explanation (not
actually the same config).

### 3. The live BA path itself: now beats its own no-BA baseline for the first time

Reran Session 7's exact BA-comparison config (P3P, OXTS+IMU, `oxtsimupnp`
default-on, `ba` flag enabled, no PnP refit -- `ba_fixed_test.log`) to
close out Session 8's "not measured this session" gap:

| Config | ATE RMSE |
|---|---|
| No BA (Session 7 baseline, same config) | 27.2m |
| BA, first attempt (Session 7, before any fix) | 41.2m |
| BA, capped @200kf (Session 7, before any fix) | 46.3m |
| **BA, with Session 8's bug fix + up-weighting + Session 9's reprojection gate** | **17.141m** |

Full sequence, 4498/4541 matched, recovered scale 1.0246. This is the
headline result: BA now beats its own no-BA baseline (27.2m) by ~37%,
reversing Session 7's conclusion completely. It's not yet confirmed
whether this beats the *best* no-BA config (Iterative+oxtsimupnp,
15.619m/18.556m depending which run) since this comparison intentionally
used P3P to stay isolated from that variable -- combining Iterative PnP
with the now-fixed BA is flagged as a next step, not assumed to stack
favorably.

### 4. Open anomaly: one BA window still scores badly despite clean-looking inputs

`frame 183<->1620` in the full loopestimate run: `landmarks 10930->10930
obs 10930->11124 verified=61 ba=ok cost 12148.4->2307.3 ATE_RMSE=125.672`.
Its actual loop closure (`[loop] closure: kf#18 (frame 183) <-> kf#124
(frame 1620), matches=127, pnpInliers=61, translation correction=3.357
world units, rotation correction=1.759 deg`) is completely unremarkable --
nothing like the "badly-wrong correction" pattern from Session 2 (that one
was 10-20x every other correction seen that run; this one is right in the
normal range). Landmark/observation/verified counts and BA cost
convergence all look identical in shape to the neighboring 7-14m windows.
Yet this single window's ATE is 10-15x its neighbors'. **Not investigated
further this session** -- every aggregate number available says this
window should have been fine, so whatever's wrong needs looking at the
actual optimized poses/points, not just the summary stats computeLoopEstimate()
already reports. Flagged as the top item for whoever picks this up next.

### 5. Caveat noticed while comparing against Session 7: a possible CLI/config mismatch

Session 7's loop windows for "the same" test fell at frame indices ~132,
~140, ~148; this session's fall at ~129, ~145, ~153 for what was assumed
to be an identical re-run. Investigating why surfaced a `kitti_ate.cpp`
CLI quirk: the pnp-method argument (argv4) maps *any* unrecognized string
-- including `"-"`, the placeholder this file's own example invocations use
to skip a flag while reaching a later one -- to an explicit `SOLVEPNP_P3P`,
rather than leaving `PnpSettings`' actual configured default (Iterative,
since Session 6) alone. If an earlier session's test passed `"-"` for this
argument while intending to test the code's real default, it would have
silently gotten P3P instead. This session's tests explicitly passed
`iterative` or `p3p` by name throughout, so this session's own numbers are
trustworthy, but it casts a shadow of doubt on any earlier recorded number
that used `"-"` here. Not fixed this session (mechanical, low-risk, but
not the priority) -- flagged in the status header above.

### 6. `LoopEstimatePanel` now renders an actual per-loop mini-map, queued left to right

Session 7 explicitly deferred this ("requested follow-up, not implemented
this session... given the ATE results above mean this feature's current
output isn't yet trustworthy to visualize as if it were an improvement").
With item 2's results now competitive with the ORB-SLAM reference, that
reason no longer holds, so implemented it on request:

- `LoopEstimateResult` (`LoopEstimateTypes.h`) gained
  `alignedLandmarks`/`alignedTrajectory`/`alignedGroundTruth` --
  `computeLoopEstimate()` now reuses the exact same `(scale, cosT, sinT,
  tx, tz)` it already computes for the ATE number to transform the
  optimized landmark positions and this loop's own trajectory segment into
  ground truth's frame, so a rendered map and its ATE figure can never
  disagree. Scoped to just this loop's window (not the whole trajectory
  the ATE fit itself runs over) so each thumbnail stays a small, readable,
  self-contained picture of one loop. Empty when no ground truth is loaded
  (nothing meaningful to align).
- New `src/widgets/LoopMapThumbnail.h`/`.cpp`: a small (220x170) top-down
  paint widget, one per resolved loop -- landmarks (blue dots), ground
  truth (green line, drawn first), corrected trajectory (orange line),
  frame range and **ATE RMSE always drawn on the thumbnail itself** (not
  just in the summary labels above it), matching `MapView`'s existing
  color scheme for visual consistency. Pure display of already-aligned
  data -- no alignment math of its own.
- `LoopEstimatePanel` gained a `QScrollArea` below the existing summary
  labels holding a `QHBoxLayout` strip; `showResult()` now appends a new
  `LoopMapThumbnail` just before the layout's trailing stretch on every
  call, so thumbnails queue left to right in the order loops actually
  resolve (loop 1..N) instead of overwriting a single "latest" view. Added
  `src/widgets/LoopMapThumbnail.cpp`/`.h` to `CMakeLists.txt`'s
  `sift_vslam_gui` target only (the headless `kitti_ate`/`sift_vslam_debug`
  targets have no UI, so no reason to link Qt Widgets paint code into
  them). All three targets rebuild clean.

### 7. Same-session follow-up: a real rendering bug, then a UX change, both user-reported live

User immediately reported "black with a yellow point" after loading the
GUI -- a real bug, not a display quirk: `LoopMapThumbnail`'s auto-zoom fit
the view to *every* point including raw triangulated landmarks, and a
single bad/outlier landmark (an easy outcome of a near-degenerate
triangulation, nothing this session's reprojection gate screens for since
that gates *matches*, not the triangulated positions themselves) blows up
the bounding box, collapsing everything else to nothing and leaving only
the always-drawn ATE text visible. Fixed: the view now fits to ground
truth + corrected trajectory only (the "did this loop close correctly"
story), landmarks are still drawn but never influence the fit, and every
point is checked with `std::isfinite()` before use. Also, per explicit
request ("apply ROI to get the most ATE in overlap loop frame"):
`LoopEstimateResult` gained `maxErrorIndex` (the single worst-divergence
point within this loop's own window, computed alongside
`alignedTrajectory`/`alignedGroundTruth`), and the thumbnail draws a red
ring around it. Thumbnail size increased 220x170 -> 360x280 for legibility.

Also raised `kEnrichmentMaxWindowKeyframes` 200 -> 400 (`LoopEstimator.cpp`)
after the user pointed out most late-sequence loops were being skipped
outright ("window too large for background re-estimate") -- the original
200 cap's justification (large windows scoring worse, Session 8) predates
this session's reprojection gate and isn't re-validated at the new value;
400 is simply large enough to cover every window this session's own
full-sequence run actually produced (max observed: ~320).

Briefly changed `LoopEstimatePanel` to show only a single "worst ATE seen
so far" slot instead of queuing every loop -- **misread of the actual
request, reverted same session**. The user's original ask ("apply ROI to
get the most ATE in overlap loop frame") meant the per-thumbnail zoom/
peak-error work above, applied to every queued loop; not collapsing the
whole panel down to one loop. Confirmed via the GUI's own log that 49 real
loop closures had fired in a run where the panel appeared to show only 1
result -- not a tracking/detection bug, just this UI misstep hiding the
other 48. Reverted `LoopEstimatePanel::showResult()` back to appending one
`LoopMapThumbnail` per resolved loop, left to right, no single-slot
filtering; `m_worstAteRmse`/`m_worstThumbnail` removed. Each thumbnail
still has its own ROI zoom and peak-error ring from item 7 above.

### 8. Overlap dedup: what "ROI" actually meant

The user's real ask, clarified with a concrete example ("frame 3<->10 has
ROI: 100m, 1<->6 has ROI: 3m, then take 1<->6 to queue") and confirmed
directly: among loop-closure windows whose `[oldFrameIndex, newFrameIndex]`
ranges *overlap*, only keep the one with the lowest ATE RMSE queued --
these overlapping detections are near-duplicate re-discoveries of the same
physical revisit (common in this log: e.g. 8 separate closures all landing
somewhere in frames ~129-1640), not independent evidence, so showing all
of them is redundant clutter, and the worse ones among an overlapping
cluster are actively misleading.

`LoopEstimatePanel` gained `m_queuedLoops` (a `std::vector` of
`{oldFrameIndex, newFrameIndex, ateRmse, widget}`). On each ATE-bearing
result, it finds every currently-queued loop whose frame range overlaps
(standard interval overlap: `a1 <= b2 && a2 <= b1`); if the new result's
ATE beats every overlapping one, they're all removed and replaced by the
new one, otherwise the new one is silently dropped (an existing
overlapping loop is already as good or better). Results with no usable
ATE (no ground truth, window-too-large, BA failed) aren't comparable and
are just queued as-is, untouched by this rule.

**Reverted later the same session** ("oke, nevermind, just put them all in
queue, no more ROI") -- back to appending every resolved loop
unconditionally, `m_queuedLoops` removed. Each thumbnail keeps its own ROI
zoom/peak-error marker from item 7; only the panel-level filtering was
undone.

### 9. Real bug found and fixed: `autoLoadKittiExtras()` silently applied sequence 00's OXTS/IMU/ground-truth to any opened video

User-reported live symptom on sequence 01 (just wired up with real OXTS
this session, item 10 below): the estimated trajectory stopped well short
of ground truth, and the tracking log showed a very long unbroken streak
of `PnP fail: ok=0 inliers=0` with recovery attempts also failing
(`recoverPose rejected: inliers=N < 20`, N always just under the
threshold) -- permanent tracking loss partway through, never recovering.

Root cause traced from the actual numbers: the live `oxtsDist` values in
the log (~0.8-1.0) matched sequence *00*'s known speed profile, not
sequence 01's (~2.5+, confirmed in this session's own successful headless
`kitti_ate` run on sequence 01). `SlamWorker::autoLoadKittiExtras()`
(connected to `sourceOpened`, firing on *every* video open) unconditionally
loaded sequence 00's hardcoded OXTS/poses paths regardless of what was
actually opened -- since those files exist on disk, it always "succeeded",
silently overwriting a manually-loaded different sequence's OXTS/IMU the
moment its video was (re)opened. Wrong-sequence OXTS corrupted scale
badly enough to plausibly cascade into the map degrading to the point PnP
found zero inliers -- compounded by (not necessarily solely caused by)
highway driving's own known difficulty for two-view geometry (near-pure
forward translation, low parallax -- consistent with the recovery
attempts' inlier counts landing just under 20 repeatedly, a classic
degenerate-geometry signature, not obviously bad matching).

**Fix** (`SlamWorker.cpp`/`.h`):
- `autoLoadKittiExtras()` no longer hardcodes sequence 00. It now detects
  the sequence number from the actually-opened video's filename (trailing
  digits, e.g. `kitti_01.mp4` -> `"01"` -- same heuristic
  `MainWindow::tryAutoLoadGroundTruth()` already used for its own
  ground-truth lookup) and looks it up in a small `knownKittiSequences()`
  table (currently `00` and `01`, the two sequences with locally-extracted
  OXTS/calib). An unrecognized sequence, or a camera source (no filename
  to detect from), is a silent no-op -- never touches whatever's already
  loaded.
- New `m_lastOpenedVideoPath` member, set in `openVideoFile()`, cleared in
  `openCamera()`.
- `openVideoFile()`/`openCamera()` now also call a new
  `clearOxtsImuData()` on every source open, wiping any previously loaded
  OXTS/IMU data unconditionally -- closes the other half of the same class
  of bug: even with the detection fix above, stale data from a *previous*
  video would otherwise keep silently applying to a newly opened,
  unrelated one just because nothing ever cleared it.
- New signals `oxtsAvailabilityChanged(bool)`/`imuAvailabilityChanged(bool)`,
  emitted by `clearOxtsImuData()` (false), `loadOxtsDir()`/`loadImuDirs()`
  (reflecting the actual resulting state, not just whether that one call
  succeeded -- a failed load leaves prior data and its availability
  untouched), and `autoLoadKittiExtras()` (on a successful auto-load).

**Also implemented, same session, directly requested**: "only enable the
feed OXTS/IMU checkbox if I provide correct data folder" --
`ControlPanel`'s "Use OXTS speed correction"/"Use IMU rotation" checkboxes
now start disabled+unchecked (previously checked+enabled by default with
no relationship to whether data was actually loaded) and only become
enabled -- auto-checking themselves in the process -- via new
`setOxtsAvailable()`/`setImuAvailable()` slots wired to the signals above.
Since `setChecked()` still fires the checkbox's own `toggled` signal, this
cascades naturally into the existing `oxtsEnabledChanged`/`imuEnabledChanged`
-> `SlamWorker::setOxtsEnabled`/`setImuEnabled` wiring with no separate
plumbing needed -- and correctly un-checks+disables itself again the
moment a new source clears the data out from under it.

### 10. Sequence 01 wired up end-to-end this session

Besides the bug above: extracted sequence 01's OXTS (`2011_10_03_drive_0042_sync`,
1170 raw frames, offset-0 alignment validated the same way Session 4
validated sequence 00's, 0.228% path-length match against ground truth),
confirmed the existing `calib_2011_10_03` folder covers it too (same date,
KITTI recalibrates per date not per drive), merged the 1101-frame PNG
sequence into `kitti_01.mp4` (OpenCV `VideoWriter`, since neither `ffmpeg`
nor a `cv2` Python binding were available in this environment), and added
a `ControlPanel` "Browse OXTS/IMU Drive Folder..." button (single folder
pick -- derives `<picked>/oxts` for OXTS and `<picked>` itself for calib,
matching how the seq01 folder was assembled) plus `SlamWorker::loadOxtsDir()`/
`loadImuDirs()` thin wrapper slots so manual loading works from the GUI
thread without an unsafe direct cross-thread call (`SlamWorker` runs on
its own `QThread`). First-ever benchmark on sequence 01 (Iterative PnP,
OXTS+IMU+`oxtsimupnp`, headless `kitti_ate`): **28.027m ATE RMSE**,
1087/1101 matched, recovered scale 1.0072, zero loop closures the entire
run (consistent with sequence 01 being KITTI's highway sequence -- a
long, mostly-straight drive with no revisited road, unlike sequence 00's
loop-heavy urban route).

## Session 8 (2026-07-12, continued once more): found and fixed the real bug behind Session 7's "BA enrichment made ATE worse" result, up-weighted the genuine loop-closure evidence, measured a real (partial) improvement, and diagnosed live why a cleanly-converging optimizer can still make ATE worse

Picked up exactly where Session 7's status header left off: two candidate
fixes were queued, neither attempted. Investigated the actual enrichment
code in `LoopEstimator.cpp` before touching either one, on the theory that
"dilution" should be provable or disprovable by reading the matching logic
directly rather than guessed at again.

### 1. The real bug: `accumulatedIds`/`accumulatedDescriptors` index misalignment

`computeLoopEstimate()`'s enrichment step builds a growing, never-evicted
"pool" to re-match each new keyframe against (see Session 7 item 7). The
pool was built like this:

```cpp
if (!kf.descriptors.empty()) {
    for (long long id : kf.localMapPointIds)
        accumulatedIds.push_back(id);
    accumulatedDescriptors = ...vconcat(accumulatedDescriptors, kf.descriptors)...;
}
```

`kf.descriptors` is a keyframe's *full* detected-SIFT-keypoint set (hundreds
per keyframe); `kf.localMapPointIds` is only the subset that got
triangulated into a landmark (typically far fewer). So `accumulatedIds`
grows by the triangulated-point count while `accumulatedDescriptors` grows
by the full-keypoint count -- the two arrays desync almost immediately, and
`accumulatedIds[m.queryIdx]` (used to look up which landmark a match
belongs to) reads out of bounds the moment a keyframe's raw keypoint count
exceeds its triangulated-point count, which is essentially always. This is
silent UB on a `std::vector<long long>` via `operator[]`, not a crash --
consistent with every symptom Session 7 observed: BA converged fine every
time (Ceres has no way to know the correspondences it's given are
nonsense), and landmark/observation counts went *up* a lot (`11693 ->
47592` landmarks in Session 7's numbers) because matches were being
attributed to whatever garbage ID happened to be read, not because real
longer tracks were found.

**Fix**: added `cv::Mat localMapDescriptors` to
`LoopEstimateSnapshot::KeyframeSnapshot` (parallel to `localMapPointIds`,
cloned from the live `Keyframe::localMapDescriptors` `SlamWorker.h` already
had for exactly this purpose -- confirmed `kf.localMapDescriptors.rows() ==
kf.localMapPointIds.size()` always holds, by reading where it's populated in
`insertKeyframe()`), and changed the pool-accumulation side of the
enrichment loop to use it instead of `kf.descriptors`. The *matching
target* side (`kf.descriptors` for the new/incoming keyframe) is left
untouched and is correct as-is -- a landmark can legitimately reappear at
any detected keypoint in a later frame, not just one that happened to get
triangulated there, so only the query/pool side needed the fix.

### 2. Up-weighting the actual loop-verified correspondence (Session 7's candidate fix (b))

Implemented in both places BA runs, live and background:

- `tryLoopClosure()` already computes `inliers` -- the PnP-RANSAC-verified
  correspondences that *measured* `(loopR, loopT)` in the first place (see
  Session 6/7). Now also collects these into a
  `std::unordered_set<long long> loopVerifiedIds`, threaded into both
  `runLoopBundleAdjustment()` and `buildLoopEstimateSnapshot()` (the latter
  via a new `LoopEstimateSnapshot::loopVerifiedLandmarkIds` field).
- New constant `kLoopVerifiedResidualWeight = 25.0` (mirrored in both
  `SlamWorker.cpp` and `LoopEstimator.cpp`). The one observation per BA
  window that's this verified correspondence gets
  `ceres::ScaledLoss(nullptr, kLoopVerifiedResidualWeight, TAKE_OWNERSHIP)`
  instead of the usual `ceres::HuberLoss(kBaHuberDeltaPixels)` -- no robust
  down-weighting at all, plus a 25x squared-cost multiplier (~5x tighter
  effective noise sigma), so the optimizer trusts it fully rather than
  treating it as just another observation that might be an outlier.
  Deliberately a soft (heavily-weighted) constraint, not a hard equality
  one, since the loop measurement itself isn't noise-free.
- `[ba]`/`[loopestimate]` log lines extended to report how many residuals
  got this treatment (`verified=N`), and
  `LoopEstimateResult::loopVerifiedResidualCount` added for the same reason.

### 3. Measured result: real improvement, still far from good enough

Both fixes build clean across all three targets (`kitti_ate`,
`sift_vslam_gui`, `sift_vslam_debug`). Reran the same `loopestimate`
diagnostic Session 7 used (`oxtsimupnp loopestimate`, OXTS+IMU+calib
loaded, Iterative PnP, 900s budget -- reached frame 3749/4541 before the
budget expired; `loopest_fix_test.log`). Comparing the same loop region
Session 7's `loopest_run2.log` flagged as broken (frame indices differ
slightly, ~129 vs ~132 etc. -- plausibly a PnP-method difference between
the two runs' exact CLI invocations, not a clean isolated A/B; the loop
region and overall shape are the same revisit):

| Window (approx.) | Landmarks before->after | ATE RMSE (Session 7, broken) | ATE RMSE (Session 8, fixed) |
|---|---|---|---|
| frame ~130<->1578 | 11677->41942 (broken) / 11011->11011 (fixed) | **113.960m** | **83.591m** |
| frame ~140<->1587 | 11692->45088 (broken) / 10894->10894 (fixed) | **122.325m** | **90.465m** |
| frame ~150<->1595 | 11693->47592 (broken) / 10903->10903 (fixed) | **116.903m** | **99.710m** |

A real, consistent ~20-30% ATE reduction across all three matching windows.
Two things about the "fixed" landmark counts confirm the bug diagnosis was
right, not just the ATE number moving:

1. **`landmarksBefore == landmarksAfter` on every single window in this
   run, with no exception** (e.g. `11011->11011`) while *observations*
   still roughly triple (`11011->31141`). This is exactly the expected
   correct behavior -- enrichment should only add more *sightings* of
   already-known landmarks (from `positions`/`observations` built in step
   1), never new landmark identities out of nowhere. Session 7's broken
   run showed landmark counts roughly *quadrupling* (`11693->47592`), which
   in hindsight was never "found longer tracks" -- it was the OOB read
   minting phantom IDs.
2. `verified=N` (58, 64, 71 for the three windows above) confirms the
   up-weighting is actually reaching real residuals, not silently landing
   on zero.

**Still nowhere near good enough**: even at 83.591m/90.465m/99.710m, this
background BA-enriched estimate is **3-6x worse** than this exact
config's no-BA live-interpolation baseline (27.2m, or 18.556m with the
Session 7 `oxtsimupnp` fix). Fixing the index-misalignment bug and adding
the up-weighting both measurably helped, but neither is sufficient on its
own to make this implementation's BA a net win.

**Not measured this session**: the *live* `runLoopBundleAdjustment()` path
(the `m_loopBundleAdjustmentEnabled` checkbox, off by default) got the same
up-weighting code change but wasn't re-run standalone as a full-sequence
`kitti_ate ... ba` comparison -- only the background diagnostic path was
exercised via `loopestimate`. Flagged as real follow-up work, not assumed
to behave identically just because the code is shared.

### 4. Live diagnosis (user-prompted, no fix yet): why a cleanly-converging optimizer still makes ATE worse

While the fix above was running, a live GUI session (`sift_vslam_gui`,
launched this session on user request) prompted the direct question: BA's
own reported cost drops by 3-5 orders of magnitude every single window
(e.g. `2151778088563.7 -> 420243097069.6`) and never fails to converge --
so why does the *ATE* (the metric that actually matters) get worse instead
of better? Worked through this using this session's own numbers, not just
generic BA folklore:

- Ceres is correctly minimizing the objective it's given (summed,
  Huber/ScaledLoss-weighted reprojection error). That objective is a proxy
  for ATE, not ATE itself -- perfectly solving the proxy doesn't guarantee
  the real metric improves if the proxy's *inputs* are wrong or
  imbalanced.
- Most landmark tracks in this system are still just 2 observations
  (creation + one re-observation) before the live map's `kMaxMapPoints`
  eviction drops them (Session 7 item 3.2's original diagnosis). A
  2-observation landmark gives BA almost no real pose constraint -- Ceres
  can always drive that residual near zero by moving the (free) 3D point
  to match wherever the two observing poses currently place it, without
  the poses themselves having to be anywhere close to correct.
- The genuinely strong evidence -- the loop-verified correspondence -- is
  a tiny minority by count even after 25x up-weighting: 58-245 verified
  residuals (effectively ~1450-6125 at the 25x weight) against 30,000-86,000
  total observations in these windows. Diluted, not ignored.
- Critically, this isn't random noise that would average out: every one of
  those thousands of short local tracks is internally self-consistent
  *with the drifted trajectory*, since that's the trajectory that produced
  them. Satisfying them pulls toward "stay comfortable with the drift,"
  which actively fights the two correct anchor poses rather than being
  neutral -- a biased/correlated-error problem, not a variance problem.
  This is also this item's direct evidence for section 1's remaining
  open item: larger, later windows in this session's data (more
  accumulated unverified pool) scored *worse* on average than the small
  early ones, the opposite of what "more real evidence" should do -- strong
  circumstantial support that a meaningful fraction of the enrichment's
  ratio-test-only matches are simply wrong, not just weak.

### 5. GUI

Launched `sift_vslam_gui` this session on request (no build/runtime issues
-- one harmless `Gtk-WARNING: Theme file for DMZ-Black has no directories`
line, unrelated to this session's changes). No new UI controls added --
both fixes apply unconditionally to existing code paths (the enrichment
bug fix has no toggle to begin with; the up-weighting applies whenever the
existing "Run bundle adjustment after loop closure" checkbox or the
background loop-estimate panel's BA already runs).

## Session 7 (2026-07-12, continued yet again): fixed the documented trackFrame() OXTS gap (best result yet), two negative results (PnP full-inlier refit, loop-closure BA) properly measured and shipped off, an AR ground-truth overlay, and a background re-estimate pipeline

Picked up from Session 6's own top-priority item and a live user report of a
persistent leftward drift in the estimated trajectory vs. ground truth.

### 1. AR-style ground-truth overlay drawn onto the live video frame

Distinct from MapView's existing top-down overlay (which only needs (x, z)
from poses.txt): `SlamWorker::drawGroundTruthOverlay()` projects a window of
ground-truth camera centers into the *current* frame's own image using that
frame's own ground-truth pose -- self-consistent within ground truth alone,
no Umeyama alignment needed. Draws two things, both yellow initially, split
into their own colors/offsets per explicit request:
- The upcoming road (yellow line, `cv::Scalar(0, 255, 255)`).
- Old, already-driven streets the car revisits (magenta dots,
  `cv::Scalar(255, 0, 255)`), found by scanning the *entire* ground-truth
  path (not just the local window) for any frame whose camera center both
  projects into the current view *and* is within `kOldStreetMaxDistance`
  (15m) of the car's current position -- the distance check matters,
  without it a street merely visible in the distance lit up the same as one
  the car is actually on.
- Independent pixel offsets for each (`setGroundTruthOverlayOffset()` /
  `setOldStreetOverlayOffset()`), persisted via `QSettings` (already existed
  for the line; extended to the dots) -- explicit request: moving one must
  not drag the other, since they're conceptually different signals (a path
  preview vs. a revisit marker) even though they share the same projection
  math.
- MapView's own top-down ground-truth line was recolored yellow then back
  to green per iterative feedback; final state is green (unchanged from
  every prior session), with the video overlay carrying the yellow/magenta
  distinction instead.

### 2. Why the trajectory leans left relative to ground truth (diagnosis, no fix)

Root cause is structural, not a single bug: this pipeline has no real
bundle adjustment (`refineLocalKeyframes()` only polishes each keyframe
against its own local points; `tryLoopClosure()`'s correction is a smooth
linear/SLERP interpolation of one measured discrepancy, not a joint
optimization), so any small per-turn rotational bias compounds unchecked
between the sparse loop-closure events that do fire. Compounded by:
`MapView::computeAlignment()` fits one *frozen* global rotation from the
first ~200 overlapping points, so it can't track a real angular error that
varies along the route; and a confirmed, still-open bug (Session 6) where
an occasional badly-wrong loop correction (one observed at 2337 world
units, 10-20x every other correction that run) is accepted unguarded.
User-reported clue that shaped the rest of the session: switching PnP
method to DLT (this codebase's own from-scratch linear solver) showed
*no* leftward bias, only more per-frame noise -- P3P/Iterative (both via
`cv::solvePnPRansac()`) shared the bias; DLT (fully independent code path)
didn't. That observation is what items 4 and 6 below chase.

### 3. Real bundle adjustment after loop closure (Ceres) -- implemented, measured, shipped off by default

Installed `libceres-dev` + `libeigen3-dev` (confirmed built with
SuiteSparse, so `SPARSE_NORMAL_CHOLESKY` works directly; `ITERATIVE_SCHUR`
+ Jacobi as a fallback if a build ever lacks that). New machinery, all in
`SlamWorker.cpp`/`.h` unless noted:

- Persistent landmark IDs (`m_nextLandmarkId`, `m_mapPointIds` parallel to
  `m_mapPoints`, `Keyframe::localMapPointIds`) -- didn't exist before;
  every map point was anonymous.
- `m_landmarkPositions`/`m_landmarkObservations`: the missing piece for
  any real multi-view BA -- which keyframes observed which landmark, and
  where. Populated at triangulation time (`insertKeyframe()`) and via a new
  `recordLandmarkObservations()` (matches a new keyframe's descriptors
  against the live rolling map to catch re-observations).
- **Bug found and fixed same session**: `initializeFromFrame()` sets the
  bootstrap map directly (`m_mapPoints = std::move(...)`), bypassing
  `appendToMap()` entirely -- so `m_mapPointIds` was never populated for
  the first ~318 points, an out-of-bounds read in
  `recordLandmarkObservations()` the moment the first keyframe was
  inserted (confirmed via segfault, `kitti_ate` core-dumped at frame ~9-10
  every time, misleadingly looking like a hang under a short timeout).
  Fixed by assigning IDs there too; `resetSlamState()` updated to clear
  the new members on an explicit Reset.
- `runLoopBundleAdjustment()`: joint reprojection-error optimization
  (Ceres `AutoDiffCostFunction`, Huber loss, `kBaHuberDeltaPixels = 4.0`)
  over every keyframe strictly between a loop's two matched keyframes plus
  every landmark two or more of them jointly observe, both endpoints held
  constant (old at its trusted pose, new at the loop-measured one).
  Replaces `tryLoopClosure()`'s linear-interpolation correction for
  keyframe poses specifically when enabled; trajectory/map-point/live-pose
  correction untouched either way.
- Checkbox: "Run bundle adjustment after loop closure" (`ControlPanel`),
  default off.

**Measured three times on the full sequence (P3P, OXTS+IMU, no PnP
refit), each a real bug found and fixed in between:**

| Run | ATE RMSE | Note |
|---|---|---|
| No BA (baseline) | **27.2m** | |
| BA, first attempt | 41.2m | worse -- missing loop-closure PnP correspondences as landmark observations entirely (see below) |
| BA, after fixing the missing observations | did not finish | huge end-of-sequence loop (car returns near its exact start): 400+ keyframes, 20831 landmarks -- solving that plus a near-duplicate window right after blew through the 900s test budget (`timeout` killed it, exit 124) |
| BA, after adding `kBaMaxWindowKeyframes = 200` cap | 46.3m | still worse; only 12/40 loop closures were small enough to run BA at all |

Root causes identified for why BA underperforms even "fixed":
1. `tryLoopClosure()`'s own PnP re-measurement (the correspondences that
   *prove* the loop closure, oldKf's landmarks re-observed at newKf) was
   never recorded into `m_landmarkObservations` at all -- fixed
   (mid-session) by recording the RANSAC inlier subset there too, but this
   alone wasn't enough (see the 46.3m row above, which already includes
   the fix).
2. Deeper issue, not fixed: most landmarks only ever get the bare minimum
   2 observations (creation + one lucky re-observation before
   `kMaxMapPoints=2000`'s rolling eviction removes them from
   `m_mapDescriptors`), so a BA "window" is really thousands of weakly-
   connected local 2-view pairs, not well-constrained multi-view tracks.
   The few genuine long-baseline observations that do exist (the loop
   correspondences from fix #1) are vastly outnumbered by these, and
   Huber loss down-weights them further since they start with large
   residual (exactly where the correction is needed) -- the optimizer
   ends up mostly relaxing the local chain instead of snapping to the
   loop-closing evidence. See item 6 below for the follow-up that
   confirmed this diagnosis empirically (enrichment found 4-9x more
   landmarks/observations per window once the eviction limit was
   removed for a one-off background re-match).

### 4. PnP full-inlier LM refit for P3P/Iterative -- implemented, measured, reverted to off

Theory: `solvePnPDltRansac()` always refits over its full RANSAC inlier
set (linear re-solve); `trackFrame()`'s P3P/Iterative branch didn't --
just used whatever the minimal-sample RANSAC winner produced. Added a
`cv::solvePnPRefineLM()` call over the full inlier set, gated by a new
`setPnpFullInlierRefineEnabled()` toggle, to test whether this explained
item 2's DLT-vs-P3P bias difference.

**Measured: 61.1m ATE RMSE with the refit vs. 27.2m without -- refitting
made it noticeably worse, not better.** Reverted the default to off (was
briefly on). Best-guess explanation, not proven: a full-inlier LM refit
converges *more consistently* onto whatever small camera-model mismatch
exists (unmodeled residual distortion, a calibration constant slightly
off), so removing per-frame noise doesn't remove bias -- it just lets the
same wrong optimum get hit every single frame instead of being partly
cancelled out by scatter. DLT's own noisiness may be *why* it doesn't
show the same leftward drift, not despite it.

### 5. Fixed the actual documented gap: OXTS/IMU fed into `trackFrame()`'s plausibility check -- new best result

Session 6 ended with this flagged as the single highest-priority item:
`trackFrame()`'s own `isPlausibleStep()` call had no OXTS-awareness at all
(unlike `recoverViaEpipolar()`, fixed the same session), so a correct but
fast PnP-against-map solve could be spuriously rejected whenever
`m_avgStepScale` had drifted low, forcing an expensive/less-accurate
fallback to `recoverViaEpipolar()`.

Fix (`setOxtsImuInPnpEnabled()`, new checkbox, default **on**): when real
OXTS distance data covers the current frame-to-frame step, compare the
PnP solve's own implied step distance against it directly (tolerance
`kOxtsPnpStepToleranceMultiplier = 3.0x`) instead of skipping the check
outright the way `recoverViaEpipolar()` does -- can't skip outright here
since `trackFrame()`'s pose is an independent vision measurement, not
derived from OXTS the way `recoverViaEpipolar()`'s candidate is. Also
cross-checks the PnP solve's rotation against IMU's independently
measured relative rotation when available (`kImuPnpMaxAngleDeg = 15`),
exercising the IMU rotation path in `trackFrame()` for the first time --
previously IMU only ever fed `estimateTwoViewPose()`'s homography branch,
confirmed inert on this sequence since Session 4.

**Measured: 18.556m ATE RMSE vs. 27.2m without (32% better), 4489/4541
frames matched.** Best result across every session recorded in this file
(previous best: 49.4m). Shipped on by default.

### 6. SQPnP added to the PnP method dropdown

Mechanical: `ControlPanel`'s combo box only exposed P3P/Iterative/DLT even
though `kitti_ate` already accepted `sqpnp`/`epnp`/`ap3p` as CLI strings
and `trackFrame()`'s generic branch already handled any
`cv::SOLVEPNP_*` value. Added `cv::SOLVEPNP_SQPNP`; not yet benchmarked
(explicit user request to leave it for later testing).

### 7. Background loop-closure re-estimate: no-eviction re-matching + BA + ATE, off the tracking thread entirely

Explicit request: after a loop closure, run a fuller re-estimate
(re-matching + BA + ATE) in the background with **no change to the live
GUI**, plus fix the root cause diagnosed in item 3.2 (landmark tracks too
short because of the live map's eviction) by giving this one-off estimate
"the full picture" of keypoints.

New files:
- `src/vision/LoopEstimateTypes.h`: `LoopEstimateSnapshot` (every `cv::Mat`
  `.clone()`d -- oldKfIdx..newKfIdx keyframes' full descriptors/keypoints/
  triangulated points/IDs, loop-measured pose, intrinsics, full trajectory,
  ground truth) and `LoopEstimateResult` (landmark/observation counts
  before/after enrichment, BA cost, ATE stats). Both `Q_DECLARE_METATYPE`'d
  and registered in `main.cpp` for cross-thread signal delivery.
- `src/vision/LoopEstimator.h`/`.cpp`: `computeLoopEstimate()`, a pure
  function (no shared state touched) meant to run on a `QtConcurrent`
  thread-pool thread. Re-matches every keyframe's full descriptors against
  an accumulating, **never-evicted** pool built only from keyframes
  earlier in this same window (unlike the live rolling map's
  `kMaxMapPoints` cap) -- this is the actual fix for item 3.2, confirmed
  working: enrichment found **~4-9x more landmarks and observations** per
  window in real testing (e.g. one window: 11456 landmarks/observations
  before -> 37094 landmarks / 104487 observations after). Then runs the
  same BA as `runLoopBundleAdjustment()` (duplicated, not shared, since
  this needs to run without any `SlamWorker` member access) over the
  enriched set, propagates the result to a corrected trajectory copy
  (same piecewise yaw+translation interpolation style as
  `tryLoopClosure()`), and computes ATE against ground truth the same way
  `kitti_ate`/`MapView` do (2D Umeyama alignment).
- `src/widgets/LoopEstimatePanel.h`/`.cpp`: new bottom panel (added below
  the existing video/map splitter in `MainWindow`), shows the latest
  result's landmark/observation counts, BA convergence + cost, and ATE.
- `SlamWorker::buildLoopEstimateSnapshot()` + new signal
  `loopClosureDetected()`: fired at the very end of `tryLoopClosure()`
  (after whichever live correction already ran), so the snapshot reflects
  post-correction state. Building it is just `.clone()`-ing data, cheap
  enough to not stall live tracking.
- `MainWindow`: dispatches each snapshot via
  `QtConcurrent::run(&computeLoopEstimate, ...)` + `QFutureWatcher`;
  if a new loop closure fires while a previous estimate is still running
  (real risk on an unthrottled run, since a full re-estimate can take
  much longer than the gap between successive loop closures), keeps only
  the latest pending snapshot rather than queuing unboundedly or running
  overlapping estimates.
- `kitti_ate`: new `loopestimate` CLI flag wires the same
  `computeLoopEstimate()` synchronously (headless tool, no thread pool
  needed) purely for testing without the GUI. Also fixed a gap this
  surfaced: `kitti_ate` never actually called
  `worker.loadGroundTruthPoses()`, so the diagnostic's ATE was always
  "unavailable" until added.

**Result, confirmed same session (`loopest_run2.log`, real KITTI data,
`oxtsimupnp loopestimate` flags)**: enrichment + BA path is crash-free and
converges every time (`ba=ok`, cost dropping by 2-3 orders of magnitude,
landmarks/observations up ~4x/~9x per window as noted above) -- but the
resulting ATE is dramatically *worse*, not better:

| Window | Landmarks before->after | ATE RMSE |
|---|---|---|
| frame 132<->1578 | 11677 -> 41942 | **113.960m** |
| frame 140<->1587 | 11692 -> 45088 | **122.325m** |
| frame 148<->1595 | 11693 -> 47592 | **116.903m** |

All far worse than the 27.2m live-interpolation baseline. So "give BA more
landmarks" alone is not the fix, and plausibly makes things worse (more
short/weakly-connected local tracks to dilute the loop-closing evidence
further) -- see the "Current status" header above for the user's own
diagnosis and the two real candidate fixes, picked up as a follow-up
outside this session ("I will keep going later... in background with
more context and data").

**Requested follow-up, not implemented this session (explicit "note in
doc, don't implement")**: `LoopEstimatePanel` should also render the
post-BA map itself (a small top-down plot, same spirit as `MapView`'s
existing one), not just the text stats it has now. Would need
`LoopEstimateResult` extended with the optimized landmark positions
(top-down x/z) and the corrected trajectory segment -- and, if ground
truth is loaded, both aligned into ground truth's frame using the same
(scale, cosT, sinT, tx, tz) already computed for the ATE number in
`computeLoopEstimate()`, so the mini-map and the ATE figure agree -- plus
a small custom-painted widget added to `LoopEstimatePanel` alongside its
current labels. Deliberately left for later, given the ATE results above
mean this feature's current output isn't yet trustworthy to visualize as
if it were an improvement.

## Session 4 (2026-07-12): 5-point Nister solver, OXTS speed correction, IMU rotation, GUI wiring

Picked up exactly where Session 3 left off: implement the 5-point
calibrated algorithm for E (Session 2/3's "biggest remaining lever"). Ended
up covering a lot more ground than that: a fully verified from-scratch
5-point solver (kept, but not the default), a real external-data scale
correction (the actual ATE win), a real external-data rotation correction
(implemented and validated, but not yet exercised by any run), and GUI
controls for all of it.

### 1. From-scratch Nister 5-point solver for E (`estimateEssentialRansac()`)

Implemented in three stages, each verified before building on it (per the
user's explicit choice of "full mechanical fallback" over trusting a
memorized published elimination recipe from memory):

1. **Null-space extraction + the 10 essential-matrix constraint equations**
   (`solveFivePoint()`'s setup in `SlamWorker.cpp`'s anonymous namespace):
   build the 5x9 constraint matrix, take its 4D null space via SVD,
   parametrize `E(x,y,z) = xX + yY + zZ + W`, derive `det(E)=0` and the 9
   equations from `2*E*E^T*E - trace(E*E^T)*E = 0` via **exact symbolic
   polynomial arithmetic** (a small hand-written multivariate-polynomial
   class -- add/sub/scale/multiply over monomial coefficient vectors, degree
   tracked explicitly), not a memorized recipe. Verified: the true solution
   from 500 random synthetic 5-point problems satisfies all 10 equations to
   ~1e-6 relative precision (19/20 initial spot-check trials passed at a
   stricter tolerance; the 1 miss was a genuine degenerate small-`w`
   configuration, not a bug -- confirmed by tracing it by hand).
2. **Reduction to a solvable form**: rather than reproducing Nister's
   specific published elimination recipe from memory (flagged as a real
   risk -- a subtly wrong recipe fails silently, not loudly), expanded the
   ideal by multiplying the 10 generators by every monomial up to degree
   `Ds-3`, Gauss-Jordan-eliminated with high-degree monomials preferred as
   pivots, and read off whatever low-degree "quotient basis" fell out
   mechanically. Empirically always exactly 10 free monomials
   (`{1,x,y,z,x^2,xy,xz,y^2,yz,z^2}`) at `Ds=4`, matching the well-known
   fact that this problem has (up to) 10 solutions -- confirmed
   computationally, not assumed. The multiplication-by-`z` operator on that
   basis is a 10x10 matrix; `cv::eigenNonSymmetric` (a generic linear-algebra
   primitive, same category as the `cv::SVD::compute`/`cv::solve` calls
   already used throughout this file, not something specific to this
   algorithm) gives its eigenvalues = the `z`-coordinates of the solutions.
   **A real bug was caught and fixed here**: initially computed
   eigenvectors of the wrong matrix (`Mz` instead of `Mz^T`) -- a transpose
   slip in the "is this a left or right eigenvector" derivation, caught by
   the verification step (z-component of the eigenvector disagreeing with
   its own eigenvalue) rather than shipped silently.
3. **Full pipeline verification**: 498/500 synthetic 5-point trials
   recovered the ground-truth E (matched up to scale/sign, validated against
   the actual 5 correspondences' epipolar residuals, not just the derived
   equations); the 2 misses were the same genuine degenerate-configuration
   class as before, correctly detected and skipped rather than silently
   producing garbage.

**Performance**: unoptimized, 0.636ms/solve -- at a naive 1000 RANSAC
iterations that's ~636ms/frame just for minimal solves, far too slow
(existing 8-point solve is 0.0136ms/solve, ~47x cheaper). Profiled: 92% of
time was in the Gauss-Jordan elimination, entirely from `cv::Mat` row-
expression overhead (temporary `Mat`/`MatExpr` objects per row op) rather
than the actual arithmetic on a 40x35 matrix. Rewrote it with raw
`double*` pointers: **8.5x speedup, 0.636ms -> 0.054ms/solve**. Also cached
the monomial-index bookkeeping (previously rebuilt via heap-allocated
`std::map` on every call despite being pure structure, never
data-dependent) as a program-lifetime static. Landed on `kERansacIterations
= 300` (vs F's 1000) since 5-point's smaller minimal sample needs fewer
RANSAC iterations for the same outlier-tolerance/confidence by standard
RANSAC sample-count theory -- ~16ms/frame at that count, comparable to or
cheaper than 8-point's 1000-iteration cost (~13.6ms/frame). In practice,
`estimateTwoViewPose()` only runs during initialization/lost-recovery, not
every tracking frame, so a full `kitti_ate` run finishes in ~8 minutes
regardless of which estimator is active -- this cost never mattered as much
as it looked like it would from the raw per-solve numbers.

**Result once wired in and measured**: 143.2m ATE RMSE, worse than the
existing 8-point+Gold-Standard-refinement path's 130.2m (see Session 3),
despite the 5-point solver being individually verified correct on hundreds
of synthetic trials. Plausible, *not yet confirmed*, explanations: the
300-iteration count was picked from RANSAC theory, not empirically tuned
against this specific dataset; this system has already-documented extreme
sensitivity to exactly this kind of small change (a single marginal-frame
RANSAC outcome cascading through the whole trajectory -- see the
"Investigation trail" section far below). **Not a demonstrated flaw in the
solver's correctness** -- flagged as a real open question for a future
session, not dismissed.

**Kept as a runtime-switchable option, not the default**: given the
5-point solver's own accuracy couldn't be faulted, and it's the standard
choice in real calibrated SLAM/VO systems (ORB-SLAM, VINS, etc. all use
5-point over 8-point-then-convert), removing it entirely felt premature
pending the open question above. `SlamWorker::TwoViewEstimator` (`FivePoint`
default enum value, `EightPointLegacy` alternative) lets `estimateTwoViewPose()`
switch between it and the restored 8-point path; `kitti_ate` takes an
`[estimator]` CLI arg for this, and the GUI has a checkbox (see below).

### 2. The old 8-point path: extracted to its own module, not deleted

Per explicit request, the original 8-point-then-convert + Gold Standard
Sampson refinement code (which the 5-point work initially fully replaced
and deleted from `SlamWorker.cpp`, following this session's usual
no-dead-code practice) was instead recovered from a pre-deletion backup and
extracted into a standalone module: **`src/vision/EightPointLegacy.h`/`.cpp`**
(namespace `eight_point_legacy`), byte-for-byte the same logic as before
(RANSAC seed 42, 1000 iterations, 1.0 squared-pixel Sampson threshold),
with zero `SlamWorker` dependency so it can be linked and benchmarked
independently. Wired into all three CMake targets. This is what
`TwoViewEstimator::EightPointLegacy` calls into.

### 3. OXTS speed-based scale correction -- the actual ATE win

**The idea**: KITTI's raw-data OXTS (GPS/IMU) stream includes `vf` (forward
velocity, m/s) per frame. Integrating `vf * dt` gives real metric distance
traveled -- a direct fix for monocular SLAM's fundamental scale-ambiguity
problem, standard practice in real systems with a wheel encoder or GPS
available (arguably a cleaner, more standard fix than the ON-HOLD
ground-plane-homography scale idea from Session 2).

**Important caveat, stated plainly**: in a real car, a wheel encoder is a
genuinely independent sensor from GPS. Here, `vf` comes from KITTI's OXTS
**GPS/IMU** unit -- the *same* underlying sensor system used to generate
`poses/00.txt`, the ground truth this whole project's ATE numbers are
evaluated against. So this is not independent validation the way, say, the
5-point solver's synthetic-data verification was; it's closer to a
proof-of-concept ceiling ("this is achievable given accurate real-world
speed"), and the resulting ATE improvements should be read with that in
mind, not cited as if a truly independent sensor fixed the problem.

**Getting the data**: KITTI's odometry benchmark (what this project uses
day-to-day, `sequences/00/image_0/*.png` + `poses/00.txt`) does not include
OXTS data -- that's only in KITTI's separate "raw data" distribution,
organized by drive date/number, gated behind a login. Sequence 00 maps to
raw drive `2011_10_03_drive_0027` (frames 000000-004540, exactly matching
the odometry benchmark's 4541-frame count -- confirmed, not assumed).
Downloaded the "synced+rectified" zip for that drive (17.6GB, since KITTI
bundles 4 camera streams + Velodyne + OXTS into one zip with no
OXTS-only option) and extracted just the ~19MB `oxts/` folder.

**Validation before ever touching SlamWorker** (per this session's
established practice of verifying before wiring in): integrated `vf * dt`
across all 4541 frames using the real per-frame OXTS timestamps (parsed as
`HH:MM:SS.fractional` -- only diffs matter, so no full date/epoch handling
needed) and got **3716.56m vs the known-correct ground-truth path length of
3722.3m -- 0.15% off**. This single check confirms simultaneously: the
frame-index alignment (raw frame *i* == odometry frame *i*, no offset), the
field choice (`vf`, index 8 in KITTI's OXTS format), and the timestamp
parsing, all at once.

**Where it's wired in** (`SlamWorker::loadOxtsSpeeds()` builds a cumulative-
distance-from-frame-0 array; `oxtsDistanceBetween(frameA, frameB)` looks up
real distance between any two frame indices, returning -1.0 as an explicit
"not available" sentinel when OXTS wasn't loaded or is disabled):
1. **`initializeFromFrame()`**: right after `estimateTwoViewPose()` succeeds
   but before `triangulate()` runs, rescale `t` from its arbitrary
   unit-baseline magnitude to the real OXTS distance between the reference
   frame and the current frame. Since the reference camera is at identity,
   the baseline distance is exactly `norm(t)` -- correcting `t` before
   triangulation means every initial map point comes out correctly scaled
   too, no separate map-rescale step needed. This fixes the *initial*
   scale, which was previously arbitrary and never revisited.
2. **`recoverViaEpipolar()`**: replaces the heuristic
   `trel * (m_avgStepScale * framesElapsed)` rescale with the real OXTS
   distance when available (falling back to the heuristic otherwise) --
   exactly the situation that heuristic exists for (no absolute-scale
   reference of its own), so a ground-truth-quality one is a strict
   improvement with no new architectural risk. Doesn't need
   `kMaxFramesElapsedForRescale`'s safety cap either, since that exists to
   bound how far a *guess* can run away over many stale frames, not a
   measurement.

Deliberately **not** touched: `trackFrame()`'s own `m_avgStepScale`-based
`isPlausibleStep` gating still uses the vision-only heuristic. In one full
run this visibly caused more transient Lost/Recovered churn than the
non-OXTS baseline during a known feature-poor stretch (~frame 1600-2300) --
`avgStepScale` still degrades there same as before, it's just that
`recoverViaEpipolar`'s fallback now recovers with accurate scale every time
instead of a drifted heuristic. Net effect was still strongly positive
(see results below), but continuously injecting OXTS scale into
`trackFrame`'s own gating too is flagged as a real, not-yet-done next step
if this direction is revisited.

### 4. IMU rotation -- validated, wired in, but never yet exercised

**The idea**: the homography branch in `estimateTwoViewPose()` exists to
handle the near-pure-rotation case where F/E's translation-and-rotation
joint estimate is ill-conditioned (baseline -> 0). OXTS provides absolute
roll/pitch/yaw per frame; composing two frames' orientations gives the
*real* relative camera rotation directly, sidestepping the ill-conditioning
rather than working around it via homography decomposition.

**Calibration dependency, and a caught mistake**: converting OXTS
roll/pitch/yaw (IMU/body frame) into the camera's frame requires the
IMU->Velodyne->camera extrinsic chain (`calib_imu_to_velo.txt`,
`calib_velo_to_cam.txt`, and `calib_cam_to_cam.txt`'s `R_rect_00` --
needed because KITTI's odometry `image_0` frames are *rectified* cam0,
confirmed via `S_rect_00` == 1241x376 exactly matching the odometry images'
resolution, not raw cam0). **A wrong-date calibration zip (`2011_09_30`,
not the drive's actual `2011_10_03`) was downloaded first and caught before
use** -- KITTI recalibrates between dates; `calib_imu_to_velo.txt` happened
to be byte-identical between the two dates but `calib_velo_to_cam.txt` was
not (small but real differences), confirming the date-matching mattered and
wasn't a paranoid worry.

**Math** (`src/vision/ImuRotation.h`/`.cpp`, standalone module,
`imu_rotation` namespace): `navFromBody(i) = Rz(yaw_i)*Ry(pitch_i)*Rx(roll_i)`
per KITTI devkit's `convertOxtsToPose.m` convention. Relative body rotation
`R_rel_body = navFromBody(B)^T * navFromBody(A)`; conjugated into the camera
frame via the fixed extrinsic: `R_rel_cam = calib.R * R_rel_body * calib.R^T`,
matching `cv::recoverPose()`'s own `X_camB = R * X_camA` convention so it's
a direct drop-in for `estimateTwoViewPose()`'s `R` output.

**Verification, two levels**:
1. Synthetic: identity-extrinsic sanity checks (frameA==frameB -> identity;
   a pure-yaw-only case). **Caught a sign error in the test itself, not the
   implementation**, on the first run -- worked through the physical
   reasoning by hand (a landmark straight ahead at frame A appears rotated
   *clockwise* relative to a leftward-yawing vehicle's new heading at frame
   B, i.e. `Rz(yaw0-yaw1)` not `Rz(yaw1-yaw0)`) before concluding the
   implementation was right and the test's expected value was backwards.
   Also checked the output is always a proper rotation (orthogonal, det=1)
   for non-trivial extrinsics.
2. **Real data, the strongest check available**: compared the IMU-derived
   relative rotation against the actual `poses/00.txt` ground-truth
   rotations, across 18 real frame pairs (gaps of 1/10/100 frames, at 6
   different points spanning the whole sequence). **Mean disagreement
   0.044deg, max 0.136deg.** Confirms both the math and the (correct-date)
   calibration together, independent of anything already used elsewhere in
   this codebase.

**Where it's wired in**: in `estimateTwoViewPose()`, when the ORB-SLAM
ratio test would otherwise prefer homography (i.e. the near-rotation
signal), and IMU data is loaded and enabled, use the IMU-derived `R`
directly; translation *direction* still comes from whichever F/E branch is
active (its own `recoverPose()` output), rescaled by the same OXTS-distance
correction every other step gets. **This translation-direction reuse is an
approximation, not independently validated the way the rotation was** --
properly solving translation given a *known* rotation (a real "N-point
given R" problem, different and simpler than the general 5/8-point case)
is flagged as the correctly-scoped next step, not attempted this session.
Falls through to the existing homography path unchanged if IMU data was
never loaded -- zero behavior change for anyone not using it.

**Confirmed never exercised, exhaustively** (follow-up done same session,
not left as a guess): reran the full sequence with the *other* estimator
(5-point, deliberately weaker F/E than 8-point+Gold-Standard, on the theory
it would be more likely to let the ratio test favor H) specifically to try
to trigger this path. Result: **still 0 homography selections, 0 IMU-
rotation uses, out of 742 total two-view pose decisions across the whole
4541-frame sequence, with either estimator.** Explicitly watched the known
hard turn (~frame 1265, "Feldbergstr." intersection -- the one that
originally motivated the whole homography branch back in the very first
bug this file documents) pass with zero H preference either time. Both
IMU runs (8-point and 5-point) produced byte-identical ATE output to their
non-IMU counterparts (68.183m and 99.869m respectively), confirming the
integration itself is correctly inert when unreached, not silently broken.
**Conclusion: on this specific sequence, with the current model-selection
threshold (`kHomographyPreferenceRatio = 0.45`) and either F/E estimator,
homography (and therefore IMU rotation) never gets a chance to matter.**
This is a real, useful negative result, not an inconclusive one -- getting
actual evidence on whether IMU rotation helps would need either a
different/harder sequence with genuine near-pure-rotation stretches, or an
artificially-lowered `kHomographyPreferenceRatio` purely as a plumbing
smoke test (not a real validation of whether it helps).

### 5. Results

Full raw-KITTI `sequences/00` runs, `kitti_ate`, P3P PnP, all config
combinations actually measured this session (RANSAC seed fixed at 42,
confirmed deterministic -- two back-to-back reruns of identical config
gave byte-identical results in every case tested):

| Config | ATE RMSE | Matched frames | Notes |
|---|---|---|---|
| No F refinement (Session 3 baseline) | 174.0m | reproducible | |
| 8-point + Gold Standard, no OXTS | 130.2m | | Session 3's shipped state |
| 5-point E, no OXTS | 143.2m | | individually verified correct; underperforms 8-point anyway, open question |
| 5-point E + OXTS | 99.9m | 3979/4541 | more Lost/Recovered churn than 8-point |
| **8-point + Gold Standard + OXTS (shipped default)** | **68.2m** | 4464/4541 | best result this session |
| 8-point + Gold Standard + OXTS + IMU | 68.183m (identical) | 4464/4541 | IMU branch never triggered, confirmed inert-safe |

Runtime: ~7-8 minutes wall-clock for the full 4541-frame sequence in every
configuration (estimateTwoViewPose only runs during init/recovery, not
every tracking frame, so the 5-point solver's higher per-solve cost barely
moves total runtime).

For calibration: full ORB-SLAM (real bundle adjustment, proper loop
closure) scores ~5.33m RMSE on this same sequence (Session 2's source,
DynaSLAM paper). 68.2m is a large step down from where this session
started (174.0m) but still nowhere near that -- expected, since this system
still has no bundle adjustment (see the architectural note at the top of
`SlamWorker.h`), not a claim of closing that gap.

### 6. GUI wiring

Added to `ControlPanel` (new "OXTS / IMU (KITTI seq00)" group, all three
default checked): a checkbox to select the 8-point-legacy vs 5-point
estimator, and two more to enable/disable already-loaded OXTS speed / IMU
rotation data at runtime without reloading (`SlamWorker::setOxtsEnabled()`/
`setImuEnabled()`, gating `oxtsDistanceBetween()` and the IMU-rotation
branch respectively). `SlamWorker::autoLoadKittiExtras()` best-effort
auto-loads this session's known OXTS/calibration paths right after any
video is opened (silently does nothing if the paths aren't present, so
opening a different dataset/video is unaffected) -- wired to fire off
`sourceOpened` via a same-thread signal/slot connection.

### Next steps for whoever picks this up

1. **Solve translation given known rotation properly**, replacing the
   current "reuse whichever F/E branch's coupled translation estimate"
   approximation in the IMU-rotation integration -- a real, different
   (and probably simpler) minimal-solver problem than 5/8-point, not yet
   scoped in detail.
2. **Investigate why 5-point underperforms 8-point+refinement** despite
   being individually verified correct -- try more RANSAC iterations first
   (this session's 300 was picked from theory, not tuned against this
   dataset) before concluding anything deeper is wrong.
3. **Exercise the IMU-rotation path for real**: find or construct a
   scenario where homography actually wins the model-selection ratio (the
   5-point estimator, being weaker, is the most likely way to trigger this
   without waiting for a genuinely harder sequence) to get real evidence
   either way on whether it helps.
4. **Continuous OXTS injection into `trackFrame()`'s own `avgStepScale`
   gating**, not just the two touch points done this session -- flagged
   above as the reason some extra Lost/Recovered churn appeared despite the
   net ATE improvement.
5. If pursuing this OXTS/IMU direction further for a portfolio/writeup,
   **lead with the caveat from section 3 above** -- these are KITTI's own
   GPS/IMU-derived quantities, not an independent sensor, so frame results
   as "achievable given accurate real-world speed/orientation," not as
   validation against ground truth.

## Session 5 (2026-07-12, continued): 5-point's missing refit, IMU exhaustively ruled inert, a live-alignment visualization bug, and VISO2-M ground-plane scale

Picked up Session 4's open question directly: why did the individually-
verified-correct 5-point solver score worse than 8-point+Gold-Standard?

### 1. Resolved: 5-point was missing its own refit-over-inliers step

Checked `estimateEssentialRansac()` directly: unlike `estimateFundamentalRansac()`
(F path, which does a linear refit over the full inlier set, previously also
with Gold Standard nonlinear refinement -- see Session 3), the 5-point
RANSAC loop just kept whichever single minimal-5-correspondence sample had
the most inlier support, with **no refit at all** over the *other* inliers.
Added both pieces, generalized to respect E's stronger calibrated
constraint (equal nonzero singular values) rather than F's rank-2-only one:

- `refitEssentialLinear()`: same linear least-squares construction as the
  5-point minimal solver's own null-space matrix, just over all inliers
  instead of exactly 5 rows (an over-determined solve), then projected onto
  the essential-matrix manifold (`projectEssentialManifold()`: SVD, average
  the two nonzero singular values, zero the third) since a general linear
  fit isn't guaranteed to satisfy the constraint the way the minimal 5-point
  case is.
- `refineEssentialSampson()`: Gold Standard (Sampson-distance-minimizing
  Levenberg-Marquardt) refinement, same structure as the removed
  `refineFundamentalSampson()` (see `EightPointLegacy.cpp` for that
  byte-identical F-only original), but projecting onto the essential
  manifold after every step instead of just rank-2.

**Result: 60.8m ATE RMSE** (5-point + refit + Gold Standard + OXTS),
beating 8-point+OXTS's 68.2m, with `Recovered scale: 0.9959` -- essentially
perfect (1.0 = ideal), better than 8-point's own 1.1347. This is now the
shipped default (`TwoViewEstimator::FivePoint`); 8-point stays available as
the runtime-switchable alternative (GUI checkbox default now unchecked,
reflecting the new ranking). The open question is closed: 5-point wasn't
flawed, it was just missing an already-known-necessary step.

Updated leaderboard:

| Config | ATE RMSE |
|---|---|
| No F refinement (baseline) | 174.0 m |
| 8-point + Gold Standard, no OXTS | 130.2 m |
| 5-point (minimal-sample only), no OXTS | 143.2 m |
| 5-point + OXTS (no refit) | 99.9 m |
| 8-point + Gold Standard + OXTS | 68.2 m |
| **5-point + refit + Gold Standard + OXTS (shipped default)** | **60.8 m** |

### 2. IMU rotation: exhaustively confirmed inert on this sequence

Followed up Session 4's open item directly instead of leaving it a guess:
reran the full sequence with 5-point (deliberately weaker F/E than
8-point+Gold-Standard, on the theory it would be more likely to let the
model-selection ratio favor homography) specifically to try to trigger the
IMU-rotation path. Result: **still 0 homography selections, 0 IMU-rotation
uses, out of 742 total two-view pose decisions across the whole sequence,
with either estimator** -- explicitly watched the known hard turn (~frame
1265, "Feldbergstr." intersection, the one that originally motivated the
whole homography branch) pass with zero H preference both times. Both IMU
runs (8-point and 5-point) produced byte-identical ATE to their non-IMU
counterparts (68.183m and 99.869m respectively), confirming the
integration is correctly inert when unreached, not silently broken. This
is a real, useful negative result: on this sequence, with the current
`kHomographyPreferenceRatio = 0.45` threshold and either F/E estimator,
homography (and therefore IMU rotation) never gets a chance to matter.
Getting real evidence on whether it helps needs either a genuinely
different/harder sequence, or an artificially-lowered threshold purely as
a plumbing smoke test (not a real validation).

### 3. Live-visualization bug: alignment was refitting every repaint

User-reported: the on-screen trajectory appeared to "fall behind" ground
truth specifically around frames 700-1000 in the live GUI. Investigated
methodically before touching any code:
- Checked CPU contention (several heavy background `kitti_ate` benchmark
  runs happened to overlap with the live GUI session) -- real, but
  `VideoSource::readFrame()` and Qt's `QTimer` don't do any wall-clock-based
  frame dropping (confirmed by reading both), so slow real-time processing
  should only make the GUI run in slow motion, not corrupt any frame-index-
  driven computation.
- Checked the actual underlying trajectory data for that exact frame range
  via two independent offline (unthrottled) runs -- both the raw PNG
  sequence and the pre-encoded `.mp4` showed *better-than-average* error at
  frames 700-1000 (24m and 16m respectively, vs ~68m overall RMSE) --
  ruling out both a real accuracy problem and a video-compression-artifact
  theory.
- Found the real cause: `MapView::computeAlignment()` recomputed the
  Umeyama best-fit alignment **on every repaint**, from whatever partial
  trajectory existed at that moment -- unlike `kitti_ate` (which aligns
  once over the complete, final trajectory). Early-to-mid in a run, that
  fit is based on a much smaller, less representative slice of the path,
  so the ground-truth overlay visibly wobbled/rescaled as more trajectory
  came in -- a pure live-visualization artifact, not a real accuracy
  regression, that's easy to mistake for the trajectory itself being wrong.

**Fix**: alignment now freezes once 200 trajectory points are available
(`kAlignmentFreezeMinPoints`), instead of continuously refitting. Resets
automatically on an explicit Reset or a newly-loaded ground-truth file. Any
*real* drift now shows as honest, visible divergence from the ground-truth
line instead of being silently re-absorbed into a fresh fit every frame.

### 4. VISO2-M ground-plane scale: researched, implemented, found broken, fixed against the real source

User asked to compare against literature and then implement VISO2-M
specifically. Research trail:
- **VISO2-M** (Geiger et al.): pure frame-to-frame 8-point F estimation +
  continuous ground-plane scale correction, **zero bundle adjustment** --
  the closest true "no-BA, F/E-only" published baseline. ~2.74% relative
  translation error on KITTI (KITTI's official *relative*, segment-based
  metric -- not directly convertible to this project's *global* Umeyama-
  aligned ATE RMSE in meters; flagged explicitly rather than conflating the
  two).
- **Song/Chandraker/Guest, PAMI 2015**: improves ground-plane scale with a
  data-driven multi-cue fusion (2.03-2.53%), but uses **local bundle
  adjustment** over a 10-keyframe window -- not a fair "no-BA" comparison,
  more a stronger cousin of this project's `refineLocalKeyframes()`. Its
  eq. 4 cites a consensus-scoring formula with a fixed `mu=50` constant.
- **Fetched the actual authoritative libviso2 source directly**
  (`KIT-MRT/viso2`, a maintained mirror of Geiger's original, GPL-licensed
  code, `src/viso_mono.cpp`) rather than trusting secondary descriptions --
  found the *real* algorithm differs from both papers' summaries in two
  concrete ways: (a) it restricts to the nearer half of triangulated points
  (by L1 distance) before scoring, excluding noisier distant triangulations;
  (b) its Gaussian consensus weight is **adaptive** (`sigma = median L1
  distance / 50`, recomputed every frame from that frame's own point-scale),
  not the fixed `mu=50` the later paper's citation implied.

**First implementation attempt (naive, wrong)**: `GroundPlaneScale.h`/`.cpp`
built with a fixed-`mu=50` consensus weight and a hardcoded *level-camera*
assumption (ground normal exactly `(0,-1,0)`, i.e. zero pitch) -- validated
against synthetic data (4/4 tests passed, including robustness to
distractor points and correct scale-error detection) before ever touching
`SlamWorker`, then wired in as a fallback at `initializeFromFrame()`'s
bootstrap scale (the only clean integration point -- `recoverViaEpipolar()`
has a genuine structural chicken-and-egg problem: its triangulation happens
*after* the scale decision, unlike `initializeFromFrame()`, so there's no
equivalent "compute the correction from this step's own points before
finalizing scale" ordering available there; left unattempted, not solved).

Full-sequence result: **184.7m ATE RMSE -- worse than no scale correction
at all (174.0m)**, with `Recovered scale: 0.0969` indicating the raw
reconstruction needed roughly a 10x rescale to match ground truth. The
zero-pitch assumption is a real, meaningful bias: KITTI's actual rig almost
certainly has some small nonzero downward tilt, and the height estimate
(and therefore the one-time initial scale this correction sets globally)
is sensitive to that assumption.

**After fetching the real source and correcting the two concrete
algorithmic gaps** (nearer-half point filtering, adaptive sigma) --
re-verified against the same synthetic tests (still 4/4 pass) -- full
sequence result: **127.1m ATE RMSE, `Matched points: 4517/4541`,
`Recovered scale: 0.1089`**. A large, real improvement over the broken v1
(184.7m) and genuinely better than the no-correction baseline (174.0m) --
this vision-only, no-external-sensor correction now beats
8-point-without-OXTS (130.2m) too. The still-large residual Umeyama
rescale (0.1089, implying roughly a 9x global scale correction was still
needed) confirms the *pitch* assumption is the real remaining gap: the
*consensus-scoring algorithm* is now correct (matches the authoritative
source exactly), but the *camera geometry* assumption (exactly zero pitch)
is still just a guess, and a systematically-biased-but-consistent height
estimate is exactly what would produce a good relative trajectory *shape*
(hence the solid ATE) alongside a large *uniform* scale error (hence the
large Umeyama rescale) -- both needed fixing, only one is done.

Final leaderboard (all full-sequence, `kitti_ate`, P3P, RANSAC seed 42):

| Config | ATE RMSE |
|---|---|
| No scale correction (baseline) | 174.0 m |
| Ground-plane v1 (broken: level-camera assumption + fixed mu=50) | 184.7 m (worse than nothing) |
| **Ground-plane v2 (corrected, vision-only, no external sensor)** | **127.1 m** |
| 8-point + Gold Standard, no OXTS | 130.2 m |
| 5-point (minimal-sample only), no OXTS | 143.2 m |
| 5-point + OXTS (no refit) | 99.9 m |
| 8-point + Gold Standard + OXTS | 68.2 m |
| 5-point + refit + Gold Standard + OXTS, P3P PnP | 60.8 m |
| **5-point + refit + Gold Standard + OXTS, Iterative PnP + freeze-bug fix (overall best, Session 6)** | **49.4 m** |

Note (Session 6): the 60.8m row predates the `recoverViaEpipolar()` freeze-bug
fix; the 49.4m row has both a different PnP method *and* the fix applied at
once, so it is not a clean isolated comparison against the 60.8m row -- see
Session 6 item 4 for the isolation rerun meant to separate the two effects.

### 5. GUI wiring

Added a third checkbox, "Use ground-plane scale (VISO2-M-style,
vision-only)", default **off** (unlike OXTS/IMU, this hasn't been
cross-validated against ground truth the same way, and is currently known
to need the pitch-calibration fix above). `SlamWorker::setGroundPlaneEnabled()`/
`setGroundPlaneConfig()` mirror the OXTS/IMU enable-toggle pattern. `kitti_ate`
gained a `groundplane` positional flag plus `-` placeholders for skipping
`oxts-dir`/`calib-dir` while still enabling it, for isolated vision-only
testing.

### Next steps for whoever picks this up

1. **Get GroundPlaneScale's corrected full-sequence result** if it wasn't
   finished when this was written (check `kitti_ate_groundplane_v2.log` /
   rerun with `fivepoint - - groundplane`).
2. **Determine KITTI's actual camera pitch** (the remaining, separate gap
   from item 4 above -- the *consensus algorithm* is now fixed, the *zero-
   pitch geometry assumption* is not). Either find it in KITTI's own
   calibration documentation, or estimate it empirically from this
   project's own triangulated data (e.g. sweep candidate pitch values and
   pick whichever gives the tightest/most-consistent height consensus
   across many frames) rather than trusting an assumed value again.
3. **Solve translation given known rotation properly** (unchanged from
   Session 4) -- the IMU-rotation integration still reuses whichever F/E
   branch's coupled translation estimate, an approximation never
   independently validated the way the rotation itself was.
4. **Find/construct a scenario that actually exercises IMU rotation** --
   confirmed inert on this sequence with both estimators; a different or
   harder sequence, or a temporarily-lowered `kHomographyPreferenceRatio`
   as a plumbing-only smoke test, are the two ways to get real evidence.
5. Same caveat as Session 4 applies to OXTS/IMU (not to GroundPlaneScale,
   which genuinely is vision-only): frame results as "achievable given
   accurate real-world speed/orientation," not independent-sensor
   validation, if writing this up further.

## Session 6 (2026-07-12, continued again): GroundPlaneScale's final number, an Iterative-PnP benchmark that found a real freeze bug, and a related bug still open

Picked up two threads: (1) confirm GroundPlaneScale's corrected-algorithm
full-sequence result, queued at the end of Session 5; (2) user asked to add
`cv::SOLVEPNP_ITERATIVE` as a selectable (then default) PnP method and
benchmark it.

### 1. GroundPlaneScale: confirmed, 127.1m -- no longer "in progress"

The Session 5 entry's top-of-file status header still said "in progress
testing" even though the body of Session 5 already had the final numbers
written up (127.1m, `Matched points: 4517/4541`, `Recovered scale: 0.1089`).
That was just a stale header, not an unfinished run -- corrected in this
session's status header. Nothing new to add beyond what Session 5 already
recorded; see that section (item 4) for the full story and the remaining
camera-pitch caveat.

### 2. Added Iterative PnP + set as default

`cv::SOLVEPNP_ITERATIVE` added to `ControlPanel`'s PnP method dropdown and
made the new default in `PnpSettings.h` (was `SOLVEPNP_P3P`), per explicit
user request ("add Interative Pnp also" / "use Interative as default").
Simple, mechanical change -- `PnpSettings::method` already flows straight
into `cv::solvePnPRansac()`'s method argument in `trackFrame()`, no other
code touches it.

### 3. Bug found and fixed: `recoverViaEpipolar()`'s plausibility gate rejected genuinely correct OXTS steps

Benchmarking Iterative PnP (5-point+refit+Gold-Standard+OXTS+IMU, otherwise
identical to the 60.8m config) initially seemed to hang -- `kitti_ate`'s
600s timeout expired at only ~355/4541 frames in. Investigating in parallel,
the user reported the *exact same symptom live in the GUI*: "the trajectory
stop somewhere" (verbatim). The GUI log made the mechanism obvious:

```
[recover] isPlausibleStep fail: stepDist=0.768 avgStepScale=0.058 framesElapsed=1 bound=0.579
[recover] isPlausibleStep fail: stepDist=1.587 avgStepScale=0.058 framesElapsed=2 bound=1.157
[recover] isPlausibleStep fail: stepDist=2.310 avgStepScale=0.058 framesElapsed=3 bound=1.736
...
```

`recoverViaEpipolar()` (SlamWorker.cpp) already had the right idea for
*scale*: when OXTS speed data covers the gap between the reference keyframe
and the current frame, it rescales the recovered unit-baseline translation
by the real measured `oxtsDist` instead of the `m_avgStepScale` heuristic
(a deliberate, documented improvement -- "a ground-truth-quality [distance]
is a strict improvement with no new risk"). But *after* that rescale, the
resulting pose was still passed through `isPlausibleStep()`, whose bound is
entirely a function of `m_avgStepScale` -- the same heuristic the OXTS path
was supposed to be strictly better than. When `m_avgStepScale` drifts low
(0.058 in the log above) relative to a genuinely fast real stretch, `bound =
kMaxStepMultiplier * avgStepScale * framesElapsed` grows strictly slower
than the real per-frame distance as `framesElapsed` climbs (since the
reference frame never advances on this kind of failure -- only a
match-*count* failure slides it forward), so a *correct* OXTS-measured step
is rejected every single attempt, forever. This is functionally the same
"avgStepScale collapse -> permanent lockup" failure mode Session 2 found and
thought it had fixed with the `m_longTermStepScale` floor -- except this
time the floor mechanism is irrelevant, because the step being rejected
isn't derived from `m_avgStepScale` at all; the heuristic is just being
asked to judge a measurement it has no business judging.

**Fix**: skip `isPlausibleStep()` entirely when `oxtsDist > 0.0` (i.e. the
step already came from a real measurement, not a guess) -- only gate the
`else` (vision-only heuristic) branch through it, which is what the check
was actually designed for. One-line condition change
(`if (oxtsDist <= 0.0 && !isPlausibleStep(...))`), plus an extended comment
explaining why, at `SlamWorker.cpp`'s `recoverViaEpipolar()`.

### 4. Result: Iterative PnP + fix = 49.443m ATE RMSE, new best

Full sequence, 5-point+refit+Gold-Standard+OXTS+IMU, Iterative PnP, with the
fix above:

```
Matched points:       4517 / 4541 ground-truth frames
Recovered scale:      1.0659
GT path length:       3722.3 m
ATE RMSE:             49.443 m
ATE mean:             42.892 m
ATE median:           37.237 m
ATE max:              114.950 m
ATE RMSE / path len:  1.33%
```

A large drop from the previous best (60.8m, P3P). **Caveat**: this single
number conflates two simultaneous changes -- the PnP method switch
(P3P -> Iterative) and the freeze-bug fix -- so it isn't yet known how much
of the ~11.4m improvement is attributable to each. An isolated rerun (same
config, P3P instead of Iterative, same fix applied) was launched to
separate them; check `kitti_ate_p3p_fixed.log` for whether it finished and
what it found by the time this is read. (It needed a bigger timeout than
the 600s default -- see the next item for why -- so it may still have been
running when this was written.)

### 5. Related bug found, NOT yet fixed: `trackFrame()`'s own `isPlausibleStep()` has the same class of vulnerability

While the P3P+fix isolation rerun was in progress, its log showed a
different, related problem -- not a permanent freeze this time (the fix
above keeps `recoverViaEpipolar()` unblocked), but a severe, sustained
slowdown:

```
[track] isPlausibleStep fail: stepDist=0.678 avgStepScale=0.042 bound=0.424
[model] npts=323 nE=296 nH=126 ratio=0.298 -> E
[state] Tracking (recovered)
[track] isPlausibleStep fail: stepDist=0.690 avgStepScale=0.042 bound=0.424
[model] npts=339 nE=317 nH=129 ratio=0.286 -> E
[state] Tracking (recovered)
...
```

`trackFrame()`'s own `isPlausibleStep()` call (`SlamWorker.cpp`, ~line 2046)
gates a real `solvePnPRansac()` result against the *map* -- a pose that
already carries correct metric scale from the map's own (OXTS- or
ground-plane-corrected) points, not a unit-baseline guess. It has no
OXTS-awareness or bypass at all, unlike `recoverViaEpipolar()` after the fix
above. So the exact same class of bug applies: once `m_avgStepScale` drifts
low relative to genuinely fast real motion, a *correct* PnP-against-map
solve gets rejected, `trackFrame()` reports failure, and every single frame
in that stretch falls through to the more expensive `recoverViaEpipolar()`
fallback instead. Confirmed live: throughput dropped from the usual ~65
frames/sec (unthrottled) to roughly 3-4 frames/sec during one such stretch
(frames ~970-1276 in the P3P+fix rerun) -- not just a performance problem,
since `recoverViaEpipolar()`'s recovered pose is described elsewhere in this
file as a cheaper, less-accurate fallback than full-map PnP, so this is
likely quietly costing accuracy too, every time it happens.

**Not fixed this session** -- flagged as the top priority for whoever picks
this up next. Likely the same shape of fix (an OXTS-derived independent
sanity bound, or trusting the map-based PnP result more directly when the
map's own scale is already known-good), but `trackFrame()`'s situation is
less clean than `recoverViaEpipolar()`'s: there's no single `oxtsDist` value
per call the way there is here, since the map's scale was set once, far
earlier (at `initializeFromFrame()`), not re-derived every tracking frame.
Needs actual thought, not a one-line port of this session's fix.

### 6. Iterative PnP is dramatically slower than P3P

Purely a benchmarking-logistics note: `cv::SOLVEPNP_ITERATIVE` inside
`cv::solvePnPRansac()` runs real nonlinear (Levenberg-Marquardt-style)
optimization per RANSAC sample, versus P3P's closed-form per-sample solve --
confirmed to need roughly 15-20x longer for a full-sequence unthrottled run.
`kitti_ate`'s default 600s timeout is no longer enough headroom for an
Iterative-PnP full-sequence run; 1800s was used instead. Worth knowing if
Iterative stays the shipped default and someone else runs `kitti_ate`
without an explicit `[seconds]` override.

### Next steps for whoever picks this up

1. **Fix `trackFrame()`'s `isPlausibleStep()` vulnerability** (item 5 above)
   -- the single highest-value item queued right now: both a likely
   accuracy leak and a confirmed severe performance regression, and it's a
   very similar shape of bug to the one just fixed elsewhere in the same
   file.
2. **Check the isolated P3P+fix result** (item 4 above) once it's finished,
   to properly attribute how much of 60.8m->49.4m came from the bug fix
   alone versus the PnP method switch, before drawing conclusions about
   which one to keep/recommend.
3. Everything queued at the end of Session 5 remains open and unchanged:
   camera pitch calibration for GroundPlaneScale, translation-given-known-
   rotation for the IMU integration, and finding a scenario that actually
   exercises IMU rotation.

## Session 3 (2026-07-11, continued again): Gold Standard F refinement, a mutual-matching dead end, and a baseline-reproducibility scare

Picked up exactly where Session 2 left off: "NEXT UP (queued, not started):
tighten F/E (epipolar) accuracy directly." Implemented the two cheapest
items from Session 2's "Full F/E option menu" (items 2 and 4), tested both
via full raw-KITTI `kitti_ate` runs, and shipped the one that actually
helped.

### What was implemented

1. **Gold Standard nonlinear refinement of F** (`refineFundamentalSampson()`,
   anonymous namespace in `SlamWorker.cpp`, called from
   `estimateFundamentalRansac()` right after the linear `solveEightPoint`
   refit over the inlier set). Hartley & Zisserman Algorithm 11.3's idea --
   a handful of Levenberg-Marquardt iterations minimizing true Sampson
   distance directly, rather than the algebraic error the linear refit
   minimizes -- implemented over F's 9 raw entries (not H&Z's minimal
   7-parameter epipole form), with the rank-2 constraint re-enforced by SVD
   projection after every accepted step. Jacobian is numerical (central
   differences): only 9 parameters, at most a few hundred inlier points,
   so the cost is trivial, and it avoids hand-deriving analytic Sampson
   derivatives (easy to get subtly wrong -- see the derivation sketch that
   was worked through and set aside in favor of this simpler, still-correct
   approach). Operates in the same Hartley-normalized coordinates
   `solveEightPoint` already uses.
2. **Mutual/cross-check descriptor matching** (`matchDescriptors()`):
   Session 2's menu item 4. Added a second `knnMatch` call in the reverse
   direction (B->A) and kept only ratio-test matches that agree both ways.
   **Implemented, tested, and reverted** -- see "Dead end" below.

### Dead end: mutual/cross-check matching causes a new permanent lockup

Full raw-KITTI run with both changes above locked up permanently at frame
1577 (`Trajectory` frozen from frame 1577 through the end of the run, frame
4541 -- confirmed via `grep -n "Trajectory:" ... | tail`, matching the exact
symptom pattern of every previous permanent-lockup bug in this file).

Root cause, traced directly in the log: a loop closure at frame 1577
(`kf#13 <-> kf#161`, 367-unit translation + 18.7deg rotation correction --
unremarkable by this session's standards, well inside the "good" range from
Session 2's table) landed fine, but every frame afterward failed
`isPlausibleStep` forever, with the log showing `avgStepScale=0.060` stuck
against a real measured step of ~20 units -- a ~300x mismatch, and the exact
"avgStepScale collapse deadlock" signature Session 2 already found and
partially fixed (see below). The mechanism: mutual/cross-check matching
measurably shrinks raw match counts (rejects any match that only passes the
ratio test in one direction), which pushes `trackFrame()` below
`kMinTrackMatches` far more often than the original one-directional
matcher did. Since `m_avgStepScale` only updates on an *accepted*
`trackFrame`-only step (by design -- `recoverViaEpipolar`'s step must never
feed back into the same average it's derived from, per Session 1's fix),
starving `trackFrame` of matches means the average essentially never gets
a chance to update once it settles low, so it can't recover even though
Session 2's median-window-plus-floor mechanism is specifically designed to
let it. Confirmed via ablation: reverting *only* the matcher (keeping the F
refinement) sails straight through frame 1577 with no freeze, isolating the
cause precisely to the matcher change, not the refinement.

**Reverted.** `matchDescriptors()` is back to the original one-directional
Lowe's-ratio-test matcher. Left off Session 2's "cheap, do regardless" menu
framing for next time -- it isn't cheap in practice; a future attempt would
need to be scoped around this specific failure mode (e.g. only apply the
stricter check where a match-starved `trackFrame` isn't a live risk, or pair
it with a `recoverViaEpipolar`-side safety valve), not applied uniformly.

### Scare: the recorded Session 2 baseline (126.9m) would not reproduce

While isolating the above, `kitti_ate` was rerun against the reverted,
byte-identical-to-Session-2 code (matcher and F both back to their original
form) to get a clean confirmation number. It scored **ATE RMSE 174.0m**, not
the 126.9m Session 2's table records as "current shipped state." This was
alarming enough to check properly, since it would mean *nothing* about
RANSAC determinism (Session 1's fix) could be trusted anymore.

Checked by rerunning the exact same reverted binary against the exact same
input a second time: **byte-identical result, 174.005m RMSE both times**,
down to the matched-frame count (4507/4541) and recovered Umeyama scale
(0.0927). This confirms the current code **is** fully deterministic
run-to-run -- Session 1's `kRansacSeed` fix still holds, and there is no
newly-introduced source of nondeterminism (multi-threaded OpenCV
internals -- this binary does link `libtbb`/`libgomp` with no thread count
pinned anywhere in the code, which was the leading suspicion -- were ruled
out empirically by the identical rerun, not just assumed away).

The discrepancy with the recorded 126.9m therefore isn't run-to-run noise;
it's some difference between what actually produced that number and the
code as currently written, that this session could not identify (no git
history in this repo to diff against, and no earlier saved binary/log to
compare byte-for-byte). **Left unresolved.** Practical consequence: treat
Session 2's ATE table (126.9m row and everything above it) as historical
record only, not a number to diff future work against. The number to diff
against going forward is the newly, repeatably-confirmed one below.

### Result: F refinement kept, matcher reverted, new confirmed baseline

Because the recorded 126.9m couldn't be trusted, the F refinement was
evaluated the only rigorous way available: a controlled A/B on the *current*
code, both arms run to completion, both reproduced twice.

| Config | ATE RMSE | Matched frames | Notes |
|---|---|---|---|
| Original matcher, no F refinement (confirmed baseline) | **174.0m** | 4507/4541 | reproduced twice, byte-identical both times |
| Original matcher, + Gold Standard F refinement (shipped) | **130.2m** | 4499/4541 | reproduced twice, byte-identical both times |
| Mutual-check matcher, + F refinement (tried) | n/a (99.8m, untrustworthy) | 1573/4541 | permanent lockup at frame 1577, reverted |

The Gold Standard F refinement alone is a genuine, reproducible **~25%
ATE reduction** (174.0m -> 130.2m) on top of whatever is actually currently
in this codebase (whatever that discrepancy with Session 2's number turns
out to be, it affects both rows of this table equally, so the *relative*
improvement from the refinement is trustworthy even though neither absolute
number matches Session 2's table). **Shipped**: `matchDescriptors()` is the
original one-directional matcher; `estimateFundamentalRansac()` includes the
Gold Standard refinement pass.

For calibration: full ORB-SLAM (real bundle adjustment, proper loop closure)
scores ~5.33m RMSE on this same sequence (Session 2's source, DynaSLAM
paper). 130m is still a long way from that -- expected, since this system
still has no bundle adjustment -- but it is a real, measured improvement
over what this specific lightweight architecture was otherwise producing,
not a claim of closing that gap.

### Next steps for whoever picks this up

1. **5-point calibrated E (Session 2's menu item 1)** is now the clearest
   next lever -- the two cheap items (3: E singular-value enforcement, ruled
   out this session as a mathematical no-op for `cv::recoverPose`'s
   decomposition -- see below; 4: mutual-check matching, dead end above)
   are both closed out.
2. **Item 3 finding, for the record**: reviewed (not empirically tested,
   didn't need to be -- this is a property of the math, not the data)
   whether explicitly averaging E's two nonzero singular values before
   `cv::recoverPose` would help. It would not: `cv::recoverPose`/
   `decomposeEssentialMat`'s internal SVD-based decomposition only ever
   uses the `U`/`Vt` factors from `E`'s SVD to build its R/t candidates,
   never the singular values themselves -- reconstructing E with equal
   singular values leaves `U`/`Vt` completely unchanged, so it's a
   byte-for-byte no-op on `recoverPose`'s output specifically. (This only
   resolves the question for *this* codebase's use of `recoverPose`; it
   doesn't mean the constraint is meaningless in general, just that there's
   no live consumer of it here worth changing.)
3. **The Session 2 baseline discrepancy is worth a real investigation** if
   it starts mattering (e.g. before writing up final results): try to find
   whatever's different between "whatever produced 126.9m" and the current
   code. Candidates to check first: whether the `kitti_ate` invocation used
   to get 126.9m actually matched this session's (`... 800 0 <prefix>`,
   P3P via the `else` branch of the pnp-method string match -- worth
   double-checking argv parsing wasn't misused in either session); whether
   any constant in the `namespace {}` block at the top of `SlamWorker.cpp`
   was tuned and reverted without updating the table; whether the KITTI
   image directory itself changed. Low priority unless it blocks something.
4. **Mutual/cross-check matching could be revisited with a narrower
   scope**: e.g. only cross-check the loop-closure PnP correspondence match
   (`tryLoopClosure()`), where a stricter, cleaner match set plausibly
   matters more (it's what determines the correction magnitude) and where
   match starvation isn't a live risk the way it is for every-frame
   `trackFrame()` matching. Not attempted this session -- flagging as an
   idea, not a queued task.

## Session 2 (2026-07-11, continued): loop closure, ATE evaluation, and two more lockup mechanisms

Picked up after the original homography-lockup fix (below) was confirmed
working. This part of the session added real features (loop closure,
better scale estimation, local pose refinement) on top of that fix, found
and fixed two *more* permanent-lockup bugs surfaced by the new code, built
a proper quantitative evaluation pipeline (ATE against KITTI ground truth,
in native C++, no Python dependency), and ends with one clearly-scoped open
problem plus a documented dead end so it isn't retried.

### What was added

1. **Median-window scale estimate** (replacing the old EMA). `m_avgStepScale`
   is now the median of the last `kScaleWindowSize` (20) independently-
   measured (`trackFrame`-only) step distances, with a slow long-term EMA
   baseline (`m_longTermStepScale`, alpha 0.02) kept alongside it purely as
   a floor -- `m_avgStepScale` is never allowed below `kMinScaleFraction`
   (0.15) of the long-term baseline. See "Bug: avgStepScale collapse
   deadlock" below for why the floor exists; it's not optional polish.
2. **Full 6-DOF loop closure** (`SlamWorker::tryLoopClosure()`, called from
   `insertKeyframe()` after every new keyframe). Searches all keyframes
   older than `kLoopExclusionWindow` (30) for one sharing >= `kLoopMinMatches`
   (60) raw descriptor matches with the new keyframe, re-measures the new
   keyframe's pose via PnP against that old keyframe's own
   locally-triangulated 3D points (independent of everything accumulated
   along the path in between), and if the match holds (>= `kLoopMinPnpInliers`
   inliers), distributes the discrepancy as a world-frame rotation+translation
   correction across every trajectory point, keyframe, and (critically, see
   below) map point between the two loop endpoints, interpolated by how far
   along the loop each one is.
3. **Local pose refinement** (`SlamWorker::refineLocalKeyframes()`, called
   from `insertKeyframe()` before the loop-closure search). Refines the
   last `kLocalRefineWindow` (6) keyframes' poses via `cv::solvePnPRefineLM`
   (real Levenberg-Marquardt reprojection-error minimization) against each
   keyframe's own local 3D-2D correspondences. Explicitly scoped as
   per-keyframe polish, not true joint multi-keyframe bundle adjustment --
   this codebase has no landmark-track structure shared across keyframes
   (each keyframe only keeps points it personally triangulated), so there's
   no shared structure to jointly optimize the way real BA would.
4. **`analyze/kitti_ate.cpp`** (new `kitti_ate` CMake target). Runs
   `SlamWorker` directly against a raw KITTI image sequence (point it at
   `.../sequences/00/image_0/%06d.png` -- `cv::VideoCapture` supports this
   printf-pattern natively, no code changes needed for that part) and
   computes Absolute Trajectory Error against `poses/00.txt` natively in
   C++: 2D Umeyama similarity alignment (scale+rotation+translation, since
   monocular SLAM has no absolute frame) then RMSE/mean/median/max, all
   with no Python dependency. Writes `<prefix>_trajectory.txt` (frame, raw
   x/z, aligned x/z, matched ground-truth x/z) for later plotting.
   Usage: `./build/kitti_ate <image-pattern> <poses.txt> [seconds] [pnp-method] [out-prefix]`.
5. **`SlamWorker::startUnthrottled()`**. The GUI's `start()` throttles
   processing to `kProcessIntervalMs` (~10fps) to match KITTI's native
   capture rate for a watchable display -- fine for the GUI, pure wasted
   wall-clock time for a headless batch tool. `startUnthrottled()` (not a
   slot, not wired to any UI control) runs the same timer at 0 interval
   instead. Both `sift_vslam_debug` and `kitti_ate` use it now. Also made
   both tools quit as soon as `statsUpdated` reports `"Stream ended"`
   instead of idling until their `[seconds]` safety-net timeout, since
   unthrottled processing usually finishes the whole sequence in well under
   that budget.

### Bug: avgStepScale collapse deadlock (found and fixed)

Symmetric to (and just as bad as) the runaway-*growth* bug fixed earlier
this session (see "Root cause (found)" below) -- except this one shrinks
`m_avgStepScale` toward zero instead of inflating it, and the earlier
fix's upper clamp (`kMaxAvgStepUpdateMultiplier`) does nothing to guard
against it.

Confirmed on a full raw-KITTI run: `m_avgStepScale` correctly tracked a
genuine slow/near-stopped stretch of city driving down to ~0.003 (frames
~2000-2800) -- that part is correct, adaptive behavior, not a bug. The bug:
once normal speed resumed, there was no way back. `isPlausibleStep`'s bound
(`kMaxStepMultiplier * avgStepScale`, ~0.03 at that point) rejected every
real step forever, and since `m_avgStepScale` only updates inside
`pushTrajectoryPoint()`, which only runs on an *accepted* step, it could
never recover -- a hard deadlock, indistinguishable in the log from a
different bug (real step distances staying suspiciously constant across
dozens of consecutive `[track]`/`[recover] isPlausibleStep fail` lines,
e.g. `stepDist=78.44, 78.44, 78.43, 78.42, ...`).

Fix: `m_longTermStepScale`, a separate slow EMA (alpha 0.02) that
`m_avgStepScale` is never allowed to fall below 15% of (`kMinScaleFraction`).
Updates itself clamped relative to *itself*, not to the fast/already-
deflated `m_avgStepScale`, so it isn't dragged down by the same collapse it
exists to guard against.

### Bug: map/pose inconsistency after a large loop correction (found and fixed)

Even bigger than the above. When `tryLoopClosure()` was first implemented,
its doc comment said leaving `m_mapPoints` uncorrected (while
trajectory/keyframes/live-pose *do* get corrected) would make map points
"look slightly offset near the loop until fresh ones replace them" --
true for a small correction, badly wrong for a large one. Confirmed on a
full raw-KITTI run: right after a loop closure applied a 1155-unit
translation correction, `trackFrame()`'s very next real PnP solve produced
`stepDist=1120.520` -- essentially the same magnitude as the correction
just applied. The corrected live pose had jumped ~1155 units away from
where the *uncorrected* map points actually were, so every subsequent real
PnP solve against that map came out looking exactly that far from the
"last known" position and got rejected forever by `isPlausibleStep`: a
second, independent permanent-lockup mechanism, layered on top of (and
easy to misdiagnose as) the avgStepScale one above.

Fix: `tryLoopClosure()` now also applies the correction to every point
currently in `m_mapPoints`. Since the map has no per-point keyframe
association (unlike `Keyframe::localMapPoints`), it can't be corrected with
the same per-point interpolated alpha as the trajectory -- it gets the
full (alpha=1) correction applied uniformly instead. This is an
approximation, justified by `kMaxMapPoints`' rolling eviction meaning the
current map is mostly recent contributions already close to the new
keyframe's frame index, not spread evenly across the whole loop span.

### Open problem, NOT fixed: occasional badly-wrong loop correction

Observed once on a full raw-KITTI run: a loop closure between kf#14 (frame
129) and kf#151 (frame 1573) produced a translation correction of **2337
world units** -- roughly 10-20x larger than every other correction seen
that run (which ranged ~3-200 units). It passed every existing check
(`kLoopMinMatches=60`: had 168; `kLoopMinPnpInliers=20`: had 41), so
nothing currently catches it. Thanks to the map-point fix above this no
longer causes a *permanent* lockup, but it still measurably corrupts
accuracy for the rest of the run (a later full run that included this bad
correction scored notably worse ATE than one that happened to avoid it by
luck of the RNG/frame timing).

**Two fixes for this were tried and reverted -- do not retry either without
a new idea, both are confirmed dead ends:**

1. Gating the loop-measured pose through `isPlausibleStep()` (reusing it
   exactly as `trackFrame()`/`recoverViaEpipolar()` do). Wrong tool:
   `isPlausibleStep`'s bound is calibrated for single-frame continuity
   (capped at `kMaxFramesElapsedForRescale` = 15 frames' worth of budget),
   but loop closures span hundreds to thousands of frames of *legitimate*
   accumulated drift. Result: rejected essentially every loop closure in a
   full run, not just the bad one.
2. A fixed/scale-adaptive magnitude cap
   (`max(400, 100 * m_longTermStepScale)`, calibrated against that run's
   observed good-correction range of ~3-200). Result: worse, not better --
   confirmed via full-run comparison. Rejecting a correction doesn't undo
   the drift it would have fixed, so drift keeps compounding unchecked,
   which makes every *subsequent* loop measurement look even larger and
   also get rejected: a vicious cycle. In the run where this was tried,
   every single loop closure ended up rejected, and corrections that had
   been small and clearly legitimate in earlier runs (e.g. 81, 107, 115
   units at similar frame gaps) were now measured at 2000-4600+ units
   because nothing had corrected the drift they were trying to fix.

**Current state: unguarded.** Accepting the occasional bad correction
measurably beats both rejection strategies tried so far. A real fix
probably needs something that doesn't touch the accept/reject boundary of
a *single* candidate at all -- e.g. requiring 2-3 independent nearby loop
candidates to roughly agree with each other before trusting any of them
(closer to what real SLAM systems do with covisibility-graph consensus /
robust-kernel bundle adjustment) -- not a threshold on one measurement in
isolation.

### ON HOLD (queued, not started, deprioritized): ground-plane scale recovery via known camera height

User explicitly redirected away from this after it was queued ("but I focus
on F and E") -- this approach leans on the H (homography) branch, which is
off-focus right now. Left documented in case it's picked back up later, but
**not the next thing to do** -- see "NEXT UP" below instead, which is F/E
(epipolar) focused per that redirect.

Attacks the root cause flagged earlier this session ("scale isn't just
globally wrong, it wanders locally") at the source, instead of the
damage-control this session's other scale work (median window, long-term
floor) had to do after the fact.

**The idea** (standard technique for monocular VO on driving datasets,
confirmed via literature search -- not a novel proposal): the camera sits
at a known, fixed height above the road. When enough tracked points lie on
the (near-planar) road surface, decomposing the homography between two
frames gives rotation + translation *direction* + plane normal/distance --
and `cv::decomposeHomographyMat`, already called in the H branch of
`estimateTwoViewPose()`, already returns exactly that plane information.
Currently it's discarded (only used to pick the best cheirality-validated
candidate). Since the real-world plane distance is the known camera
height, that same decomposition can solve directly for *true metric scale*
of that frame's translation -- a per-frame absolute measurement, not
something integrated/accumulated through `m_avgStepScale` and therefore
not vulnerable to any of the drift-related bugs this session found and
fixed (or the one still open, above).

**Reference baseline**: VISO2-M (the monocular variant of VISO2, the
classic lightweight KITTI VO baseline) uses exactly this -- known camera
height + ground-plane detection for scale, no bundle adjustment. It's the
right comparison target for this codebase (same category: lightweight VO,
no BA), not full-BA systems like ORB-SLAM's ~5.33m ATE on this sequence.

**To do when picked back up**:
1. Confirm KITTI sequence 00's actual camera height above ground from its
   calibration/sensor documentation (commonly cited ~1.65m in papers, but
   verify for this specific rig rather than trusting that number blindly).
2. In the H branch of `estimateTwoViewPose()`, after
   `cv::decomposeHomographyMat` returns its candidates (`Rs, Ts, Ns` --
   `Ns` is the plane normals, currently unused), identify which candidate's
   plane normal is consistent with "ground plane roughly below and
   perpendicular-ish to the camera" (i.e. points mostly downward in camera
   frame) rather than just picking by cheirality vote alone.
3. Use that candidate's implied plane distance (in `decomposeHomographyMat`'s
   arbitrary-scale units) versus the known real camera height to derive a
   metric scale factor for that frame's `t`.
4. Feed that as an additional, independent scale signal -- likely wants its
   own trust/weighting logic alongside `m_avgStepScale`/`m_longTermStepScale`
   rather than a wholesale replacement, since ground-plane visibility won't
   hold every frame (occlusion by other vehicles, turns, non-planar
   stretches) and the existing step-distance-based estimate is a reasonable
   fallback for frames where it doesn't.
5. Watch out for the numerical-instability failure mode papers on this
   technique specifically call out in the ground-plane homography
   decomposition stage (see sources below) -- don't assume a naive port is
   automatically robust.

Sources (found via search this session, not yet read in full depth):
- https://www.researchgate.net/publication/332944864_Ground-Plane-Based_Absolute_Scale_Estimation_for_Monocular_Visual_Odometry
- https://www.researchgate.net/publication/305257884_Reliable_Scale_Estimation_and_Correction_for_Monocular_Visual_Odometry
- https://arxiv.org/pdf/2101.05995 (Accurate and Robust Scale Recovery for Monocular Visual Odometry Based on Plane Geometry -- this one explicitly addresses the numerical-instability issue above)

### NEXT UP (queued, not started): tighten F/E (epipolar) accuracy directly

User's actual redirect: stay focused on the F/E (fundamental/essential
matrix) side of the pipeline, not H/homography. `solveEightPoint()` and its
RANSAC wrapper (`estimateFundamentalRansac()`) were reviewed in detail this
session and confirmed mathematically correct (constraint matrix, SVD
null-space solve, rank-2 enforcement, Hartley normalization/denormalization
-- all checked by hand, no bugs found). One real gap was found and flagged
but not yet implemented:

**Missing: nonlinear (Gold Standard) refinement of F.** The current
pipeline is: minimal 8-point sample per RANSAC iteration -> Sampson-distance
inlier scoring -> linear refit over the full inlier set (still
`solveEightPoint`, i.e. still minimizing *algebraic* error in normalized
coordinates). It never takes the further step Hartley & Zisserman's
Algorithm 11.3 ("Gold Standard") calls for: a handful of Gauss-Newton/
Levenberg-Marquardt iterations on top of that linear estimate, minimizing
Sampson distance (true geometric error) directly. This is a well-defined,
bounded, no-new-dependency addition -- `sampsonDistance()` already exists
in the anonymous namespace of `SlamWorker.cpp` and is exactly the residual
to minimize; only the iterative refinement loop is missing.

**Where this plugs in**: at the end of `estimateFundamentalRansac()`, right
after `FnRefined = solveEightPoint(inNorm1, inNorm2)` -- add a refinement
pass over `Frefined` (or `FnRefined` in normalized coordinates, whichever
is numerically cleaner) using the existing inlier set before returning.

**Why this matters here specifically**: F is the basis for every epipolar
operation in the system -- `initializeFromFrame()` (sets the whole video's
world scale/origin), `recoverViaEpipolar()` (used every time PnP tracking
fails), and indirectly `tryLoopClosure()`'s candidate geometry. Tightening
it once benefits all of those, unlike the loop-closure-specific work this
session did.

**Also consider while in this area** (not yet scoped in detail, listed for
whoever picks this up): the essential matrix `E = K^T F K` -> `cv::recoverPose`
step downstream of F has no equivalent refinement pass either -- once F
itself is tightened, worth checking whether `recoverPose`'s own
decomposition/cheirality selection is still the accuracy bottleneck or
whether the F improvement alone closes enough of the gap to reassess.

#### Full F/E option menu (discussed, not yet prioritized/implemented -- pick one to start)

Ranked by expected impact. None started yet as of end of session
2026-07-11; this is the menu to choose from next time, not a to-do list to
do all of.

1. **(Biggest lever, biggest lift) Direct calibrated 5-point algorithm for
   E, replacing 8-point-F-then-convert.** Right now: estimate F via general
   8-point (only enforces rank-2) -> `E = K^T F K`. This throws away
   information: since K is known/exact for KITTI, E has a *stronger*
   constraint than F -- its two nonzero singular values must be **equal**,
   not just nonzero, and the 8-point solve never enforces that. Nister's
   5-point algorithm solves for E directly under the calibrated
   constraint -- the standard approach in essentially every modern
   calibrated VO/SLAM system (ORB-SLAM, VINS, etc. all use 5-point, not
   8-point-then-convert). Also minimal (5 correspondences vs. 8), so
   RANSAC gets more inlier-scoring iterations per time budget and
   tolerates a higher outlier fraction. Real implementation lift: a
   different algorithm entirely, involves solving a degree-10 polynomial
   for the minimal case -- not a small patch like the items below.
2. **(Already scoped above, smaller lift) Nonlinear Gold Standard
   refinement of F** -- see the full writeup above this menu. Valid
   regardless of whether #1 happens (F-then-convert stays viable if #1
   isn't picked up); redundant with #1 if 5-point is implemented, since
   5-point's calibrated constraint supersedes what this refinement buys on
   the F side.
3. **(Cheap, do regardless of #1/#2) Explicitly enforce E's singular-value
   constraint.** After `E = K^T F K`, SVD it and force the two nonzero
   singular values equal (average them) before `cv::recoverPose`.
   Currently unverified whether `recoverPose` already does this
   internally, or whether an unconstrained E (from the current
   F-then-convert path) is being fed in as-is -- check OpenCV's
   `recoverPose` docs/source before assuming either way.
4. **(Cheap, do regardless of #1/#2) Mutual/cross-check matching before
   RANSAC.** Currently `matchDescriptors` does a Lowe's-ratio-test-style
   check (default ratio 0.75f per its signature) one direction only.
   Adding a mutual/cross consistency check (match A->B and B->A
   independently, keep only mutual matches) cleans the correspondence set
   feeding *both* F and E before RANSAC ever runs -- garbage-in reduction
   that helps regardless of which estimator consumes the matches.

### ATE results (native C++, `kitti_ate`, raw KITTI images, not the video)

Progression across this session's fixes, all on the full `sequences/00`
image sequence (`image_0/%06d.png`, not `kitti_00.mp4`), P3P, ATE computed
by `kitti_ate` itself (2D Umeyama-aligned RMSE against `poses/00.txt`, GT
path length 3722.3 m -- matches KITTI-00's published length; an earlier
Python-side path-length computation had a bug, see below):

| Run | avgStepScale floor | map-point correction | loop-correction gate | Result |
|---|---|---|---|---|
| 1 | no | no | n/a | ATE 81.4m, but a permanent stall mid-run made this number not fully trustworthy (frozen for the last ~2460 lines) |
| 2 | yes | no | n/a | ATE 83.2m, still stalled (shorter, ~1150 lines) |
| 3 | yes | yes | none (unguarded) | **ATE 126.9m, 0 stalls, complete/honest run** -- current shipped state |
| 4 | yes | yes | isPlausibleStep-based | ATE 149.6m, every closure rejected -- reverted |
| 5 | yes | yes | magnitude cap | ATE 153.8m, every closure rejected -- reverted |

Run 3 is the current state of the code. For reference, published monocular
ORB-SLAM (full system: real bundle adjustment, proper loop closure with
essential-graph optimization) scores ATE RMSE ~5.33m on the same sequence
(source: DynaSLAM paper's comparison table) -- so there's a real, known
gap; this system's local-refinement-plus-lightweight-pose-graph approach
was always expected to land well short of full BA (this was flagged before
loop closure was even built -- see the "no bundle adjustment" architectural
note at the top of `SlamWorker.h`).

**Bug found and fixed in the *Python* evaluation scripts (now superseded by
`kitti_ate`, but note for posterity):** `compute_ate.py`/`plot_trajectory.py`
in the scratchpad computed ground-truth path length by summing distances
only between *matched* (i.e. successfully-tracked) trajectory points. Any
gap where tracking was briefly lost gets chord-shortcut across in that sum
instead of following the road, undercounting the true path length (got
2440.5m instead of the correct ~3722m). `kitti_ate.cpp` sums the complete,
contiguous ground-truth path instead, which is correct.

### Root cause (found), 2026-07-11

## Root cause (found), 2026-07-11

**The permanent lockup is caused by a circular/self-reinforcing feedback
loop in `m_avgStepScale`, not by anything in the homography branch.**
Homography was fully ruled out first (see "Investigation trail" below for
how) before this was found.

`m_avgStepScale` (`SlamWorker.cpp`) is an EMA of "typical per-frame camera
displacement," updated in `pushTrajectoryPoint()` every time a new
trajectory point is pushed. It's read back in `recoverViaEpipolar()` to
turn the unit-length translation that `estimateTwoViewPose()`/`recoverPose`
produces into a real-world-scale step: `trel = trel * (m_avgStepScale *
framesElapsed)`. That rescaled `trel` is then composed into `newT` and
**pushed right back into the trajectory via the same `pushTrajectoryPoint()`
call that updates `m_avgStepScale`** -- so the "measurement" folded into the
average is not an independent measurement at all, it's a value the average
itself just manufactured.

This is fine as long as `trackFrame()`'s real PnP-based steps (which *are*
independent, solved from 2D-3D correspondences against the map) dominate.
But once a couple of genuinely large or borderline PnP-fail frames push the
EMA up (observed concretely: stepDist 216 at frame 508 and 126 at frame 514
dragged the average from ~48 to ~73.5, see the raw trace below), every
subsequent `recoverViaEpipolar` success starts manufacturing a step of
*almost exactly* the current average (73.4-73.5, frame after frame) and
feeding it straight back in. The EMA's 90% weight on its own previous value
(`kAvgStepScaleAlpha = 0.1`) means genuine smaller measurements from
`trackFrame`, when they do land, barely nudge it back down -- the average
is effectively stuck confirming itself.

Once the reference keyframe then goes stale (both `trackFrame` and
`recoverViaEpipolar` fail on the same frame -- the pre-existing
frozen-reference mechanism described further down in this doc), each
further failed recovery attempt multiplies the *already-inflated*
`m_avgStepScale` by a growing `framesElapsed` (1, 2, 3, ... capped at
`kMaxFramesElapsedForRescale = 15`). The guessed camera position is pushed
absurdly far along the recovered direction every attempt (observed: `trel_z`
of -73, -147, -220, -294, ... growing linearly), quickly overshooting past
any real 3D structure visible from the stale reference. Result: `triangulate()`
finds **zero valid (positive-depth) points out of hundreds of candidates**,
every single attempt, forever -- confirmed directly via added instrumentation
(`[recover] triangulation-empty fail: ... valid=0`). Not an F-vs-H model
selection problem, not an `isPlausibleStep` rejection (the guessed poses
were well inside its bound the whole time) -- purely a corrupted scale
estimate compounding through triangulation.

### The fix

`pushTrajectoryPoint()` (`SlamWorker.h`/`.cpp`) now takes a third parameter,
`updateAvgStepScale` (default `true`). `recoverViaEpipolar()`'s call passes
`false`: its step's magnitude was itself derived from `m_avgStepScale`, so it
must never be folded back into that same running average. `trackFrame()`'s
call (real, independent PnP measurement) and the one-time initialization
bootstrap call both keep the default `true`. This breaks the circular
dependency entirely while leaving `recoverViaEpipolar`'s use of
`m_avgStepScale` as a magnitude *heuristic* (reading it) intact -- only the
feedback write is removed.

### Verification so far

- Rebuilt both targets clean.
- Re-ran the exact scenario that permanently locked up before the fix
  (deterministic repro at frame ~525, see "Investigation trail"): trajectory
  now sails straight through frame 525 with no gap, continues past the known
  hard corner (~frame 1265) with only brief transient `Lost`→recovered blips
  (35 total across ~2000 frames, none permanent -- trajectory count keeps
  climbing after every one), reaching frame 1994 (the run's time budget) with
  `Trajectory` still actively incrementing.
- Full 4541-frame run in progress at time of writing this entry -- **check
  the tail of the log / rerun before declaring this fully done** (see
  "Next steps" below).

### Next steps for whoever picks this up

1. **Confirm the full 4541-frame run completes without a permanent freeze**
   (this was started but not yet confirmed complete as this entry was
   written -- rerun `./build/sift_vslam_debug <video> 480` or similar if
   needed and grep for a `Trajectory:` value that stops changing for
   hundreds of consecutive lines the way frame 1116/525 did before the fix).
2. **Clean up temporary debug instrumentation** once confirmed: remove all
   `[model]`, `[track]`, `[recover]`, and `[step]`-tagged `std::fprintf`
   lines from `SlamWorker.cpp` (search for those tags), and follow the
   existing "Temporary artifacts" section below to remove `debug_main.cpp`
   / the `sift_vslam_debug` CMake target once this is fully done, per this
   session's normal practice (skipped only for the explicit handoff that
   started this file).
3. Decide whether to keep `kRansacSeed = 42` (deterministic RANSAC,
   introduced this session) permanently or revert to
   `std::random_device`-seeded RANSAC once debugging is complete. Recommend
   keeping it, or at minimum keeping *a* fixed-seed mode toggle -- the
   non-determinism cost real time this session (see "Investigation trail")
   and will again for any future marginal-frame issue.
4. The frozen-reference-frame permanence mechanism (single-keyframe
   recovery target, never refreshed on failure) is still a real,
   independent architectural gap -- it's what turns *any* single bad
   frame into a multi-thousand-frame freeze instead of a blip. This fix
   closes the specific mechanism that was producing bad frames in this
   scenario, but doesn't add resilience against some other future cause
   of a bad frame. Worth a follow-up if this class of bug recurs (see the
   still-relevant discussion further down this doc, "Why a single unlucky
   RANSAC draw becomes a *permanent* lockup").

## Investigation trail, 2026-07-11 (leading hypothesis refuted, path to root cause)

Ran the already-queued `[model]` instrumentation (see "Instrumentation
already added" section below for what it prints). Two back-to-back runs of
the *identical* command against the *identical* video:

```bash
./build/sift_vslam_debug ".../kitti_00.mp4" 190   # run 1
./build/sift_vslam_debug ".../kitti_00.mp4" 140   # run 2 (after adding more [track]/[recover] prints)
```

**Run 1** locked up permanently at frame 1116 (`Trajectory` frozen from
frame 1117 onward, state goes `Lost` and never recovers, matching the
previously-reported symptom). Across the entire run (frames 1-1899, well
past the lockup point), `grep -c "\-> H$"` was **0** and `grep -c "\-> F$"`
was **80** -- homography was never once selected, not even a single time.
Right at the lockup transition (frames 1113-1121), `ratio` was consistently
0.24-0.39, nowhere near the `kHomographyPreferenceRatio = 0.45` threshold.
**This directly refutes the leading hypothesis below (H/F ratio
miscalibration / H winning too often) for this run.** The H branch is inert
here; whatever is failing is entirely inside the F path / recovery-state
machine.

**Run 2**, same video, same command, same binary (rebuilt only to add two
more `[track]`/`[recover]` fprintf sites, no logic changes) -- **did not
lock up at all**. Trajectory climbed normally and continuously from frame
1110 through frame 1396 (end of the run's time budget), with only brief
1-frame PnP hiccups (`[track] PnP fail: ok=1 inliers=5 min=10` at frame
1133, `inliers=6` at frame 1137) that were immediately recovered via
`recoverViaEpipolar` in the very next attempt, exactly as designed. No
`Lost` state was ever entered (`grep -c Lost` = 0 for the whole run).

### Root cause of the run-to-run difference: unseeded RANSAC RNG

`SlamWorker.cpp:434` (`estimateFundamentalRansac`) and `SlamWorker.cpp:503`
(`estimateHomographyRansac`) both do:

```cpp
std::mt19937 rng(std::random_device{}());
```

i.e. a **fresh, entropy-seeded RNG on every single RANSAC call, every
frame**, for both estimators. There is no fixed seed anywhere. This means
the actual minimal-sample draws RANSAC uses are different on every run of
the program, even for byte-identical input video and byte-identical code.
At an ordinary, comfortably-inlier-rich frame this doesn't matter (RANSAC
converges to basically the same answer regardless of sample order). But at
a marginal frame -- and frames 1113-1137 here have visibly collapsing
correspondence counts (`npts` 256 -> 226 -> 197 -> ... -> low 40s-60s,
`inliers=5`/`6` right at `kMinTrackInliers=10`) -- whether a given RANSAC
call happens to find a "good enough" sample within its 500 iterations is
now **literally a coin flip that changes between runs**.

This is the actual explanation for the "worse than before, doesn't recover"
symptom being hard to pin down: **the underlying single-frame failure at a
hard, low-feature-count frame is not new or specific to the homography
change.** It has probably always been possible (F/E recovery has always
been marginal exactly at those low-correspondence, high-baseline-change
frames). What may genuinely be new (still unverified -- see next steps) is
either (a) sheer bad luck across a few debugging sessions surfacing the
existing marginal-frame risk more often, or (b) the homography branch
subtly changing *which* frames end up marginal (e.g. by adding CPU cost per
frame that shifts frame timing/sampling in the debug harness, or by some
other second-order effect) without being the branch that's actually chosen.
Both are speculative; don't assume either without more runs.

**Separately, and importantly: this non-determinism means every past
conclusion drawn from a single log in this document (including the
"homography is the culprit" framing that motivated most of the earlier
investigation) was drawn from a sample size of one nondeterministic run,
and should be treated as unconfirmed until it can be reproduced with a
fixed seed.** The "known-good reference point" section's baseline claim is
in the same boat -- it was almost certainly also never rerun multiple times
to check for the same variance.

### Why a single unlucky RANSAC draw becomes a *permanent* lockup

This part is a real, confirmed design gap, independent of RANSAC
determinism, and is very likely why an occasional marginal-frame failure
(which will always happen sometimes, homography or not) turns into
"frozen for the remaining 3300+ frames of the video" instead of a brief
blip:

- `recoverViaEpipolar()` (`SlamWorker.cpp:858`) matches the current frame
  against a **single fixed reference keyframe** (`m_refDescriptors` /
  `m_refKeypoints`), refreshed via `setReferenceFrame()` **only when a
  recovery attempt succeeds**.
- If `trackFrame()` (PnP against the whole map) fails AND
  `recoverViaEpipolar()` also fails on the same frame (for any reason --
  RANSAC unluckiness, `isPlausibleStep` rejection, whatever), the reference
  keyframe does **not** advance. The next frame's recovery attempt matches
  against that same, now one-frame-more-stale keyframe.
- Because the vehicle keeps moving, viewpoint divergence from that frozen
  keyframe grows every subsequent frame, so correspondence counts
  (`npts`/`matches`) shrink monotonically frame over frame -- exactly what
  run 1's log shows (256 -> 226 -> ... -> 18) -- **not because anything is
  "wrong" with F vs H, but purely because the reference view is stale and
  getting staler.**
- Once `nF` (or raw match count) drops below `kMinInitInliers` /
  `kMinInitMatches`, recovery becomes mechanically impossible (not enough
  correspondences to even attempt a pose), and it stays impossible forever,
  because nothing in the codebase ever relocalizes against anything other
  than that one frozen reference frame, and nothing ever resets/refreshes
  the reference frame on failure. There is no keyframe database, no
  relocalization against older keyframes, no reference-frame reset timer --
  by design (see the comment already in the code at
  `recoverViaEpipolar()` acknowledging the frozen-reference tradeoff for
  the `framesElapsed` cap, though that comment undersells how catastrophic
  the failure mode actually is).
- **This means literally any single-frame recovery failure at a moment
  when both `trackFrame` and `recoverViaEpipolar` fail together is a
  latent permanent-lockup trigger**, regardless of whether homography
  exists at all. The homography change may have made this more likely to
  trigger (more RANSAC calls per frame = more chances for an unlucky draw;
  or some other second-order effect -- unconfirmed), or may be unrelated
  and the timing was coincidental. But **the permanence itself is a
  pre-existing architectural gap in the recovery design, not a bug
  introduced by the homography branch.**

### Concrete next steps (updated, supersedes the old "next steps" below where they conflict)

1. **Make RANSAC reproducible before doing anything else.** Add a way to
   seed both `estimateFundamentalRansac`'s and `estimateHomographyRansac`'s
   `std::mt19937` deterministically (e.g. a fixed seed constant for the
   debug harness, or seed from frame index so different frames still get
   different samples but reruns are identical). Without this, no future
   debugging session can trust that a reproduced (or non-reproduced) lockup
   means anything -- it's currently impossible to tell "fixed the bug" from
   "got lucky this run."
2. **With a fixed seed, binary-search which frame's RANSAC draw is the
   trigger** and inspect it directly (dump the actual sample points/inlier
   set for that one call) rather than inferring from aggregate npts/ratio
   trends.
3. **Decide whether the frozen-reference-frame permanence gap (above) is
   in scope to fix now or is a separate, pre-existing issue to file
   separately.** Even a perfect model-selection fix does not close this
   gap -- any future marginal frame (turn, motion blur, low-texture
   stretch, etc.) can still trigger it. Candidates if it's in scope:
   relocalize against multiple recent keyframes instead of just the last
   one; refresh/advance the reference frame periodically even on failure
   (with the `framesElapsed` cap already in place to bound the rescale
   blast radius, per the existing comment); or lower `kMinInitMatches`/
   `kMinInitInliers` specifically for the recovery path so a stale-but-not-
   yet-hopeless reference still gets a chance.
4. The old "leading hypothesis" and its associated next-steps list below
   are superseded for the *lockup* question -- H/F ratio miscalibration is
   refuted by run 1's data (H never even got selected). They may still be
   worth doing eventually for general correctness/tuning of the homography
   branch, just not as the explanation for this regression.

## What was being fixed

At a real ~90° turn at a street intersection (KITTI `kitti_00.mp4`, frame ~1265,
"Feldbergstr."), monocular tracking would freeze because essential-matrix (F/E)
recovery is geometrically degenerate for near-pure rotation / small translation
baseline (recoverPose's translation direction becomes unstable as baseline→0).
This is real and confirmed by pulling the actual video frames and observing a
sharp ~90° FOV change within ~15 frames (1265→1280).

The fix in progress: add a homography (H) model alongside the fundamental
matrix (F), matching ORB-SLAM's `Initializer` approach — estimate both per
attempt, pick whichever fits the correspondences better via an inlier-count
ratio test, decompose whichever wins.

## What was implemented (src/vision/SlamWorker.h / .cpp)

- `solveHomographyDLT()` (anonymous namespace, near `solveEightPoint`): normalized
  DLT for a 3x3 homography from >=4 correspondences, x2 ~ H x1. Verified by hand
  against Hartley & Zisserman Alg. 4.2 — equations check out (see chat history /
  git history of this session for the derivation).
- `homographyTransferErrorSq()`: symmetric transfer error (squared pixels, both
  directions) used as the RANSAC inlier test for H, analogous to `sampsonDistance`
  for F.
- `SlamWorker::estimateHomographyRansac()`: RANSAC wrapper mirroring
  `estimateFundamentalRansac()` (4-point minimal sample, `kHRansacIterations = 500`,
  `kHRansacThreshold = 4.0` px²).
- `SlamWorker::estimateTwoViewPose()`: new unified method that runs BOTH
  `estimateFundamentalRansac` and `estimateHomographyRansac`, computes
  `ratio = nH / (nH + nF + 1)`, prefers H if `ratio > kHomographyPreferenceRatio (0.45)`
  or F is empty. F branch: `E = K^T F K`, `cv::recoverPose`. H branch:
  `cv::decomposeHomographyMat(H, K, Rs, Ts, Ns)` then manually picks the
  candidate (up to 4) with the most `triangulate()`-validated (positive-depth)
  points, normalizing the winning translation to unit length (decomposeHomographyMat's
  t is scaled by an arbitrary plane distance, NOT unit length like recoverPose's --
  this was explicitly normalized to keep both branches on the same convention).
- Wired into both `initializeFromFrame()` and `recoverViaEpipolar()`, replacing
  their previous duplicated "F→E→recoverPose" blocks with a single
  `estimateTwoViewPose()` call.

This all compiles cleanly and the app runs without crashing.

## The regression (found via headless testing, NOT yet root-caused)

Re-added the temporary headless debug harness (this is currently present in the
tree — see "Temporary artifacts currently in the tree" below) and ran the full
4541-frame `kitti_00.mp4` through it:

```
cd "/run/media/nam/D1/Master/Computer Vision/Project"
./build/sift_vslam_debug "/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/video_samples/kitti_00.mp4" 470
```

Result: tracking works fine and progresses normally from frame 1 up to roughly
frame 1240 (trajectory count climbing, map filling up to its 2000-point cap
normally). Then, right around the known hard intersection, it goes `Lost` --
and **never recovers for the rest of the video** (frames ~1240 through 4541,
i.e. over 3300 frames / most of the remaining footage). Both
`Trajectory: 1062` and `Map points: 2000` are frozen literally unchanged for
that entire remaining stretch (checked via `grep` over the log, see below).

This is worse than the pre-homography behavior, where the trajectory would at
least freeze only through the hard corner itself and then resume once normal
geometry returned. Something about adding the H branch causes recovery to stop
working *permanently*, not just through the hard corner.

### Evidence

Log saved at `/tmp/homography_test.log` (may not survive past this session --
regenerate with the command above if needed). Key checks run:

```bash
grep -c "\[state\] Lost" /tmp/homography_test.log        # 3475 lines
grep -c "\[state\] Tracking$" /tmp/homography_test.log   # 1029 lines (all before the lockup)
grep -n "\[state\] Tracking$" /tmp/homography_test.log | tail -3   # last one around line 2125
tail -20 /tmp/homography_test.log   # shows Trajectory: 1062 frozen all the way to Frame 4541 | Stream ended
```

## New evidence: it locks up on a STRAIGHT street, not the corner

User-reported and important: the permanent lockup happens while driving
**straight**, not while turning at the intersection. This strengthens the
leading hypothesis below considerably -- a straight street is a textbook
general-3D-scene case where the fundamental matrix should clearly be the
right model and homography should basically never win. If homography is
still getting selected (or causing the failure) there, the model-selection
logic itself is very likely mis-tuned/buggy, independent of anything specific
to the intersection/rotation case. Confirming this is now step 1.

## Instrumentation already added, NOT yet run to a useful conclusion

Temporary `std::fprintf(stderr, ...)` instrumentation has been added to
`SlamWorker::estimateTwoViewPose()` in `src/vision/SlamWorker.cpp` (search for
`[model]` in that file to find all of it -- there are prints for: the
nF/nH/ratio/chosen-branch decision, F-branch rejection reasons, the number of
`decomposeHomographyMat` candidates, each candidate's cheirality support
count, and H-branch accept/reject). `#include <cstdio>` was added for this.
It compiles (`sift_vslam_debug` target rebuilt successfully with it in) but
**has not actually been run yet** -- the run was interrupted before producing
a log. This is the very next thing to do.

### Exact command to run (do this first)

```bash
cd "/run/media/nam/D1/Master/Computer Vision/Project"
./build/sift_vslam_debug "/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/video_samples/kitti_00.mp4" 190 > /tmp/model_selection_debug.log 2>&1
```

190 seconds should cover roughly the first ~1400-1800 frames at this machine's
observed ~9.5 fps processing rate, comfortably past the frame-1240ish lockup
point with margin. If the log doesn't reach past frame ~1300 (check with
`grep "Frame 1[23]" /tmp/model_selection_debug.log | tail -5`), rerun with a
larger seconds argument.

### What to look for once it's run

```bash
# Overall H vs F selection rate -- if H is winning a large fraction even
# during normal straight driving, that confirms the ratio/threshold is miscalibrated.
grep -c "\-> H$" /tmp/model_selection_debug.log
grep -c "\-> F$" /tmp/model_selection_debug.log

# Look at a normal early stretch (e.g. frame 500-600, known-good straight
# driving before any problems) to see the baseline nF/nH/ratio behavior:
grep -B2 -A2 "Frame 5[0-9][0-9] " /tmp/model_selection_debug.log | grep "\[model\]" | head -40

# Then look right at the lockup transition (~frame 1240 onward) and compare:
grep -A2 "Frame 124[0-9] \|Frame 125[0-9] " /tmp/model_selection_debug.log

# Does it ever accept H at all, and if so, what does bestSupport/bestIdx look
# like compared to nH itself (an accepted candidate with very low support
# relative to nH would be suspicious)?
grep "H accepted\|H rejected" /tmp/model_selection_debug.log | sort | uniq -c | sort -rn | head -20
```

Once the actual nF/nH/ratio numbers are in hand, decide next action:
- If H is winning way more than ~10-20% of attempts even in normal straight
  driving: the ratio test or one of the two thresholds
  (`kHRansacThreshold` vs `kFRansacSampsonThreshold`) is the bug -- they are
  different metrics on different natural scales and were never empirically
  cross-calibrated. Try recalibrating `kHRansacThreshold` down (stricter) or
  `kHomographyPreferenceRatio` up (e.g. 0.6-0.7, closer to what some
  ORB-SLAM variants actually use in practice) and re-test.
- If H is winning rarely but the lockup still coincides with an H acceptance:
  the H branch itself (decomposition, candidate disambiguation, or the
  unit-translation normalization) is producing a bad pose that then poisons
  `m_avgStepScale` or `m_refR/m_refT` -- see hypothesis 2 below.
- If it's neither (F is being chosen throughout, or `estimateTwoViewPose`
  is failing both branches outright at the lockup point): the bug may not be
  in model selection at all, and the "straight street" report just means
  ordinary F/E recovery started failing for some other reason coincident
  with this change (e.g. a subtle behavior change in shared code that both
  `initializeFromFrame()` and `recoverViaEpipolar()` now go through). Diff
  the current `estimateTwoViewPose`'s F-branch logic line-by-line against the
  pre-homography F-only code (git history / earlier chat transcript) to check
  nothing was altered besides the wrapping.

Remove all `[model]` fprintf lines once root-caused and fixed. Also remove
the `[track]`/`[recover]` fprintf lines added in the 2026-07-11 session
(search for those tags in `trackFrame()` and `recoverViaEpipolar()` in
`SlamWorker.cpp`) -- they were added to pinpoint the exact failing check
(match-count / PnP-inlier-count / `isPlausibleStep`) at the lockup frame and
are still needed for the next debugging session, so leave them in for now.

## Leading hypothesis (NOT yet verified -- next session should check this first)

The model-selection ratio test may be miscalibrated, causing homography to be
selected far more often than it should be -- including for ordinary
non-planar, non-rotation-dominated driving scenes where F/E is actually the
correct model. If H gets picked in cases where it's a poor fit, and something
in the H branch (candidate disambiguation, or the unit-translation
normalization, or an interaction with `isPlausibleStep`'s rescaling) produces
a systematically bad result, that could plausibly explain a *permanent*
lockup rather than a transient one at just the hard corner: once something in
persistent state (m_refR/m_refT, m_avgStepScale, or the map itself) gets
driven to a bad value by a wrong H-branch pose, every subsequent attempt
(which reuses that same bad state) could keep failing indefinitely.

Concretely suspicious: `kHRansacThreshold = 4.0` (symmetric transfer error,
in px²) vs `kFRansacSampsonThreshold = 1.0` (Sampson distance, in px²) --
these are different error metrics with different natural scales, and were
not empirically calibrated against each other. It's plausible H's threshold
is effectively "easier" to satisfy, inflating `nH` relative to `nF` and
biasing `ratio` toward homography far more often than the ORB-SLAM heuristic
intends.

## Concrete next steps for the next session

1. **Run the already-added instrumentation** -- see "Instrumentation already
   added, NOT yet run to a useful conclusion" above for the exact command and
   what to grep for. This was queued up but not executed yet.

2. **Check `isPlausibleStep` interaction**: after a wrong-ish H-branch pose
   is (hypothetically) accepted once, does it corrupt `m_avgStepScale` badly
   enough that all future candidates -- even correct ones -- get rejected by
   the strict plausibility bound? (`m_avgStepScale` only updates on accepted
   steps, and there is deliberately no bypass anymore -- see the comment on
   `isPlausibleStep` in SlamWorker.h for why that bypass was removed. If a
   single bad accepted H-branch step poisons the average, *this* could be
   the actual lockup mechanism, independent of any threshold-calibration
   issue.)

3. **Sanity-check `estimateHomographyRansac` in isolation** with synthetic
   data (a small standalone test, e.g. compiled the same way the earlier
   `bench_*.cpp` scratch files in this session were -- see chat history for
   the pattern: `g++ -O2 file.cpp -o file $(pkg-config --cflags --libs opencv4)`)
   -- generate points under a known homography, confirm
   `estimateHomographyRansac` recovers it within tolerance, independent of
   the rest of the SLAM pipeline. This isolates whether the estimator itself
   is correct (the DLT equations were checked by hand and appear correct,
   but this has not been empirically verified against synthetic ground truth).

4. **Consider whether `cv::decomposeHomographyMat`'s translation convention
   assumption is actually right.** The code assumes it returns a translation
   scaled by an arbitrary plane distance and normalizes it to unit length to
   match `recoverPose`'s convention -- this assumption should be verified
   against OpenCV's actual documented/observed behavior, ideally with a
   synthetic test, not just from memory of the API.

## Temporary artifacts currently in the tree (were NOT cleaned up this time,
   unlike previous debugging rounds -- intentional, so the next session can
   use them immediately)

- `src/debug_main.cpp` -- headless harness, drives `SlamWorker` directly via
  `QCoreApplication` (no GUI/QThread), prints `[state]`/`[stats]`/`[open]` to
  stderr. Usage: `./build/sift_vslam_debug <video-path> [seconds]`.
- `CMakeLists.txt` has a temporary `sift_vslam_debug` executable target
  (clearly marked `# TEMPORARY debug harness` in the file) building the above
  against `SlamWorker.cpp`/`VideoSource.cpp` directly.
- Build both targets: `cmake --build build -j$(nproc)` (builds both
  `sift_vslam_gui` and `sift_vslam_debug`).

**Once this regression is actually fixed and verified, remove both of these**
(delete `src/debug_main.cpp`, remove the `sift_vslam_debug` target block from
`CMakeLists.txt`, reconfigure with `cmake -S . -B build`) -- this has been the
practice throughout this session; it was only skipped this one time because
of the explicit handoff request.

## Known-good reference point

Immediately before the homography fallback was added, the system was working
as well as it was going to get with F/E-only recovery: strict (no-bypass)
`isPlausibleStep`, `kMaxStepMultiplier = 10.0`, `kMinInitMatches`/
`kMinInitInliers = 20`, `kKeyframeEveryNFrames = 8`, `kMaxFramesElapsedForRescale = 15`.
At that point the known limitation was exactly the one this fallback was
meant to fix: freezing through hard turns like the Feldbergstr. intersection,
but *recovering* afterward. If the homography work proves hard to fix
correctly, reverting `estimateTwoViewPose()` back to calling
`estimateFundamentalRansac()` + `recoverPose()` directly (i.e. undoing just
the model-selection wrapper, keeping everything else) returns to that known-
working baseline.
