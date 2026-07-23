# DEBUGGING v3 — Backend #1/#2 + new SIFT-DBoW2 vocab (session 2026-07-23)

Continues `DEBUGGING_v2.md` (best so far: **41.782m ATE**, seq00, custom `kitti_ate`
pipeline). This session: implemented the two backend levers from v2's NEXT list and
wired in the newly-available retrained vocabulary. **Code done + compile-validated;
no new ATE numbers yet — the full run was started then stopped before finishing.**

## What changed (code) — all default-OFF, so the 41.8m recipe is byte-for-byte unchanged unless opted in

### Backend #1 — local-BA pose-prior weights now CLI-overridable (argv60/61)
- `kLocalBaPosePriorRotWeight`(20)/`TransWeight`(3) constexpr → runtime members
  `m_localBaPosePriorRotWeight/TransWeight` + `setLocalBaPosePriorWeights()`
  (`SlamWorker.h`/`.cpp`), same migration precedent as `kLocalBaWindowKeyframes`.
- CLI: **argv60/61** = `<rot> <trans>` (`-`/0 keeps defaults 20/3). Mirrors the FE
  leash-weight override (argv58/59). The map-side analogue of the leash: lower =
  looser prior = local BA free to move window poses further from PnP (more map-fit
  accuracy, more scale-collapse risk).

### Backend #2 — map-quality lever (two independent flags)
- **`retriangulate` (argv62)**: once a landmark has ≥`kRetriangulateMinViews`(3)
  cross-keyframe observations, re-triangulate from ALL of them via
  `triangulateMultiView()` (was only used in the closed Phase-B merge), accepting
  only a **strict reprojection improvement** over the current position (BA-safe;
  can't regress a landmark local BA already refined). New method
  `retriangulateKeyframeLandmarks()`, called after fuse / before local BA so BA
  refines the better seed. Reports `[retri] landmarks re-triangulated this run: N`.
- **`parallaxgate` (argv63)**: reject newly-triangulated landmarks whose two viewing
  rays are more parallel than `kMinTriangulationParallaxCos`(~1°) — noise-dominated
  depth that pollutes the map (item-40 backfire mechanism). Added at landmark
  creation in `trackFrame()`'s newPoints loop.

### Bug caught + fixed during implementation
`triangulateMultiView()` dereferences `m_keyframeHistory[obs.first]`, but the first
draft called re-triangulation from inside `recordLandmarkObservations()` — which runs
BEFORE the current keyframe is pushed to `m_keyframeHistory`. That would index one
past the end. Fixed by moving re-triangulation into its own pass invoked AFTER the
`m_keyframeHistory.push_back()`.

### Vocab — no code needed
The retrained SIFT-DBoW2 vocabulary (`sift_dbow_vocab (2).txt`, K=10/L=5, RootSIFT)
drops into the existing `siftdbow <path>` flag (**argv36/37**); takes precedence over
`vlad` for loop-candidate search. File name has a space + `(2)` — quote it.

## Combined recipe (41.8m + vocab + backend knobs)
Start from the exact 41.8m command; two edits:
1. Set argv36/37 (the two `-` before `fuse`) → `siftdbow "/…/sift_dbow_vocab (2).txt"`.
2. After the final `poseonlyleash` (argv57) append: `- - - - retriangulate parallaxgate`
   (dashes = argv58/59 leash weights + argv60/61 local-BA weights, both default).

Isolate each lever: vocab-only first, then +retriangulate, then +parallaxgate, then
sweep argv60/61 (e.g. `15 2.5`). Watch ATE + scale + max-error + the `[retri]` count
(0 = strict-improvement guard rejected everything, expected if BA is already tight).

## Validation done
- **Docker `-fsyntax-only` compile-check** (g++ 13.3, Qt6/OpenCV4.6/Ceres/g2o headers):
  BOTH changed translation units (`SlamWorker.cpp`, `analyze/kitti_ate.cpp`) PASS —
  only pre-existing Eigen `AlignedBit` deprecation warnings. So all Backend #1/#2 edits
  are type-correct.

## RESULTS (2026-07-23, Linux box, local dataset + local build — no Docker needed)

Isolated each lever on top of the 41.782m leash config:

| Config | ATE | Note |
|---|---|---|
| baseline (no leash) | 51.273m | |
| + poseonlyba + poseonlyleash | 41.782m | v2's best |
| **+ retriangulate** | **37.003m** | **NEW BEST.** 25157 landmarks re-triangulated. |
| + siftdbow (new vocab) | 110.5m | NEGATIVE — see root cause below |
| + parallaxgate | 131.2m | NEGATIVE — starves tracking, same pattern as every other "reject more" gate this project has tried |
| retriangulate + local-BA-prior 15/2.5 (looser) | 169.5m | NEGATIVE — scalar tuning is chaotic here, don't retune |
| retriangulate + local-BA-prior 60/10 (tighter) | 116.4m | NEGATIVE, same reason |

**NEW BEST CONFIG: baseline + `poseonlyba` + `poseonlyleash` + `retriangulate` = 37.003m**
(argv50=poseonlyba, argv57=poseonlyleash, argv62=retriangulate; leave argv60/61
local-BA-prior weights at default `-`/`-` — sweeping them only made it worse).

### Why siftdbow (new vocab) is still negative — root-caused from the log, not guessed
Direct comparison of the SAME physical loop revisit (~frame 405) in the siftdbow
run vs a VLAD run: VLAD measured a clean correction (54 world units, scaleMeas
1.23, 20 Sim3 inliers); siftdbow measured the same event as 340.7 world units with
**scaleMeas clamped at the 3.0 safety ceiling** (i.e. computed scale exceeded 3x —
a degenerate measurement) on only 15 inliers, then fired a near-duplicate closure
on the same weak signal shortly after (another 335-unit correction). Root cause:
siftdbow's raw candidate scores are still weakly discriminative — full distribution
min=0.13, **median=0.246**, max=0.62 (VLAD's accepted-closure scores run 0.5-0.75).
Also notable: this new vocab has **7216 words, actually FEWER than the OLD
under-trained vocab's 7928** despite 3x more training data (`stride` 10→3) — the
retraining did not clearly fix the word-count starvation item 29 diagnosed. Worth
checking the Kaggle training run's own log (did it complete cleanly / hit a
timeout?) before retraining again. `siftdbow` stays off; `vlad` remains the
loop-candidate search of choice for now.

### Why parallaxgate is negative
Same pattern as every other "reject more, trust less" gate tried this project
(reprojErr tightening, loopqualitygate, pnpfullrefine's implicit tightening): SIFT
is already sparse on KITTI, and rejecting low-parallax landmarks at creation time
removes real, needed observations rather than only noise -- net starves tracking.
Consistent with the project's now well-established rule: gates that REJECT data
tend to backfire on this sparse-SIFT pipeline; gates that ADD/REFINE data
(retriangulate, the leash) tend to help.

## Item 42: tried to extend retriangulate further — ALL extensions negative, original stays best

Implemented (all default-off, additive, byte-identical unless opted in):
- `retriangulateWindowLandmarks()` — re-triangulate the union of every landmark
  touched anywhere in the loop window, called right after loop-BA/correction lands
  (in addition to the existing per-keyframe pass). Shares implementation with the
  original via new `retriangulateLandmarkIds()` helper.
- `setRetriangulateMinViews()` — CLI-tunable min-views threshold (argv64, was a
  hardcoded constexpr=3).
- Tested combos with the already-existing `cull` (argv20) and `covismap` (argv27)
  flags, neither previously tried alongside retriangulate/leash.

| Config (on top of the 37.003m base) | ATE |
|---|---|
| **retriangulate, original (per-keyframe only, min-views=3)** | **37.003m — still best** |
| + post-loop-BA window pass | 55.709m | NEGATIVE |
| + post-loop-BA window pass + cull | 55.709m | NEGATIVE, byte-identical to the line above -- `cull` is a no-op in this config (consistent with it being inert in earlier session tests too) |
| min-views=2 (instead of 3) | 68.433m | NEGATIVE |
| + covismap | 118.180m | NEGATIVE, worst of the four |

**All four extensions made things worse. The original retriangulate design — one
pass per newly-inserted keyframe, min-views=3, BEFORE local BA — remains the best
known config at 37.003m.** Likely reason the window pass hurts: re-triangulating
right after loop-BA pulls in observations from keyframes loop-BA hasn't finished
reconciling yet, and the "accept only if reprojection strictly improves" guard
isn't strict enough to filter out a worse-but-locally-better-fitting solution in
that transient state. Not pursued further — same "changing a working config
usually backfires on this sparse-SIFT pipeline" pattern as octaveweight/
parallaxgate/prior-tuning. New code stays in the tree, default off
(`setRetriangulateMinViews()`/window pass unused by the recommended recipe).

**Recommended config unchanged: baseline + `poseonlyba` + `poseonlyleash` +
`retriangulate` (defaults) = 37.003m.**

## Tested whether the "two negatives combine positively" precedent generalizes — it does NOT for this pair

Precedent (item 40's footnote): `reprojErr=4` (starves matches, 140.8m alone) +
`pnpfullrefine` (overfits noise, 175.7m alone) combined to 73.0m -- reprojErr=4
filtered to a clean inlier set that fullrefine could then safely fit tightly to.

Tested the most analogous untried pair here: `parallaxgate` (filters noisy
landmarks at CREATION, 131.2m alone with leash) + `retriangulate` (re-fits
survivors' positions, 37.003m alone -- the current best). **Result: 142.077m --
WORSE than either alone, not a cancel-out.** Only 17250 landmarks got
re-triangulated (vs the ~25k the retriangulate-alone run touches) -- parallaxgate
removed landmarks retriangulate needed rather than handing it a cleaner set.
`Recovered scale` also overshot to 1.2576 (every other negative result this
session undershot below 1.0) -- a qualitatively different failure mode, not
the same "map compression" collapse seen elsewhere.

**Lesson: the cancel-out precedent is NOT a general pattern -- it worked for
reprojErr/fullrefine because both act on the SAME axis (the live PnP solve's
inlier set). parallaxgate and retriangulate act on DIFFERENT axes (landmark
creation-time filtering vs. post-hoc position refinement), so one doesn't
"clean up for" the other -- parallaxgate's rejects were exactly the marginal
landmarks retriangulate could have improved.** Don't assume two negatives will
combine well without a mechanistic reason they share an axis. Best config
still baseline + `poseonlyba` + `poseonlyleash` + `retriangulate` = 37.003m.

## ⚠️ CORRECTION: the 37.003m "new best" does NOT generalize — confirmed regression on seq01

Tested baseline vs. the leash+retriangulate "best" config on **seq01** (held-out,
never tuned against, KITTI's known-hard high-speed highway sequence, 1101 frames,
**zero loop closures** in either run):

| Config | seq01 ATE | Recovered scale |
|---|---|---|
| baseline (plain SQPnP, no leash/retriangulate) | **122.622m** | 1.0529 (near-perfect already) |
| leash + retriangulate ("best" on seq00) | **292.784m** | 0.6431 (degraded) |

**The seq00 winner LOSES to baseline by ~2.4x on seq01.** Root cause: seq01 has
zero loop closures over 1101 frames, so nothing ever resets accumulated
per-frame bias. Baseline's plain SQPnP already reaches near-perfect scale (1.05)
on this fast, low-parallax highway sequence; the leash's per-frame pose-only-BA
refinement (motion-model NEVER lost track here -- 0 fallback events, so it ran
unaltered on every single frame) apparently introduces a small systematic bias
each frame that, with no loop closure to periodically correct it, compounds
over the full sequence into large drift. On seq00, dense urban loop closures
mask/reset this same tendency, which is likely why it looked like a clean win
there.

**This means the whole leash+retriangulate direction was validated on ONE
sequence only and is NOT safe to claim as a general improvement.** The 37.003m
number stands as a real, verified seq00 result, but "NEW BEST CONFIG" language
in this file and `DEBUGGING_v2.md` overstates it — it should be read as
"seq00-specific, untested/regressing elsewhere" until validated (or fixed) on a
proper held-out set (seq01 is now a clear counter-example; seq05/08 also
available locally in `kitti_poses/` + the kagglehub dataset for further checks).
**Recommendation: do not adopt leash+retriangulate as a default; keep it as an
opt-in, seq00/loop-rich-scenario flag until a fix (e.g. gate the leash refinement
by some confidence/consistency check so it can't silently accumulate bias over
long loop-free stretches) is found and re-validated across sequences.**

## Generalization attempt (item 43): Fix A/B v2 anchor-gates — fixed seq01, broke seq00, direction closed

Following the seq01 regression above, implemented and tested loop-closure
anchor-gates for both mechanisms:
- **Fix A** (`poseonlyleashanchor`, argv65): leash's per-frame pose-only-BA
  refinement doesn't run until `m_everHadLoopClosure` is true.
- **Fix B v1** (`retriparallaxgate`, argv66): rejected — wrong hypothesis
  (parallax gate on retriangulate's accept decision). Measured WORSE on
  seq01 (312.4m vs 186.6m unfixed). Refuted, kept in tree off by default.
- **Fix B v2** (`retrianchor`, argv67): same anchor-gate pattern as Fix A,
  applied to retriangulate's accept decision instead.

**seq01 (held-out, zero loop closures) — both fixes confirmed clean:**

| Config | seq01 ATE |
|---|---|
| baseline | 122.622m |
| leash + `poseonlyleashanchor` (Fix A alone) | **122.622m** — exact match |
| retriangulate + `retrianchor` (Fix B v2 alone) | **122.622m** — exact match, `[retri] re-triangulated: 0` |
| leash+retri + both anchors (Fix A+B v2 combined) | **122.622m** — exact match |

Both mechanisms now degrade gracefully to plain baseline when no loop
closure ever fires — confirmed, not just hypothesized.

**seq00 (loop-rich, the sequence the 37.003m win was found on) — regression:**

Re-tested the combined Fix A+B v2 config (`poseonlyba poseonlyleash
poseonlyleashanchor ... retriangulate retrianchor`) on seq00 expecting the
37.003m win to be preserved (loop closures happen early/often on seq00, so
the gate should barely matter in practice). **It did not preserve the win:**

| Config | seq00 ATE | Note |
|---|---|---|
| baseline (no leash) | 51.273m | |
| leash + retriangulate, ungated | **37.003m** | the real, standing win |
| leash + retriangulate, both anchor-gated (Fix A+B v2) | **113.815m** | WORSE than plain baseline; run also didn't finish (truncated at frame 3775/4541, no `Stream ended`, likely hit the 1200s wall-clock cutoff) |

**Root cause, visible directly in the log**: the first loop closure on
seq00 doesn't fire until frame 1580 (`kf#15<->kf#178`). Gating means leash
+retriangulate are both OFF for those first 1580 frames -- plain SQPnP
runs alone, accumulating more raw drift than the ungated 37.003m run ever
saw (leash was correcting every frame from frame 1 there). When the first
loop closure finally lands, it has to absorb a much larger accumulated
error: **translation correction=912.523 world units, scaleMeas clamped at
the 0.3 floor** -- an unusually violent correction. The run then shows
repeated large loop corrections with scaleMeas oscillating between the 0.3
and 3.0 clamps (48 loop closures total, several with BA costs in the
hundreds of millions), the same "vicious cycle" signature
`loopqualitygate` produced earlier this project -- consistent with the
first correction's violence destabilizing the map for the rest of the run,
which is also the likely reason it blew through the time budget.

**Conclusion: the anchor-gate mechanism is not a free generalization fix.**
It trades seq01 safety for a real seq00 cost -- and that cost isn't just
"loses the 37.003m improvement," it's worse than doing nothing (51.273m
plain baseline). Since anchor-gated leash+retriangulate is never better
than plain baseline on either tested sequence (matches it exactly on
seq01, loses to it on seq00), **plain baseline already dominates it as the
"safe, unknown-sequence" choice** -- there's no scenario where deploying
the anchor-gated recipe beats just using plain baseline. The anchor-gate
code (`poseonlyleashanchor`/`retrianchor`) stays in the tree, default off,
as a diagnostic tool and for any future redesign, but is **not
recommended for deployment**.

### Final recommendation: two-tier choice, not one generalized recipe

- **Sequence unknown, or known to be loop-closure-sparse**: plain baseline
  (SQPnP + localba + guided + vlad + fuse), no leash, no retriangulate. The
  only config confirmed to never be worse than itself on any tested
  sequence.
- **Sequence known to be loop-rich (dense urban revisits, like seq00)**:
  baseline + `poseonlyba` + `poseonlyleash` + `retriangulate`, **ungated**
  = 37.003m, the real, standing win. Opt-in only -- do not default this on
  for a sequence of unknown character, per the seq01 regression above.

## Two more directions tried this session (both closed, negative)

### SIFT `contrastThreshold` sweep — closed, default (0.04) stays best

Prompted by a real observation from this session's own logs: seq01
(highway) keypoint counts run ~150-200/frame vs seq00 (urban) ~600-1200/
frame, a 6-7x gap, raising the question of whether feature starvation
specifically explains seq01's fragility. Checked project history first --
`detectionScale` (raw resolution, 0.5->1.0) was already tried and reverted
(72.550m->105.692m, DEBUGGING.md item 13) with a documented hint that a
*stricter*, not looser, `contrastThreshold` might be the untested
promising direction. Added `contrastThreshold` as a new CLI override
(argv68) and swept both directions on seq01 baseline (122.622m default):

| contrastThreshold | seq01 ATE | Recovered scale | Note |
|---|---|---|---|
| 0.04 (default, OpenCV stock) | **122.622m** | 1.0529 | still best |
| 0.015 (looser) | 415.828m | 2.5297 | overshoot |
| 0.08 (stricter) | 213.798m | 1.0525 | scale fine, trajectory noisier anyway |
| 0.15 (much stricter) | 308.598m | 39.24 | degenerate -- 88 keypoints, 0 map points by frame 3 |

Neither direction helps; 0.04 stays optimal. Note the 0.08 row: scale
calibration came out almost identical to baseline (1.0525 vs 1.0529) yet
ATE nearly doubled -- confirms the earlier `detectionScale` finding's
mechanism generalizes to this lever too (fewer correspondences -> noisier
per-frame solve, independent of whether global scale happens to land
right). SIFT feature-density tuning is now closed on **both** known levers
(`detectionScale`, `contrastThreshold`) -- don't revisit without a new
hypothesis. `argv[68]` plumbing kept in tree, default off (harmless).

### Continuous GroundPlaneScale anchor — negative as tested, real bug identified, not re-tested

New mechanism (`setGroundPlaneContinuousEnabled()`, argv69/70): unlike the
existing `groundplane` flag's one-shot bootstrap-only correction, this
re-estimates the ground-plane-implied scale from the window's latest
keyframe every local-BA call and adds a soft `PosePriorCost` nudging just
that keyframe's translation toward the corrected step length -- intended
as a loop-closure-independent scale signal (would help exactly the
stretches leash/retriangulate's anchor-gates can only decline to worsen,
not actually correct). Implementation: `SlamWorker.h`/`.cpp`
(`runLocalBundleAdjustment()`), reuses `GroundPlaneScale.h`'s existing
VISO2-M algorithm and `m_groundPlaneConfig` (1.65m height, level camera).

**Tested on seq01 baseline, 3 weights -- all negative:**

| Weight | seq01 ATE | Recovered scale |
|---|---|---|
| baseline (off) | 122.622m | 1.0529 |
| 1.0 (default) | 377.476m | 3.9941 |
| 0.3 (soft) | 413.392m | 4.0061 |
| 3.0 (strong) | 312.352m | 0.1713 |

**Known bug in how this was tested, not necessarily in the mechanism
itself**: the test enabled `groundplanecontinuous` (argv69) but NOT
`groundplane` (argv9, the one-shot bootstrap correction). The design's
whole rationale -- "doesn't need the height/pitch assumption to be exactly
right, because the same assumption is already baked into the metric scale
from bootstrap onward" -- only holds if bootstrap actually used the same
ground-plane assumption. Without `groundplane` also enabled at bootstrap,
the continuous corrector spent the whole run pulling the trajectory toward
an absolute 1.65m/level-camera assumption unrelated to whatever scale the
vision-only bootstrap actually established -- consistent with the
weight-dependent sign flip (4x overshoot at weight 1.0, 0.17x undershoot
at weight 3.0 -- not a stable trend, a sign of fighting the wrong
reference rather than being under/over-tuned). **Not re-tested with both
flags enabled together** (the theoretically-valid config) -- deprioritized
after the anchor-gate seq00 regression finding shifted focus back to
closing out generalization work with the simpler two-tier recommendation
above. Left as a real, not-yet-invalidated direction for a future session:
re-run with `groundplane groundplanecontinuous` both set, on seq01, before
drawing any conclusion about the mechanism itself. Code kept in tree,
default off.

## Run status — INCOMPLETE (stopped by user)
- **Dataset located locally** (no dev-machine / Linux box needed): KITTI odometry gray
  is cached via kagglehub at
  `~/.cache/kagglehub/datasets/xuehu12/kitti-odometry-gray/versions/2/dataset/sequences/00/image_0/%06d.png`
  (4541 frames). Poses in `archive.zip` / `kitti_poses/00.txt`.
- **Full Docker build** of `kitti_ate` (root CMakeLists target) was set up
  (`Dockerfile.fullbuild`): builds ORB-SLAM3's g2o from source, overrides the
  dev-machine CMake paths (G2O_ROOT, BOOST_INCLUDE_DIR=/usr/include, CONDA_ENV_ROOT
  shim for libboost_serialization, ORBSLAM3_VOCAB_PATH), then builds only kitti_ate.
  Build reached the ORB-SLAM3 lib compile (step 12/8, `LoopClosing.cc`, warnings only,
  no errors) then was **stopped**. No image produced, no run, **no ATE numbers**.

## NEXT
1. Finish the build (`docker build -f Dockerfile.fullbuild -t orbslam-kitti-run .`)
   or build on the Linux dev box, then run the combined recipe on seq00.
2. Record ATE per lever (vocab-only → +retriangulate → +parallaxgate → argv60/61 sweep)
   the same way v2 verified 41.8m (median, max, coverage, scale).

## Standing reference
- Real ORB-SLAM3 (vendored): 6.4–10.7m seq00. Goal: <20m. Best: 41.782m.
- Files touched this session: `src/vision/SlamWorker.cpp`, `src/vision/SlamWorker.h`,
  `analyze/kitti_ate.cpp`, `DEBUGGING_v2.md` (recipe table), this file. All uncommitted.
