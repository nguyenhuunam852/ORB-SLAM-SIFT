# Debugging log: homography fallback caused a permanent tracking lockup

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
