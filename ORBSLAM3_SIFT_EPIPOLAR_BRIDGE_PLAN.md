# ORB_SLAM3_SIFT — Epipolar-bridge recovery transplant (scoping plan)

**Problem scope: the WHOLE KITTI set (00–10 with GT poses), not seq00.** Success =
higher coverage with a CONTINUOUS trajectory AND ATE not worse, measured across the
set — not an seq00-specific win.

## Root cause (confirmed by measurement, both efforts converge here)

ORB_SLAM3_SIFT's recovery paths all rely on **existing** geometry:
- `TrackWithKLT()` — KLT-flows the *last frame's existing* map points, then PnP.
- `Relocalization()` — VLAD-database match against *existing* keyframes.

Both fail the moment the car drives into **new territory** — measured directly (part 29):
`dbCandidates==0` **74.3%** of dropout time, reloc success 0.65%, dead-reckoning 0.41%.
There is nothing to recover *to*. Result: `mState=LOST` → `ResetActiveMap`/
`CreateMapInAtlas` (Tracking.cc ~2155–2160) → map fragments, coverage collapses.

The custom pipeline's `recoverViaEpipolar()` survives this exact regime because it does
the one thing ORB-SLAM3's recovery never does: **CREATE new geometry** (2D-2D triangulation
from the current view against the last keyframe) to bridge the map *forward* into unseen
scenery. Proven load-bearing: disabling it (custom pipeline `norecover` ablation) →
1097/1101 frames fail, 4 trajectory points — i.e. ORB_SLAM3_SIFT's exact failure mode.

The current fork's workaround is **fail-fast** (recovery window 3s→0.1s): reset and
re-init a fresh map quickly (coverage 498m→1278m). This spawns many short fragments,
each with its own arbitrary scale → a patchwork trajectory with scale jumps. This plan
replaces "reset fast" with "**never reset — bridge the same map forward**".

## Insertion point

`Tracking::Track()`, the `RECENTLY_LOST` branch (Tracking.cc ~2038–2130), AFTER
`TrackWithKLT()` + `Relocalization()` both fail and BEFORE the flip to `LOST`/reset.

## New method: `Tracking::TrackWithEpipolarBridge()`

Mirrors `recoverViaEpipolar()` + `CreateInitialMapMonocular()`'s map-creation pattern:

1. Match current frame ↔ `mpLastKeyFrame` (existing `ORBmatcher`, CudaSIFT/RootSIFT
   descriptors). Need ≥ N inliers or fall through to reset.
2. Two-view Essential/Homography (existing `TwoViewReconstruction`) → `Rrel`, `trel`
   (unit length).
3. **Scale anchor — the key improvement over the custom pipeline.** Instead of the
   custom pipeline's `m_avgStepScale` step-size *guess* (the drift source we spent a
   whole session fighting), scale the new triangulation so its **median depth matches
   `mpLastKeyFrame->ComputeSceneMedianDepth()`** — the same normalization mono-init
   already uses. This ties the bridge's scale to the *existing map's local scale*, so
   it is geometrically continuous, not an invented number.
4. Create a `KeyFrame` from the current frame + triangulated `MapPoint`s, add to the
   **active** map (the `CreateInitialMapMonocular` API: `new KeyFrame(...)`,
   `ComputeBoW`, `AddKeyFrame`, `new MapPoint(...)`, `AddObservation`,
   `ComputeDistinctiveDescriptors`, `UpdateNormalAndDepth`, `AddMapPoint`,
   `UpdateConnections`). Pose = last-KF pose ∘ (`Rrel`, scaled `trel`).
5. `mpLocalMapper->InsertKeyFrame(...)`; `mState = OK`; continue. Map stays alive.

## Why this can beat the fail-fast workaround (and both current efforts)

- **Trajectory continuity**: one map, no fragment scale-jumps (vs the current 8-fragment
  patchwork).
- **Drift correction**: the residual scale drift the bridge introduces is corrected by
  ORB-SLAM3's **local BA + loop-closure Sim3** — machinery the custom pipeline entirely
  lacks. Strongest on loop-rich sequences; the median-depth anchor bounds drift even on
  loop-free ones (seq01-style highway).

## Risks / open questions (address during implementation, not assume away)

1. **LocalMapping thread interaction** — creating KFs/MPs while the background BA thread
   runs. Follow `CreateNewKeyFrame()`'s `InsertKeyFrame` protocol exactly; may need to
   `RequestStop`/wait as mono-init does.
2. **Bridge-KF culling** — LocalMapping may cull these sparse bridge KFs immediately,
   undoing the bridge. May need a protected flag or a min-lifetime.
3. **Two-view degeneracy on turns** — near-pure-rotation is degenerate for the essential
   matrix (same failure the custom pipeline has). Guard with inlier + parallax checks;
   if degenerate, fall through to the existing reset (no regression vs today).
4. **Scale-anchor validity** — median depth assumes the last KF's scene depth is
   representative; robust but not perfect across a sharp depth change. Guard the scale
   factor to a plausible band.

## Staged plan

- **Stage 0 (prep):** local-build `orbslam3_sift_kitti_ate` (non-CUDA SIFT fallback path,
  no GPU). Record the CURRENT fail-fast baseline — coverage + ATE — on **seq00 + seq01**
  (the fast local iteration pair). This is the number to beat. NO code change yet.
- **Stage 1:** implement `TrackWithEpipolarBridge()` behind a flag (default OFF →
  byte-identical to today). Short-run sanity: confirm it creates valid KFs/MPs and keeps
  `mState=OK` through a dropout.
- **Stage 2:** wire into the `RECENTLY_LOST` path before reset. Re-measure the Stage-0
  subset.
- **Stage 3:** whole-KITTI (00–10) coverage + ATE; compare vs Stage-0 baseline. Report
  per-sequence, no aggregate-hiding.
- **Stage 4:** tune trigger conditions (min matches, parallax gate, scale band,
  culling protection).

## Evaluation criteria

- **Local iteration: seq00 + seq01 only** (fast). seq00 = loop-rich (tests drift
  correction), seq01 = highway/loop-free/new-territory (tests the bridge's core case).
- **Final estimate: on Kaggle** (user-run) — full whole-KITTI + the CudaSIFT GPU build.
  The local non-CUDA build is for iterating the logic; the final number is Kaggle's.

Per-sequence coverage (tracked/total) AND ATE. A result counts only if it holds across
BOTH local sequences — a seq00 win that regresses seq01 is NOT success (the standing
lesson from the custom-pipeline generalization work).

## Results table — SQPnP + init-fix + epipolar bridge (Kaggle, CudaSIFT GPU, nFeatures=5000)

Commit `ca00d94` bundles all three changes (SQPnP tracker + mono-init stale-reference
fix + epipolar bridge), so the "new" column below is the combined recipe. The
"baseline" column is the pre-commit fail-fast state (KLT + reset, no bridge) — take it
from the last Kaggle run at the same nFeatures, or re-run once with the pre-`ca00d94`
tree (`git stash`/checkout `ca00d94^`) for a matched comparison.

**Metrics to fill (extract from each run's stdout):**
- `Coverage` = summed aligned fragment pathLen / GT path length (`[fragment … pathLen=…]`
  lines summed, over `GT path length`). Report meters and %.
- `ATE` = per-fragment RMSE (`[fragment … ATE(rmse/…)]`); note worst fragment + whether
  any exceeds the 20 m budget.
- `Resets` = `grep -c 'A new map is started\|Reseting current map'`.
- `Bridge` = `grep -c '\[bridge\].*SUCCESS'` (fires that kept a map alive; 0 = bridge
  never helped, e.g. a sequence that never reaches RECENTLY_LOST).
- `Init%` = `mono-init SUCCESS` / (`SUCCESS`+`reconstruct FAILED`) — did the stale-ref fix
  raise it on seq01?

| Seq | Config | Coverage (m / %) | ATE (worst frag / budget) | Resets | Bridge SUCCESS | Init% | Notes |
|----|--------|------------------|---------------------------|--------|----------------|-------|-------|
| 00 | baseline (KLT, reset) |  |  |  | 0 |  | prior fail-fast |
| 00 | **new (SQPnP+bridge+initfix)** |  |  |  |  |  |  |
| 01 | baseline (KLT, reset) |  |  |  | 0 |  | highway, prior |
| 01 | **new (SQPnP+bridge+initfix)** |  |  |  |  |  | watch init% + degeneracy |
| 02 | baseline |  |  |  | 0 |  |  |
| 02 | **new** |  |  |  |  |  |  |
| 05 | baseline |  |  |  | 0 |  |  |
| 05 | **new** |  |  |  |  |  |  |
| 08 | baseline |  |  |  | 0 |  |  |  |
| 08 | **new** |  |  |  |  |  |  |  |

**Local (CPU cv::SIFT, nFeatures=5000) reference points already measured this session:**
- seq01 baseline (KLT): coverage ~255 m / 2451 m (10.4%), 39 resets, 3 fragments (ATE 0.5–1.9 m/frag).
- seq01 new: INCONCLUSIVE — stuck in mono-init (bridge never reached; the init-fix + SQPnP
  are what target this, re-measure with `ca00d94`).
- seq00 new (partial, killed at ~42%): bridge fired 316× @ 99% success; resets still
  accrued in the hard zone (~64 by frame 1917) — coverage impact needs a full run.

**Open question the table should answer:** does the bridge's map-keep-alive raise *coverage*
even where reset *count* stays near baseline (longer-lived maps between resets)? And does
SQPnP+init-fix rescue seq01's init failure? If resets stay high on hard zones, the next
lever is wiring the bridge into the `LOST` path too (for maps that die with <10 KFs and
currently bypass the RECENTLY_LOST/bridge path).
