# DEBUGGING v2 — SlamWorker/kitti_ate (KITTI seq00) — current state & forward plan

Clean checkpoint. Full historical detail (items 1–41) stays in `DEBUGGING.md`;
this file is the actionable current state + next steps. Dataset:
`.../Dataset/KITTI/dataset/sequences/00/image_0/%06d.png` + `poses/00.txt`.
Pipeline runs on **RootSIFT** everywhere (toRootSift right after detection,
`SlamWorker.cpp`), VLAD codebook `vocabulary_sift/vlad_codebook_all_rootsift.yml`.

## NEW BEST: 41.782m ATE (2026-07-23) — was 51.273m

**Config: baseline + pose-only BA leashed to SQPnP** (`poseonlyba` + `poseonlyleash`).

```
kitti_ate <img> <poses> 1200 sqpnp <out> fivepoint - - - ba \
  - - - - - - - - - - - - - - localba - - guided - vlad <vlad-codebook> \
  - - - - - - - fuse - - - - - - - - - - - poseonlyba - - - - - - poseonlyleash
```
(argv50=poseonlyba, argv57=poseonlyleash, on top of the prior 51.273m recipe.)

Verified genuine: median 35.3m, **max 90.8m** (< baseline 120.4), coverage
4518/4541 (≥ baseline 4515), scale 0.459 (healthier than 0.232), no per-region
scale collapse.

### What the leash is (the winning mechanism)
"Pose-only BA" = optimize only the 6-DOF camera pose, **map held fixed** (ORB-SLAM3
`Optimizer::PoseOptimization`); it runs AFTER a PnP estimate, refining it. Plain
pose-only BA (`poseonlyba`, 97.2m) is BETTER than baseline in 3/4 seq00 regions but
COLLAPSES scale in frames ~2500-3300: it holds the map fixed and rigidly follows a
loop-closure-compressed map into a local scale collapse (step ratio → 0.07).

**Leash (`poseonlyleash`)** fixes this: switch to **SQPnP-PRIMARY** tracking, then
refine with pose-only BA anchored to the SQPnP pose via a soft `PosePriorCost`
(weights `m_poseOnlyLeashRotWeight`/`TransWeight`, default 30/5, CLI-overridable via
argv58/59). Pose-only BA sharpens the pose but can't drift/collapse away from the
robust, loosely-map-coupled SQPnP estimate. This is the per-frame analogue of
soft-prior local BA's live-pose leash, and resolves the deep item-40 tension:
"any tighter map-fit collapses scale UNLESS anchored to a robust estimate."

## Negative / closed this thread (don't repeat)
- `poseonlystepgate` (per-frame step gate, argv56): **159.4m**, negative.
- leash + `octaveweight` (argv52): **79.8m** — octave down-weighting removes
  constraints on sparse SIFT, hurts. octaveweight closed.
- leash + `reprojErr=6.5`: **132.3m** — reprojErr6.5 is a fragile knife-edge.
- `pnpfullrefine` (argv54) alone: 175.7m; `reprojErr=4`: 140.8m. Tighter map-fit
  collapses scale. reprojErr+fullrefine sweep chaotic (r6.5=46.9 is a lucky spike,
  neighbors r6.25=128/r6.75=71 catastrophic — NOT usable).
- `loopqualitygate` (argv55, reject low-inlier extreme-scale loops): **113.7m** —
  rejecting loop corrections triggers the documented drift vicious cycle.

## IN PROGRESS (this checkpoint)
- **FE leash-weight sweep** (argv58/59): {60/10, 20/3.5, 15/2.5, 8/1.5} vs default
  30/5=41.8m. Looser = more pose-only refinement (accuracy) at higher collapse risk.
  RESULTS: _pending — fill in when the runs finish._
- **Backend #1 + #2 implemented (2026-07-23), runs pending.** Code landed; all
  three levers are default-OFF CLI flags so the validated 41.8m recipe is byte-for-
  byte unchanged unless opted in. Sweep/verify on the Linux box, fill in below.
- **Retrained SIFT-DBoW2 vocabulary is now on disk** (`sift_dbow_vocab (2).txt`,
  10 MB, header `10 5` = K=10/L=5 as trained). No longer an external blocker — see
  the combined recipe.

## Combined recipe (41.8m + new SIFT-DBoW2 vocab + backend knobs)
Start from the exact 41.8m recipe and set these positional slots (all others as in
the 41.8m recipe above; `-` = leave default):

| argv | word / value              | what it does                                        |
|------|---------------------------|-----------------------------------------------------|
| 36/37| `siftdbow <vocab-path>`   | new retrained vocab for loop-candidate search. Takes precedence over `vlad` (argv30/31) when both are passed — so you can drop it onto the 41.8m recipe as-is. RootSIFT-trained, matches this build. |
| 60/61| `<rot> <trans>`           | **Backend #1**: local-BA pose-prior weights (default 20/3). `-`/0 = keep. Lower = looser = local BA moves poses further from PnP (more map-fit accuracy, more collapse risk). |
| 62   | `retriangulate`           | **Backend #2a**: re-triangulate a landmark from ALL its views once it has ≥3, accept only a strict reprojection improvement (BA-safe). Reported at shutdown: `[retri] landmarks re-triangulated this run`. |
| 63   | `parallaxgate`            | **Backend #2b**: reject new landmarks whose two rays are <~1° apart (noise-dominated depth). |

The vocab path has a space and `(2)` — quote it: `"sift_dbow_vocab (2).txt"` (or
rename to `sift_dbow_vocab.txt`). Sweep 60/61 like the FE leash; try
`retriangulate`/`parallaxgate` independently first, then together, so each lever's
effect on ATE + scale + coverage is attributable.

## NEXT (user's plan: finish FE, then Backend)
1. **FE**: pick best leash weights from the sweep above; if a looser weight beats
   41.8m without collapse, that's the new FE optimum.
2. **Backend #1** (DONE, code): local-BA pose-prior weights are now CLI-overridable
   via argv60/61 (`setLocalBaPosePriorWeights()`), mirroring the leash weights.
   Same soft-prior principle the leash validated, applied map-side. Sweep pending.
3. **Backend #2** (DONE, code): map QUALITY — `retriangulate` (argv62) re-triangulates
   NEW landmarks multi-view via `triangulateMultiView()` (was only used in the closed
   Phase-B merge), `parallaxgate` (argv63) culls low-parallax garbage at creation.
   Item 40's <20m gate: better map quality lets tight-fitting stop backfiring. The
   new vocab feeds the same lever (cleaner loops → fewer map-compressing corrections).
   Runs pending.

## Standing reference points
- Real ORB-SLAM3 (vendored, `third_party/ORB_SLAM3`): 6.4–10.7m on seq00.
- User goal: <20m. Current best: 41.782m. Gap to goal ~2.1x.
