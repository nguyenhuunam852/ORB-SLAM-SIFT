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

## NEXT (user's plan: finish FE, then Backend)
1. **FE**: pick best leash weights from the sweep above; if a looser weight beats
   41.8m without collapse, that's the new FE optimum.
2. **Backend #1 (cheap, same soft-prior principle the leash just validated)**:
   tune local-BA pose-prior weights `kLocalBaPosePriorRotWeight`(20)/`TransWeight`(3)
   — never tuned. Consider making them CLI-overridable like the leash weights.
3. **Backend #2 (deeper, the real lever toward <20m)**: landmark/map QUALITY —
   multi-view triangulation of NEW landmarks (reuse `triangulateMultiView()`, today
   only used in the closed Phase-B merge) and/or a triangulation quality gate.
   Item 40 established better map quality is what lets tight-fitting stop
   backfiring — this is the gate to the <20m goal.

## Open external item
- **Retrained SIFT-DBoW2 vocabulary** (K=10/L=5/stride=3, trained on Kaggle, file
  currently left at the office). When available: drop in as `sift_dbow_vocab.txt`,
  run with `siftdbow <path>` (argv36/37). Cleaner loop-closure candidate search →
  fewer map-compressing garbage loops → directly helps the map-quality lever.
  Must be RootSIFT-trained (notebook's FRootSift adapter does this).

## Standing reference points
- Real ORB-SLAM3 (vendored, `third_party/ORB_SLAM3`): 6.4–10.7m on seq00.
- User goal: <20m. Current best: 41.782m. Gap to goal ~2.1x.
