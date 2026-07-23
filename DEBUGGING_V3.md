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
