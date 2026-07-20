"""Ground-truth-verified precision analysis for SIFT+LightGlue matches on
real KITTI frame pairs.

TwoViewReconstruction's own [reconstruct-diag] logging showed nmatches in
the 700-1600 range collapsing to N=20-60 RANSAC-consistent inliers and then
maxGood=0 chirality-verified points -- i.e. LightGlue's own 0.1 confidence
filter lets through matches that are ~95-97% geometrically wrong on this
segment. But that number is confounded by RANSAC's own behavior (a bad
minimal-sample draw, iteration budget, etc.), not a direct measurement.

This script measures precision directly against KNOWN ground truth: KITTI's
poses/00.txt gives the exact metric relative pose between any two frames (no
scale ambiguity, no RANSAC estimation error) -- from that we can build the
TRUE fundamental matrix and classify every LightGlue match as a genuine
inlier/outlier by symmetric epipolar distance, independent of RANSAC. This
also lets us sweep the match-score threshold (LightGlue's own
filter_threshold, hardcoded to 0.1 in the exported ONNX graph) to see
whether a stricter threshold recovers acceptable precision, and at what
recall cost -- using the PyTorch model directly (not the ONNX export) so we
can inspect scores for ALL candidate pairs before the threshold filter
whittles them down to just one hardcoded cutoff.

CAVEAT (found via the ratio-test control experiment below, not yet fixed):
on KITTI seq00 frames 0/8, the traditional ratio-test matcher -- known from
this project's own extensive prior measurement (DEBUGGING.md part 56) to be
~82-94% precise -- ALSO scores only ~9% "precision" under this script's
fixed-2px symmetric-epipolar-distance check, essentially identical to
LightGlue's ~7-11%. That means THIS SCRIPT'S absolute precision numbers are
not trustworthy as an apples-to-apples LightGlue-vs-ratio-test comparison --
likely the classic forward-motion near-degenerate epipolar geometry (KITTI's
translation is close to parallel with the optical axis, which makes a fixed-
pixel-threshold symmetric-epipolar-distance check unreliable) rather than a
real difference in matcher quality. Left in the tree as-is (not fixed) --
the threshold-sweep SHAPE (precision ~flat across threshold 0->0.9) is still
informative on its own (rules out "just raise filter_threshold" as a fix
regardless of the absolute-number caveat), but do not quote the absolute
precision percentages from this script without first fixing the epipolar
check (e.g. switch to reprojection error via triangulation, or a much
looser threshold) and re-validating the ratio-test control lands in the
expected ~82-94% range. The genuinely trustworthy finding remains
[reconstruct-diag]'s live RANSAC measurement (TwoViewReconstruction.cc,
production code): nmatches 700-1600 collapsing to N=20-60 RANSAC-consistent
inliers, then maxGood=0 chirality-verified points, on real mono-init
attempts -- that number does not depend on this script's epipolar check.

Usage: python3 analyze/lightglue_precision_gt.py <frame0_idx> <frame1_idx>
    [--seq-dir DIR] [--poses PATH]
"""
import argparse
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "analyze")
from export_lightglue_sift import LightGlueSift, filter_matches  # noqa: E402

SIFT_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/sift_lightglue.pth"

NFEATURES = 5000
N_OCTAVE_LAYERS = 8
CONTRAST_THRESHOLD = 0.04
EDGE_THRESHOLD = 10.0
SIGMA = 1.6

# settings_sift/KITTI00-02-sift.yaml
FX, FY, CX, CY = 718.856, 718.856, 607.193, 185.216
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

EPIPOLAR_INLIER_PX = 2.0  # symmetric epipolar distance threshold, pixels


def to_rootsift(desc: np.ndarray) -> np.ndarray:
    eps = 1e-6
    l1 = np.sum(np.abs(desc), axis=1, keepdims=True) + eps
    desc = desc / l1
    desc = np.sqrt(np.maximum(desc, eps))
    l2 = np.sqrt(np.sum(desc * desc, axis=1, keepdims=True)) + eps
    return (desc / l2).astype(np.float32)


def extract(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    sift = cv2.SIFT_create(
        nfeatures=NFEATURES, nOctaveLayers=N_OCTAVE_LAYERS,
        contrastThreshold=CONTRAST_THRESHOLD, edgeThreshold=EDGE_THRESHOLD, sigma=SIGMA,
    )
    kps, desc = sift.detectAndCompute(img, None)
    desc = to_rootsift(desc.astype(np.float32))
    return img.shape, kps, desc


def pack(img_shape, kps, desc, N):
    h, w = img_shape
    shift_x, shift_y = w / 2.0, h / 2.0
    scale = max(w, h) / 2.0
    kp_arr = np.zeros((N, 4), dtype=np.float32)
    d_arr = np.zeros((N, 128), dtype=np.float32)
    for i, kp in enumerate(kps):
        kp_arr[i, 0] = (kp.pt[0] - shift_x) / scale
        kp_arr[i, 1] = (kp.pt[1] - shift_y) / scale
        kp_arr[i, 2] = kp.size
        kp_arr[i, 3] = kp.angle * np.pi / 180.0
    d_arr[: desc.shape[0]] = desc
    return kp_arr, d_arr


def load_pose(poses_path: str, idx: int) -> np.ndarray:
    """Returns 4x4 T_world_cam (world = frame 0's coordinate system)."""
    with open(poses_path) as f:
        lines = f.readlines()
    vals = [float(x) for x in lines[idx].split()]
    T = np.eye(4)
    T[:3, :4] = np.array(vals).reshape(3, 4)
    return T


def relative_pose(T0: np.ndarray, T1: np.ndarray):
    """R,t such that X_cam1 = R @ X_cam0 + t."""
    R0, t0 = T0[:3, :3], T0[:3, 3]
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R = R1.T @ R0
    t = R1.T @ (t0 - t1)
    return R, t


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def symmetric_epipolar_dist(F: np.ndarray, pt0: np.ndarray, pt1: np.ndarray) -> float:
    p0 = np.array([pt0[0], pt0[1], 1.0])
    p1 = np.array([pt1[0], pt1[1], 1.0])
    l1 = F @ p0  # epipolar line in image 1
    l0 = F.T @ p1  # epipolar line in image 0
    num = (p1 @ l1) ** 2
    d1 = num / (l1[0] ** 2 + l1[1] ** 2 + 1e-12)
    d0 = num / (l0[0] ** 2 + l0[1] ** 2 + 1e-12)
    return float(np.sqrt(d1) + np.sqrt(d0)) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame0", type=int)
    ap.add_argument("frame1", type=int)
    ap.add_argument("--seq-dir", default="/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/dataset/sequences/00/image_0")
    ap.add_argument("--poses", default="/run/media/nam/D1/Master/Computer Vision/Dataset/KITTI/dataset/poses/00.txt")
    args = ap.parse_args()

    img0_path = f"{args.seq_dir}/{args.frame0:06d}.png"
    img1_path = f"{args.seq_dir}/{args.frame1:06d}.png"

    T0 = load_pose(args.poses, args.frame0)
    T1 = load_pose(args.poses, args.frame1)
    R, t = relative_pose(T0, T1)
    baseline_m = np.linalg.norm(t)
    print(f"GT relative pose {args.frame0}->{args.frame1}: baseline={baseline_m:.3f}m")

    E = skew(t) @ R
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ E @ Kinv

    shape0, kps0, desc0 = extract(img0_path)
    shape1, kps1, desc1 = extract(img1_path)
    print(f"real SIFT counts: N0={len(kps0)} N1={len(kps1)}")

    # Control experiment: traditional brute-force ratio-test matching on the
    # SAME frame pair, scored by the SAME ground-truth epipolar check. If
    # this methodology is correct, ratio-test matching (known-reliable, per
    # this project's own Part-56-era measurements: ~82-94% inlier rate)
    # should score high precision here -- validating the epipolar-distance
    # code before trusting LightGlue's low numbers above.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc0, desc1, k=2)
    ratio_matches = [m for m, n in knn if m.distance < 0.8 * n.distance]
    n_inlier_rt = 0
    for m in ratio_matches:
        d = symmetric_epipolar_dist(F, kps0[m.queryIdx].pt, kps1[m.trainIdx].pt)
        if d < EPIPOLAR_INLIER_PX:
            n_inlier_rt += 1
    rt_precision = n_inlier_rt / max(1, len(ratio_matches))
    print(f"\n[control] ratio-test (0.8) on same pair: nmatches={len(ratio_matches)} "
          f"gt_inliers={n_inlier_rt} precision={rt_precision:.3f}")

    N = max(len(kps0), len(kps1))
    kp0_arr, d0_arr = pack(shape0, kps0, desc0, N)
    kp1_arr, d1_arr = pack(shape1, kps1, desc1, N)
    keypoints = np.stack([kp0_arr, kp1_arr], axis=0)
    descriptors = np.stack([d0_arr, d1_arr], axis=0)

    print("Loading PyTorch cvg/LightGlue SIFT weights and running forward pass to raw scores...")
    model = LightGlueSift(SIFT_URL).eval()
    with torch.no_grad():
        kp_t = torch.from_numpy(keypoints)
        d_t = torch.from_numpy(descriptors)
        d_proj = model.input_proj(d_t)
        enc = model.posenc(kp_t)
        for i in range(model.n_layers):
            d_proj = model.transformers[i](d_proj, enc)
        scores = model.log_assignment[model.n_layers - 1](d_proj)

    thresholds = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    print(f"\n{'threshold':>10} {'nmatches':>9} {'gt_inliers':>10} {'precision':>10}")
    for thr in thresholds:
        with torch.no_grad():
            matches, mscores = filter_matches(scores, thr)
        matches = matches.numpy()
        n = matches.shape[0]
        if n == 0:
            print(f"{thr:>10.2f} {n:>9} {0:>10} {'n/a':>10}")
            continue
        n_inlier = 0
        for m in matches:
            idx0, idx1 = int(m[1]), int(m[2])
            if idx0 >= len(kps0) or idx1 >= len(kps1):
                continue
            d = symmetric_epipolar_dist(F, kps0[idx0].pt, kps1[idx1].pt)
            if d < EPIPOLAR_INLIER_PX:
                n_inlier += 1
        precision = n_inlier / n
        print(f"{thr:>10.2f} {n:>9} {n_inlier:>10} {precision:>10.3f}")


if __name__ == "__main__":
    main()
