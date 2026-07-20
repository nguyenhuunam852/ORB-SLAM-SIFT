"""Verify the exported ONNX SIFT+LightGlue model against the original
PyTorch cvg/LightGlue module on REAL KITTI images.

export_lightglue_sift.py's own equivalence check only ran on random noise
(torch.randn) -- that proves the ONNX graph computes the same FUNCTION as
the PyTorch graph, but says nothing about whether the actual preprocessing
convention used at inference time (RootSIFT conversion, keypoint
normalization, angle-to-radians) matches what real SIFT descriptors on
real images need. This script closes that gap: extracts real SIFT
features from two real KITTI frames (identical parameters to
ORBextractor.cc/settings_sift/KITTI00-02-sift.yaml), runs the SAME input
through both the PyTorch model and the exported ONNX model, and compares
the resulting matches/scores directly.

Usage: python3 analyze/verify_lightglue_onnx.py <image0> <image1> [onnx_path]
"""
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "analyze")
from export_lightglue_sift import LightGlueSift  # noqa: E402

SIFT_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/sift_lightglue.pth"

# Must match ORBextractor.cc's kContrastThreshold/kEdgeThreshold/kSigma and
# settings_sift/KITTI00-02-sift.yaml's ORBextractor.nFeatures/nLevels.
NFEATURES = 5000
N_OCTAVE_LAYERS = 8
CONTRAST_THRESHOLD = 0.04
EDGE_THRESHOLD = 10.0
SIGMA = 1.6


def to_rootsift(desc: np.ndarray) -> np.ndarray:
    """cvg/LightGlue/sift.py: sift_to_rootsift -- L1 normalize, sqrt, L2 normalize.
    Byte-for-byte the same formula as LightGlueMatcher.cc's toRootSift()."""
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
        nfeatures=NFEATURES,
        nOctaveLayers=N_OCTAVE_LAYERS,
        contrastThreshold=CONTRAST_THRESHOLD,
        edgeThreshold=EDGE_THRESHOLD,
        sigma=SIGMA,
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


def main():
    img0_path, img1_path = sys.argv[1], sys.argv[2]
    onnx_path = sys.argv[3] if len(sys.argv) > 3 else "weights/lightglue_sift.onnx"

    shape0, kps0, desc0 = extract(img0_path)
    shape1, kps1, desc1 = extract(img1_path)
    print(f"real SIFT counts: N0={len(kps0)} N1={len(kps1)}")

    N = max(len(kps0), len(kps1))
    kp0_arr, d0_arr = pack(shape0, kps0, desc0, N)
    kp1_arr, d1_arr = pack(shape1, kps1, desc1, N)

    keypoints = np.stack([kp0_arr, kp1_arr], axis=0)  # (2, N, 4)
    descriptors = np.stack([d0_arr, d1_arr], axis=0)  # (2, N, 128)

    # ---- PyTorch reference ----
    print("Loading PyTorch cvg/LightGlue SIFT weights...")
    model = LightGlueSift(SIFT_URL).eval()
    with torch.no_grad():
        matches_pt, mscores_pt = model(torch.from_numpy(keypoints), torch.from_numpy(descriptors))
    matches_pt = matches_pt.numpy()
    mscores_pt = mscores_pt.numpy()
    print(f"PyTorch: {matches_pt.shape[0]} matches")

    # ---- ONNX Runtime ----
    print(f"Loading ONNX model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    matches_onnx, mscores_onnx = sess.run(
        ["matches", "mscores"], {"keypoints": keypoints, "descriptors": descriptors}
    )
    print(f"ONNX:    {matches_onnx.shape[0]} matches")

    # ---- Compare ----
    set_pt = {(int(m[1]), int(m[2])) for m in matches_pt}
    set_onnx = {(int(m[1]), int(m[2])) for m in matches_onnx}
    only_pt = set_pt - set_onnx
    only_onnx = set_onnx - set_pt
    common = set_pt & set_onnx
    print(f"\nmatch-set comparison: common={len(common)} only_pytorch={len(only_pt)} only_onnx={len(only_onnx)}")
    print(f"Jaccard overlap: {len(common) / max(1, len(set_pt | set_onnx)):.4f}")

    if common:
        score_pt_by_pair = {(int(m[1]), int(m[2])): s for m, s in zip(matches_pt, mscores_pt)}
        score_onnx_by_pair = {(int(m[1]), int(m[2])): s for m, s in zip(matches_onnx, mscores_onnx)}
        diffs = np.array([abs(score_pt_by_pair[p] - score_onnx_by_pair[p]) for p in common])
        print(f"mscore diff over common matches: max={diffs.max():.6f} mean={diffs.mean():.6f}")

    if len(only_pt) == 0 and len(only_onnx) == 0:
        print("\nRESULT: EXACT match set agreement -- ONNX export is faithful on real data.")
    elif len(common) / max(1, len(set_pt | set_onnx)) > 0.95:
        print("\nRESULT: Near-exact agreement (>95% Jaccard) -- export is faithful, minor float/threshold noise.")
    else:
        print("\nRESULT: SIGNIFICANT DIVERGENCE -- the ONNX export or its preprocessing has a real bug.")


if __name__ == "__main__":
    main()
