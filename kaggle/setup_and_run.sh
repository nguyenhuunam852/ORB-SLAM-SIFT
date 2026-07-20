#!/usr/bin/env bash
# Kaggle setup + run for the CudaSIFT+RootSIFT ORB-SLAM3 fork's headless
# KITTI benchmark (orbslam3_sift_kitti_ate). See kaggle/CMakeLists.txt for
# why this is a separate, minimal build (no Qt/Ceres, no dev-machine-
# specific paths). This fork's earlier LightGlue matcher experiment was
# reverted (negative result -- see DEBUGGING.md/git log) in favor of GPU-
# accelerating the SIFT EXTRACTOR itself instead.
#
# Usage (from a Kaggle notebook cell, GPU accelerator + internet ON,
# repo already cloned/attached so this script's own directory is
# <repo>/kaggle):
#   !bash kaggle/setup_and_run.sh
#
# Env vars (all optional, override any default):
#   USE_CUDASIFT    1 (default) to build the GPU CudaSift extractor, 0 to
#                    build the plain CPU cv::SIFT path instead
#   KITTI_SEQ_DIR   Directory containing image_0/*.png for the sequence
#                    (default: auto-search common /kaggle/input layouts)
#   KITTI_POSES     Path to poses/00.txt-style ground truth
#                    (default: auto-search alongside KITTI_SEQ_DIR)
#   START_FRAME     First frame index (default: 0)
#   MAX_FRAMES      Frame count to run (default: 1000)
#   OUT_PREFIX      Output file prefix (default: /kaggle/working/lightglue_run)
#   SKIP_BUILD      If set to 1, skip the whole build step and just run
#                    (reuse a previous session's build/ dir)
#   BUILD_JOBS      Parallel compile jobs for `make`/`cmake --build` (default:
#                    4, not $(nproc)) -- this codebase's Eigen/Sophus/g2o-heavy
#                    translation units (e.g. LoopClosing.cc, Optimizer.cc) can
#                    each use 1GB+ RAM per job; on a memory-constrained host
#                    (this project's own dev machine hit an OOM-killed VSCode
#                    process building locally at -j8) a lower default is
#                    safer. Kaggle's shared CPU/RAM makes this worth capping
#                    there too. Raise it explicitly if you have RAM to spare.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${SCRIPT_DIR}"

# USE_CUDASIFT=1 (default): build the GPU CudaSift extractor (see
# ORBextractor.cc's USE_CUDASIFT-gated path) instead of CPU cv::SIFT.
# Set to 0 to build the plain CPU path instead (e.g. for a quick sanity
# comparison run without touching CUDA at all).
USE_CUDASIFT="${USE_CUDASIFT:-1}"
BUILD_JOBS="${BUILD_JOBS:-4}"

echo "=== [1/7] GPU + CUDA toolkit check ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found -- make sure the Kaggle notebook's" \
         "Accelerator is set to a GPU." >&2
fi
if [ "${USE_CUDASIFT}" = "1" ]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "ERROR: USE_CUDASIFT=1 but nvcc (CUDA toolkit compiler) was not found." >&2
        echo "Kaggle's GPU images normally ship it; if missing, set USE_CUDASIFT=0" >&2
        echo "to build the CPU cv::SIFT path instead, or install the toolkit manually." >&2
        exit 1
    fi
    nvcc --version | tail -1
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then

echo "=== [2/7] apt dependencies ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    build-essential cmake git wget \
    libopencv-dev \
    libeigen3-dev \
    libboost-dev libboost-serialization-dev \
    libssl-dev libcurl4-openssl-dev libxml2-dev \
    > /tmp/apt-install.log 2>&1 || { echo "apt-get install failed, see /tmp/apt-install.log"; tail -50 /tmp/apt-install.log; exit 1; }
# cv::SIFT needs OpenCV >=4.4 (moved out of the patented xfeatures2d module).
OPENCV_VER=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || pkg-config --modversion opencv4 2>/dev/null || echo "unknown")
echo "OpenCV version seen by pkg-config/python: ${OPENCV_VER}"

echo "=== [3/7] Build g2o (ORB-SLAM3's own Thirdparty/g2o fork) ==="
G2O_BUILD_ROOT="${WORK_DIR}/g2o_build"
if [ ! -f "${G2O_BUILD_ROOT}/lib/libg2o.so" ]; then
    rm -rf /tmp/orbslam3_upstream
    git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git /tmp/orbslam3_upstream
    rm -rf "${G2O_BUILD_ROOT}"
    cp -r /tmp/orbslam3_upstream/Thirdparty/g2o "${G2O_BUILD_ROOT}"
    mkdir -p "${G2O_BUILD_ROOT}/build"
    ( cd "${G2O_BUILD_ROOT}/build" && cmake -DCMAKE_BUILD_TYPE=Release .. > /tmp/g2o-cmake.log 2>&1 \
        && make -j"${BUILD_JOBS}" > /tmp/g2o-make.log 2>&1 ) \
        || { echo "g2o build failed, see /tmp/g2o-cmake.log / /tmp/g2o-make.log"; exit 1; }
else
    echo "g2o already built at ${G2O_BUILD_ROOT}, skipping (set SKIP_BUILD=0 and rm -rf it to force)"
fi
# third_party/ORB_SLAM3_SIFT/Thirdparty/g2o is committed as a symlink to an
# absolute path on the original dev machine (/home/nam/ORB_SLAM3/Thirdparty/g2o)
# -- it only resolved there because that machine actually has a real g2o
# checkout at that path; on a fresh clone (Kaggle included) it's a dangling
# symlink. The vendored source does literal `#include
# "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"`-style includes (not
# `<g2o/...>`), so no amount of extra `-I` flags fixes it -- a real
# `Thirdparty/g2o` directory has to exist under the SIFT fork. Point it at
# the g2o we just built instead.
ln -sfn "${G2O_BUILD_ROOT}" "${REPO_ROOT}/third_party/ORB_SLAM3_SIFT/Thirdparty/g2o"

echo "=== [4/7] Clone CudaSift source ==="
CUDASIFT_SRC_DIR="${WORK_DIR}/cudasift_src"
if [ "${USE_CUDASIFT}" = "1" ]; then
    if [ ! -f "${CUDASIFT_SRC_DIR}/cudaSift.h" ]; then
        rm -rf "${CUDASIFT_SRC_DIR}"
        git clone --depth 1 https://github.com/Celebrandil/CudaSift.git "${CUDASIFT_SRC_DIR}"
        echo "CudaSift source cloned to ${CUDASIFT_SRC_DIR}"
    else
        echo "CudaSift source already present at ${CUDASIFT_SRC_DIR}, skipping"
    fi
    # NOTE: CudaSift's own CMakeLists.txt/build system is deliberately NOT
    # used (hardcodes an outdated -arch=sm_20 and only builds a demo
    # executable, no library/install rules) -- kaggle/CMakeLists.txt globs
    # its .cu/.cpp sources directly into orbslam3_sift_ext instead, same
    # "vendor source, build it ourselves" pattern as DBoW2/g2o above.
else
    echo "USE_CUDASIFT=0, skipping CudaSift clone"
fi

echo "=== [5/7] Download ONNX Runtime GPU ==="
ORT_ROOT="${WORK_DIR}/onnxruntime"
if [ ! -f "${ORT_ROOT}/lib/libonnxruntime.so" ]; then
    echo "Querying GitHub API for the latest onnxruntime release..."
    ORT_API_RESPONSE=$(curl -sS -w '\nHTTP_STATUS:%{http_code}' https://api.github.com/repos/microsoft/onnxruntime/releases/latest)
    ORT_HTTP_STATUS=$(echo "${ORT_API_RESPONSE}" | grep -o 'HTTP_STATUS:[0-9]*' | cut -d: -f2)
    if [ "${ORT_HTTP_STATUS}" != "200" ]; then
        echo "GitHub API request failed (HTTP ${ORT_HTTP_STATUS}), likely rate-limited. Response:" >&2
        echo "${ORT_API_RESPONSE}" | head -20 >&2
        exit 1
    fi
    # Asset naming has changed across onnxruntime releases -- older ones
    # ship a single CUDA-agnostic "onnxruntime-linux-x64-gpu-<ver>.tgz";
    # newer ones (>=1.20ish) split by CUDA major version instead, e.g.
    # "onnxruntime-linux-x64-gpu_cuda12-<ver>.tgz" /
    # "..._cuda13-<ver>.tgz" (underscore before "cuda", not the old
    # hyphen-then-digit pattern). Try cuda12 first (broadest driver
    # compatibility -- CUDA 12 runs fine under the newer 580.x drivers
    # Kaggle's T4/P100 images currently ship), then cuda13, then the old
    # plain naming as a last resort for pinned older releases.
    ORT_ASSET_URL=""
    for pattern in \
        'onnxruntime-linux-x64-gpu_cuda12-[0-9][^"]*\.tgz' \
        'onnxruntime-linux-x64-gpu_cuda13-[0-9][^"]*\.tgz' \
        'onnxruntime-linux-x64-gpu-[0-9][^"]*\.tgz'
    do
        ORT_ASSET_URL=$(echo "${ORT_API_RESPONSE}" \
            | grep -o "\"browser_download_url\": *\"[^\"]*${pattern}\"" \
            | head -1 | sed -E 's/.*"(https[^"]+)"/\1/')
        [ -n "${ORT_ASSET_URL}" ] && break
    done
    if [ -z "${ORT_ASSET_URL}" ]; then
        echo "GitHub API responded but no recognized onnxruntime-linux-x64-gpu*.tgz asset was found." >&2
        echo "Actual linux-x64 asset names in this release:" >&2
        echo "${ORT_API_RESPONSE}" | grep -o '"name": *"[^"]*linux-x64[^"]*"' >&2
        echo "Check https://github.com/microsoft/onnxruntime/releases and set ORT_ASSET_URL logic manually." >&2
        exit 1
    fi
    echo "Resolved URL: ${ORT_ASSET_URL}"
    echo "Downloading (typically 150-300MB, progress shown below)..."
    wget --progress=dot:giga "${ORT_ASSET_URL}" -O /tmp/onnxruntime-gpu.tgz
    echo "Download complete, extracting..."
    rm -rf "${ORT_ROOT}" /tmp/onnxruntime-extracted
    mkdir -p /tmp/onnxruntime-extracted
    tar -xzf /tmp/onnxruntime-gpu.tgz -C /tmp/onnxruntime-extracted
    mv /tmp/onnxruntime-extracted/onnxruntime-linux-x64-gpu* "${ORT_ROOT}"
    echo "ONNX Runtime ready at ${ORT_ROOT}"
else
    echo "ONNX Runtime already present at ${ORT_ROOT}, skipping"
fi

echo "=== [6/7] Configure + build orbslam3_sift_kitti_ate ==="
BUILD_DIR="${WORK_DIR}/build"
CMAKE_USE_CUDASIFT="OFF"
[ "${USE_CUDASIFT}" = "1" ] && CMAKE_USE_CUDASIFT="ON"
cmake -S "${WORK_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DG2O_ROOT="${G2O_BUILD_ROOT}" \
    -DONNXRUNTIME_ROOT="${ORT_ROOT}" \
    -DUSE_CUDASIFT="${CMAKE_USE_CUDASIFT}" \
    -DCUDASIFT_SRC_DIR="${CUDASIFT_SRC_DIR}"
cmake --build "${BUILD_DIR}" --target orbslam3_sift_kitti_ate -j"${BUILD_JOBS}"
if [ "${USE_CUDASIFT}" = "1" ]; then
    cmake --build "${BUILD_DIR}" --target cudasift_probe -j"${BUILD_JOBS}"
    echo "Built cudasift_probe -- STRONGLY recommended to run it first:"
    echo "  \"${BUILD_DIR}/cudasift_probe\" \"\${KITTI_SEQ_DIR}\" 200"
    echo "and confirm the printed octave/scale ranges before trusting a full benchmark run" \
         "(ORBextractor.cc's USE_CUDASIFT path is a placeholder pending this probe's measurement)."
fi

else
    echo "=== [2-6/7] SKIP_BUILD=1, reusing existing build ==="
    BUILD_DIR="${WORK_DIR}/build"
fi

echo "=== [7/7] Locate KITTI dataset and run ==="
if [ -z "${KITTI_SEQ_DIR:-}" ]; then
    # Common layouts for a Kaggle "KITTI odometry" dataset attachment.
    CANDIDATE=$(find /kaggle/input -maxdepth 6 -type d -iname "image_0" 2>/dev/null | head -1)
    if [ -n "${CANDIDATE}" ]; then
        KITTI_SEQ_DIR="$(dirname "${CANDIDATE}")"
    fi
fi
if [ -z "${KITTI_SEQ_DIR:-}" ] || [ ! -d "${KITTI_SEQ_DIR}/image_0" ]; then
    echo "Could not find a KITTI sequence directory (expected .../image_0/*.png)." >&2
    echo "Attach a KITTI odometry (sequence 00) Kaggle Dataset, or set KITTI_SEQ_DIR explicitly." >&2
    exit 1
fi
if [ -z "${KITTI_POSES:-}" ]; then
    CANDIDATE=$(find /kaggle/input -maxdepth 6 -type f -iname "00.txt" -path "*poses*" 2>/dev/null | head -1)
    KITTI_POSES="${CANDIDATE:-}"
fi
if [ -z "${KITTI_POSES:-}" ] || [ ! -f "${KITTI_POSES}" ]; then
    echo "Could not find ground-truth poses/00.txt." >&2
    echo "Set KITTI_POSES explicitly (needed for the final ATE report; tracking itself still runs without it)." >&2
    exit 1
fi
echo "KITTI_SEQ_DIR=${KITTI_SEQ_DIR}"
echo "KITTI_POSES=${KITTI_POSES}"

START_FRAME="${START_FRAME:-0}"
MAX_FRAMES="${MAX_FRAMES:-1000}"
OUT_PREFIX="${OUT_PREFIX:-/kaggle/working/lightglue_run}"

# ORBextractor.cc's USE_CUDASIFT path applies RootSIFT permanently to every
# stored descriptor (see its own doc comment) -- vocabulary_sift/
# vlad_codebook_all.yml was trained on raw-SIFT descriptors, so it's stale
# for a CudaSift build. Retrain via:
#   cmake --build "${WORK_DIR}/build" --target orbslam3_vlad_train
#   "${WORK_DIR}/build/orbslam3_vlad_train" <settings.yaml> <one-or-more-sequence-dirs...> \
#       vocabulary_sift/vlad_codebook_all_rootsift.yml
# (needs KITTI sequences attached, matching Session 14's original 22-
# sequence training methodology -- see DEBUGGING.md). Not run automatically
# here since it needs a much larger dataset attachment than a single
# sequence and takes real wall-clock time; falls back to the raw-SIFT
# codebook with a loud warning if the RootSIFT one doesn't exist yet --
# tracking/ATE are unaffected (VLAD only feeds loop-closure/relocalization
# candidate search), just with unreliable loop-closure scores until retrained.
if [ "${USE_CUDASIFT}" = "1" ]; then
    VLAD_CODEBOOK="${VLAD_CODEBOOK:-vocabulary_sift/vlad_codebook_all_rootsift.yml}"
    if [ ! -f "${REPO_ROOT}/${VLAD_CODEBOOK}" ]; then
        echo "WARNING: ${VLAD_CODEBOOK} does not exist (not yet retrained for RootSIFT)." >&2
        echo "Falling back to vocabulary_sift/vlad_codebook_all.yml (raw-SIFT-trained --" >&2
        echo "loop-closure/relocalization scores will be unreliable until retrained; see" >&2
        echo "the retraining command in this script's comments just above)." >&2
        VLAD_CODEBOOK="vocabulary_sift/vlad_codebook_all.yml"
    fi
else
    VLAD_CODEBOOK="vocabulary_sift/vlad_codebook_all.yml"
fi
echo "VLAD_CODEBOOK=${VLAD_CODEBOOK}"

cd "${REPO_ROOT}"
LD_LIBRARY_PATH="${WORK_DIR}/onnxruntime/lib:${WORK_DIR}/g2o_build/lib:${LD_LIBRARY_PATH:-}" \
"${WORK_DIR}/build/orbslam3_sift_kitti_ate" \
    "${VLAD_CODEBOOK}" \
    settings_sift/KITTI00-02-sift.yaml \
    "${KITTI_SEQ_DIR}" \
    "${KITTI_POSES}" \
    "${OUT_PREFIX}" \
    "${START_FRAME}" \
    0 \
    "${MAX_FRAMES}"
