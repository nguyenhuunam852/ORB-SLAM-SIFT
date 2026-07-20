cmake_minimum_required(VERSION 3.18)
project(orbslam3_sift_kaggle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU SIFT extraction via Celebrandil/CudaSift, replacing this session's
# earlier (now-reverted, see Tracking.cc/DEBUGGING.md) LightGlue matcher
# attempt -- that was a matcher-accuracy problem, not an extraction-speed
# one, so this targets the extractor instead. Off by default so the plain
# CPU cv::SIFT path (kaggle_build's original, still the default) is
# untouched; setup_and_run.sh passes -DUSE_CUDASIFT=ON explicitly.
option(USE_CUDASIFT "Build orbslam3_sift_ext with the GPU CudaSift extractor instead of CPU cv::SIFT" OFF)
if(USE_CUDASIFT)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Broad coverage for Kaggle's two GPU offerings (P100=sm_60, T4=sm_75)
        # plus a few newer archs in case the notebook lands on something else.
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
    endif()

    # CudaSift's own CMakeLists.txt hardcodes an outdated -arch=sm_20 (pre-
    # Kepler, unsupported by modern CUDA toolkits) and only builds a demo
    # executable from mainSift.cpp -- no library or install rules at all.
    # Rather than fight its build system, compile its .cu/.cpp sources
    # directly into orbslam3_sift_ext below (same "vendor source, build it
    # ourselves" pattern already used for DBoW2 in this same file), cloned
    # fresh into CUDASIFT_SRC_DIR by setup_and_run.sh.
    set(CUDASIFT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cudasift_src" CACHE PATH
        "Root of a fresh git clone of Celebrandil/CudaSift (cloned by setup_and_run.sh)")
    # Explicit allowlist, NOT a glob -- the repo's root also contains
    # cudaSiftD.cu (device kernels, #included by cudaSiftH.cu, not an
    # independent translation unit -- compiling it separately produced
    # duplicate-symbol nvlink errors for every kernel/global), the
    # cudaSiftD_old.cu/cudaSiftH_old.cu/cudaSiftH_old.h legacy duplicates
    # (same problem), cudaSift_gold.cpp (a CPU reference/test harness), and
    # mainSift.cpp (the repo's own demo main(), conflicting with this
    # project's own main() in orbslam3_kitti_ate.cpp -- confirmed live via
    # a duplicate-`main`-symbol link error). This exact file list matches
    # what CudaSift's own upstream CMakeLists.txt compiles into its
    # `cudasift` executable (verified by fetching it directly), minus
    # mainSift.cpp.
    set(CUDASIFT_SOURCES
        "${CUDASIFT_SRC_DIR}/cudaImage.cu"
        "${CUDASIFT_SRC_DIR}/cudaSiftH.cu"
        "${CUDASIFT_SRC_DIR}/matching.cu"
        "${CUDASIFT_SRC_DIR}/singular.cu"
        "${CUDASIFT_SRC_DIR}/geomFuncs.cpp"
    )
endif()

# Minimal, Kaggle-clean build of just the SIFT ORB-SLAM3 fork's headless
# KITTI benchmark (orbslam3_sift_kitti_ate) -- deliberately NOT the
# root CMakeLists.txt at the repo root, which unconditionally requires Qt6/
# Qt5 (find_package(... REQUIRED ...)) and Ceres for the GUI/custom-pipeline
# targets neither this binary nor a Kaggle container needs, and which
# hardcodes this project's own dev machine's absolute paths (a conda env,
# /home/nam/ORB_SLAM3, Ubuntu multiarch RPATH-poisoning workarounds for a
# conda-vs-system library conflict that doesn't exist in a fresh Kaggle
# container). See kaggle/setup_and_run.sh for the full build sequence this
# file expects to run inside.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio features2d calib3d)

# Built fresh by setup_and_run.sh from the official ORB-SLAM3 Thirdparty/g2o
# source (a small, ORB-SLAM3-specific g2o subset -- core/solvers/types/stuff
# only) into G2O_ROOT/{g2o,lib}. Not vendored in this repo (the dev machine's
# copy is an external symlink outside the project), so it's built fresh here
# instead of assumed present.
set(G2O_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/g2o_build" CACHE PATH
    "Root containing g2o/core, g2o/types, g2o/solvers headers and lib/libg2o.so (built by setup_and_run.sh)")
add_library(g2o_ext SHARED IMPORTED)
set_target_properties(g2o_ext PROPERTIES
    IMPORTED_LOCATION "${G2O_ROOT}/lib/libg2o.so"
    INTERFACE_INCLUDE_DIRECTORIES "${G2O_ROOT};/usr/include/eigen3"
)
# Matches the flag g2o's own CMakeLists.txt hardcodes for Release builds --
# built in the same container/session as its consumer here, so there's no
# cross-machine Eigen-alignment mismatch risk (see the root CMakeLists.txt's
# g2o_ext comment for the segfault this avoids when the two are built
# separately).
target_compile_options(g2o_ext INTERFACE -march=native)

# ONNX Runtime GPU release tarball, downloaded and extracted by
# setup_and_run.sh (official prebuilt include/+lib/ layout -- not apt-
# installable, and the pip onnxruntime-gpu package doesn't ship the C++
# headers this project's LightGlueMatcher.cc needs).
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime" CACHE PATH
    "Root containing include/onnxruntime_cxx_api.h and lib/libonnxruntime.so")

set(SIFT_FORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ORB_SLAM3_SIFT")
set(DBOW2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/DBoW2")

file(GLOB ORB_SLAM3_SIFT_SOURCES
    "${SIFT_FORK_DIR}/src/*.cc"
    "${SIFT_FORK_DIR}/src/CameraModels/*.cc"
)
add_library(orbslam3_sift_ext STATIC
    ${ORB_SLAM3_SIFT_SOURCES}
    ${DBOW2_DIR}/DBoW2/BowVector.cpp
    ${DBOW2_DIR}/DBoW2/FORB.cpp
    ${DBOW2_DIR}/DBoW2/FeatureVector.cpp
    ${DBOW2_DIR}/DBoW2/ScoringObject.cpp
    ${DBOW2_DIR}/DUtils/Random.cpp
    ${DBOW2_DIR}/DUtils/Timestamp.cpp
    ${CUDASIFT_SOURCES}
)
# Deliberately built at the project default C++17 (NOT the C++14 override
# the untouched original ORB-SLAM3 vendored copy needs) -- LoopClosing.cc's
# mnFullBAIdx++ on a bool member was changed to mnFullBAIdx = true (identical
# semantics under the old pre-C++17 bool-increment rule) specifically so this
# target can build under C++17, which LightGlueMatcher.cc's
# onnxruntime_cxx_api.h headers require (newer onnxruntime releases use
# structured bindings internally -- C++14 fails outright, not just a
# deprecation warning).
target_include_directories(orbslam3_sift_ext PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${SIFT_FORK_DIR}
    ${SIFT_FORK_DIR}/include
    ${SIFT_FORK_DIR}/include/CameraModels
    ${SIFT_FORK_DIR}/Thirdparty/Sophus
    ${ONNXRUNTIME_ROOT}/include
)
target_link_libraries(orbslam3_sift_ext PUBLIC
    ${OpenCV_LIBS}
    g2o_ext
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    ssl
    crypto
    curl
    xml2
    boost_serialization
)
if(USE_CUDASIFT)
    target_include_directories(orbslam3_sift_ext PUBLIC ${CUDASIFT_SRC_DIR})
    target_link_libraries(orbslam3_sift_ext PUBLIC CUDA::cudart CUDA::cufft)
    target_compile_definitions(orbslam3_sift_ext PUBLIC USE_CUDASIFT)
    set_target_properties(orbslam3_sift_ext PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(orbslam3_sift_kitti_ate ${CMAKE_CURRENT_SOURCE_DIR}/../analyze/orbslam3_kitti_ate.cpp)
target_include_directories(orbslam3_sift_kitti_ate PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_sift_kitti_ate PRIVATE orbslam3_sift_ext)

# So the built binary finds libonnxruntime.so/libg2o.so at runtime without
# needing LD_LIBRARY_PATH set manually in every notebook cell.
set_target_properties(orbslam3_sift_kitti_ate PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${G2O_ROOT}/lib"
)

# Stage-0 probe (see analyze/cudasift_probe.cpp) -- measures CudaSift's real
# scale/subsampling encoding against KITTI frames. Standalone (only needs
# CudaSift + OpenCV, no g2o/ORB_SLAM3 dependency); only buildable/meaningful
# with USE_CUDASIFT=ON. Run this FIRST on Kaggle and confirm the printed
# octave/scale ranges before trusting ORBextractor.cc's USE_CUDASIFT path
# (currently a placeholder pending this probe's real measurement).
if(USE_CUDASIFT)
    add_executable(cudasift_probe ${CMAKE_CURRENT_SOURCE_DIR}/../analyze/cudasift_probe.cpp ${CUDASIFT_SOURCES})
    target_include_directories(cudasift_probe PRIVATE ${OpenCV_INCLUDE_DIRS} ${CUDASIFT_SRC_DIR})
    target_link_libraries(cudasift_probe PRIVATE ${OpenCV_LIBS} CUDA::cudart CUDA::cufft)
    set_target_properties(cudasift_probe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
