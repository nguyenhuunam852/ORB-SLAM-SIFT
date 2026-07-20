cmake_minimum_required(VERSION 3.16)
project(sift_vslam_gui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Any target linking libg2o.so (IMPORTED_LOCATION under
# /home/nam/ORB_SLAM3/Thirdparty/g2o) or libboost_serialization.so (under the
# conda env, see CONDA_ENV_ROOT below) gets those directories auto-added to
# its RPATH/RUNPATH by CMake, since neither lives in a default system
# library path. That RUNPATH is then searched at *runtime* for every NEEDED
# library the executable directly links -- including ones the conda
# directory ALSO happens to contain a (different, ABI-incompatible) build
# of: libQt6Core.so.6 (conda 6.9.3 vs. this system's 6.10.2), libcurl.so.4,
# libxml2.so.16, libssl/libcrypto.so.3. Confirmed at runtime as
# "version `Qt_6.10' not found" when the conda copy got picked over the
# system one. Explicitly listing the system lib directory FIRST here (before
# CMake's own auto-computed entries, which conftest confirmed get appended
# after manually-specified CMAKE_BUILD_RPATH entries) makes the dynamic
# linker try it first for every target -- resolving anything that exists in
# both places (Qt, curl, xml2, ssl, crypto) to the correct system copy, while
# still falling through to the g2o/conda directories for the two libraries
# that only exist there.
set(CMAKE_BUILD_RPATH "/usr/lib/x86_64-linux-gnu")

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets Concurrent)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets Concurrent)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio features2d calib3d)
find_package(Ceres REQUIRED)

# DBoW2 (bag-of-words place recognition, ORB-SLAM2's own loop-closure
# candidate search) -- vendored from third_party/DBoW2 (the original
# https://github.com/dorian3d/DBoW2, via ORB-SLAM3's lightly-modified copy
# which adds loadFromTextFile()/saveToTextFile() for ORB-SLAM's own
# ORBvoc.txt format) and compiled here from source against THIS project's
# own OpenCV, rather than linking a prebuilt libDBoW2.so from elsewhere --
# a prebuilt one on this machine (from an unrelated ORB_SLAM3 checkout) is
# built against a different OpenCV build (conda's 4.12 vs. this project's
# system apt OpenCV4), and mixing two OpenCV ABIs' cv::Mat in one process
# risks silent corruption, not just a clean link error.
# BowVector.h/FeatureVector.h include <boost/serialization/...> headers for
# an optional (de)serialization path this project never calls -- compiles
# fine with just the headers present (no boost library link needed), but
# the include path must be found; BOOST_INCLUDE_DIR lets it be overridden
# for a machine without this exact conda layout.
set(BOOST_INCLUDE_DIR "$ENV{HOME}/miniconda3/envs/myenv/include" CACHE PATH
    "Directory containing boost/serialization/serialization.hpp (DBoW2 needs the headers only, not the compiled library)")

add_library(DBoW2 STATIC
    third_party/DBoW2/DBoW2/BowVector.cpp
    third_party/DBoW2/DBoW2/FORB.cpp
    third_party/DBoW2/DBoW2/FeatureVector.cpp
    third_party/DBoW2/DBoW2/ScoringObject.cpp
    third_party/DBoW2/DUtils/Random.cpp
    third_party/DBoW2/DUtils/Timestamp.cpp
)
target_include_directories(DBoW2 PUBLIC third_party/DBoW2 ${OpenCV_INCLUDE_DIRS})
# PUBLIC, not PRIVATE: SlamWorker.h itself (compiled directly into every
# executable target, not just through the DBoW2 library) includes
# DBoW2/BowVector.h, so every consumer of DBoW2 needs this path too, not
# just DBoW2.cpp's own translation units.
target_include_directories(DBoW2 SYSTEM PUBLIC ${BOOST_INCLUDE_DIR})
target_link_libraries(DBoW2 PUBLIC ${OpenCV_LIBS})

# g2o (general graph optimization) -- ORB-SLAM2/3's own backend for its
# Sim(3) essential-graph loop-closure optimization. pose_graph::optimizePoseGraph()'s
# Sim(3) path (see PoseGraphOptimizer.cpp) uses g2o::VertexSim3Expmap/EdgeSim3
# directly (types_seven_dof_expmap.h) instead of a hand-rolled Ceres cost
# functor, matching real ORB-SLAM2/3 exactly. Linked directly against a
# prebuilt libg2o.so (unlike DBoW2, NOT recompiled from source here) --
# verified via ldd that libg2o.so has no OpenCV dependency at all (just
# libstdc++/libm/libc), so none of the ABI-mismatch risk that ruled out
# DBoW2's prebuilt .so applies here. G2O_ROOT lets this be overridden for a
# machine without this exact ORB_SLAM3 checkout.
set(G2O_ROOT "/home/nam/ORB_SLAM3/Thirdparty/g2o" CACHE PATH
    "Root containing g2o/core, g2o/types, g2o/solvers headers and lib/libg2o.so")
add_library(g2o_ext SHARED IMPORTED)
set_target_properties(g2o_ext PROPERTIES
    IMPORTED_LOCATION "${G2O_ROOT}/lib/libg2o.so"
    INTERFACE_INCLUDE_DIRECTORIES "${G2O_ROOT};/usr/include/eigen3"
)
# libg2o.so was built with -march=native (its own CMakeLists.txt hardcodes
# this for CMAKE_BUILD_TYPE=Release). Without matching it here, Eigen's
# vectorization-dependent alignment requirements can differ between this
# prebuilt .so and code compiled against these same headers in THIS
# project, which manifested as a segfault inside g2o::EdgeSim3's
# constructor (confirmed via gdb backtrace on a synthetic reproduction) --
# fixed entirely by this flag. INTERFACE so it propagates to every g2o_ext
# consumer, not just this file.
target_compile_options(g2o_ext INTERFACE -march=native)


set(CONDA_ENV_ROOT "$ENV{HOME}/miniconda3/envs/myenv" CACHE PATH
    "Conda environment root -- only used here for libboost_serialization.so (System.cc's Atlas save/load)")

# Path to the DBoW2 ORB vocabulary (ORBvoc.txt) ORB_SLAM3::System's
# constructor loads at startup -- same file format/content as the one
# SlamWorker::loadOrbVocabulary() can already load for the custom pipeline's
# own DBoW2 loop closure, just needed unconditionally here since System's
# constructor requires a vocabulary path up front. Overridable for a machine
# without this exact ORB_SLAM3 checkout.
set(ORBSLAM3_VOCAB_PATH "/home/nam/ORB_SLAM3/Vocabulary/ORBvoc.txt" CACHE PATH
    "Path to ORB_SLAM3's ORBvoc.txt, loaded by SlamWorker's vendored-ORB-SLAM3 GUI mode")

# Real ORB-SLAM3 reproduction (analyze/orbslam3_kitti_ate.cpp) -- the actual
# ORB-SLAM3 algorithm source vendored into third_party/ORB_SLAM3 (copied from
# an existing ORB_SLAM3 checkout), compiled from source here rather than
# linking its prebuilt libORB_SLAM3.so -- that .so (like DBoW2's own prebuilt
# one) was built against a conda environment's OpenCV, not this project's
# system OpenCV, and mixing the two cv::Mat ABIs in one process is exactly
# the risk that ruled out linking DBoW2's prebuilt .so earlier. Viewer.cc/.h
# were deliberately NOT vendored (GUI/Pangolin-only, not needed for headless
# benchmarking -- see System.cc's bUseViewer check and MapDrawer's own doc
# comment on the 4 GL-drawing methods removed from it), which drops the hard
# Pangolin dependency entirely, so -- unlike DBoW2 -- this compiles cleanly
# against this project's OWN system OpenCV/Eigen with no ABI concern at all.
# DBoW2 sources are recompiled fresh here (not the project's own `DBoW2`
# target reused) purely so ORB_SLAM3's own KeyFrameDatabase/ORBVocabulary
# headers resolve via their expected "Thirdparty/DBoW2/DBoW2/..." include
# path (see third_party/ORB_SLAM3/Thirdparty/DBoW2's symlink) -- it's the
# exact same vendored source either way. g2o is reused directly via the
# existing g2o_ext target (safe: g2o has no OpenCV dependency at all, see
# g2o_ext's own doc comment above).
file(GLOB ORB_SLAM3_SOURCES
    "third_party/ORB_SLAM3/src/*.cc"
    "third_party/ORB_SLAM3/src/CameraModels/*.cc"
)
# Shared STATIC lib so both the headless CLI benchmark (orbslam3_kitti_ate)
# and the live GUI (sift_vslam_gui, via SlamWorker's ORB-SLAM3 mode) link the
# exact same vendored algorithm instead of duplicating ~28,500 lines of
# compilation. Kept as its own target (rather than folding these sources
# directly into sift_vslam_gui) specifically so the CXX_STANDARD 14 override
# below applies only to this vendored code, not the whole project (which is
# C++17, see top of file).
add_library(orbslam3_ext STATIC
    ${ORB_SLAM3_SOURCES}
    third_party/DBoW2/DBoW2/BowVector.cpp
    third_party/DBoW2/DBoW2/FORB.cpp
    third_party/DBoW2/DBoW2/FeatureVector.cpp
    third_party/DBoW2/DBoW2/ScoringObject.cpp
    third_party/DBoW2/DUtils/Random.cpp
    third_party/DBoW2/DUtils/Timestamp.cpp
)
# ORB-SLAM3's own vendored source (unmodified logic -- LoopClosing.h's
# mnFullBAIdx is declared bool and incremented via mnFullBAIdx++ in several
# places, legal-but-deprecated pre-C++17, removed entirely in C++17) needs
# its ORIGINAL C++ standard (matching ORB_SLAM3's own CMakeLists.txt, which
# targets C++11), not this project's global C++17 -- overridden per-target
# rather than changing the vendored source's types, to keep the vendored
# logic a faithful, unmodified copy.
set_target_properties(orbslam3_ext PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
target_include_directories(orbslam3_ext PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    third_party/ORB_SLAM3
    third_party/ORB_SLAM3/include
    third_party/ORB_SLAM3/include/CameraModels
    third_party/ORB_SLAM3/Thirdparty/Sophus
)
target_include_directories(orbslam3_ext SYSTEM PUBLIC
    "${BOOST_INCLUDE_DIR}"
)
target_link_libraries(orbslam3_ext PUBLIC
    ${OpenCV_LIBS}
    g2o_ext
    # Full paths to the SYSTEM's own libssl/libcrypto (not bare -lssl/-lcrypto)
    # -- bare names risked resolving to the conda environment's OpenSSL 3.x
    # copy instead (both exist), which conflicts with system libcurl's own
    # (transitively linked via OpenCV's imgcodecs -> libhdf5_serial) older
    # OpenSSL-versioned symbols, producing undefined-reference link errors
    # unrelated to anything this target itself needs OpenSSL for (only
    # System.cc's MD5 checksum for Atlas save/load).
    /usr/lib/x86_64-linux-gnu/libssl.so
    /usr/lib/x86_64-linux-gnu/libcrypto.so
    "${CONDA_ENV_ROOT}/lib/libboost_serialization.so"
    # linking libboost_serialization.so (above) and g2o_ext's IMPORTED_LOCATION
    # both live in directories CMake auto-adds to this target's RPATH, and that
    # RPATH doubles as an rpath-link hint when the linker resolves *transitive*
    # NEEDED entries -- e.g. OpenCV's imgcodecs -> libhdf5_serial -> libcurl.so.4.
    # That caused libhdf5_serial's curl symbol references to resolve against
    # the conda env's libcurl.so.4 (which has NO CURL_OPENSSL_4 version node at
    # all) instead of the system's (which does), producing undefined-reference
    # link errors. Explicitly linking the system libcurl/libxml2 here forces the
    # linker to satisfy those versioned symbols from the correct copy.
    /usr/lib/x86_64-linux-gnu/libcurl.so.4
    /usr/lib/x86_64-linux-gnu/libxml2.so
    # Same RPATH-poisoning mechanism, hitting Qt itself once a target linking
    # orbslam3_ext also links Qt (sift_vslam_gui/sift_vslam_debug/kitti_ate):
    # Qt6Widgets/Gui transitively NEEDs libQt6DBus.so.6, and the poisoned
    # rpath-link resolved it against the conda env's Qt 6.9.3 copy instead of
    # the system's Qt 6.10.2 -- a private-ABI symbol version mismatch
    # (QObjectPrivate::QObjectPrivate(int)@Qt_6_PRIVATE_API undefined at link
    # time). Explicitly linking the system libQt6DBus.so here forces the
    # correct copy to be used, same fix shape as libcurl/libxml2 above.
    /usr/lib/x86_64-linux-gnu/libQt6DBus.so
)

add_executable(orbslam3_kitti_ate analyze/orbslam3_kitti_ate.cpp)
target_include_directories(orbslam3_kitti_ate PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_kitti_ate PRIVATE orbslam3_ext)

# Keypoint-location visualizer (analyze/orbslam3_visualize_keypoints.cpp,
# see DEBUGGING.md part 56) -- draws every detected keypoint on a single
# image, to directly compare WHERE SIFT vs ORB actually find features.
add_executable(orbslam3_visualize_keypoints analyze/orbslam3_visualize_keypoints.cpp)
target_include_directories(orbslam3_visualize_keypoints PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_visualize_keypoints PRIVATE orbslam3_ext)

# Standalone ASIFT (cv::AffineFeature-wrapped SIFT) feasibility test (see
# DEBUGGING.md part 56 / next-session plan) -- pure OpenCV, no dependency
# on either ORB_SLAM3 fork's own extractor code.
add_executable(asift_test analyze/asift_test.cpp)
target_include_directories(asift_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(asift_test PRIVATE ${OpenCV_LIBS})

# Standalone SIFT+LightGlue feasibility test (ASIFT investigation closed as
# a NEGATIVE RESULT, see DEBUGGING.md part 57 -- LightGlue is the
# follow-up: a learned matcher, official SIFT-trained weights exist, kept
# extractor untouched, only replacing the matching step). No official ONNX
# export of SIFT+LightGlue exists anywhere -- the model this loads comes
# from analyze/export_lightglue_sift.py (weights/lightglue_sift.onnx),
# verified numerically equivalent to the original PyTorch module. Requires
# ONNX Runtime (libonnxruntime-dev, installed via apt).
add_executable(lightglue_test analyze/lightglue_test.cpp)
target_include_directories(lightglue_test PRIVATE ${OpenCV_INCLUDE_DIRS} /usr/include/onnxruntime)
target_link_libraries(lightglue_test PRIVATE ${OpenCV_LIBS} onnxruntime)

# SIFT-based fork of the vendored ORB-SLAM3, built from an entirely separate
# copy (third_party/ORB_SLAM3_SIFT, `cp -r` of third_party/ORB_SLAM3) rather
# than modifying the original in place -- per explicit request, so the
# original, already-validated ORB-based algorithm (orbslam3_ext above, 6.4-
# 10.7m ATE RMSE on KITTI seq00, see DEBUGGING.md) stays completely untouched
# and independently buildable/runnable. Both copies define the same
# ORB_SLAM3::* class names, so this target and orbslam3_ext must never both
# be linked into the same executable (ODR violation) -- kept in a fully
# separate static lib + its own CLI validation executable for exactly that
# reason. Mirrors orbslam3_ext's own target definition immediately above;
# see its doc comments for why each piece (CXX_STANDARD 14 override, the
# explicit system-lib link paths working around RPATH poisoning, DBoW2
# recompiled fresh for the Thirdparty/DBoW2 include path) is there -- same
# reasoning applies unchanged, just against the new source tree.
file(GLOB ORB_SLAM3_SIFT_SOURCES
    "third_party/ORB_SLAM3_SIFT/src/*.cc"
    "third_party/ORB_SLAM3_SIFT/src/CameraModels/*.cc"
)
add_library(orbslam3_sift_ext STATIC
    ${ORB_SLAM3_SIFT_SOURCES}
    third_party/DBoW2/DBoW2/BowVector.cpp
    third_party/DBoW2/DBoW2/FORB.cpp
    third_party/DBoW2/DBoW2/FeatureVector.cpp
    third_party/DBoW2/DBoW2/ScoringObject.cpp
    third_party/DBoW2/DUtils/Random.cpp
    third_party/DBoW2/DUtils/Timestamp.cpp
)
set_target_properties(orbslam3_sift_ext PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
target_include_directories(orbslam3_sift_ext PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    third_party/ORB_SLAM3_SIFT
    third_party/ORB_SLAM3_SIFT/include
    third_party/ORB_SLAM3_SIFT/include/CameraModels
    third_party/ORB_SLAM3_SIFT/Thirdparty/Sophus
    /usr/include/onnxruntime
)
target_include_directories(orbslam3_sift_ext SYSTEM PUBLIC
    "${BOOST_INCLUDE_DIR}"
)
target_link_libraries(orbslam3_sift_ext PUBLIC
    ${OpenCV_LIBS}
    g2o_ext
    onnxruntime
    /usr/lib/x86_64-linux-gnu/libssl.so
    /usr/lib/x86_64-linux-gnu/libcrypto.so
    "${CONDA_ENV_ROOT}/lib/libboost_serialization.so"
    /usr/lib/x86_64-linux-gnu/libcurl.so.4
    /usr/lib/x86_64-linux-gnu/libxml2.so
    /usr/lib/x86_64-linux-gnu/libQt6DBus.so
)

add_executable(orbslam3_sift_kitti_ate analyze/orbslam3_kitti_ate.cpp)
target_include_directories(orbslam3_sift_kitti_ate PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_sift_kitti_ate PRIVATE orbslam3_sift_ext)

add_executable(orbslam3_sift_visualize_keypoints analyze/orbslam3_visualize_keypoints.cpp)
target_include_directories(orbslam3_sift_visualize_keypoints PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_sift_visualize_keypoints PRIVATE orbslam3_sift_ext)

# VLAD codebook trainer (see analyze/orbslam3_vlad_train.cpp) -- linked
# against orbslam3_sift_ext (not orbslam3_ext) so the training-time
# ORBextractor/VladVocabulary are the exact same compiled code the runtime
# SIFT-based System uses.
add_executable(orbslam3_vlad_train analyze/orbslam3_vlad_train.cpp)
target_include_directories(orbslam3_vlad_train PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_vlad_train PRIVATE orbslam3_sift_ext)

# Stage 4 threshold-calibration tool (see analyze/orbslam3_sift_calibrate.cpp).
add_executable(orbslam3_sift_calibrate analyze/orbslam3_sift_calibrate.cpp)
target_include_directories(orbslam3_sift_calibrate PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_sift_calibrate PRIVATE orbslam3_sift_ext)

set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/widgets/ControlPanel.cpp
    src/widgets/VideoView.cpp
    src/widgets/MapView.cpp
    src/widgets/LoopEstimatePanel.cpp
    src/widgets/LoopMapThumbnail.cpp
    src/vision/VideoSource.cpp
    src/vision/SlamWorker.cpp
    src/vision/EightPointLegacy.cpp
    src/vision/ImuRotation.cpp
    src/vision/GroundPlaneScale.cpp
    src/vision/LoopEstimator.cpp
    src/vision/PoseGraphOptimizer.cpp
    src/vision/FeatureDetector.cpp
)

set(HEADERS
    src/MainWindow.h
    src/widgets/ControlPanel.h
    src/widgets/VideoView.h
    src/widgets/MapView.h
    src/widgets/LoopEstimatePanel.h
    src/widgets/LoopMapThumbnail.h
    src/vision/CameraIntrinsics.h
    src/vision/SiftSettings.h
    src/vision/PnpSettings.h
    src/vision/VideoSource.h
    src/vision/SlamWorker.h
    src/vision/EightPointLegacy.h
    src/vision/ImuRotation.h
    src/vision/GroundPlaneScale.h
    src/vision/LoopEstimateTypes.h
    src/vision/LoopEstimator.h
    src/vision/PoseGraphOptimizer.h
    src/vision/FeatureDetector.h
)

add_executable(sift_vslam_gui ${SOURCES} ${HEADERS})

target_include_directories(sift_vslam_gui PRIVATE src ${OpenCV_INCLUDE_DIRS})

target_link_libraries(sift_vslam_gui PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Concurrent
    ${OpenCV_LIBS}
    Ceres::ceres
    DBoW2
    g2o_ext
    orbslam3_ext
)
target_compile_definitions(sift_vslam_gui PRIVATE
    ORBSLAM3_VOCAB_PATH="${ORBSLAM3_VOCAB_PATH}"
)

# TEMPORARY debug harness -- drives SlamWorker headlessly to verify the
# homography fallback. Remove once done.
add_executable(sift_vslam_debug
    src/debug_main.cpp
    src/vision/VideoSource.cpp
    src/vision/SlamWorker.cpp
    src/vision/SlamWorker.h
    src/vision/EightPointLegacy.cpp
    src/vision/EightPointLegacy.h
    src/vision/ImuRotation.cpp
    src/vision/ImuRotation.h
    src/vision/GroundPlaneScale.cpp
    src/vision/GroundPlaneScale.h
    src/vision/PoseGraphOptimizer.cpp
    src/vision/PoseGraphOptimizer.h
    src/vision/FeatureDetector.cpp
    src/vision/FeatureDetector.h
)
target_include_directories(sift_vslam_debug PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sift_vslam_debug PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    ${OpenCV_LIBS}
    Ceres::ceres
    DBoW2
    g2o_ext
    orbslam3_ext
)
target_compile_definitions(sift_vslam_debug PRIVATE
    ORBSLAM3_VOCAB_PATH="${ORBSLAM3_VOCAB_PATH}"
)

# Runs SlamWorker against a raw KITTI image sequence and computes ATE
# against ground truth natively in C++ -- see analyze/kitti_ate.cpp.
add_executable(kitti_ate
    analyze/kitti_ate.cpp
    src/vision/VideoSource.cpp
    src/vision/SlamWorker.cpp
    src/vision/SlamWorker.h
    src/vision/EightPointLegacy.cpp
    src/vision/EightPointLegacy.h
    src/vision/ImuRotation.cpp
    src/vision/ImuRotation.h
    src/vision/GroundPlaneScale.cpp
    src/vision/GroundPlaneScale.h
    src/vision/LoopEstimator.cpp
    src/vision/LoopEstimator.h
    src/vision/PoseGraphOptimizer.cpp
    src/vision/PoseGraphOptimizer.h
    src/vision/FeatureDetector.cpp
    src/vision/FeatureDetector.h
)
target_include_directories(kitti_ate PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(kitti_ate PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    ${OpenCV_LIBS}
    Ceres::ceres
    DBoW2
    g2o_ext
    orbslam3_ext
)
target_compile_definitions(kitti_ate PRIVATE
    ORBSLAM3_VOCAB_PATH="${ORBSLAM3_VOCAB_PATH}"
)
