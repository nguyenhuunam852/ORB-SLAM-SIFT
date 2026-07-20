cmake_minimum_required(VERSION 3.16)
project(orbslam3_sift_kaggle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Minimal, Kaggle-clean build of just the SIFT+LightGlue ORB-SLAM3 fork's
# headless KITTI benchmark (orbslam3_sift_kitti_ate) -- deliberately NOT the
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
)
# Matches the vendored ORB-SLAM3 source's own original target (LoopClosing.h's
# mnFullBAIdx++ on a bool member is legal pre-C++17, removed in C++17) -- see
# the root CMakeLists.txt's identical override for the full explanation.
set_target_properties(orbslam3_sift_ext PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
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

add_executable(orbslam3_sift_kitti_ate ${CMAKE_CURRENT_SOURCE_DIR}/../analyze/orbslam3_kitti_ate.cpp)
target_include_directories(orbslam3_sift_kitti_ate PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(orbslam3_sift_kitti_ate PRIVATE orbslam3_sift_ext)

# So the built binary finds libonnxruntime.so/libg2o.so at runtime without
# needing LD_LIBRARY_PATH set manually in every notebook cell.
set_target_properties(orbslam3_sift_kitti_ate PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${G2O_ROOT}/lib"
)
