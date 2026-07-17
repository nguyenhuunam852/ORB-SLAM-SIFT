#pragma once

#include <QMetaType>
#include <QPointF>
#include <QString>
#include <QVector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <unordered_set>
#include <vector>

#include "CameraIntrinsics.h"

// Self-contained, deep-copied snapshot of one loop-closure window, handed
// from SlamWorker (worker thread) to a background QtConcurrent task (a
// thread-pool thread) so the heavy re-matching + bundle adjustment + ATE
// computation in computeLoopEstimate() (LoopEstimator.h) never touches
// live SlamWorker state and never blocks the live tracking loop -- "no
// change in the GUI" while it runs, per explicit request. Every cv::Mat
// here is a clone(), never a shallow/shared reference into anything the
// worker thread might still be mutating.
struct LoopEstimateSnapshot
{
    struct KeyframeSnapshot
    {
        cv::Mat R, t; // world-to-camera, cloned
        cv::Mat descriptors; // this keyframe's full SIFT descriptors (not just its own
                              // triangulated points) -- computeLoopEstimate() re-matches
                              // these against every earlier keyframe in the window to
                              // build longer landmark tracks than live tracking's
                              // eviction-bounded rolling map ever accumulates
        std::vector<cv::KeyPoint> keypoints;
        std::vector<cv::Point3f> localMapPoints; // this keyframe's own triangulated points
        cv::Mat localMapDescriptors; // parallel to localMapPoints/localMapPointIds (NOT the same
                                      // rows as `descriptors` above, which is the full detected-
                                      // keypoint set) -- computeLoopEstimate()'s enrichment pass
                                      // must accumulate its re-matching *pool* from this array, not
                                      // `descriptors`, or the pool's ID bookkeeping desyncs from its
                                      // row count (see DEBUGGING.md Session 8: this was a real,
                                      // silent out-of-bounds bug, not a dilution/weighting issue)
        std::vector<cv::Point2f> localMapImagePoints; // parallel to localMapPoints
        std::vector<long long> localMapPointIds; // parallel to localMapPoints
        int frameIndex = 0;
    };

    std::vector<KeyframeSnapshot> keyframes; // window [oldKfIdx, newKfIdx], reindexed 0..N-1
    cv::Mat loopR, loopT; // loop-measured pose at the new (last) endpoint
    CameraIntrinsics intrinsics;

    // Landmark IDs whose observation at the LAST keyframe in `keyframes`
    // (index keyframes.size()-1, i.e. the new/closing keyframe) is the
    // actual PnP-RANSAC-verified correspondence tryLoopClosure() used to
    // measure (loopR, loopT) -- not just any re-match the enrichment pass
    // happens to find. computeLoopEstimate() gives these residuals a much
    // higher weight and no robust loss, the same special treatment
    // runLoopBundleAdjustment() gives them live.
    std::unordered_set<long long> loopVerifiedLandmarkIds;

    QVector<QPointF> trajectory; // full live trajectory at the moment the loop closed
    QVector<int> trajectoryFrameIndex; // parallel to trajectory

    QVector<QPointF> groundTruth; // (x, z) per poses.txt line; empty if none loaded
    bool hasGroundTruth = false;

    int oldFrameIndex = 0;
    int newFrameIndex = 0;
};

// Result of computeLoopEstimate(), delivered back to the GUI thread via
// QFutureWatcher for display in LoopEstimatePanel.
struct LoopEstimateResult
{
    bool ok = false;
    QString message; // set when ok == false (e.g. "no ground truth loaded")

    int oldFrameIndex = 0;
    int newFrameIndex = 0;

    // Landmark-track richness before vs. after the in-window re-matching
    // pass -- directly shows whether the enrichment step actually found
    // the longer tracks live tracking's rolling-map eviction misses.
    int landmarksBefore = 0;
    int observationsBefore = 0;
    int landmarksAfter = 0;
    int observationsAfter = 0;

    bool baConverged = false;
    double baInitialCost = 0.0;
    double baFinalCost = 0.0;
    int loopVerifiedResidualCount = 0; // how many residuals got the special
                                        // loop-verified weighting -- should be
                                        // close to loopVerifiedLandmarkIds'
                                        // size if enrichment is finding them

    int ateMatchedPoints = 0;
    double ateRmse = 0.0;
    double ateMean = 0.0;
    double ateMedian = 0.0;
    double ateMax = 0.0;
    double recoveredScale = 0.0;

    // Post-BA map data for LoopEstimatePanel's per-loop mini-map, aligned
    // into ground truth's frame using the exact same (scale, cosT, sinT,
    // tx, tz) computed for the ATE numbers above, so the rendered map and
    // the ATE figure always agree. Empty if no ground truth was loaded
    // (ateMatchedPoints == 0) -- nothing meaningful to align against.
    QVector<QPointF> alignedLandmarks; // optimized landmark positions (world X/Z)
    QVector<QPointF> alignedTrajectory; // corrected trajectory segment, same points averaged into ATE
    QVector<QPointF> alignedGroundTruth; // matching ground-truth segment (already in GT frame)
    int maxErrorIndex = -1; // index into alignedTrajectory/alignedGroundTruth of this window's single
                             // worst-error point -- the "most ATE" location, highlighted by
                             // LoopMapThumbnail. -1 if alignedTrajectory is empty.
};

Q_DECLARE_METATYPE(LoopEstimateSnapshot)
Q_DECLARE_METATYPE(LoopEstimateResult)
