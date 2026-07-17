#include "GroundPlaneScale.h"

#include <algorithm>
#include <cmath>

namespace ground_plane_scale {

double estimateCameraHeight(const std::vector<cv::Point3f> &pointsInCameraFrame, const GroundPlaneConfig &config)
{
    const int n = static_cast<int>(pointsInCameraFrame.size());
    if (n < 3 || config.groundNormalCam.empty())
        return -1.0;

    // Restrict to the nearer half of points (by L1 distance |X|+|Y|+|Z|)
    // before doing anything else -- matches the real libviso2 source
    // exactly (VisualOdometryMono::smallerThanMedian(), confirmed against
    // the authoritative implementation, KIT-MRT/viso2, src/viso_mono.cpp).
    // Distant triangulations are noisier and less likely to be genuine
    // nearby road points, so this is a real, deliberate filtering step in
    // the original algorithm, not an approximation added here.
    std::vector<double> l1(n);
    for (int i = 0; i < n; ++i) {
        const auto &p = pointsInCameraFrame[i];
        l1[i] = std::abs(p.x) + std::abs(p.y) + std::abs(p.z);
    }
    std::vector<double> sortedL1 = l1;
    std::sort(sortedL1.begin(), sortedL1.end());
    const double median = sortedL1[n / 2];

    const double nx = config.groundNormalCam.at<double>(0);
    const double ny = config.groundNormalCam.at<double>(1);
    const double nz = config.groundNormalCam.at<double>(2);

    // Per-point implied camera height: the ground-plane equation is
    // n^T X + h = 0 for a point X *on* the ground (n points from ground up
    // towards the camera, h = camera height above it), so a point's own
    // implied height (were it exactly on the ground) is h_i = -n^T X_i.
    // Only the near-half (l1[i] <= median) are kept, per the filtering step
    // above.
    std::vector<double> h;
    h.reserve(n / 2 + 1);
    for (int i = 0; i < n; ++i) {
        if (l1[i] > median)
            continue;
        const auto &p = pointsInCameraFrame[i];
        h.push_back(-(nx * p.x + ny * p.y + nz * p.z));
    }
    if (h.size() < 3)
        return -1.0;

    // VISO2-M's robust consensus vote (confirmed verbatim against
    // KIT-MRT/viso2, src/viso_mono.cpp's estimateMotion()): score each
    // point's height estimate by how consistent it is with every other
    // point's, then take the height with the best score. Points far from
    // the road (parked cars, curbs, building facades) naturally score low
    // since their implied height disagrees with the (larger) group of
    // genuine road points. sigma is *adaptive*, scaled to this frame's own
    // median L1 distance (median/50 in the original), not a fixed
    // constant -- a different, later paper's citation of this formula
    // (Song/Chandraker/Guest PAMI 2015 eq. 4) uses a fixed mu=50 instead,
    // which is NOT what the original libviso2 source actually does; this
    // was corrected here to match the authoritative implementation after
    // fetching and reading it directly, rather than trusting the
    // secondary citation.
    const double sigma = median / 50.0;
    const double weight = (sigma > 1e-9) ? 1.0 / (2.0 * sigma * sigma) : 0.0;

    double bestScore = -1.0;
    double bestH = h[0];
    for (size_t i = 0; i < h.size(); ++i) {
        double score = 0.0;
        for (size_t j = 0; j < h.size(); ++j) {
            if (i == j)
                continue;
            const double dh = h[i] - h[j];
            score += std::exp(-dh * dh * weight);
        }
        if (score > bestScore) {
            bestScore = score;
            bestH = h[i];
        }
    }

    if (!(bestH > 0.0) || !std::isfinite(bestH))
        return -1.0; // a "height" at or behind the camera, or non-finite, isn't a real ground point
    return bestH;
}

double estimateScaleCorrection(const std::vector<cv::Point3f> &pointsInCameraFrame, const GroundPlaneConfig &config)
{
    const double estimatedHeight = estimateCameraHeight(pointsInCameraFrame, config);
    if (estimatedHeight <= 0.0)
        return -1.0;
    return config.knownCameraHeight / estimatedHeight;
}

} // namespace ground_plane_scale
