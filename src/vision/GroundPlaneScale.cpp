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

bool estimateGroundNormal(const std::vector<cv::Point3f> &pointsInCameraFrame, cv::Vec3d &normalOut)
{
    // Road-candidate selection: below the camera (Y > 0 in the X-right,
    // Y-down, Z-forward frame), in front (Z > 0), and among the nearer half
    // by L1 distance (same near-half filter estimateCameraHeight() uses --
    // distant triangulations are noisy and less likely genuine road). This
    // deliberately does NOT assume a normal (that is what we are solving
    // for); it only assumes the road is roughly below and ahead, which is
    // true regardless of the exact pitch.
    std::vector<cv::Point3f> cand;
    cand.reserve(pointsInCameraFrame.size());
    std::vector<double> l1;
    l1.reserve(pointsInCameraFrame.size());
    for (const auto &p : pointsInCameraFrame) {
        if (p.y <= 0.0f || p.z <= 0.0f)
            continue; // above camera or behind it -- not a road point
        cand.push_back(p);
        l1.push_back(std::abs(p.x) + std::abs(p.y) + std::abs(p.z));
    }
    const int m = static_cast<int>(cand.size());
    if (m < 8)
        return false; // too few road candidates to fit a plane robustly

    std::vector<double> sortedL1 = l1;
    std::sort(sortedL1.begin(), sortedL1.end());
    const double medianL1 = sortedL1[static_cast<size_t>(m) / 2];
    std::vector<cv::Point3f> road;
    road.reserve(static_cast<size_t>(m) / 2 + 1);
    for (int i = 0; i < m; ++i)
        if (l1[static_cast<size_t>(i)] <= medianL1)
            road.push_back(cand[static_cast<size_t>(i)]);
    const int r = static_cast<int>(road.size());
    if (r < 8)
        return false;

    // Deterministic RANSAC plane fit (fixed LCG seed -- no Math.random /
    // std::rand nondeterminism, so a given point set always calibrates to
    // the same normal). Inlier threshold is scaled to the point cloud's own
    // size (medianL1/50), the same adaptive scale estimateCameraHeight()'s
    // consensus vote uses.
    const double inlierThresh = medianL1 / 50.0;
    unsigned int lcg = 0x9e3779b9u; // fixed seed
    auto nextRand = [&lcg]() {
        lcg = lcg * 1664525u + 1013904223u;
        return lcg;
    };

    cv::Vec3d bestNormal(0.0, -1.0, 0.0);
    int bestInliers = -1;
    const int kIterations = 200;
    for (int it = 0; it < kIterations; ++it) {
        const cv::Point3f &a = road[nextRand() % static_cast<unsigned int>(r)];
        const cv::Point3f &b = road[nextRand() % static_cast<unsigned int>(r)];
        const cv::Point3f &c = road[nextRand() % static_cast<unsigned int>(r)];
        const cv::Vec3d ab(b.x - a.x, b.y - a.y, b.z - a.z);
        const cv::Vec3d ac(c.x - a.x, c.y - a.y, c.z - a.z);
        cv::Vec3d nrm = ab.cross(ac);
        const double len = cv::norm(nrm);
        if (len < 1e-9)
            continue; // collinear sample
        nrm /= len;
        const double d = -(nrm[0] * a.x + nrm[1] * a.y + nrm[2] * a.z);
        int inliers = 0;
        for (const auto &p : road) {
            const double dist = std::abs(nrm[0] * p.x + nrm[1] * p.y + nrm[2] * p.z + d);
            if (dist <= inlierThresh)
                ++inliers;
        }
        if (inliers > bestInliers) {
            bestInliers = inliers;
            bestNormal = nrm;
        }
    }
    if (bestInliers < 6)
        return false;

    // Orient the normal to point UP towards the camera (ny < 0), matching
    // GroundPlaneConfig::groundNormalCam's convention.
    if (bestNormal[1] > 0.0)
        bestNormal = -bestNormal;
    normalOut = bestNormal;
    return true;
}

double pitchDegreesFromNormal(const cv::Vec3d &normal)
{
    cv::Vec3d n = normal;
    const double len = cv::norm(n);
    if (len < 1e-9)
        return 0.0;
    n /= len;
    // Level normal is (0,-1,0). A downward (nose-down) pitch tilts it
    // towards +Z: (0, -cos, sin). So pitch = atan2(nz, -ny), signed.
    return std::atan2(n[2], -n[1]) * 180.0 / M_PI;
}

cv::Mat groundNormalFromPitchDegrees(double pitchDeg)
{
    const double th = pitchDeg * M_PI / 180.0;
    return (cv::Mat_<double>(3, 1) << 0.0, -std::cos(th), std::sin(th));
}

} // namespace ground_plane_scale
