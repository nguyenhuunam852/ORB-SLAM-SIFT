#include "EightPointLegacy.h"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace eight_point_legacy {
namespace {

constexpr unsigned kRansacSeed = 42; // matches SlamWorker.cpp's kRansacSeed -- fixed so results are
                                      // reproducible and directly comparable run-to-run
constexpr int kFRansacIterations = 1000;
constexpr double kFRansacSampsonThreshold = 1.0; // squared-pixel Sampson error

// Hartley normalization: a similarity transform moving the point set's
// centroid to the origin with mean distance sqrt(2) from it.
cv::Mat computeNormalizationTransform(const std::vector<cv::Point2f> &pts)
{
    double meanX = 0.0, meanY = 0.0;
    for (const auto &p : pts) {
        meanX += p.x;
        meanY += p.y;
    }
    meanX /= static_cast<double>(pts.size());
    meanY /= static_cast<double>(pts.size());

    double meanDist = 0.0;
    for (const auto &p : pts) {
        const double dx = p.x - meanX;
        const double dy = p.y - meanY;
        meanDist += std::sqrt(dx * dx + dy * dy);
    }
    meanDist /= static_cast<double>(pts.size());
    const double scale = (meanDist > 1e-8) ? std::sqrt(2.0) / meanDist : 1.0;

    return (cv::Mat_<double>(3, 3) << scale, 0.0, -scale * meanX, 0.0, scale, -scale * meanY, 0.0, 0.0, 1.0);
}

std::vector<cv::Point2f> applyTransform(const cv::Mat &T, const std::vector<cv::Point2f> &pts)
{
    std::vector<cv::Point2f> out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        const double x = T.at<double>(0, 0) * pts[i].x + T.at<double>(0, 1) * pts[i].y + T.at<double>(0, 2);
        const double y = T.at<double>(1, 0) * pts[i].x + T.at<double>(1, 1) * pts[i].y + T.at<double>(1, 2);
        out[i] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    }
    return out;
}

// Linear 8-point algorithm (Hartley & Zisserman, Alg. 11.1) on already
// normalized correspondences. Requires pts1.size() == pts2.size() >= 8.
cv::Mat solveEightPoint(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2)
{
    const int n = static_cast<int>(pts1.size());
    cv::Mat A(n, 9, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double x1 = pts1[i].x, y1 = pts1[i].y;
        const double x2 = pts2[i].x, y2 = pts2[i].y;
        A.at<double>(i, 0) = x2 * x1;
        A.at<double>(i, 1) = x2 * y1;
        A.at<double>(i, 2) = x2;
        A.at<double>(i, 3) = y2 * x1;
        A.at<double>(i, 4) = y2 * y1;
        A.at<double>(i, 5) = y2;
        A.at<double>(i, 6) = x1;
        A.at<double>(i, 7) = y1;
        A.at<double>(i, 8) = 1.0;
    }

    // F is the singular vector of A associated with its smallest singular value.
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);
    cv::Mat F = vt.row(8).clone().reshape(0, 3);

    // Enforce the rank-2 constraint by zeroing the smallest singular value of F.
    cv::Mat w2, u2, vt2;
    cv::SVD::compute(F, w2, u2, vt2, cv::SVD::FULL_UV);
    w2.at<double>(2) = 0.0;
    return u2 * cv::Mat::diag(w2) * vt2;
}

// Plain scalar coefficients of a 3x3 F, pulled out once per RANSAC candidate
// so the per-correspondence inlier test below never touches cv::Mat (each
// cv::Mat construction heap-allocates; doing that inside an
// iterations x correspondences loop is what was causing multi-second stalls).
struct FCoeffs
{
    double f00, f01, f02;
    double f10, f11, f12;
    double f20, f21, f22;
};

FCoeffs extractF(const cv::Mat &F)
{
    return FCoeffs{F.at<double>(0, 0), F.at<double>(0, 1), F.at<double>(0, 2),
                    F.at<double>(1, 0), F.at<double>(1, 1), F.at<double>(1, 2),
                    F.at<double>(2, 0), F.at<double>(2, 1), F.at<double>(2, 2)};
}

// First-order (Sampson) approximation of squared geometric distance to the
// epipolar line, in squared pixels. Pure scalar arithmetic, no allocation.
double sampsonDistance(const FCoeffs &f, const cv::Point2f &p1, const cv::Point2f &p2)
{
    const double x1 = p1.x, y1 = p1.y;
    const double x2 = p2.x, y2 = p2.y;

    const double Fx1_0 = f.f00 * x1 + f.f01 * y1 + f.f02;
    const double Fx1_1 = f.f10 * x1 + f.f11 * y1 + f.f12;
    const double Fx1_2 = f.f20 * x1 + f.f21 * y1 + f.f22;

    const double Ftx2_0 = f.f00 * x2 + f.f10 * y2 + f.f20;
    const double Ftx2_1 = f.f01 * x2 + f.f11 * y2 + f.f21;

    const double x2tFx1 = x2 * Fx1_0 + y2 * Fx1_1 + Fx1_2;

    const double denom = Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 + Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1;
    if (denom < 1e-12)
        return std::numeric_limits<double>::max();
    return (x2tFx1 * x2tFx1) / denom;
}

// Signed version of sampsonDistance's residual: r = (x2^T F x1) / sqrt(denom),
// so r*r == sampsonDistance(...). Needed (rather than just the squared
// distance) because the Gauss-Newton/LM refinement below builds a proper
// per-point Jacobian, not just a gradient of the summed squared cost.
double sampsonResidual(const FCoeffs &f, const cv::Point2f &p1, const cv::Point2f &p2)
{
    const double x1 = p1.x, y1 = p1.y;
    const double x2 = p2.x, y2 = p2.y;

    const double Fx1_0 = f.f00 * x1 + f.f01 * y1 + f.f02;
    const double Fx1_1 = f.f10 * x1 + f.f11 * y1 + f.f12;
    const double Fx1_2 = f.f20 * x1 + f.f21 * y1 + f.f22;

    const double Ftx2_0 = f.f00 * x2 + f.f10 * y2 + f.f20;
    const double Ftx2_1 = f.f01 * x2 + f.f11 * y2 + f.f21;

    const double e = x2 * Fx1_0 + y2 * Fx1_1 + Fx1_2;
    const double denom = Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 + Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1;
    if (denom < 1e-12)
        return 0.0;
    return e / std::sqrt(denom);
}

// Re-enforces F's rank-2 constraint (SVD, zero smallest singular value) and
// renormalizes to unit Frobenius norm. Needed after every LM step below since
// an unconstrained 9-parameter update walks F off the rank-2 manifold.
cv::Mat projectFundamentalRank2(const cv::Mat &F)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(F, w, u, vt, cv::SVD::FULL_UV);
    w.at<double>(2) = 0.0;
    cv::Mat Fp = u * cv::Mat::diag(w) * vt;
    const double n = cv::norm(Fp);
    return (n > 1e-12) ? Fp / n : Fp;
}

// Nonlinear (Gold Standard, Hartley & Zisserman Algorithm 11.3) refinement of
// a linear-least-squares F estimate: a handful of Levenberg-Marquardt steps
// minimizing true Sampson distance directly, rather than the algebraic error
// solveEightPoint minimizes. Operates on F's 9 raw entries (not H&Z's minimal
// 7-parameter epipole form) with the rank-2 constraint re-enforced by SVD
// projection after every accepted step -- simpler to implement correctly than
// the minimal parameterization, and equivalent in practice for a handful of
// iterations starting from an already-good linear estimate. Jacobian is
// numerical (central differences): F has only 9 parameters and inlier counts
// here are at most a few hundred, so the cost is trivial, and it avoids
// hand-deriving analytic Sampson derivatives (easy to get subtly wrong).
// Expects (and returns F in) already Hartley-normalized coordinates, same as
// solveEightPoint, for numerical conditioning.
cv::Mat refineFundamentalSampson(const cv::Mat &Finit, const std::vector<cv::Point2f> &pts1,
                                  const std::vector<cv::Point2f> &pts2)
{
    constexpr int kMaxIters = 10;
    constexpr int kMaxLmTries = 10;
    constexpr double kFdStep = 1e-6;

    const int n = static_cast<int>(pts1.size());

    cv::Mat F = Finit.clone();
    {
        const double fn = cv::norm(F);
        if (fn > 1e-12)
            F /= fn;
    }

    auto residuals = [&](const cv::Mat &Fc) {
        std::vector<double> r(n);
        const FCoeffs fc = extractF(Fc);
        for (int i = 0; i < n; ++i)
            r[i] = sampsonResidual(fc, pts1[i], pts2[i]);
        return r;
    };
    auto costOf = [](const std::vector<double> &r) {
        double c = 0.0;
        for (double v : r)
            c += v * v;
        return c;
    };

    std::vector<double> r0 = residuals(F);
    double cost = costOf(r0);
    double lambda = 1e-3;

    for (int iter = 0; iter < kMaxIters; ++iter) {
        cv::Mat J(n, 9, CV_64F);
        double *fptr = F.ptr<double>();
        for (int j = 0; j < 9; ++j) {
            const double orig = fptr[j];
            fptr[j] = orig + kFdStep;
            const std::vector<double> rPlus = residuals(F);
            fptr[j] = orig - kFdStep;
            const std::vector<double> rMinus = residuals(F);
            fptr[j] = orig;
            for (int i = 0; i < n; ++i)
                J.at<double>(i, j) = (rPlus[i] - rMinus[i]) / (2.0 * kFdStep);
        }

        cv::Mat rVec(n, 1, CV_64F);
        for (int i = 0; i < n; ++i)
            rVec.at<double>(i) = r0[i];

        const cv::Mat JtJ = J.t() * J;
        const cv::Mat Jtr = J.t() * rVec;

        bool improved = false;
        for (int lmTry = 0; lmTry < kMaxLmTries; ++lmTry) {
            const cv::Mat A = JtJ + lambda * cv::Mat::diag(JtJ.diag());
            cv::Mat delta;
            if (!cv::solve(A, -Jtr, delta, cv::DECOMP_SVD)) {
                lambda *= 10.0;
                continue;
            }

            cv::Mat Fcand = F.clone();
            double *fcand = Fcand.ptr<double>();
            for (int j = 0; j < 9; ++j)
                fcand[j] += delta.at<double>(j);
            Fcand = projectFundamentalRank2(Fcand);

            const std::vector<double> rCand = residuals(Fcand);
            const double candCost = costOf(rCand);
            if (candCost < cost) {
                F = Fcand;
                r0 = rCand;
                cost = candCost;
                lambda = std::max(lambda * 0.3, 1e-8);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if (!improved)
            break; // converged (or stuck): further iterations won't help
    }

    return F;
}

} // namespace

cv::Mat estimateFundamentalRansac(const std::vector<cv::Point2f> &pts1, const std::vector<cv::Point2f> &pts2,
                                   cv::Mat &mask)
{
    const int n = static_cast<int>(pts1.size());
    mask = cv::Mat::zeros(n, 1, CV_8U);
    if (n < 8)
        return cv::Mat();

    const cv::Mat T1 = computeNormalizationTransform(pts1);
    const cv::Mat T2 = computeNormalizationTransform(pts2);
    const std::vector<cv::Point2f> normPts1 = applyTransform(T1, pts1);
    const std::vector<cv::Point2f> normPts2 = applyTransform(T2, pts2);

    std::mt19937 rng(kRansacSeed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    std::vector<int> bestInliers;

    for (int iter = 0; iter < kFRansacIterations; ++iter) {
        std::vector<int> sample;
        sample.reserve(8);
        while (sample.size() < 8) {
            const int idx = dist(rng);
            if (std::find(sample.begin(), sample.end(), idx) == sample.end())
                sample.push_back(idx);
        }

        std::vector<cv::Point2f> sample1(8), sample2(8);
        for (int i = 0; i < 8; ++i) {
            sample1[i] = normPts1[sample[i]];
            sample2[i] = normPts2[sample[i]];
        }

        const cv::Mat Fn = solveEightPoint(sample1, sample2);
        const cv::Mat F = T2.t() * Fn * T1;
        const FCoeffs fc = extractF(F);

        std::vector<int> inliers;
        inliers.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (sampsonDistance(fc, pts1[i], pts2[i]) < kFRansacSampsonThreshold)
                inliers.push_back(i);
        }

        if (inliers.size() > bestInliers.size())
            bestInliers = std::move(inliers);
    }

    if (bestInliers.size() < 8)
        return cv::Mat();

    // Refit F over the full inlier set (over-determined least squares).
    std::vector<cv::Point2f> inNorm1, inNorm2;
    inNorm1.reserve(bestInliers.size());
    inNorm2.reserve(bestInliers.size());
    for (int idx : bestInliers) {
        inNorm1.push_back(normPts1[idx]);
        inNorm2.push_back(normPts2[idx]);
    }
    const cv::Mat FnRefined = solveEightPoint(inNorm1, inNorm2);
    // Gold Standard nonlinear refinement: solveEightPoint above only
    // minimizes algebraic error; this takes the further Gauss-Newton/LM
    // step (H&Z Algorithm 11.3) minimizing true Sampson distance over the
    // same inlier set. See refineFundamentalSampson()'s doc comment.
    const cv::Mat FnGold = refineFundamentalSampson(FnRefined, inNorm1, inNorm2);
    const cv::Mat Frefined = T2.t() * FnGold * T1;

    for (int idx : bestInliers)
        mask.at<uchar>(idx) = 1;

    return Frefined;
}

} // namespace eight_point_legacy
