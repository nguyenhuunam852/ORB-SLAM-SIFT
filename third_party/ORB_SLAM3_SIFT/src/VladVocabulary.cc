#include "VladVocabulary.h"

#include <opencv2/core.hpp>

#include <limits>

namespace ORB_SLAM3
{

bool VladVocabulary::loadFromTextFile(const std::string &path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    cv::Mat centroids;
    fs["centroids"] >> centroids;
    if (centroids.empty() || centroids.type() != CV_32F)
        return false;

    mCentroids = centroids;
    return true;
}

int VladVocabulary::nearestCentroid(const cv::Mat &descriptorRow) const
{
    int best = 0;
    float bestDist = std::numeric_limits<float>::max();
    for (int c = 0; c < mCentroids.rows; ++c) {
        const float d = static_cast<float>(cv::norm(descriptorRow, mCentroids.row(c), cv::NORM_L2SQR));
        if (d < bestDist) {
            bestDist = d;
            best = c;
        }
    }
    return best;
}

cv::Mat VladVocabulary::computeVlad(const cv::Mat &descriptors) const
{
    cv::Mat vlad = cv::Mat::zeros(mCentroids.rows, mCentroids.cols, CV_32F);

    for (int r = 0; r < descriptors.rows; ++r) {
        const cv::Mat row = descriptors.row(r);
        const int c = nearestCentroid(row);
        vlad.row(c) += (row - mCentroids.row(c));
    }

    for (int c = 0; c < vlad.rows; ++c) {
        cv::Mat centroidResidual = vlad.row(c);
        const double n = cv::norm(centroidResidual, cv::NORM_L2);
        if (n > std::numeric_limits<float>::epsilon())
            centroidResidual /= n;
    }

    cv::Mat flat = vlad.reshape(1, 1).clone();
    const double n = cv::norm(flat, cv::NORM_L2);
    if (n > std::numeric_limits<float>::epsilon())
        flat /= n;

    return flat;
}

float VladVocabulary::score(const cv::Mat &v1, const cv::Mat &v2) const
{
    if (v1.empty() || v2.empty() || v1.total() != v2.total())
        return 0.0f;
    return static_cast<float>(v1.dot(v2));
}

unsigned int VladVocabulary::size() const
{
    return static_cast<unsigned int>(mCentroids.rows);
}

} // namespace ORB_SLAM3
