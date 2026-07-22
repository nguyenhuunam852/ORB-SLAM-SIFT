/**
 * File: FRootSift.cpp
 * Description: functions for RootSIFT (and plain SIFT) descriptors, see
 *   FRootSift.h's own doc comment.
 */

#include <vector>
#include <string>
#include <sstream>

#include "FRootSift.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

const int FRootSift::L = 128;

void FRootSift::meanValue(const std::vector<FRootSift::pDescriptor> &descriptors,
  FRootSift::TDescriptor &mean)
{
  if(descriptors.empty())
  {
    // Deliberately NOT mean.release() here (this project, 2026-07-22):
    // HKmeansStep passes clusters[c] as `mean`, already holding that
    // cluster's centroid from the previous iteration/initiateClusters --
    // releasing it turns clusters[c] into a 0x0 cv::Mat, and the very next
    // line in HKmeansStep's reassignment loop calls F::distance() against
    // it, which does b.ptr<float>() on an empty Mat -> null pointer,
    // dereferenced 128 times -> SIGSEGV. A cluster losing all its assigned
    // descriptors mid-iteration is routine for RootSIFT's continuous
    // 128-dim float space (far more common than for FORB's Hamming
    // distance, which is why the vendored FORB.cpp has this identical
    // release()-on-empty pattern but rarely trips it). Keeping the old,
    // still-valid centroid instead of releasing it fixes the crash and
    // gives that cluster a chance to reattract points on a later
    // iteration.
    return;
  }
  else if(descriptors.size() == 1)
  {
    mean = descriptors[0]->clone();
  }
  else
  {
    mean = cv::Mat::zeros(1, FRootSift::L, CV_32F);
    float *pMean = mean.ptr<float>();

    for(size_t i = 0; i < descriptors.size(); ++i)
    {
      const cv::Mat &d = *descriptors[i];
      const float *p = d.ptr<float>();
      for(int j = 0; j < FRootSift::L; ++j)
        pMean[j] += p[j];
    }

    const float N = static_cast<float>(descriptors.size());
    for(int j = 0; j < FRootSift::L; ++j)
      pMean[j] /= N;
  }
}

// --------------------------------------------------------------------------

double FRootSift::distance(const FRootSift::TDescriptor &a,
  const FRootSift::TDescriptor &b)
{
  // Squared L2 -- see the doc comment on the declaration in FRootSift.h
  // for why this deliberately skips the sqrt.
  const float *pa = a.ptr<float>();
  const float *pb = b.ptr<float>();

  double sqSum = 0.0;
  for(int i = 0; i < FRootSift::L; ++i)
  {
    const double diff = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
    sqSum += diff * diff;
  }
  return sqSum;
}

// --------------------------------------------------------------------------

std::string FRootSift::toString(const FRootSift::TDescriptor &a)
{
  stringstream ss;
  const float *p = a.ptr<float>();

  for(int i = 0; i < a.cols; ++i, ++p)
  {
    ss << *p << " ";
  }

  return ss.str();
}

// --------------------------------------------------------------------------

void FRootSift::fromString(FRootSift::TDescriptor &a, const std::string &s)
{
  a.create(1, FRootSift::L, CV_32F);
  float *p = a.ptr<float>();

  stringstream ss(s);
  for(int i = 0; i < FRootSift::L; ++i, ++p)
  {
    float n;
    ss >> n;

    if(!ss.fail())
      *p = n;
  }
}

// --------------------------------------------------------------------------

void FRootSift::toMat32F(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }

  const size_t N = descriptors.size();
  mat.create(static_cast<int>(N), FRootSift::L, CV_32F);

  for(size_t i = 0; i < N; ++i)
    descriptors[i].copyTo(mat.row(static_cast<int>(i)));
}

// --------------------------------------------------------------------------

} // namespace DBoW2
