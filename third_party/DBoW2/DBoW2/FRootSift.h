/**
 * File: FRootSift.h
 * Description: functions for RootSIFT (and plain SIFT) descriptors, for
 *   DBoW2::TemplatedVocabulary. Written for this project (Session
 *   2026-07-22, see DEBUGGING.md) -- the vendored DBoW2 only shipped
 *   FORB.h (Hamming distance over packed-bit ORB descriptors). Mirrors
 *   FORB.h/.cpp's structure exactly, swapped for a 128-dim CV_32F
 *   descriptor and L2 (Euclidean) distance, the standard choice for
 *   SIFT-family descriptors in bag-of-words vocabularies. License: see
 *   the DBoW2 LICENSE.txt file (this file follows the same license as
 *   the rest of DBoW2, being a derivative of FORB.h).
 */

#ifndef __D_T_F_ROOT_SIFT__
#define __D_T_F_ROOT_SIFT__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

#include "FClass.h"

namespace DBoW2 {

/// Functions to manipulate RootSIFT/SIFT descriptors (128-dim CV_32F rows,
/// this project's own FeatureDetector.h/toRootSift() output format --
/// SlamWorker's m_detector/m_mapDescriptors rows are already in exactly
/// this shape, so no conversion is needed to feed them straight into
/// vocabulary training or transform()).
class FRootSift: protected FClass
{
public:

  /// Descriptor type: a single 1xL CV_32F row (same convention as FORB's
  /// TDescriptor being a single cv::Mat row, just float instead of
  /// packed-bit).
  typedef cv::Mat TDescriptor;
  /// Pointer to a single descriptor
  typedef const TDescriptor *pDescriptor;
  /// Descriptor length (in floats) -- 128 for both plain SIFT and RootSIFT
  /// (RootSIFT is the same-length descriptor with an L1-normalize/sqrt/
  /// L2-normalize transform already baked in by the caller, see
  /// FeatureDetector.h's own toRootSift() doc comment).
  static const int L;

  /**
   * Calculates the mean value of a set of descriptors -- plain
   * per-dimension arithmetic mean, the standard choice for real-valued
   * descriptors (unlike FORB's majority-bit-vote scheme, which only makes
   * sense for binary descriptors).
   * @param descriptors
   * @param mean mean descriptor
   */
  static void meanValue(const std::vector<pDescriptor> &descriptors,
    TDescriptor &mean);

  /**
   * Calculates the (squared) L2 distance between two descriptors. Squared
   * (not sqrt'd) deliberately: DBoW2's k-median/k-means clustering and
   * scoring only ever compare distances against each other, never against
   * an absolute threshold, so the monotonic squared form avoids N sqrt()
   * calls per comparison during vocabulary training (the actual hot loop)
   * for zero behavioral difference.
   * @param a
   * @param b
   * @return squared L2 distance
   */
  static double distance(const TDescriptor &a, const TDescriptor &b);

  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const TDescriptor &a);

  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(TDescriptor &a, const std::string &s);

  /**
   * Returns a mat with the descriptors in float format (already float
   * here, so this is just a vertical stack into one Nx128 matrix).
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
  static void toMat32F(const std::vector<TDescriptor> &descriptors,
    cv::Mat &mat);

};

} // namespace DBoW2

#endif
