/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <cmath>
#include <opencv2/opencv.hpp>


namespace ORB_SLAM3
{

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    // Reconfigures the live cv::SIFT detector's target feature count and
    // contrast threshold without touching nOctaveLayers or any of the
    // per-level scale/sigma arrays (nlevels-sized, and already relied on
    // elsewhere -- resizing them mid-sequence would be a much bigger,
    // riskier change). Cheap enough to call only when entering/leaving a
    // high-angular-velocity window, not every frame -- see Tracking.cc's
    // motion-model hook. See DEBUGGING.md for why this exists.
    void SetDynamicDensity(int nfeatures_, double contrastThreshold_);

    int inline GetLevels(){
        return nlevels;}

    // SIFT's own per-octave sub-layer count (see ORBextractor.cc's doc
    // comments on flatLevel()/the constructor) -- flat levels
    // [0, GetOctaveLayers()) are the finest octave's layers. Used by
    // ORBmatcher::SearchForInitialization() to admit same-octave,
    // cross-layer matches instead of requiring an exact flat-level-0 match;
    // see DEBUGGING.md's BA-sigma-weighting investigation for why that's
    // now safe to do.
    int inline GetOctaveLayers(){
        return nOctaveLayers;}

    // Part 56 fix (see DEBUGGING.md): this used to return the raw
    // constructor-argument `scaleFactor` (the settings file's
    // ORBextractor.scaleFactor=1.2), which is otherwise UNUSED by this SIFT
    // reimplementation (the constructor's own comment already says
    // _scaleFactor is "accepted for signature compatibility but not used").
    // But Frame/KeyFrame's mfScaleFactor/mfLogScaleFactor -- fed straight
    // from this getter -- ARE real consumers: MapPoint::PredictScale()
    // divides by mfLogScaleFactor to convert a real-world distance ratio
    // into a flat-level array index, and LocalMapping::CreateNewMapPoints()
    // multiplies mfScaleFactor into a scale-consistency tolerance band.
    // Both need the TRUE per-flat-level scale step this SIFT
    // reimplementation actually uses (mvScaleFactor[lvl] jumps by 2.0x per
    // *octave*, i.e. per nOctaveLayers flat levels -- see the constructor),
    // not the vestigial 1.2 config value. Using 1.2 made PredictScale's
    // denominator ~2.1x too large (log(1.2)=0.182 vs the true per-flat-level
    // log(2^(1/nOctaveLayers))=log(2)/nOctaveLayers=0.087 for nOctaveLayers=8),
    // so every predicted flat level came out roughly half of where the point
    // actually was -- systematically mis-centering SearchByProjection's
    // search window regardless of how wide that window is. Measured to be
    // the dominant cause of both empty_window (58.3%) and dist_reject
    // (75.9% conditional) in the sbp-diag breakdown.
    float inline GetScaleFactor(){
        return std::pow(2.0f, 1.0f / static_cast<float>(nOctaveLayers));}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels; // now the FLAT scale-array size (kMaxOctaveSpan * nOctaveLayers), matching every
                 // consumer's "nlevels = valid mvScaleFactor/mvLevelSigma2 array length" assumption --
                 // see the SIFT reimplementation's doc comment in ORBextractor.cc
    int iniThFAST;
    int minThFAST;

    // SIFT reimplementation (see ORBextractor.cc's constructor/operator() doc
    // comments): the constructor's own _nlevels parameter is reinterpreted as
    // nOctaveLayers (SIFT's own per-octave sub-layer count) instead of a
    // pyramid level count.
    int nOctaveLayers;
    cv::Ptr<cv::SIFT> mSift;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

} //namespace ORB_SLAM

#endif

