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


#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>

#include "KeyFrame.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "Map.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>

#include<mutex>


namespace ORB_SLAM3
{

class KeyFrame;
class Frame;
class Map;


// Candidate search internals rewritten for VLAD (see VladVocabulary.h and
// DEBUGGING.md's ORB->SIFT swap session): DBoW2's inverted file
// (mvInvertedFile, word-id -> KeyFrame list) has no VLAD equivalent -- VLAD
// has no discrete "word" to bucket by -- so mvDatabase is a flat list of
// every added KeyFrame, and every DetectXxxCandidates() method scores the
// query against every entry directly via VladVocabulary::score() (a cheap
// dot product) instead of a word-sharing prefilter. Confirmed via
// full-tree grep that only DetectNBestCandidates() (LoopClosing.cc) and
// DetectRelocalizationCandidates() (Tracking.cc) are ever actually called
// in this project's monocular path; DetectLoopCandidates() (already marked
// DEPRECATED upstream), DetectCandidates(), and DetectBestCandidates() are
// kept working (same brute-force-scoring shape) for API completeness, not
// because anything here calls them. All public method signatures are
// otherwise unchanged from the original.
class KeyFrameDatabase
{
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        // mvBackupInvertedFileId (the DBoW2-era word-id-based save format)
        // is gone along with mvInvertedFile -- Atlas save/load was already
        // confirmed dormant in this project's actual usage (neither
        // existing caller sets the settings keys that trigger it), so this
        // is intentionally a no-op rather than inventing a new persisted
        // format for something nothing here exercises.
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KeyFrameDatabase(){}
    KeyFrameDatabase(const ORBVocabulary &voc);

    void add(KeyFrame* pKF);

    void erase(KeyFrame* pKF);

    void clear();
    void clearMap(Map* pMap);

    // Loop Detection(DEPRECATED)
    std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);

    // Loop and Merge Detection
    void DetectCandidates(KeyFrame* pKF, float minScore,vector<KeyFrame*>& vpLoopCand, vector<KeyFrame*>& vpMergeCand);
    void DetectBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nMinWords);
    void DetectNBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nNumCandidates);

    // Relocalization
    std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F, Map* pMap);

    void PreSave();
    void PostLoad(map<long unsigned int, KeyFrame*> mpKFid);
    void SetORBVocabulary(ORBVocabulary* pORBVoc);

protected:

   // Associated vocabulary
   const ORBVocabulary* mpVoc;

   // Flat list of every added KeyFrame -- replaces mvInvertedFile. Brute-
   // force linear scan is fine at KITTI-per-sequence scale (hundreds of
   // keyframes, 959 for all of seq00 per DEBUGGING.md); revisit only if
   // profiling (see the plan's Stage 5) shows a real problem.
   std::vector<KeyFrame*> mvDatabase;

   // Mutex
   std::mutex mMutex;

};

} //namespace ORB_SLAM

#endif
