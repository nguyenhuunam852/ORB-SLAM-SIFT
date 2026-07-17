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

// Rewritten for VLAD (see KeyFrameDatabase.h's class doc comment and
// VladVocabulary.h) -- brute-force score-every-candidate instead of DBoW2's
// inverted-file word-sharing prefilter. Every public method keeps its
// original signature.

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM3
{

namespace {
bool compFirst(const pair<float, KeyFrame*> & a, const pair<float, KeyFrame*> & b)
{
    return a.first > b.first;
}
}

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
}


void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);
    mvDatabase.push_back(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(vector<KeyFrame*>::iterator vit=mvDatabase.begin(), vend=mvDatabase.end(); vit!=vend; vit++)
    {
        if(*vit==pKF)
        {
            mvDatabase.erase(vit);
            break;
        }
    }
}

void KeyFrameDatabase::clear()
{
    unique_lock<mutex> lock(mMutex);
    mvDatabase.clear();
}

void KeyFrameDatabase::clearMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutex);

    for(vector<KeyFrame*>::iterator vit=mvDatabase.begin(), vend=mvDatabase.end(); vit!=vend;)
    {
        if(pMap == (*vit)->GetMap())
        {
            // Dont delete the KF because the class Map cleans all the KFs when it is destroyed
            vit = mvDatabase.erase(vit);
        }
        else
        {
            ++vit;
        }
    }
}

vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    const cv::Mat queryVlad = pKF->mVladVec;

    list<pair<float,KeyFrame*> > lScoreAndMatch;
    {
        unique_lock<mutex> lock(mMutex);
        for(KeyFrame* pKFi : mvDatabase)
        {
            if(pKFi==pKF || pKFi->GetMap()!=pKF->GetMap() || spConnectedKeyFrames.count(pKFi))
                continue;

            float si = mpVoc->score(queryVlad, pKFi->mVladVec);
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        for(KeyFrame* pKF2 : vpNeighs)
        {
            if(pKF2==pKF || pKF2->GetMap()!=pKF->GetMap() || spConnectedKeyFrames.count(pKF2))
                continue;

            float si2 = mpVoc->score(queryVlad, pKF2->mVladVec);
            accScore += si2;
            if(si2>bestScore)
            {
                pBestKF = pKF2;
                bestScore = si2;
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore = accScore;
    }

    float minScoreToRetain = 0.75f*bestAccScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}

void KeyFrameDatabase::DetectCandidates(KeyFrame* pKF, float minScore,vector<KeyFrame*>& vpLoopCand, vector<KeyFrame*>& vpMergeCand)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
    const cv::Mat queryVlad = pKF->mVladVec;

    list<pair<float,KeyFrame*> > lScoreAndMatchLoop, lScoreAndMatchMerge;
    {
        unique_lock<mutex> lock(mMutex);
        for(KeyFrame* pKFi : mvDatabase)
        {
            if(pKFi==pKF || spConnectedKeyFrames.count(pKFi))
                continue;

            float si = mpVoc->score(queryVlad, pKFi->mVladVec);
            if(si<minScore)
                continue;

            if(pKFi->GetMap()==pKF->GetMap())
                lScoreAndMatchLoop.push_back(make_pair(si,pKFi));
            else if(!pKFi->GetMap()->IsBad())
                lScoreAndMatchMerge.push_back(make_pair(si,pKFi));
        }
    }

    if(!lScoreAndMatchLoop.empty())
    {
        list<pair<float,KeyFrame*> > lAccScoreAndMatch;
        float bestAccScore = minScore;

        for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatchLoop.begin(), itend=lScoreAndMatchLoop.end(); it!=itend; it++)
        {
            KeyFrame* pKFi = it->second;
            vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float bestScore = it->first;
            float accScore = it->first;
            KeyFrame* pBestKF = pKFi;
            for(KeyFrame* pKF2 : vpNeighs)
            {
                if(pKF2==pKF || pKF2->GetMap()!=pKF->GetMap() || spConnectedKeyFrames.count(pKF2))
                    continue;
                float si2 = mpVoc->score(queryVlad, pKF2->mVladVec);
                accScore += si2;
                if(si2>bestScore)
                {
                    pBestKF = pKF2;
                    bestScore = si2;
                }
            }

            lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
            if(accScore>bestAccScore)
                bestAccScore = accScore;
        }

        float minScoreToRetain = 0.75f*bestAccScore;
        set<KeyFrame*> spAlreadyAddedKF;
        vpLoopCand.reserve(lAccScoreAndMatch.size());
        for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
        {
            if(it->first>minScoreToRetain)
            {
                KeyFrame* pKFi = it->second;
                if(!spAlreadyAddedKF.count(pKFi))
                {
                    vpLoopCand.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }
    }

    if(!lScoreAndMatchMerge.empty())
    {
        list<pair<float,KeyFrame*> > lAccScoreAndMatch;
        float bestAccScore = minScore;

        for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatchMerge.begin(), itend=lScoreAndMatchMerge.end(); it!=itend; it++)
        {
            KeyFrame* pKFi = it->second;
            vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

            float bestScore = it->first;
            float accScore = it->first;
            KeyFrame* pBestKF = pKFi;
            for(KeyFrame* pKF2 : vpNeighs)
            {
                if(pKF2==pKF || pKF2->GetMap()==pKF->GetMap() || spConnectedKeyFrames.count(pKF2))
                    continue;
                float si2 = mpVoc->score(queryVlad, pKF2->mVladVec);
                accScore += si2;
                if(si2>bestScore)
                {
                    pBestKF = pKF2;
                    bestScore = si2;
                }
            }

            lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
            if(accScore>bestAccScore)
                bestAccScore = accScore;
        }

        float minScoreToRetain = 0.75f*bestAccScore;
        set<KeyFrame*> spAlreadyAddedKF;
        vpMergeCand.reserve(lAccScoreAndMatch.size());
        for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
        {
            if(it->first>minScoreToRetain)
            {
                KeyFrame* pKFi = it->second;
                if(!spAlreadyAddedKF.count(pKFi))
                {
                    vpMergeCand.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }
    }
}

void KeyFrameDatabase::DetectBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nMinWords)
{
    // nMinWords (a DBoW2 word-count floor) has no VLAD equivalent -- see
    // this class's own doc comment -- so it's accepted for signature
    // compatibility but unused; every candidate is scored regardless.
    (void)nMinWords;

    set<KeyFrame*> spConnectedKF;
    const cv::Mat queryVlad = pKF->mVladVec;

    list<pair<float,KeyFrame*> > lScoreAndMatch;
    {
        unique_lock<mutex> lock(mMutex);
        spConnectedKF = pKF->GetConnectedKeyFrames();
        for(KeyFrame* pKFi : mvDatabase)
        {
            if(pKFi==pKF || spConnectedKF.count(pKFi))
                continue;
            float si = mpVoc->score(queryVlad, pKFi->mVladVec);
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return;

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(KeyFrame* pKF2 : vpNeighs)
        {
            if(pKF2==pKF || spConnectedKF.count(pKF2))
                continue;
            float si2 = mpVoc->score(queryVlad, pKF2->mVladVec);
            accScore += si2;
            if(si2>bestScore)
            {
                pBestKF = pKF2;
                bestScore = si2;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore = accScore;
    }

    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vpLoopCand.reserve(lAccScoreAndMatch.size());
    vpMergeCand.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                if(pKF->GetMap() == pKFi->GetMap())
                    vpLoopCand.push_back(pKFi);
                else
                    vpMergeCand.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }
}

void KeyFrameDatabase::DetectNBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nNumCandidates)
{
    set<KeyFrame*> spConnectedKF;
    const cv::Mat queryVlad = pKF->mVladVec;

    list<pair<float,KeyFrame*> > lScoreAndMatch;
    {
        unique_lock<mutex> lock(mMutex);
        spConnectedKF = pKF->GetConnectedKeyFrames();
        for(KeyFrame* pKFi : mvDatabase)
        {
            if(pKFi==pKF || spConnectedKF.count(pKFi))
                continue;
            float si = mpVoc->score(queryVlad, pKFi->mVladVec);
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return;

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;

    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(KeyFrame* pKF2 : vpNeighs)
        {
            if(pKF2==pKF || spConnectedKF.count(pKF2))
                continue;
            float si2 = mpVoc->score(queryVlad, pKF2->mVladVec);
            accScore += si2;
            if(si2>bestScore)
            {
                pBestKF = pKF2;
                bestScore = si2;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
    }

    lAccScoreAndMatch.sort(compFirst);

    vpLoopCand.reserve(nNumCandidates);
    vpMergeCand.reserve(nNumCandidates);
    set<KeyFrame*> spAlreadyAddedKF;
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(static_cast<int>(vpLoopCand.size())>=nNumCandidates && static_cast<int>(vpMergeCand.size())>=nNumCandidates)
            break;

        KeyFrame* pKFi = it->second;
        if(pKFi->isBad())
            continue;

        if(spAlreadyAddedKF.count(pKFi))
            continue;

        if(pKF->GetMap() == pKFi->GetMap() && static_cast<int>(vpLoopCand.size()) < nNumCandidates)
        {
            vpLoopCand.push_back(pKFi);
            spAlreadyAddedKF.insert(pKFi);
        }
        else if(pKF->GetMap() != pKFi->GetMap() && static_cast<int>(vpMergeCand.size()) < nNumCandidates && !pKFi->GetMap()->IsBad())
        {
            vpMergeCand.push_back(pKFi);
            spAlreadyAddedKF.insert(pKFi);
        }
    }
}

vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F, Map* pMap)
{
    const cv::Mat queryVlad = F->mVladVec;

    list<pair<float,KeyFrame*> > lScoreAndMatch;
    {
        unique_lock<mutex> lock(mMutex);
        for(KeyFrame* pKFi : mvDatabase)
        {
            float si = mpVoc->score(queryVlad, pKFi->mVladVec);
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(KeyFrame* pKF2 : vpNeighs)
        {
            float si2 = mpVoc->score(queryVlad, pKF2->mVladVec);
            accScore += si2;
            if(si2>bestScore)
            {
                pBestKF = pKF2;
                bestScore = si2;
            }
        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore = accScore;
    }

    // Capped at kMaxRelocCandidates (sorted by score, best first) -- unlike
    // DBoW2's word-sharing prefilter, VLAD's brute-force scoring above has
    // no built-in bound on how many keyframes can score above
    // minScoreToRetain, and each candidate returned here triggers an
    // expensive brute-force ORBmatcher::SearchByBoW() call in
    // Tracking::Relocalization() (called every frame while tracking is
    // lost). Confirmed as a real, not just theoretical, cost: an
    // uncapped run stalled for 20+ minutes with no progress once tracking
    // got stuck in a relocalization loop against a large keyframe database
    // -- see DEBUGGING.md's ORB->SIFT swap session. DetectNBestCandidates()
    // (the loop-closing path) already had an equivalent cap via its
    // nNumCandidates parameter; this one didn't.
    constexpr int kMaxRelocCandidates = 20;
    lAccScoreAndMatch.sort(compFirst);

    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(std::min<size_t>(lAccScoreAndMatch.size(), kMaxRelocCandidates));
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(static_cast<int>(vpRelocCandidates.size()) >= kMaxRelocCandidates)
            break;

        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if(pKFi->GetMap() != pMap)
                continue;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

void KeyFrameDatabase::SetORBVocabulary(ORBVocabulary* pORBVoc)
{
    mpVoc = pORBVoc;
}

} //namespace ORB_SLAM
