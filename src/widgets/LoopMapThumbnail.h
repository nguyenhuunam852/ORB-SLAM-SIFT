#pragma once

#include <QWidget>

#include "vision/LoopEstimateTypes.h"

// One small top-down map per resolved background loop-closure re-estimate,
// queued left-to-right in LoopEstimatePanel's scroll strip as each loop
// resolves (order == resolution order, i.e. loop 1..N). Painting style
// mirrors MapView's (dark background, green ground truth, orange estimate)
// for visual consistency, but this widget only ever draws data that's
// already aligned into ground truth's frame (see
// LoopEstimateResult::alignedLandmarks/alignedTrajectory/alignedGroundTruth,
// computed once in computeLoopEstimate()) -- no alignment math here.
class LoopMapThumbnail : public QWidget
{
    Q_OBJECT

public:
    explicit LoopMapThumbnail(const LoopEstimateResult &result, QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    LoopEstimateResult m_result;
};
