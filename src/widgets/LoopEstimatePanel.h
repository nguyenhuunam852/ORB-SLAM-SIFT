#pragma once

#include <QWidget>

#include "vision/LoopEstimateTypes.h"

class QLabel;
class QScrollArea;
class QHBoxLayout;

// Bottom panel showing the latest background loop-closure re-estimate (see
// SlamWorker::loopClosureDetected(), computeLoopEstimate() in
// LoopEstimator.h). Purely a display -- never feeds back into live
// tracking/the map view, by construction (it only ever receives a
// LoopEstimateResult value, never a reference into live state). Below the
// latest-result summary labels, a horizontally-scrollable strip queues one
// LoopMapThumbnail per resolved loop, left to right in resolution order --
// every loop is queued, no filtering/dedup (an earlier overlap-dedup
// experiment was tried and explicitly reverted -- "just put them all in
// queue, no more ROI"). Each thumbnail still applies its own ROI zoom (fit
// to ground truth + trajectory, peak-error point ringed in red; see
// LoopMapThumbnail).
class LoopEstimatePanel : public QWidget
{
    Q_OBJECT

public:
    explicit LoopEstimatePanel(QWidget *parent = nullptr);

public slots:
    // Called once a background estimate finishes (see MainWindow's
    // QtConcurrent/QFutureWatcher wiring). Also used to show "running..."
    // immediately when a new loop closure is detected, before the
    // background result comes back.
    void showResult(const LoopEstimateResult &result);
    void showRunning(int oldFrameIndex, int newFrameIndex);

private:
    QLabel *m_statusLabel = nullptr;
    QLabel *m_enrichmentLabel = nullptr;
    QLabel *m_baLabel = nullptr;
    QLabel *m_ateLabel = nullptr;

    QScrollArea *m_mapScrollArea = nullptr;
    QHBoxLayout *m_mapStripLayout = nullptr; // thumbnails inserted before the trailing stretch
};
