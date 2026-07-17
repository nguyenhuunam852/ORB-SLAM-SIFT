#pragma once

#include <QElapsedTimer>
#include <QFutureWatcher>
#include <QMainWindow>

#include "vision/LoopEstimateTypes.h"

class QThread;
class ControlPanel;
class VideoView;
class MapView;
class SlamWorker;
class LoopEstimatePanel;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void handleSourceOpened(bool ok, const QString &message);
    void tryAutoLoadGroundTruth(const QString &videoPath);

    // See loopClosureDetected()'s doc comment (SlamWorker.h): dispatches
    // the snapshot to a background thread-pool thread via QtConcurrent,
    // queuing it instead if a previous estimate is still running (loop
    // closures on an unthrottled run can fire faster than a full estimate
    // takes to compute). handleLoopEstimateFinished() reads the result off
    // m_loopEstimateWatcher and starts the next queued snapshot, if any.
    void handleLoopClosureDetected(LoopEstimateSnapshot snapshot);
    void handleLoopEstimateFinished();

private:
    void buildUi();
    void wireSignals();

    ControlPanel *m_controlPanel = nullptr;
    VideoView *m_videoView = nullptr;
    MapView *m_mapView = nullptr;
    LoopEstimatePanel *m_loopEstimatePanel = nullptr;

    QThread *m_workerThread = nullptr;
    SlamWorker *m_worker = nullptr;

    QFutureWatcher<LoopEstimateResult> *m_loopEstimateWatcher = nullptr;
    bool m_loopEstimateBusy = false;
    bool m_hasPendingLoopEstimateSnapshot = false;
    LoopEstimateSnapshot m_pendingLoopEstimateSnapshot;

    // Diagnostic only (stderr logging in handleLoopClosureDetected()/
    // handleLoopEstimateFinished()) -- added to investigate a reported
    // stuck/"pending" LoopEstimatePanel (frame ~330<->400): with no prior
    // logging at all in this path, a background computeLoopEstimate() call
    // that hangs or just takes a very long time (large window -- see
    // DEBUGGING.md Session 7's 900s-budget note) was indistinguishable from
    // one that crashed, from the GUI's perspective.
    QElapsedTimer m_loopEstimateTimer;
};
