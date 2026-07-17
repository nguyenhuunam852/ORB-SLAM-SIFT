#include "MainWindow.h"

#include <cstdio>

#include <QDir>
#include <QFileInfo>
#include <QMessageBox>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSplitter>
#include <QStatusBar>
#include <QThread>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrent>

#include "vision/LoopEstimator.h"
#include "vision/SlamWorker.h"
#include "widgets/ControlPanel.h"
#include "widgets/LoopEstimatePanel.h"
#include "widgets/MapView.h"
#include "widgets/VideoView.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("SIFT / vSLAM Viewer"));

    buildUi();

    m_workerThread = new QThread(this);
    m_worker = new SlamWorker();
    m_worker->moveToThread(m_workerThread);
    connect(m_workerThread, &QThread::finished, m_worker, &QObject::deleteLater);
    m_workerThread->start();

    m_loopEstimateWatcher = new QFutureWatcher<LoopEstimateResult>(this);
    connect(m_loopEstimateWatcher, &QFutureWatcher<LoopEstimateResult>::finished, this,
            &MainWindow::handleLoopEstimateFinished);

    wireSignals();

    statusBar()->showMessage(QStringLiteral("Ready. Open a video file or camera to begin."));
}

MainWindow::~MainWindow()
{
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait();
    }
}

void MainWindow::buildUi()
{
    m_controlPanel = new ControlPanel(this);

    auto *scrollArea = new QScrollArea(this);
    scrollArea->setWidget(m_controlPanel);
    scrollArea->setWidgetResizable(true);
    scrollArea->setMinimumWidth(360);
    scrollArea->setMaximumWidth(520);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    m_videoView = new VideoView(this);
    m_mapView = new MapView(this);

    auto *splitter = new QSplitter(Qt::Horizontal, this);
    splitter->addWidget(scrollArea);
    splitter->addWidget(m_videoView);
    splitter->addWidget(m_mapView);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 3);
    splitter->setStretchFactor(2, 2);
    splitter->setSizes({400, 900, 420});

    m_loopEstimatePanel = new LoopEstimatePanel(this);

    auto *central = new QWidget(this);
    auto *centralLayout = new QVBoxLayout(central);
    centralLayout->setContentsMargins(0, 0, 0, 0);
    centralLayout->addWidget(splitter, 1);
    centralLayout->addWidget(m_loopEstimatePanel, 0);

    setCentralWidget(central);
    resize(1600, 940);
}

void MainWindow::wireSignals()
{
    // Left panel -> worker thread (auto-queued: connection crosses threads).
    connect(m_controlPanel, &ControlPanel::openVideoFileSelected, m_worker, &SlamWorker::openVideoFile);
    connect(m_controlPanel, &ControlPanel::openVideoFileSelected, this, &MainWindow::tryAutoLoadGroundTruth);
    connect(m_controlPanel, &ControlPanel::openCameraRequested, m_worker, &SlamWorker::openCamera);
    connect(m_controlPanel, &ControlPanel::startRequested, m_worker, &SlamWorker::start);
    connect(m_controlPanel, &ControlPanel::stopRequested, m_worker, &SlamWorker::stop);
    connect(m_controlPanel, &ControlPanel::resetRequested, m_worker, &SlamWorker::reset);
    connect(m_controlPanel, &ControlPanel::intrinsicsChanged, m_worker, &SlamWorker::setIntrinsics);
    connect(m_controlPanel, &ControlPanel::siftSettingsChanged, m_worker, &SlamWorker::setSiftSettings);
    connect(m_controlPanel, &ControlPanel::pnpSettingsChanged, m_worker, &SlamWorker::setPnpSettings);
    connect(m_controlPanel, &ControlPanel::detectorTypeChanged, m_worker, &SlamWorker::setDetectorType);
    connect(m_controlPanel, &ControlPanel::orbSettingsChanged, m_worker, &SlamWorker::setOrbSettings);
    connect(m_controlPanel, &ControlPanel::orbSlam3EnabledChanged, m_worker, &SlamWorker::setOrbSlam3Enabled);
    connect(m_controlPanel, &ControlPanel::orbSlam3EnabledChanged, m_mapView, &MapView::setContinuousAlignmentEnabled);

    // OXTS/IMU controls (KITTI sequence 00 only -- see
    // SlamWorker::autoLoadKittiExtras()). useEightPointChanged maps a bool
    // checkbox onto the estimator enum; m_worker is passed as the context
    // object so the lambda runs queued on the worker thread, same as any
    // other cross-thread connection here.
    connect(m_controlPanel, &ControlPanel::useEightPointChanged, m_worker, [this](bool useEightPoint) {
        m_worker->setTwoViewEstimator(useEightPoint ? SlamWorker::TwoViewEstimator::EightPointLegacy
                                                     : SlamWorker::TwoViewEstimator::FivePoint);
    });
    connect(m_controlPanel, &ControlPanel::oxtsEnabledChanged, m_worker, &SlamWorker::setOxtsEnabled);
    connect(m_controlPanel, &ControlPanel::imuEnabledChanged, m_worker, &SlamWorker::setImuEnabled);
    connect(m_controlPanel, &ControlPanel::oxtsDirSelected, m_worker, &SlamWorker::loadOxtsDir);
    connect(m_controlPanel, &ControlPanel::imuDirsSelected, m_worker, &SlamWorker::loadImuDirs);
    connect(m_worker, &SlamWorker::oxtsAvailabilityChanged, m_controlPanel, &ControlPanel::setOxtsAvailable);
    connect(m_worker, &SlamWorker::imuAvailabilityChanged, m_controlPanel, &ControlPanel::setImuAvailable);
    connect(m_controlPanel, &ControlPanel::groundPlaneEnabledChanged, m_worker, &SlamWorker::setGroundPlaneEnabled);
    connect(m_controlPanel, &ControlPanel::groundTruthOverlayEnabledChanged, m_worker,
            &SlamWorker::setGroundTruthOverlayEnabled);
    connect(m_controlPanel, &ControlPanel::groundTruthOverlayOffsetChanged, m_worker,
            &SlamWorker::setGroundTruthOverlayOffset);
    connect(m_controlPanel, &ControlPanel::oldStreetOverlayOffsetChanged, m_worker,
            &SlamWorker::setOldStreetOverlayOffset);
    connect(m_controlPanel, &ControlPanel::loopBundleAdjustmentEnabledChanged, m_worker,
            &SlamWorker::setLoopBundleAdjustmentEnabled);
    connect(m_controlPanel, &ControlPanel::pnpFullInlierRefineEnabledChanged, m_worker,
            &SlamWorker::setPnpFullInlierRefineEnabled);
    connect(m_controlPanel, &ControlPanel::oxtsImuInPnpEnabledChanged, m_worker,
            &SlamWorker::setOxtsImuInPnpEnabled);

    // Best-effort auto-load of KITTI seq00's OXTS/IMU data right after any
    // video is opened (no-op if the known paths aren't present -- see
    // autoLoadKittiExtras()'s doc comment). Connecting the worker's own
    // signal back to its own slot keeps this entirely on the worker thread.
    connect(m_worker, &SlamWorker::sourceOpened, m_worker, &SlamWorker::autoLoadKittiExtras);

    // Show the first frame (with the ground-truth overlay, if loaded)
    // immediately after opening, before Start is ever pressed -- lets the
    // GT overlay offset be tuned against a static frame. Connected after
    // autoLoadKittiExtras() above so ground truth is already loaded by the
    // time this runs (same-thread queued connections fire in connection
    // order).
    connect(m_worker, &SlamWorker::sourceOpened, m_worker, &SlamWorker::previewFirstFrame);

    // Worker thread -> UI (auto-queued).
    connect(m_worker, &SlamWorker::frameReady, m_videoView, &VideoView::setFrame);
    connect(m_worker, &SlamWorker::mapUpdated, m_mapView, &MapView::setMapData);
    connect(m_worker, &SlamWorker::trackingStateChanged, m_controlPanel, &ControlPanel::setTrackingState);
    connect(m_worker, &SlamWorker::statsUpdated, this, [this](const QString &text) {
        statusBar()->showMessage(text);
    });
    connect(m_worker, &SlamWorker::sourceOpened, this, &MainWindow::handleSourceOpened);
    connect(m_worker, &SlamWorker::loopClosureDetected, this, &MainWindow::handleLoopClosureDetected);
}

void MainWindow::handleSourceOpened(bool ok, const QString &message)
{
    statusBar()->showMessage(message);
    if (!ok) {
        QMessageBox::warning(this, QStringLiteral("Video Source"), message);
    }
}

void MainWindow::tryAutoLoadGroundTruth(const QString &videoPath)
{
    // Best-effort: look for a KITTI-style poses.txt near the opened video,
    // keyed off the trailing digits in its filename (e.g. "kitti_00.mp4"
    // -> sequence "00"). Tries this dataset's actual layout (poses live
    // under a sibling "dataset/poses/" directory next to "video_samples/")
    // plus a few more common conventions; silently does nothing if none
    // match, since ground truth is optional and this must never block
    // opening the video itself.
    const QFileInfo videoInfo(videoPath);
    const QDir videoDir = videoInfo.dir();
    const QString baseName = videoInfo.completeBaseName();

    const QRegularExpression trailingDigits(QStringLiteral("(\\d+)$"));
    const QRegularExpressionMatch match = trailingDigits.match(baseName);
    if (!match.hasMatch())
        return;
    const QString seq = match.captured(1);

    const QStringList candidates = {
        videoDir.filePath(seq + QStringLiteral(".txt")),
        videoDir.filePath(QStringLiteral("poses/") + seq + QStringLiteral(".txt")),
        videoDir.filePath(QStringLiteral("../dataset/poses/") + seq + QStringLiteral(".txt")),
        videoDir.filePath(QStringLiteral("../poses/") + seq + QStringLiteral(".txt")),
    };
    for (const QString &candidate : candidates) {
        const QString cleaned = QDir::cleanPath(candidate);
        if (QFileInfo::exists(cleaned) && m_mapView->loadGroundTruthFile(cleaned)) {
            statusBar()->showMessage(QStringLiteral("Loaded ground truth: %1").arg(cleaned), 5000);
            return;
        }
    }
}

void MainWindow::handleLoopClosureDetected(LoopEstimateSnapshot snapshot)
{
    m_loopEstimatePanel->showRunning(snapshot.oldFrameIndex, snapshot.newFrameIndex);

    // Diagnostic (see MainWindow.h's m_loopEstimateTimer comment): this path
    // previously had zero logging, so a background estimate that hangs or
    // just takes a very long time on a large window was indistinguishable
    // from a crash from the GUI's perspective.
    std::fprintf(stderr, "[mainwindow] loop-estimate frame %d<->%d: %zu keyframes in window, starting\n",
                 snapshot.oldFrameIndex, snapshot.newFrameIndex, snapshot.keyframes.size());
    std::fflush(stderr);

    if (m_loopEstimateBusy) {
        // A previous estimate is still running (possible on an unthrottled
        // run where loop closures fire faster than a full re-estimate
        // takes) -- keep only the latest pending snapshot, drop any older
        // one rather than queuing unboundedly or running them in parallel.
        std::fprintf(stderr, "[mainwindow] loop-estimate frame %d<->%d: queued, previous one still running\n",
                     snapshot.oldFrameIndex, snapshot.newFrameIndex);
        std::fflush(stderr);
        m_pendingLoopEstimateSnapshot = std::move(snapshot);
        m_hasPendingLoopEstimateSnapshot = true;
        return;
    }

    m_loopEstimateBusy = true;
    m_loopEstimateTimer.start();
    m_loopEstimateWatcher->setFuture(QtConcurrent::run(&computeLoopEstimate, std::move(snapshot)));
}

void MainWindow::handleLoopEstimateFinished()
{
    const LoopEstimateResult result = m_loopEstimateWatcher->result();
    std::fprintf(stderr, "[mainwindow] loop-estimate frame %d<->%d: finished in %lld ms (%s)\n",
                 result.oldFrameIndex, result.newFrameIndex, m_loopEstimateTimer.elapsed(),
                 result.ok ? "ok" : qPrintable(result.message));
    std::fflush(stderr);
    m_loopEstimatePanel->showResult(result);

    if (m_hasPendingLoopEstimateSnapshot) {
        m_hasPendingLoopEstimateSnapshot = false;
        LoopEstimateSnapshot next = std::move(m_pendingLoopEstimateSnapshot);
        m_loopEstimatePanel->showRunning(next.oldFrameIndex, next.newFrameIndex);
        std::fprintf(stderr, "[mainwindow] loop-estimate frame %d<->%d: %zu keyframes in window, starting "
                             "(was queued)\n",
                     next.oldFrameIndex, next.newFrameIndex, next.keyframes.size());
        std::fflush(stderr);
        m_loopEstimateTimer.start();
        m_loopEstimateWatcher->setFuture(QtConcurrent::run(&computeLoopEstimate, std::move(next)));
    } else {
        m_loopEstimateBusy = false;
    }
}
