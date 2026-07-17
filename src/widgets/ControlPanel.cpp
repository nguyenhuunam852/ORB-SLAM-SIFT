#include "ControlPanel.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

ControlPanel::ControlPanel(QWidget *parent)
    : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setAlignment(Qt::AlignTop);

    layout->addWidget(buildSourceGroup());
    layout->addWidget(buildOrbSlam3Group());
    layout->addWidget(buildIntrinsicsGroup());
    m_siftGroup = buildSiftGroup();
    layout->addWidget(m_siftGroup);
    m_pnpGroup = buildPnpGroup();
    layout->addWidget(m_pnpGroup);
    m_oxtsImuGroup = buildOxtsImuGroup();
    layout->addWidget(m_oxtsImuGroup);

    m_stateLabel = new QLabel(QStringLiteral("State: Idle"), this);
    m_stateLabel->setStyleSheet(QStringLiteral("font-weight: bold; padding: 4px;"));
    layout->addWidget(m_stateLabel);

    layout->addStretch(1);
}

QWidget *ControlPanel::buildSourceGroup()
{
    auto *group = new QGroupBox(QStringLiteral("Video Source"), this);
    auto *form = new QFormLayout(group);

    auto *openFileBtn = new QPushButton(QStringLiteral("Open Video File..."), group);
    connect(openFileBtn, &QPushButton::clicked, this, &ControlPanel::browseForVideoFile);
    form->addRow(openFileBtn);

    m_cameraIndex = new QSpinBox(group);
    m_cameraIndex->setRange(0, 16);
    auto *openCameraBtn = new QPushButton(QStringLiteral("Open Camera"), group);
    connect(openCameraBtn, &QPushButton::clicked, this, [this]() {
        emit openCameraRequested(m_cameraIndex->value());
    });
    auto *cameraRow = new QWidget(group);
    auto *cameraRowLayout = new QHBoxLayout(cameraRow);
    cameraRowLayout->setContentsMargins(0, 0, 0, 0);
    cameraRowLayout->addWidget(m_cameraIndex);
    cameraRowLayout->addWidget(openCameraBtn);
    form->addRow(QStringLiteral("Camera index"), cameraRow);

    auto *buttonsRow = new QWidget(group);
    auto *buttonsLayout = new QHBoxLayout(buttonsRow);
    buttonsLayout->setContentsMargins(0, 0, 0, 0);
    auto *startBtn = new QPushButton(QStringLiteral("Start"), buttonsRow);
    auto *stopBtn = new QPushButton(QStringLiteral("Stop"), buttonsRow);
    auto *resetBtn = new QPushButton(QStringLiteral("Reset"), buttonsRow);
    connect(startBtn, &QPushButton::clicked, this, &ControlPanel::startRequested);
    connect(stopBtn, &QPushButton::clicked, this, &ControlPanel::stopRequested);
    connect(resetBtn, &QPushButton::clicked, this, &ControlPanel::resetRequested);
    buttonsLayout->addWidget(startBtn);
    buttonsLayout->addWidget(stopBtn);
    buttonsLayout->addWidget(resetBtn);
    form->addRow(buttonsRow);

    return group;
}

QWidget *ControlPanel::buildOrbSlam3Group()
{
    // Toggles SlamWorker over to the vendored, real ORB-SLAM3 algorithm
    // (third_party/ORB_SLAM3) instead of this project's own custom pipeline
    // -- see SlamWorker::setOrbSlam3Enabled(). Checking this disables (grays
    // out, doesn't clear) every widget in the Feature Detector / PnP /
    // OXTS-IMU groups below, since none of those settings apply to
    // ORB-SLAM3's own internal tracking -- per explicit request. Camera
    // intrinsics stay enabled: ORB-SLAM3 mode still reads fx/fy/cx/cy from
    // there to build its own settings YAML.
    auto *group = new QGroupBox(QStringLiteral("ORB-SLAM3 (vendored, real algorithm)"), this);
    auto *layout = new QVBoxLayout(group);

    m_orbSlam3Enabled = new QCheckBox(QStringLiteral("Use real ORB-SLAM3 (disables custom pipeline settings below)"), group);
    m_orbSlam3Enabled->setChecked(false);
    layout->addWidget(m_orbSlam3Enabled);

    connect(m_orbSlam3Enabled, &QCheckBox::toggled, this, [this](bool enabled) {
        m_siftGroup->setEnabled(!enabled);
        m_pnpGroup->setEnabled(!enabled);
        m_oxtsImuGroup->setEnabled(!enabled);
        emit orbSlam3EnabledChanged(enabled);

        // Keep the Feature Detector combo (m_detectorType, built in
        // buildSiftGroup(), called after this group -- safe since this only
        // runs post-construction, on a user/programmatic toggle) in sync:
        // checking this box directly (not via selecting "ORB" above) should
        // still show "ORB" so the two controls never disagree, and
        // unchecking it should fall back to "SIFT" so re-enabling the
        // custom pipeline doesn't silently leave it on the
        // known-inaccurate ORB path. Signals blocked to avoid re-entering
        // the combo's own handler, which would otherwise call back into
        // this lambda.
        if (m_detectorType) {
            const int wantType = static_cast<int>(enabled ? feature_detector::DetectorType::Orb
                                                            : feature_detector::DetectorType::Sift);
            if (m_detectorType->currentData().toInt() != wantType) {
                const QSignalBlocker blocker(m_detectorType);
                m_detectorType->setCurrentIndex(m_detectorType->findData(wantType));
                applyDetectorRowVisibility(enabled);
            }
        }
    });

    return group;
}

void ControlPanel::applyDetectorRowVisibility(bool isOrb)
{
    const auto setRowVisible = [this](QWidget *field, bool visible) {
        if (QWidget *label = m_siftForm->labelForField(field))
            label->setVisible(visible);
        field->setVisible(visible);
    };
    setRowVisible(m_nOctaveLayers, !isOrb);
    setRowVisible(m_contrastThreshold, !isOrb);
    setRowVisible(m_edgeThreshold, !isOrb);
    setRowVisible(m_sigma, !isOrb);
    setRowVisible(m_orbScaleFactor, isOrb);
    setRowVisible(m_orbNLevels, isOrb);
}

QWidget *ControlPanel::buildIntrinsicsGroup()
{
    auto *group = new QGroupBox(QStringLiteral("Camera Intrinsic Matrix"), this);
    auto *form = new QFormLayout(group);

    const CameraIntrinsics defaults;

    m_fx = new QDoubleSpinBox(group);
    m_fx->setRange(1.0, 20000.0);
    m_fx->setDecimals(2);
    m_fx->setValue(defaults.fx);
    form->addRow(QStringLiteral("fx"), m_fx);

    m_fy = new QDoubleSpinBox(group);
    m_fy->setRange(1.0, 20000.0);
    m_fy->setDecimals(2);
    m_fy->setValue(defaults.fy);
    form->addRow(QStringLiteral("fy"), m_fy);

    m_cx = new QDoubleSpinBox(group);
    m_cx->setRange(0.0, 20000.0);
    m_cx->setDecimals(2);
    m_cx->setValue(defaults.cx);
    form->addRow(QStringLiteral("cx"), m_cx);

    m_cy = new QDoubleSpinBox(group);
    m_cy->setRange(0.0, 20000.0);
    m_cy->setDecimals(2);
    m_cy->setValue(defaults.cy);
    form->addRow(QStringLiteral("cy"), m_cy);

    for (auto *box : {m_fx, m_fy, m_cx, m_cy}) {
        connect(box, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &ControlPanel::emitIntrinsics);
    }

    return group;
}

QWidget *ControlPanel::buildSiftGroup()
{
    auto *group = new QGroupBox(QStringLiteral("Feature Detector"), this);
    auto *form = new QFormLayout(group);
    m_siftForm = form;

    const SiftSettings defaults;
    const feature_detector::OrbSettings orbDefaults;

    m_detectorType = new QComboBox(group);
    m_detectorType->addItem(QStringLiteral("SIFT"), static_cast<int>(feature_detector::DetectorType::Sift));
    m_detectorType->addItem(QStringLiteral("ORB"), static_cast<int>(feature_detector::DetectorType::Orb));
    form->addRow(QStringLiteral("Detector"), m_detectorType);

    m_nFeatures = new QSpinBox(group);
    m_nFeatures->setRange(0, 20000);
    m_nFeatures->setSpecialValueText(QStringLiteral("unlimited"));
    m_nFeatures->setValue(defaults.nFeatures);
    form->addRow(QStringLiteral("nFeatures"), m_nFeatures);

    m_nOctaveLayers = new QSpinBox(group);
    m_nOctaveLayers->setRange(1, 10);
    m_nOctaveLayers->setValue(defaults.nOctaveLayers);
    form->addRow(QStringLiteral("nOctaveLayers"), m_nOctaveLayers);

    m_contrastThreshold = new QDoubleSpinBox(group);
    m_contrastThreshold->setRange(0.0, 1.0);
    m_contrastThreshold->setSingleStep(0.005);
    m_contrastThreshold->setDecimals(4);
    m_contrastThreshold->setValue(defaults.contrastThreshold);
    form->addRow(QStringLiteral("contrastThreshold"), m_contrastThreshold);

    m_edgeThreshold = new QDoubleSpinBox(group);
    m_edgeThreshold->setRange(1.0, 100.0);
    m_edgeThreshold->setValue(defaults.edgeThreshold);
    form->addRow(QStringLiteral("edgeThreshold"), m_edgeThreshold);

    m_sigma = new QDoubleSpinBox(group);
    m_sigma->setRange(0.1, 10.0);
    m_sigma->setSingleStep(0.1);
    m_sigma->setDecimals(2);
    m_sigma->setValue(defaults.sigma);
    form->addRow(QStringLiteral("Gaussian sigma"), m_sigma);

    // ORB-only fields -- hidden by default (SIFT is the initial detector,
    // matching today's default behavior unchanged).
    m_orbScaleFactor = new QDoubleSpinBox(group);
    m_orbScaleFactor->setRange(1.01, 3.0);
    m_orbScaleFactor->setSingleStep(0.05);
    m_orbScaleFactor->setDecimals(2);
    m_orbScaleFactor->setValue(orbDefaults.scaleFactor);
    form->addRow(QStringLiteral("ORB scaleFactor"), m_orbScaleFactor);

    m_orbNLevels = new QSpinBox(group);
    m_orbNLevels->setRange(1, 16);
    m_orbNLevels->setValue(orbDefaults.nLevels);
    form->addRow(QStringLiteral("ORB nLevels"), m_orbNLevels);

    applyDetectorRowVisibility(false);

    // Selecting ORB here is also a shortcut for the vendored real ORB-SLAM3
    // algorithm (see buildOrbSlam3Group()): this project's own custom-
    // pipeline ORB integration was measured far less accurate than SIFT on
    // KITTI seq00 (167m vs 17m ATE RMSE, see DEBUGGING.md Session 11),
    // whereas real ORB-SLAM3 -- which is ORB-based throughout -- gets to
    // 6.4-10.7m. So rather than actually running the custom pipeline with
    // ORB, choosing "ORB" here just checks m_orbSlam3Enabled, which switches
    // SlamWorker over to the real algorithm entirely (and grays out this
    // group, PnP, and OXTS/IMU, since none of them apply there). Choosing
    // "SIFT" again unchecks it, returning to the custom pipeline.
    connect(m_detectorType, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] {
        const bool isOrb = m_detectorType->currentData().toInt() == static_cast<int>(feature_detector::DetectorType::Orb);
        applyDetectorRowVisibility(isOrb);
        emitDetectorType();
        if (m_orbSlam3Enabled && m_orbSlam3Enabled->isChecked() != isOrb)
            m_orbSlam3Enabled->setChecked(isOrb);
    });

    // nFeatures is shared conceptually between both detectors -- forward it
    // to whichever settings struct is currently active by just emitting both;
    // SlamWorker stores both regardless and rebuildDetector() only uses
    // whichever m_detectorType currently selects, so this is harmless.
    connect(m_nFeatures, QOverload<int>::of(&QSpinBox::valueChanged), this, &ControlPanel::emitSiftSettings);
    connect(m_nFeatures, QOverload<int>::of(&QSpinBox::valueChanged), this, &ControlPanel::emitOrbSettings);
    connect(m_nOctaveLayers, QOverload<int>::of(&QSpinBox::valueChanged), this, &ControlPanel::emitSiftSettings);
    connect(m_contrastThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel::emitSiftSettings);
    connect(m_edgeThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel::emitSiftSettings);
    connect(m_sigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel::emitSiftSettings);
    connect(m_orbScaleFactor, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel::emitOrbSettings);
    connect(m_orbNLevels, QOverload<int>::of(&QSpinBox::valueChanged), this, &ControlPanel::emitOrbSettings);

    return group;
}

QWidget *ControlPanel::buildPnpGroup()
{
    auto *group = new QGroupBox(QStringLiteral("Pose Estimation (PnP)"), this);
    auto *form = new QFormLayout(group);

    const PnpSettings defaults;

    m_pnpMethod = new QComboBox(group);
    m_pnpMethod->addItem(QStringLiteral("P3P"), cv::SOLVEPNP_P3P);
    m_pnpMethod->addItem(QStringLiteral("Iterative"), cv::SOLVEPNP_ITERATIVE);
    m_pnpMethod->addItem(QStringLiteral("EPnP"), cv::SOLVEPNP_EPNP);
    m_pnpMethod->addItem(QStringLiteral("SQPnP"), cv::SOLVEPNP_SQPNP);
    m_pnpMethod->addItem(QStringLiteral("DLT (custom)"), kPnpMethodDlt);
    m_pnpMethod->setCurrentIndex(m_pnpMethod->findData(defaults.method));
    form->addRow(QStringLiteral("Method"), m_pnpMethod);

    m_reprojError = new QDoubleSpinBox(group);
    m_reprojError->setRange(0.1, 50.0);
    m_reprojError->setDecimals(2);
    m_reprojError->setValue(defaults.reprojectionError);
    form->addRow(QStringLiteral("RANSAC reproj. error (px)"), m_reprojError);

    m_ransacIterations = new QSpinBox(group);
    m_ransacIterations->setRange(10, 5000);
    m_ransacIterations->setValue(defaults.iterationsCount);
    form->addRow(QStringLiteral("RANSAC iterations"), m_ransacIterations);

    // Lets P3P/Iterative's bias-vs-noise trade-off be compared live: off
    // (default) is the original raw, minimal-sample-only pose -- noisier
    // per-frame, but empirically measured this session to track ground
    // truth *better* over a full KITTI sequence (27.2m ATE RMSE) than with
    // this refit on (61.1m). The theory going in was that refitting over
    // the full RANSAC inlier set (like solvePnPDltRansac() already always
    // does) would remove a systematic bias -- instead it seems to converge
    // more consistently onto whatever small camera-model mismatch exists,
    // making that bias *stronger*, while the noisier unrefined estimate
    // apparently cancels out closer to ground truth by chance. No effect on
    // DLT, which already always does this refit. Left in as a toggle
    // because the trade-off is real and worth being able to compare live,
    // not because on is recommended.
    m_pnpFullInlierRefine = new QCheckBox(QStringLiteral("Refine PnP over full inlier set (P3P/Iterative)"), group);
    m_pnpFullInlierRefine->setChecked(false);
    form->addRow(m_pnpFullInlierRefine);

    connect(m_pnpMethod, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ControlPanel::emitPnpSettings);
    connect(m_reprojError, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ControlPanel::emitPnpSettings);
    connect(m_ransacIterations, QOverload<int>::of(&QSpinBox::valueChanged), this, &ControlPanel::emitPnpSettings);
    connect(m_pnpFullInlierRefine, &QCheckBox::toggled, this, &ControlPanel::pnpFullInlierRefineEnabledChanged);

    return group;
}

QWidget *ControlPanel::buildOxtsImuGroup()
{
    // KITTI sequence-00-only controls (see SlamWorker::autoLoadKittiExtras()):
    // real forward-velocity-derived scale correction and real IMU-orientation
    // rotation, both cross-validated against ground truth before being wired
    // in (see DEBUGGING.md). OXTS/IMU now defaults OFF (per explicit
    // request; was on) -- opt in explicitly once data is loaded. The
    // 8-point checkbox also defaults OFF, unrelated to that change (5-point
    // with its own refit + Gold Standard refinement is this codebase's
    // better-measured combination regardless of OXTS/IMU).
    auto *group = new QGroupBox(QStringLiteral("OXTS / IMU (KITTI seq00)"), this);
    auto *layout = new QVBoxLayout(group);

    m_useEightPoint = new QCheckBox(QStringLiteral("Use legacy 8-point + Gold Standard estimator"), group);
    m_useEightPoint->setChecked(false);
    layout->addWidget(m_useEightPoint);

    // Start disabled+unchecked -- only enabled once SlamWorker actually has
    // matching data loaded (see setOxtsAvailable()/setImuAvailable(), wired
    // to SlamWorker::oxtsAvailabilityChanged()/imuAvailabilityChanged() in
    // MainWindow), per explicit request: don't let these be checked (and
    // silently do nothing, or worse, apply stale data left over from a
    // previously opened source) with nothing correct loaded underneath.
    m_oxtsEnabled = new QCheckBox(QStringLiteral("Use OXTS speed correction"), group);
    m_oxtsEnabled->setChecked(false);
    m_oxtsEnabled->setEnabled(false);
    layout->addWidget(m_oxtsEnabled);

    m_imuEnabled = new QCheckBox(QStringLiteral("Use IMU rotation (near-turn frames)"), group);
    m_imuEnabled->setChecked(false);
    m_imuEnabled->setEnabled(false);
    layout->addWidget(m_imuEnabled);

    // Manual single-folder picker, for any sequence other than the
    // hardcoded seq00 paths autoLoadKittiExtras() knows about (e.g. after
    // separately extracting a different sequence's OXTS folder -- see
    // DEBUGGING.md). Expects a "drive" folder laid out as
    // <picked>/oxts/{timestamps.txt,data/} plus calib_*.txt directly inside
    // <picked> -- see browseForOxtsImu()'s doc comment.
    auto *browseOxtsImuBtn = new QPushButton(QStringLiteral("Browse OXTS/IMU Drive Folder..."), group);
    connect(browseOxtsImuBtn, &QPushButton::clicked, this, &ControlPanel::browseForOxtsImu);
    layout->addWidget(browseOxtsImuBtn);

    // Default off, per explicit request (was on -- feeds OXTS distance + IMU
    // rotation into trackFrame()'s own plausibility gate too, not just
    // recoverViaEpipolar()'s; see SlamWorker::setOxtsImuInPnpEnabled()'s doc
    // comment. Measured on a full KITTI sequence-00 run: 18.6m ATE RMSE vs.
    // 27.2m without it -- a real, substantial improvement when explicitly
    // opted into, just not the default anymore).
    m_oxtsImuInPnpEnabled =
        new QCheckBox(QStringLiteral("Use OXTS/IMU in PnP tracking plausibility check"), group);
    m_oxtsImuInPnpEnabled->setChecked(false);
    layout->addWidget(m_oxtsImuInPnpEnabled);

    // Default OFF: a much weaker, one-shot-only approximation of VISO2-M's
    // continuous ground-plane correction (see GroundPlaneScale.h), not
    // cross-validated against ground truth the way OXTS/IMU were. Only
    // takes effect where OXTS doesn't cover (i.e. useful mainly with OXTS
    // unchecked, as a vision-only fallback to compare against).
    m_groundPlaneEnabled = new QCheckBox(QStringLiteral("Use ground-plane scale (VISO2-M-style, vision-only)"), group);
    m_groundPlaneEnabled->setChecked(false);
    layout->addWidget(m_groundPlaneEnabled);

    // AR-style ground-truth path overlay drawn onto the video frame itself
    // (see SlamWorker::drawGroundTruthOverlay()) -- distinct from MapView's
    // top-down red-line overlay, which needs no ground-truth poses beyond
    // (x,z). Default on: purely visual, no effect on tracking/ATE.
    m_groundTruthOverlayEnabled =
        new QCheckBox(QStringLiteral("Show ground-truth path overlay (video)"), group);
    m_groundTruthOverlayEnabled->setChecked(true);
    layout->addWidget(m_groundTruthOverlayEnabled);

    // Manual pixel offset for the overlay above -- lets you re-center it by
    // eye (see SlamWorker::setGroundTruthOverlayOffset()'s doc comment for
    // why this exists as a separate knob from the camera intrinsics).
    // Persisted across runs (QSettings, see emitGroundTruthOverlayOffset())
    // so a value tuned once against the preview frame (see
    // SlamWorker::previewFirstFrame()) doesn't need re-tuning every launch.
    QSettings settings;
    const int savedOffsetX = settings.value(QStringLiteral("groundTruthOverlay/offsetX"), 0).toInt();
    const int savedOffsetY = settings.value(QStringLiteral("groundTruthOverlay/offsetY"), 0).toInt();

    auto *offsetRow = new QWidget(group);
    auto *offsetLayout = new QHBoxLayout(offsetRow);
    offsetLayout->setContentsMargins(0, 0, 0, 0);
    offsetLayout->addWidget(new QLabel(QStringLiteral("GT overlay offset (px):"), offsetRow));
    m_groundTruthOverlayOffsetX = new QSpinBox(offsetRow);
    m_groundTruthOverlayOffsetX->setRange(-2000, 2000);
    m_groundTruthOverlayOffsetX->setPrefix(QStringLiteral("x:"));
    offsetLayout->addWidget(m_groundTruthOverlayOffsetX);
    m_groundTruthOverlayOffsetY = new QSpinBox(offsetRow);
    m_groundTruthOverlayOffsetY->setRange(-2000, 2000);
    m_groundTruthOverlayOffsetY->setPrefix(QStringLiteral("y:"));
    offsetLayout->addWidget(m_groundTruthOverlayOffsetY);
    layout->addWidget(offsetRow);

    // Separate pixel offset for the old-street revisit dots (see
    // SlamWorker::setOldStreetOverlayOffset()) -- kept independent of the
    // road-ahead offset above so nudging one doesn't drag the other along,
    // per explicit user request. Same persistence pattern.
    const int savedOldStreetOffsetX = settings.value(QStringLiteral("oldStreetOverlay/offsetX"), 0).toInt();
    const int savedOldStreetOffsetY = settings.value(QStringLiteral("oldStreetOverlay/offsetY"), 0).toInt();

    auto *oldStreetOffsetRow = new QWidget(group);
    auto *oldStreetOffsetLayout = new QHBoxLayout(oldStreetOffsetRow);
    oldStreetOffsetLayout->setContentsMargins(0, 0, 0, 0);
    oldStreetOffsetLayout->addWidget(new QLabel(QStringLiteral("Old-street dots offset (px):"), oldStreetOffsetRow));
    m_oldStreetOverlayOffsetX = new QSpinBox(oldStreetOffsetRow);
    m_oldStreetOverlayOffsetX->setRange(-2000, 2000);
    m_oldStreetOverlayOffsetX->setPrefix(QStringLiteral("x:"));
    oldStreetOffsetLayout->addWidget(m_oldStreetOverlayOffsetX);
    m_oldStreetOverlayOffsetY = new QSpinBox(oldStreetOffsetRow);
    m_oldStreetOverlayOffsetY->setRange(-2000, 2000);
    m_oldStreetOverlayOffsetY->setPrefix(QStringLiteral("y:"));
    oldStreetOffsetLayout->addWidget(m_oldStreetOverlayOffsetY);
    layout->addWidget(oldStreetOffsetRow);

    // Real joint bundle adjustment over a loop's span (see
    // SlamWorker::runLoopBundleAdjustment()), instead of just interpolating
    // the loop-measured discrepancy across the keyframes in between.
    // Default off: new, more expensive, and not yet benchmarked against
    // every sequence the interpolated correction has been run on.
    m_loopBundleAdjustmentEnabled =
        new QCheckBox(QStringLiteral("Run bundle adjustment after loop closure"), group);
    m_loopBundleAdjustmentEnabled->setChecked(false);
    layout->addWidget(m_loopBundleAdjustmentEnabled);

    connect(m_useEightPoint, &QCheckBox::toggled, this, &ControlPanel::useEightPointChanged);
    connect(m_oxtsEnabled, &QCheckBox::toggled, this, &ControlPanel::oxtsEnabledChanged);
    connect(m_imuEnabled, &QCheckBox::toggled, this, &ControlPanel::imuEnabledChanged);
    connect(m_oxtsImuInPnpEnabled, &QCheckBox::toggled, this, &ControlPanel::oxtsImuInPnpEnabledChanged);
    connect(m_groundPlaneEnabled, &QCheckBox::toggled, this, &ControlPanel::groundPlaneEnabledChanged);
    connect(m_groundTruthOverlayEnabled, &QCheckBox::toggled, this,
            &ControlPanel::groundTruthOverlayEnabledChanged);
    connect(m_groundTruthOverlayOffsetX, &QSpinBox::valueChanged, this,
            &ControlPanel::emitGroundTruthOverlayOffset);
    connect(m_groundTruthOverlayOffsetY, &QSpinBox::valueChanged, this,
            &ControlPanel::emitGroundTruthOverlayOffset);
    connect(m_oldStreetOverlayOffsetX, &QSpinBox::valueChanged, this,
            &ControlPanel::emitOldStreetOverlayOffset);
    connect(m_oldStreetOverlayOffsetY, &QSpinBox::valueChanged, this,
            &ControlPanel::emitOldStreetOverlayOffset);
    connect(m_loopBundleAdjustmentEnabled, &QCheckBox::toggled, this,
            &ControlPanel::loopBundleAdjustmentEnabledChanged);

    // Restore the persisted offsets *after* wiring the signals above, so if
    // they differ from the spin boxes' zero starting point, the resulting
    // valueChanged fires the emit*Offset() slots and the worker actually
    // receives the restored values at startup (not just the UI).
    m_groundTruthOverlayOffsetX->setValue(savedOffsetX);
    m_groundTruthOverlayOffsetY->setValue(savedOffsetY);
    m_oldStreetOverlayOffsetX->setValue(savedOldStreetOffsetX);
    m_oldStreetOverlayOffsetY->setValue(savedOldStreetOffsetY);

    return group;
}

void ControlPanel::emitIntrinsics()
{
    CameraIntrinsics intrinsics;
    intrinsics.fx = m_fx->value();
    intrinsics.fy = m_fy->value();
    intrinsics.cx = m_cx->value();
    intrinsics.cy = m_cy->value();
    emit intrinsicsChanged(intrinsics);
}

void ControlPanel::emitSiftSettings()
{
    SiftSettings settings;
    settings.nFeatures = m_nFeatures->value();
    settings.nOctaveLayers = m_nOctaveLayers->value();
    settings.contrastThreshold = m_contrastThreshold->value();
    settings.edgeThreshold = m_edgeThreshold->value();
    settings.sigma = m_sigma->value();
    emit siftSettingsChanged(settings);
}

void ControlPanel::emitPnpSettings()
{
    PnpSettings settings;
    settings.method = m_pnpMethod->currentData().toInt();
    settings.reprojectionError = m_reprojError->value();
    settings.iterationsCount = m_ransacIterations->value();
    emit pnpSettingsChanged(settings);
}

void ControlPanel::emitDetectorType()
{
    emit detectorTypeChanged(static_cast<feature_detector::DetectorType>(m_detectorType->currentData().toInt()));
}

void ControlPanel::emitOrbSettings()
{
    feature_detector::OrbSettings settings;
    settings.nFeatures = m_nFeatures->value();
    settings.scaleFactor = static_cast<float>(m_orbScaleFactor->value());
    settings.nLevels = m_orbNLevels->value();
    emit orbSettingsChanged(settings);
}

void ControlPanel::emitGroundTruthOverlayOffset()
{
    const int dx = m_groundTruthOverlayOffsetX->value();
    const int dy = m_groundTruthOverlayOffsetY->value();

    QSettings settings;
    settings.setValue(QStringLiteral("groundTruthOverlay/offsetX"), dx);
    settings.setValue(QStringLiteral("groundTruthOverlay/offsetY"), dy);

    emit groundTruthOverlayOffsetChanged(dx, dy);
}

void ControlPanel::emitOldStreetOverlayOffset()
{
    const int dx = m_oldStreetOverlayOffsetX->value();
    const int dy = m_oldStreetOverlayOffsetY->value();

    QSettings settings;
    settings.setValue(QStringLiteral("oldStreetOverlay/offsetX"), dx);
    settings.setValue(QStringLiteral("oldStreetOverlay/offsetY"), dy);

    emit oldStreetOverlayOffsetChanged(dx, dy);
}

void ControlPanel::browseForVideoFile()
{
    const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Open Video File"),
                                                        QString(),
                                                        QStringLiteral("Video files (*.mp4 *.avi *.mkv *.mov);;All files (*)"));
    if (!path.isEmpty())
        emit openVideoFileSelected(path);
}

void ControlPanel::browseForOxtsImu()
{
    // Single folder pick, per explicit request -- expects a "drive" folder
    // laid out as <picked>/oxts/{timestamps.txt,data/} plus the three
    // calib_*.txt files directly inside <picked> itself (that's how the
    // seq01 drive folder was assembled: OXTS extracted from the raw-data
    // sync zip, calibration copied in alongside it since it's shared by
    // date, not drive). Falls back to treating <picked> itself as the OXTS
    // folder if it has no "oxts" subfolder, so pointing directly at an
    // oxts/ folder still works too.
    const QString picked = QFileDialog::getExistingDirectory(
        this, QStringLiteral("Select KITTI drive folder (oxts/ subfolder + calib_*.txt)"));
    if (picked.isEmpty())
        return;

    const QDir pickedDir(picked);
    const QString oxtsDir = pickedDir.exists(QStringLiteral("oxts/timestamps.txt"))
                                 ? pickedDir.filePath(QStringLiteral("oxts"))
                                 : picked;
    emit oxtsDirSelected(oxtsDir);
    emit imuDirsSelected(oxtsDir, picked);
}

void ControlPanel::setTrackingState(const QString &state)
{
    m_stateLabel->setText(QStringLiteral("State: %1").arg(state));
}

void ControlPanel::setOxtsAvailable(bool available)
{
    m_oxtsEnabled->setEnabled(available);
    // Only force-UNcheck when data disappears (stale data should never stay
    // silently applied to an unrelated new source). Becoming available no
    // longer auto-checks -- OXTS/IMU default off per explicit request; the
    // user must opt in explicitly now that data is loaded and the box is
    // clickable.
    if (!available)
        m_oxtsEnabled->setChecked(false);
}

void ControlPanel::setImuAvailable(bool available)
{
    m_imuEnabled->setEnabled(available);
    if (!available)
        m_imuEnabled->setChecked(false);
}
