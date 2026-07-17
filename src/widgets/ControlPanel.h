#pragma once

#include <QWidget>

#include "vision/CameraIntrinsics.h"
#include "vision/FeatureDetector.h"
#include "vision/PnpSettings.h"
#include "vision/SiftSettings.h"

class QDoubleSpinBox;
class QSpinBox;
class QComboBox;
class QLabel;
class QCheckBox;
class QFormLayout;

// Left-hand settings panel: video source controls, camera intrinsics,
// SIFT (Gaussian / octave layer) parameters and the PnP solver choice.
class ControlPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ControlPanel(QWidget *parent = nullptr);

public slots:
    void setTrackingState(const QString &state);

    // Only let "Use OXTS speed correction"/"Use IMU rotation" be checked
    // once real, matching data is actually loaded -- per explicit request.
    // Both start disabled+unchecked and default OFF even once data loads
    // (also per explicit request) -- becoming available only enables the
    // checkbox for the user to opt into manually; becoming UNavailable
    // force-unchecks it (fires the checkbox's own toggled signal, cascading
    // into oxtsEnabledChanged()/imuEnabledChanged()) so stale data never
    // stays silently applied to an unrelated new source. Connected to
    // SlamWorker::oxtsAvailabilityChanged()/imuAvailabilityChanged() (true
    // on a successful load, false whenever a new source clears previously
    // loaded data).
    void setOxtsAvailable(bool available);
    void setImuAvailable(bool available);

signals:
    void openVideoFileSelected(const QString &path);
    void openCameraRequested(int index);
    void startRequested();
    void stopRequested();
    void resetRequested();

    void intrinsicsChanged(CameraIntrinsics intrinsics);
    void siftSettingsChanged(SiftSettings settings);
    void pnpSettingsChanged(PnpSettings settings);

    // Feature-detector choice (SIFT stays the default -- see FeatureDetector.h).
    void detectorTypeChanged(feature_detector::DetectorType type);
    void orbSettingsChanged(feature_detector::OrbSettings settings);

    // OXTS/IMU toggles (KITTI sequence 00 only -- see
    // SlamWorker::autoLoadKittiExtras()). useEightPointChanged selects the
    // legacy 8-point+Gold-Standard estimator over the default 5-point one;
    // the other two enable/disable already-loaded OXTS speed / IMU
    // rotation data without needing to reload it.
    void useEightPointChanged(bool useEightPoint);
    void oxtsEnabledChanged(bool enabled);
    void imuEnabledChanged(bool enabled);
    void groundPlaneEnabledChanged(bool enabled);
    void groundTruthOverlayEnabledChanged(bool enabled);
    void groundTruthOverlayOffsetChanged(int dx, int dy);
    void oldStreetOverlayOffsetChanged(int dx, int dy);
    void loopBundleAdjustmentEnabledChanged(bool enabled);
    void pnpFullInlierRefineEnabledChanged(bool enabled);
    void oxtsImuInPnpEnabledChanged(bool enabled);

    // Manual OXTS/IMU folder selection (as opposed to
    // SlamWorker::autoLoadKittiExtras()'s hardcoded sequence-00 paths) --
    // lets any KITTI sequence with locally-extracted OXTS/calibration data
    // use it, not just seq00. oxtsDirSelected loads OXTS speed alone (only
    // needs the OXTS folder); imuDirsSelected loads IMU orientation (needs
    // both the OXTS folder and a same-date calibration folder together).
    void oxtsDirSelected(const QString &oxtsDir);
    void imuDirsSelected(const QString &oxtsDir, const QString &calibDir);

    // See buildOrbSlam3Group(): switches SlamWorker::processNext() over to
    // the vendored, real ORB-SLAM3 algorithm (third_party/ORB_SLAM3) instead
    // of this project's own custom pipeline. All of the custom pipeline's
    // own settings groups (feature detector, PnP, OXTS/IMU) are disabled
    // whenever this is checked, since none of them apply to ORB-SLAM3's own
    // internal tracking.
    void orbSlam3EnabledChanged(bool enabled);

private slots:
    void emitIntrinsics();
    void emitSiftSettings();
    void emitPnpSettings();
    void emitDetectorType();
    void emitOrbSettings();
    void browseForVideoFile();
    void browseForOxtsImu();
    void emitGroundTruthOverlayOffset();
    void emitOldStreetOverlayOffset();

private:
    QWidget *buildSourceGroup();
    QWidget *buildOrbSlam3Group();
    QWidget *buildIntrinsicsGroup();
    QWidget *buildSiftGroup();
    QWidget *buildPnpGroup();
    QWidget *buildOxtsImuGroup();

    // Shows/hides the SIFT-only vs. ORB-only parameter rows in m_siftGroup's
    // form (m_siftForm) -- shared by the detector combo's own
    // currentIndexChanged handler and buildOrbSlam3Group()'s checkbox
    // handler (see the "selecting ORB switches to real ORB-SLAM3" sync
    // below), so both keep the same rows visible without duplicating the
    // per-field setVisible() logic.
    void applyDetectorRowVisibility(bool isOrb);

    QSpinBox *m_cameraIndex = nullptr;

    // See buildOrbSlam3Group(): checking m_orbSlam3Enabled disables every
    // widget in these three groups (grayed out, values left untouched --
    // SlamWorker keeps whatever was last set, they just stop being editable
    // while ORB-SLAM3 mode is active) since none of them apply to the
    // vendored ORB-SLAM3 algorithm's own internal tracking. m_intrinsicsGroup
    // is NOT included: fx/fy/cx/cy are still read from there to build
    // ORB-SLAM3's own settings YAML (see SlamWorker::buildOrbSlam3Settings()).
    QCheckBox *m_orbSlam3Enabled = nullptr;
    QWidget *m_siftGroup = nullptr;
    QWidget *m_pnpGroup = nullptr;
    QWidget *m_oxtsImuGroup = nullptr;

    QDoubleSpinBox *m_fx = nullptr;
    QDoubleSpinBox *m_fy = nullptr;
    QDoubleSpinBox *m_cx = nullptr;
    QDoubleSpinBox *m_cy = nullptr;

    QFormLayout *m_siftForm = nullptr; // see applyDetectorRowVisibility()
    QComboBox *m_detectorType = nullptr;
    QSpinBox *m_nFeatures = nullptr;
    QSpinBox *m_nOctaveLayers = nullptr;
    QDoubleSpinBox *m_contrastThreshold = nullptr;
    QDoubleSpinBox *m_edgeThreshold = nullptr;
    QDoubleSpinBox *m_sigma = nullptr;
    QDoubleSpinBox *m_orbScaleFactor = nullptr;
    QSpinBox *m_orbNLevels = nullptr;

    QComboBox *m_pnpMethod = nullptr;
    QDoubleSpinBox *m_reprojError = nullptr;
    QSpinBox *m_ransacIterations = nullptr;

    QCheckBox *m_useEightPoint = nullptr;
    QCheckBox *m_oxtsEnabled = nullptr;
    QCheckBox *m_imuEnabled = nullptr;
    QCheckBox *m_groundPlaneEnabled = nullptr;
    QCheckBox *m_groundTruthOverlayEnabled = nullptr;
    QSpinBox *m_groundTruthOverlayOffsetX = nullptr;
    QSpinBox *m_groundTruthOverlayOffsetY = nullptr;
    QSpinBox *m_oldStreetOverlayOffsetX = nullptr;
    QSpinBox *m_oldStreetOverlayOffsetY = nullptr;
    QCheckBox *m_loopBundleAdjustmentEnabled = nullptr;
    QCheckBox *m_pnpFullInlierRefine = nullptr;
    QCheckBox *m_oxtsImuInPnpEnabled = nullptr;

    QLabel *m_stateLabel = nullptr;
};
