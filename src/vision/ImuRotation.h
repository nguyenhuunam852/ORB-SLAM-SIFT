#pragma once

// Standalone module: derives the camera's relative rotation between two
// frames directly from KITTI OXTS (GPS/IMU) orientation data, instead of
// vision (F/E or homography decomposition). Intended as a drop-in
// replacement for estimateTwoViewPose()'s R output specifically for the
// near-pure-rotation case homography currently exists to handle -- real
// measured rotation sidesteps the ill-conditioned-epipolar-geometry problem
// entirely rather than working around it. Not wired into SlamWorker; a
// standalone, independently-testable module (see DEBUGGING.md).
//
// Requires three KITTI raw-data calibration files (from the *same date* as
// the OXTS data -- KITTI recalibrates between dates, so date must match
// exactly, confirmed this session when a wrong-date calibration zip was
// caught before use):
//   calib_imu_to_velo.txt, calib_velo_to_cam.txt (IMU->Velodyne->cam0 raw),
//   calib_cam_to_cam.txt's R_rect_00 (cam0 raw -> cam0 *rectified*, since
//   KITTI's odometry-benchmark image_0 frames are the rectified ones, not
//   raw cam0 -- confirmed this session via S_rect_00 == 1241x376, matching
//   the odometry images exactly).

#include <opencv2/core.hpp>

#include <QString>

#include <vector>

namespace imu_rotation {

struct ImuToCameraCalib
{
    cv::Mat R; // 3x3: maps IMU(body)-frame vectors into the rectified camera frame
};

// calibDir must contain calib_imu_to_velo.txt, calib_velo_to_cam.txt, and
// calib_cam_to_cam.txt (KITTI raw-data per-date calibration folder).
bool loadImuToCameraCalib(const QString &calibDir, ImuToCameraCalib &out);

// Loads per-frame OXTS orientation (roll, pitch, yaw) and returns, for each
// frame i, the 3x3 rotation mapping body(IMU)-frame vectors to a fixed
// local navigation frame at that frame (KITTI's convertOxtsToPose.m
// convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)). oxtsDir must contain
// data/<frame>.txt (KITTI OXTS format, fields 3,4,5 = roll,pitch,yaw).
bool loadOxtsOrientations(const QString &oxtsDir, std::vector<cv::Mat> &navFromBody);

// Relative camera rotation between frameA and frameB, in the same
// convention cv::recoverPose()/estimateTwoViewPose() use for R: X_camB =
// R * X_camA. Derived purely from measured IMU orientation plus the fixed
// IMU->camera extrinsic (assumes a rigid mount, i.e. the same extrinsic
// applies at both frames -- true for a vehicle-mounted rig). Returns an
// empty Mat if frameA/frameB are out of range for navFromBody.
cv::Mat relativeCameraRotation(const std::vector<cv::Mat> &navFromBody, const ImuToCameraCalib &calib, int frameA,
                                int frameB);

} // namespace imu_rotation
