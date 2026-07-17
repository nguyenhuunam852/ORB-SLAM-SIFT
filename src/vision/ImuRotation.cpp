#include "ImuRotation.h"

#include <QFile>
#include <QTextStream>

#include <cmath>

namespace imu_rotation {
namespace {

// Finds a line starting with "<key>:" in a KITTI calibration text file and
// parses the 9 numbers following it into a 3x3 row-major rotation matrix.
cv::Mat parseCalibRotation(const QString &path, const QString &key)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return cv::Mat();

    QTextStream stream(&file);
    const QString prefix = key + QStringLiteral(":");
    while (!stream.atEnd()) {
        const QString line = stream.readLine();
        if (!line.startsWith(prefix))
            continue;
        const QStringList tokens = line.mid(prefix.length()).trimmed().split(QLatin1Char(' '), Qt::SkipEmptyParts);
        if (tokens.size() < 9)
            return cv::Mat();
        cv::Mat R(3, 3, CV_64F);
        for (int i = 0; i < 9; ++i)
            R.at<double>(i / 3, i % 3) = tokens[i].toDouble();
        return R;
    }
    return cv::Mat();
}

cv::Mat rotX(double a)
{
    return (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a));
}
cv::Mat rotY(double a)
{
    return (cv::Mat_<double>(3, 3) << std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a));
}
cv::Mat rotZ(double a)
{
    return (cv::Mat_<double>(3, 3) << std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1);
}

} // namespace

bool loadImuToCameraCalib(const QString &calibDir, ImuToCameraCalib &out)
{
    // imu->velo
    const cv::Mat R1 = parseCalibRotation(calibDir + QStringLiteral("/calib_imu_to_velo.txt"), QStringLiteral("R"));
    // velo->cam0 (raw, unrectified)
    const cv::Mat R2 = parseCalibRotation(calibDir + QStringLiteral("/calib_velo_to_cam.txt"), QStringLiteral("R"));
    // cam0 raw -> cam0 rectified (KITTI's odometry image_0 frames are rectified)
    const cv::Mat R3 =
        parseCalibRotation(calibDir + QStringLiteral("/calib_cam_to_cam.txt"), QStringLiteral("R_rect_00"));
    if (R1.empty() || R2.empty() || R3.empty())
        return false;

    out.R = R3 * R2 * R1; // imu -> rectified cam0, composed once
    return true;
}

bool loadOxtsOrientations(const QString &oxtsDir, std::vector<cv::Mat> &navFromBody)
{
    navFromBody.clear();
    int i = 0;
    while (true) {
        const QString path = oxtsDir + QStringLiteral("/data/") + QString::number(i).rightJustified(10, QLatin1Char('0')) +
                              QStringLiteral(".txt");
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            break;

        QTextStream stream(&file);
        const QStringList tokens = stream.readLine().split(QLatin1Char(' '), Qt::SkipEmptyParts);
        if (tokens.size() < 6) {
            navFromBody.clear();
            return false;
        }
        const double roll = tokens[3].toDouble();
        const double pitch = tokens[4].toDouble();
        const double yaw = tokens[5].toDouble();
        // KITTI devkit convertOxtsToPose.m convention: R = Rz(yaw)*Ry(pitch)*Rx(roll),
        // mapping body-frame vectors into a fixed local navigation frame.
        navFromBody.push_back(rotZ(yaw) * rotY(pitch) * rotX(roll));
        ++i;
    }
    return !navFromBody.empty();
}

cv::Mat relativeCameraRotation(const std::vector<cv::Mat> &navFromBody, const ImuToCameraCalib &calib, int frameA,
                                int frameB)
{
    if (calib.R.empty() || frameA < 0 || frameB < 0 || frameA >= static_cast<int>(navFromBody.size()) ||
        frameB >= static_cast<int>(navFromBody.size()))
        return cv::Mat();

    // Body-frame relative rotation (X_bodyB = R_rel_body * X_bodyA):
    // R_rel_body = bodyFromNav(B) * navFromBody(A) = navFromBody(B)^T * navFromBody(A).
    const cv::Mat R_rel_body = navFromBody[frameB].t() * navFromBody[frameA];

    // Conjugate into the camera frame via the fixed (rigid-mount) IMU->camera
    // rotation: X_cam = calib.R * X_body at both A and B, so
    // R_rel_cam = calib.R * R_rel_body * calib.R^T (same X_camB = R*X_camA
    // convention cv::recoverPose()/estimateTwoViewPose() use for R).
    return calib.R * R_rel_body * calib.R.t();
}

} // namespace imu_rotation
