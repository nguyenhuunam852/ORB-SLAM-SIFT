#include "MapView.h"

#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QPainter>
#include <QPaintEvent>
#include <QRegularExpression>
#include <QResizeEvent>
#include <QTextStream>

#include <algorithm>
#include <cmath>

MapView::MapView(QWidget *parent)
    : QWidget(parent)
{
    setMinimumSize(240, 240);
    setAutoFillBackground(false);

    m_loadGtButton = new QPushButton(QStringLiteral("Load Ground Truth"), this);
    m_loadGtButton->setToolTip(QStringLiteral("Load a KITTI-format poses.txt to overlay as a green line"));
    connect(m_loadGtButton, &QPushButton::clicked, this, &MapView::browseForGroundTruth);
    m_loadGtButton->move(8, 8);
    m_loadGtButton->adjustSize();
}

void MapView::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    m_loadGtButton->move(8, 8);
}

void MapView::browseForGroundTruth()
{
    const QString path = QFileDialog::getOpenFileName(this, QStringLiteral("Load KITTI ground-truth poses"),
                                                        QString(), QStringLiteral("Poses (*.txt);;All files (*)"));
    if (path.isEmpty())
        return;
    loadGroundTruthFile(path);
}

bool MapView::loadGroundTruthFile(const QString &path)
{
    if (!parseGroundTruthFile(path))
        return false;
    update();
    return true;
}

bool MapView::parseGroundTruthFile(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;

    QVector<QPointF> trajectory;
    QTextStream stream(&file);
    static const QRegularExpression whitespace(QStringLiteral("\\s+"));
    while (!stream.atEnd()) {
        const QString line = stream.readLine();
        const QStringList parts = line.split(whitespace, Qt::SkipEmptyParts);
        if (parts.size() < 12)
            continue;
        // Row-major 3x4 [R|t]: columns 3, 7, 11 are tx, ty, tz -- the
        // camera center in the ground-truth world frame. Top-down plot
        // uses (tx, tz), matching the estimated trajectory's (world X,
        // world Z) convention.
        bool okX = false, okZ = false;
        const double tx = parts[3].toDouble(&okX);
        const double tz = parts[11].toDouble(&okZ);
        if (!okX || !okZ)
            continue;
        trajectory.append(QPointF(tx, tz));
    }
    if (trajectory.isEmpty())
        return false;

    m_groundTruth = std::move(trajectory);
    m_groundTruthFileName = QFileInfo(path).fileName();
    m_alignmentFrozen = false; // new ground truth invalidates any previously frozen fit
    return true;
}

void MapView::setMapData(const QVector<QPointF> &trajectory, const QVector<QPointF> &mapPoints,
                          const QVector<int> &trajectoryFrameIndices)
{
    if (trajectory.isEmpty() && !m_trajectory.isEmpty())
        m_alignmentFrozen = false; // an explicit Reset -- start fresh once the new run has enough points again
    m_trajectory = trajectory;
    m_mapPoints = mapPoints;
    m_trajectoryFrameIndex = trajectoryFrameIndices;
    update();
}

void MapView::setContinuousAlignmentEnabled(bool enabled)
{
    m_continuousAlignmentEnabled = enabled;
    m_alignmentFrozen = false; // re-fit fresh under the new policy instead of keeping a stale frozen fit
    update();
}

namespace {
constexpr int kAlignmentFreezeMinPoints = 200; // enough overlap for a stable similarity fit without
                                                // waiting so long the overlay sits unaligned for most
                                                // of a short run
}

bool MapView::computeAlignment(double &scale, double &cosTheta, double &sinTheta, QPointF &translation) const
{
    if (m_alignmentFrozen && !m_continuousAlignmentEnabled) {
        scale = m_frozenScale;
        cosTheta = m_frozenCos;
        sinTheta = m_frozenSin;
        translation = m_frozenTranslation;
        return true;
    }

    if (m_groundTruth.isEmpty() || m_trajectory.size() != m_trajectoryFrameIndex.size())
        return false;

    // Pair each trajectory point with its corresponding ground-truth pose
    // by frame index (KITTI poses.txt line i == frame i+1, since frame
    // count starts at 1 for the first processed frame -- see
    // SlamWorker::processNext()).
    QVector<QPointF> src, dst;
    src.reserve(m_trajectory.size());
    dst.reserve(m_trajectory.size());
    for (int i = 0; i < m_trajectory.size(); ++i) {
        const int gtIdx = m_trajectoryFrameIndex[i] - 1;
        if (gtIdx < 0 || gtIdx >= m_groundTruth.size())
            continue;
        src.append(m_trajectory[i]);
        dst.append(m_groundTruth[gtIdx]);
    }
    if (src.size() < 8) // too little overlap yet for a stable fit
        return false;

    double meanSrcX = 0.0, meanSrcY = 0.0, meanDstX = 0.0, meanDstY = 0.0;
    for (int i = 0; i < src.size(); ++i) {
        meanSrcX += src[i].x(); meanSrcY += src[i].y();
        meanDstX += dst[i].x(); meanDstY += dst[i].y();
    }
    const double n = static_cast<double>(src.size());
    meanSrcX /= n; meanSrcY /= n; meanDstX /= n; meanDstY /= n;

    // Closed-form least-squares similarity fit (2D specialization of
    // Umeyama 1991): minimize sum||s*R(theta)*p_i + t - q_i||^2 over
    // centered p_i (trajectory), q_i (ground truth). The optimal rotation
    // angle is atan2(D, C) with C = sum(p.q), D = sum(p x q); scale
    // follows from the magnitude of that same (C, D) vector.
    double C = 0.0, D = 0.0, srcSqSum = 0.0;
    for (int i = 0; i < src.size(); ++i) {
        const double px = src[i].x() - meanSrcX, py = src[i].y() - meanSrcY;
        const double qx = dst[i].x() - meanDstX, qy = dst[i].y() - meanDstY;
        C += px * qx + py * qy;
        D += px * qy - py * qx;
        srcSqSum += px * px + py * py;
    }
    if (srcSqSum < 1e-9)
        return false;

    const double theta = std::atan2(D, C);
    cosTheta = std::cos(theta);
    sinTheta = std::sin(theta);
    scale = std::sqrt(C * C + D * D) / srcSqSum;
    if (!(scale > 0.0) || !std::isfinite(scale))
        return false;

    const double rx = cosTheta * meanSrcX - sinTheta * meanSrcY;
    const double ry = sinTheta * meanSrcX + cosTheta * meanSrcY;
    translation = QPointF(meanDstX - scale * rx, meanDstY - scale * ry);

    if (src.size() >= kAlignmentFreezeMinPoints && !m_continuousAlignmentEnabled) {
        m_alignmentFrozen = true;
        m_frozenScale = scale;
        m_frozenCos = cosTheta;
        m_frozenSin = sinTheta;
        m_frozenTranslation = translation;
    }
    return true;
}

void MapView::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), QColor(24, 24, 24));

    // Align the estimate into ground truth's frame before doing anything
    // else with it -- see the class comment for why (monocular SLAM's
    // world frame has no reason to match ground truth's without this).
    double alignScale = 1.0, alignCos = 1.0, alignSin = 0.0;
    QPointF alignT(0.0, 0.0);
    const bool aligned = computeAlignment(alignScale, alignCos, alignSin, alignT);
    auto alignPoint = [&](const QPointF &p) -> QPointF {
        if (!aligned)
            return p;
        const double x = alignScale * (alignCos * p.x() - alignSin * p.y()) + alignT.x();
        const double y = alignScale * (alignSin * p.x() + alignCos * p.y()) + alignT.y();
        return QPointF(x, y);
    };

    QVector<QPointF> trajectory, mapPoints;
    trajectory.reserve(m_trajectory.size());
    for (const auto &p : m_trajectory)
        trajectory.append(alignPoint(p));
    mapPoints.reserve(m_mapPoints.size());
    for (const auto &p : m_mapPoints)
        mapPoints.append(alignPoint(p));

    // Determine world-space bounds (fallback to a fixed range with no data).
    double minX = -5.0, maxX = 5.0, minZ = -5.0, maxZ = 5.0;
    bool haveData = false;
    auto expand = [&](const QPointF &p) {
        if (!haveData) {
            minX = maxX = p.x();
            minZ = maxZ = p.y();
            haveData = true;
        } else {
            minX = std::min(minX, p.x());
            maxX = std::max(maxX, p.x());
            minZ = std::min(minZ, p.y());
            maxZ = std::max(maxZ, p.y());
        }
    };
    for (const auto &p : mapPoints)
        expand(p);
    for (const auto &p : trajectory)
        expand(p);
    for (const auto &p : m_groundTruth)
        expand(p);

    const double rangeX = std::max(maxX - minX, 1.0);
    const double rangeZ = std::max(maxZ - minZ, 1.0);
    const double pad = 0.1;
    minX -= rangeX * pad; maxX += rangeX * pad;
    minZ -= rangeZ * pad; maxZ += rangeZ * pad;

    const double worldW = maxX - minX;
    const double worldH = maxZ - minZ;
    const QRectF viewport = rect().adjusted(12, 12, -12, -28);

    const double scale = std::min(viewport.width() / worldW, viewport.height() / worldH);
    auto toScreen = [&](const QPointF &p) -> QPointF {
        const double sx = viewport.left() + (p.x() - minX) * scale;
        const double sy = viewport.bottom() - (p.y() - minZ) * scale;
        return QPointF(sx, sy);
    };

    // Grid.
    painter.setPen(QPen(QColor(50, 50, 50), 1));
    const int gridLines = 8;
    for (int i = 0; i <= gridLines; ++i) {
        const double fx = viewport.left() + viewport.width() * i / gridLines;
        const double fy = viewport.top() + viewport.height() * i / gridLines;
        painter.drawLine(QPointF(fx, viewport.top()), QPointF(fx, viewport.bottom()));
        painter.drawLine(QPointF(viewport.left(), fy), QPointF(viewport.right(), fy));
    }

    // Map points.
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(120, 170, 255, 160));
    for (const auto &p : mapPoints) {
        const QPointF s = toScreen(p);
        painter.drawEllipse(s, 1.6, 1.6);
    }

    // Ground-truth path (green), drawn under the estimated path so the
    // estimate stays visually on top where they overlap.
    if (m_groundTruth.size() > 1) {
        painter.setPen(QPen(QColor(60, 220, 90), 2));
        QPolygonF path;
        path.reserve(m_groundTruth.size());
        for (const auto &p : m_groundTruth)
            path << toScreen(p);
        painter.drawPolyline(path);
    }

    // Estimated trajectory path (aligned into ground truth's frame, if an
    // alignment is available -- see computeAlignment()).
    if (trajectory.size() > 1) {
        painter.setPen(QPen(QColor(255, 180, 60), 2));
        QPolygonF path;
        path.reserve(trajectory.size());
        for (const auto &p : trajectory)
            path << toScreen(p);
        painter.drawPolyline(path);
    }

    // Current camera position marker.
    if (!trajectory.isEmpty()) {
        const QPointF s = toScreen(trajectory.last());
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 90, 90));
        painter.drawEllipse(s, 4.5, 4.5);
    }

    // ATE (Absolute Trajectory Error) RMSE -- reuses the SAME aligned
    // `trajectory` points and frame-index pairing computeAlignment() itself
    // fits against, so this is directly comparable to the offline CLI
    // benchmark's own ATE methodology (analyze/kitti_ate.cpp,
    // analyze/orbslam3_kitti_ate.cpp), just computed live instead of once
    // after a full run. Not meaningful (and not drawn) until aligned, same
    // gating as the overlay itself.
    double ateRmse = 0.0, ateMean = 0.0, ateMedian = 0.0, ateMax = 0.0;
    int ateCount = 0;
    if (aligned && !m_groundTruth.isEmpty() && trajectory.size() == m_trajectoryFrameIndex.size()) {
        QVector<double> errors;
        errors.reserve(trajectory.size());
        double sumSq = 0.0, sum = 0.0;
        for (int i = 0; i < trajectory.size(); ++i) {
            const int gtIdx = m_trajectoryFrameIndex[i] - 1;
            if (gtIdx < 0 || gtIdx >= m_groundTruth.size())
                continue;
            const double dx = trajectory[i].x() - m_groundTruth[gtIdx].x();
            const double dy = trajectory[i].y() - m_groundTruth[gtIdx].y();
            const double err = std::sqrt(dx * dx + dy * dy);
            sumSq += err * err;
            sum += err;
            ateMax = std::max(ateMax, err);
            errors.push_back(err);
        }
        ateCount = errors.size();
        if (ateCount > 0) {
            ateRmse = std::sqrt(sumSq / ateCount);
            ateMean = sum / ateCount;
            std::sort(errors.begin(), errors.end());
            ateMedian = errors[errors.size() / 2];
        }
    }

    // Labels.
    painter.setPen(QColor(160, 160, 160));
    QString label = QStringLiteral("Top-down map (world X / Z)  |  points: %1  |  path: %2")
                         .arg(mapPoints.size())
                         .arg(trajectory.size());
    if (!m_groundTruth.isEmpty()) {
        label += QStringLiteral("  |  GT (green): %1 [%2]%3")
                     .arg(m_groundTruth.size())
                     .arg(m_groundTruthFileName)
                     .arg(aligned ? QString() : QStringLiteral(" (unaligned -- need more overlap)"));
    }
    // RMSE alone can look alarming when a handful of high-error keyframes
    // (e.g. a brief tracking-loss/recovery jump) dominate it while most of
    // the trajectory tracks ground truth closely -- mean/median/max (same
    // four stats analyze/kitti_ate.cpp and orbslam3_kitti_ate.cpp report)
    // show that distribution instead of hiding it behind one number.
    if (ateCount > 0) {
        label += QStringLiteral("  |  ATE RMSE: %1 m (n=%2)").arg(ateRmse, 0, 'f', 2).arg(ateCount);
    }
    painter.drawText(rect().adjusted(8, 0, -8, -6), Qt::AlignBottom | Qt::AlignLeft, label);

    if (ateCount > 0) {
        painter.setPen(QColor(255, 220, 100));
        QFont bigFont = painter.font();
        bigFont.setPointSizeF(bigFont.pointSizeF() * 1.6);
        bigFont.setBold(true);
        painter.setFont(bigFont);
        painter.drawText(rect().adjusted(8, 8, -8, 0), Qt::AlignTop | Qt::AlignLeft,
                          QStringLiteral("ATE RMSE: %1 m").arg(ateRmse, 0, 'f', 2));

        QFont smallFont = painter.font();
        smallFont.setPointSizeF(smallFont.pointSizeF() / 1.6);
        smallFont.setBold(false);
        painter.setFont(smallFont);
        painter.setPen(QColor(200, 200, 200));
        painter.drawText(rect().adjusted(8, 34, -8, 0), Qt::AlignTop | Qt::AlignLeft,
                          QStringLiteral("mean %1 m  |  median %2 m  |  max %3 m")
                              .arg(ateMean, 0, 'f', 2)
                              .arg(ateMedian, 0, 'f', 2)
                              .arg(ateMax, 0, 'f', 2));
    }
}
