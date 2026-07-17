#include "LoopMapThumbnail.h"

#include <QPainter>
#include <QPaintEvent>

#include <algorithm>
#include <cmath>

namespace {
bool finitePoint(const QPointF &p)
{
    return std::isfinite(p.x()) && std::isfinite(p.y());
}
}

LoopMapThumbnail::LoopMapThumbnail(const LoopEstimateResult &result, QWidget *parent)
    : QWidget(parent)
    , m_result(result)
{
    setFixedSize(360, 280);
}

void LoopMapThumbnail::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), QColor(24, 24, 24));
    painter.setPen(QPen(QColor(70, 70, 70), 1));
    painter.drawRect(rect().adjusted(0, 0, -1, -1));

    const QRectF viewport = rect().adjusted(8, 22, -8, -22);

    if (!m_result.ok || m_result.alignedGroundTruth.isEmpty()) {
        painter.setPen(QColor(160, 160, 160));
        painter.drawText(rect(), Qt::AlignCenter | Qt::TextWordWrap,
                          m_result.ok ? QStringLiteral("no ground truth") : m_result.message);
    } else {
        // ROI: fit the view to ground truth + the corrected trajectory
        // only -- the actual "did this loop close correctly" story -- NOT
        // the raw triangulated landmark cloud. A single bad/outlier
        // landmark (an easy thing to get from a near-degenerate
        // triangulation) used to blow up the bounding box and collapse
        // everything else to nothing, leaving only the always-drawn text
        // labels visible ("black screen, one yellow point"). Landmarks are
        // still drawn (for context) but never affect the fit; any
        // non-finite point is skipped defensively either way.
        bool haveBounds = false;
        double minX = 0.0, maxX = 0.0, minZ = 0.0, maxZ = 0.0;
        auto expand = [&](const QPointF &p) {
            if (!finitePoint(p))
                return;
            if (!haveBounds) {
                minX = maxX = p.x();
                minZ = maxZ = p.y();
                haveBounds = true;
            } else {
                minX = std::min(minX, p.x());
                maxX = std::max(maxX, p.x());
                minZ = std::min(minZ, p.y());
                maxZ = std::max(maxZ, p.y());
            }
        };
        for (const auto &p : m_result.alignedGroundTruth)
            expand(p);
        for (const auto &p : m_result.alignedTrajectory)
            expand(p);

        if (!haveBounds) {
            painter.setPen(QColor(160, 160, 160));
            painter.drawText(rect(), Qt::AlignCenter | Qt::TextWordWrap, QStringLiteral("no finite map data"));
        } else {
            const double rangeX = std::max(maxX - minX, 1.0);
            const double rangeZ = std::max(maxZ - minZ, 1.0);
            const double pad = 0.15;
            minX -= rangeX * pad;
            maxX += rangeX * pad;
            minZ -= rangeZ * pad;
            maxZ += rangeZ * pad;
            const double worldW = maxX - minX, worldH = maxZ - minZ;
            const double scale = std::min(viewport.width() / worldW, viewport.height() / worldH);
            auto toScreen = [&](const QPointF &p) -> QPointF {
                const double sx = viewport.left() + (p.x() - minX) * scale;
                const double sy = viewport.bottom() - (p.y() - minZ) * scale;
                return QPointF(sx, sy);
            };

            // Landmarks (optimized positions) -- drawn for context, outside
            // the ROI's influence; any that fall outside the viewport
            // simply don't show, which is fine.
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(120, 170, 255, 140));
            for (const auto &p : m_result.alignedLandmarks) {
                if (finitePoint(p))
                    painter.drawEllipse(toScreen(p), 1.5, 1.5);
            }

            // Ground truth (green), under the estimate.
            if (m_result.alignedGroundTruth.size() > 1) {
                painter.setPen(QPen(QColor(60, 220, 90), 2));
                QPolygonF path;
                for (const auto &p : m_result.alignedGroundTruth)
                    path << toScreen(p);
                painter.drawPolyline(path);
            }

            // Corrected trajectory (orange).
            if (m_result.alignedTrajectory.size() > 1) {
                painter.setPen(QPen(QColor(255, 180, 60), 2));
                QPolygonF path;
                for (const auto &p : m_result.alignedTrajectory)
                    path << toScreen(p);
                painter.drawPolyline(path);
            }

            // Peak-error marker: the "most ATE" point in this window,
            // explicitly requested -- a bright red ring around wherever the
            // corrected trajectory diverges from ground truth the most.
            if (m_result.maxErrorIndex >= 0 && m_result.maxErrorIndex < m_result.alignedTrajectory.size()) {
                const QPointF worst = toScreen(m_result.alignedTrajectory[m_result.maxErrorIndex]);
                painter.setPen(QPen(QColor(255, 60, 60), 2));
                painter.setBrush(Qt::NoBrush);
                painter.drawEllipse(worst, 6.0, 6.0);
            }
        }
    }

    // Title: frame range on top, ATE (the number the user explicitly asked
    // to keep visible on every thumbnail) on the bottom.
    painter.setPen(QColor(220, 220, 220));
    QFont f = painter.font();
    f.setPointSizeF(f.pointSizeF() * 0.85);
    painter.setFont(f);
    painter.drawText(rect().adjusted(4, 2, -4, 0), Qt::AlignTop | Qt::AlignLeft,
                      QStringLiteral("frame %1<->%2").arg(m_result.oldFrameIndex).arg(m_result.newFrameIndex));

    painter.setPen(m_result.ok && m_result.ateMatchedPoints > 0 ? QColor(255, 220, 140) : QColor(200, 100, 100));
    const QString ateText = (m_result.ok && m_result.ateMatchedPoints > 0)
                                 ? QStringLiteral("ATE RMSE: %1 m").arg(m_result.ateRmse, 0, 'f', 2)
                                 : QStringLiteral("ATE: n/a");
    painter.drawText(rect().adjusted(4, 0, -4, -4), Qt::AlignBottom | Qt::AlignLeft, ateText);
}
