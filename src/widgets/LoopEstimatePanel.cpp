#include "LoopEstimatePanel.h"

#include "LoopMapThumbnail.h"

#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QVBoxLayout>

LoopEstimatePanel::LoopEstimatePanel(QWidget *parent)
    : QWidget(parent)
{
    auto *group = new QGroupBox(QStringLiteral("Background Loop-Closure Re-Estimate"), this);
    auto *layout = new QHBoxLayout(group);

    m_statusLabel = new QLabel(QStringLiteral("No loop closure yet."), group);
    m_enrichmentLabel = new QLabel(group);
    m_baLabel = new QLabel(group);
    m_ateLabel = new QLabel(group);
    for (QLabel *label : {m_statusLabel, m_enrichmentLabel, m_baLabel, m_ateLabel}) {
        label->setTextInteractionFlags(Qt::TextSelectableByMouse);
        layout->addWidget(label);
    }
    layout->addStretch(1);

    // Horizontally-scrollable strip of per-loop mini-maps, queued left to
    // right (loop 1..N) as each background re-estimate resolves -- see
    // LoopMapThumbnail. A plain QWidget + QHBoxLayout inside a QScrollArea
    // so the strip can grow arbitrarily wide without resizing this panel.
    auto *stripWidget = new QWidget(this);
    m_mapStripLayout = new QHBoxLayout(stripWidget);
    m_mapStripLayout->setContentsMargins(4, 4, 4, 4);
    m_mapStripLayout->addStretch(1);

    m_mapScrollArea = new QScrollArea(this);
    m_mapScrollArea->setWidget(stripWidget);
    m_mapScrollArea->setWidgetResizable(false);
    m_mapScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_mapScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_mapScrollArea->setFixedHeight(190);

    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->addWidget(group);
    outer->addWidget(m_mapScrollArea);
}

void LoopEstimatePanel::showRunning(int oldFrameIndex, int newFrameIndex)
{
    m_statusLabel->setText(
        QStringLiteral("Loop closure frame %1 <-> %2: re-estimating in background...")
            .arg(oldFrameIndex)
            .arg(newFrameIndex));
    m_enrichmentLabel->clear();
    m_baLabel->clear();
    m_ateLabel->clear();
}

void LoopEstimatePanel::showResult(const LoopEstimateResult &result)
{
    m_statusLabel->setText(QStringLiteral("Loop closure frame %1 <-> %2%3")
                                .arg(result.oldFrameIndex)
                                .arg(result.newFrameIndex)
                                .arg(result.ok ? QString() : QStringLiteral(" -- %1").arg(result.message)));

    m_enrichmentLabel->setText(QStringLiteral("Landmarks: %1 -> %2 | Observations: %3 -> %4")
                                    .arg(result.landmarksBefore)
                                    .arg(result.landmarksAfter)
                                    .arg(result.observationsBefore)
                                    .arg(result.observationsAfter));

    m_baLabel->setText(QStringLiteral("BA: %1, cost %2 -> %3")
                            .arg(result.baConverged ? QStringLiteral("converged") : QStringLiteral("failed"))
                            .arg(result.baInitialCost, 0, 'f', 1)
                            .arg(result.baFinalCost, 0, 'f', 1));

    if (result.ok && result.ateMatchedPoints > 0) {
        m_ateLabel->setText(QStringLiteral("ATE RMSE: %1 m (mean %2, median %3, max %4, n=%5, scale %6)")
                                 .arg(result.ateRmse, 0, 'f', 2)
                                 .arg(result.ateMean, 0, 'f', 2)
                                 .arg(result.ateMedian, 0, 'f', 2)
                                 .arg(result.ateMax, 0, 'f', 2)
                                 .arg(result.ateMatchedPoints)
                                 .arg(result.recoveredScale, 0, 'f', 3));
    } else {
        m_ateLabel->setText(QStringLiteral("ATE: unavailable (%1)").arg(result.message));
    }

    // Queue this loop's mini-map at the right end of the strip -- inserted
    // just before the trailing stretch so new thumbnails keep appending
    // left to right in resolution order (loop 1..N), no filtering/dedup.
    auto *thumbnail = new LoopMapThumbnail(result, m_mapScrollArea->widget());
    m_mapStripLayout->insertWidget(m_mapStripLayout->count() - 1, thumbnail);
    m_mapScrollArea->widget()->adjustSize();
}
