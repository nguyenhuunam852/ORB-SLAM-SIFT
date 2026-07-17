#pragma once

#include <QPointF>
#include <QPushButton>
#include <QVector>
#include <QWidget>

class QPaintEvent;
class QResizeEvent;

// Simple top-down (world X / world Z) plot of the estimated camera
// trajectory and the sparse 3D map, auto-scaled to fit the visible data.
// Optionally overlays a loaded ground-truth trajectory (e.g. a KITTI
// poses.txt) as a green line for visual comparison.
//
// Monocular SLAM's world frame has an arbitrary origin, orientation and
// scale (fixed only by whatever the first two-view initialization happened
// to produce) -- it has no reason to line up with ground truth's world
// frame. Plotting both raw would make even a perfectly accurate estimate
// look like a completely different, unrelated shape. So before drawing,
// the estimated trajectory (and map points) are aligned into the ground
// truth's frame via a least-squares similarity transform (2D Umeyama:
// scale + rotation + translation) fit from the trajectory points paired
// with ground truth by frame index -- only then does the overlay actually
// mean anything.
class MapView : public QWidget
{
    Q_OBJECT

public:
    explicit MapView(QWidget *parent = nullptr);

public slots:
    void setMapData(const QVector<QPointF> &trajectory, const QVector<QPointF> &mapPoints,
                     const QVector<int> &trajectoryFrameIndices);

    // Disables computeAlignment()'s freeze-after-200-points behavior (see
    // its own doc comment) -- the custom pipeline's early trajectory is
    // stable enough that freezing onto it works well, but the vendored
    // ORB-SLAM3 mode's first ~200-350 tracked points routinely land inside
    // its opening map-reset/loop-closure churn (confirmed this session: a
    // live run froze its alignment on that early unstable segment and then
    // stayed visibly misaligned for the rest of a run that otherwise tracked
    // a full sequence cleanly, including a successful map merge at keyframe
    // 1671). Recomputing every frame instead means the overlay stays
    // accurate to ORB-SLAM3's own internal loop-closure/GBA corrections as
    // they happen, at the cost of visible wobble/rescale each time those
    // corrections land -- an explicit, accepted tradeoff for this mode, not
    // a bug. Wired to ControlPanel's ORB-SLAM3 checkbox (see MainWindow).
    void setContinuousAlignmentEnabled(bool enabled);

    // Loads a KITTI-format poses.txt (one 3x4 [R|t] per line) and repaints
    // if successful. Returns false (no-op on the current overlay) if the
    // file doesn't exist or doesn't parse -- used both by the manual
    // "Load Ground Truth" button and MainWindow's auto-detect-on-open.
    bool loadGroundTruthFile(const QString &path);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void browseForGroundTruth();

private:
    bool parseGroundTruthFile(const QString &path);

    // Least-squares similarity (scale/rotation/translation) mapping
    // m_trajectory points into the ground-truth frame, fit from points
    // paired by frame index (trajectory frame i <-> ground truth line
    // i - 1, KITTI's convention). Returns false (no alignment available)
    // if there's not yet enough overlapping data -- callers should fall
    // back to drawing the trajectory unaligned in that case.
    //
    // Freezes once kAlignmentFreezeMinPoints points are available rather
    // than refitting on every repaint: a live-updating fit from a small,
    // early slice of the trajectory is a much worse fit than the eventual
    // complete one, so continuously refitting made the overlay visibly
    // wobble/rescale as the trajectory grew -- easily mistaken for the
    // *trajectory* going wrong when it was actually the *alignment*
    // chasing a moving target. Freezing trades that wobble for honestly
    // showing any real drift as visible divergence from the ground-truth
    // line instead of silently re-absorbing it into a new fit every frame.
    bool computeAlignment(double &scale, double &cosTheta, double &sinTheta, QPointF &translation) const;

    QVector<QPointF> m_trajectory;
    QVector<int> m_trajectoryFrameIndex;
    QVector<QPointF> m_mapPoints;
    QVector<QPointF> m_groundTruth;
    QString m_groundTruthFileName;
    QPushButton *m_loadGtButton;

    mutable bool m_alignmentFrozen = false;
    mutable double m_frozenScale = 1.0;
    mutable double m_frozenCos = 1.0;
    mutable double m_frozenSin = 0.0;
    mutable QPointF m_frozenTranslation;

    bool m_continuousAlignmentEnabled = false; // see setContinuousAlignmentEnabled()
};
