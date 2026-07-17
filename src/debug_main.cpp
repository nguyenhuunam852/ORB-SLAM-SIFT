#include <QCoreApplication>
#include <QTimer>
#include <cstdio>
#include <cstring>

#include <opencv2/calib3d.hpp>

#include "vision/FeatureDetector.h"
#include "vision/SlamWorker.h"

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <video-path> [seconds] [pnp-method] [detector]\n", argv[0]);
        std::fprintf(stderr, "  pnp-method: p3p (default), dlt, iterative, epnp, ap3p, sqpnp\n");
        std::fprintf(stderr, "  detector:   sift (default), orb\n");
        return 1;
    }
    const QString path = QString::fromLocal8Bit(argv[1]);
    const int seconds = argc > 2 ? std::atoi(argv[2]) : 60;

    SlamWorker worker;

    if (argc > 3) {
        PnpSettings settings;
        const char *m = argv[3];
        if (std::strcmp(m, "dlt") == 0)
            settings.method = kPnpMethodDlt;
        else if (std::strcmp(m, "epnp") == 0)
            settings.method = cv::SOLVEPNP_EPNP;
        else if (std::strcmp(m, "iterative") == 0)
            settings.method = cv::SOLVEPNP_ITERATIVE;
        else if (std::strcmp(m, "ap3p") == 0)
            settings.method = cv::SOLVEPNP_AP3P;
        else if (std::strcmp(m, "sqpnp") == 0)
            settings.method = cv::SOLVEPNP_SQPNP;
        else
            settings.method = cv::SOLVEPNP_P3P;
        worker.setPnpSettings(settings);
        std::fprintf(stderr, "[config] pnp method=%s\n", m);
    }

    if (argc > 4 && std::strcmp(argv[4], "orb") == 0) {
        worker.setDetectorType(feature_detector::DetectorType::Orb);
        std::fprintf(stderr, "[config] detector=orb\n");
    }

    QObject::connect(&worker, &SlamWorker::trackingStateChanged, [](const QString &s) {
        std::fprintf(stderr, "[state] %s\n", s.toLocal8Bit().constData());
        std::fflush(stderr);
    });
    QObject::connect(&worker, &SlamWorker::statsUpdated, [&app](const QString &s) {
        std::fprintf(stderr, "[stats] %s\n", s.toLocal8Bit().constData());
        std::fflush(stderr);
        // Unthrottled processing can blow through the whole sequence in
        // well under `seconds` of wall-clock time -- quit as soon as the
        // stream naturally ends instead of idling until the fallback
        // timeout below.
        if (s == QStringLiteral("Stream ended"))
            app.quit();
    });
    QObject::connect(&worker, &SlamWorker::sourceOpened, [](bool ok, const QString &msg) {
        std::fprintf(stderr, "[open] ok=%d %s\n", ok, msg.toLocal8Bit().constData());
        std::fflush(stderr);
    });

    worker.openVideoFile(path);
    worker.startUnthrottled(); // no reason to wait on real-time playback pacing headlessly

    QTimer::singleShot(seconds * 1000, &app, &QCoreApplication::quit);
    const int result = app.exec();

    // Final (post-loop-closure-correction) trajectory dump for offline ATE
    // evaluation against ground truth -- see the trajectoryPoints() doc
    // comment in SlamWorker.h.
    const QVector<QPointF> &traj = worker.trajectoryPoints();
    const QVector<int> &frames = worker.trajectoryFrameIndices();
    for (int i = 0; i < traj.size(); ++i) {
        std::fprintf(stderr, "[traj] frame=%d x=%.6f z=%.6f\n", frames[i], traj[i].x(), traj[i].y());
    }

    return result;
}
