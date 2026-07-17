#include <QApplication>

#include "MainWindow.h"
#include "vision/LoopEstimateTypes.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("SIFT vSLAM GUI"));
    QApplication::setOrganizationName(QStringLiteral("SIFT vSLAM GUI"));

    // Needed for LoopEstimateSnapshot/-Result to cross the SlamWorker
    // worker-thread <-> GUI-thread boundary via queued signal/slot
    // connections (see SlamWorker::loopClosureDetected(), MainWindow's
    // wiring of it to a background QtConcurrent task).
    qRegisterMetaType<LoopEstimateSnapshot>("LoopEstimateSnapshot");
    qRegisterMetaType<LoopEstimateResult>("LoopEstimateResult");

    MainWindow window;
    window.show();

    return QApplication::exec();
}
