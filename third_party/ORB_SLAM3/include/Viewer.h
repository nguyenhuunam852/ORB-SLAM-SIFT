// Minimal stub replacing the real (Pangolin-dependent) Viewer class in this
// vendored, headless-only copy of ORB-SLAM3 (see System.cc/MapDrawer.h's own
// doc comments on why Viewer.cc/.h were excluded). mpViewer is ALWAYS
// nullptr in this build (System's constructor throws if bUseViewer=true, see
// System.cc), so none of these methods are ever actually invoked at
// runtime -- every call site is guarded by `if (mpViewer)`. This stub exists
// purely so those (dead but still compiled) call sites resolve against a
// complete type instead of an incomplete forward declaration.
#ifndef VIEWER_H
#define VIEWER_H

namespace ORB_SLAM3
{

class Viewer
{
public:
    void RequestStop() {}
    bool isStopped() { return true; }
    void Release() {}
};

} // namespace ORB_SLAM3

#endif // VIEWER_H
