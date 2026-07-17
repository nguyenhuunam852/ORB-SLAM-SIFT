#pragma once

#include "LoopEstimateTypes.h"

// Runs entirely on whatever thread calls it (meant to be invoked via
// QtConcurrent::run() from a background thread-pool thread, never the
// SlamWorker worker thread or the GUI thread) -- pure function over a
// self-contained snapshot, touches no shared/live state, safe to run
// concurrently with live tracking. Two things it does that live tracking's
// runLoopBundleAdjustment() can't, both explicitly requested after
// diagnosing why that live BA underperforms:
//   1. Re-matches every keyframe's full descriptors against every EARLIER
//      keyframe's in the same window (an unevicted, in-memory-only pool,
//      unlike the live rolling map's kMaxMapPoints cap) -- giving BA the
//      longer, better-connected landmark tracks it was missing, without
//      slowing down live tracking's own hot path at all.
//   2. Computes ATE against ground truth (if loaded), so the effect of (1)
//      is measured, not just assumed.
LoopEstimateResult computeLoopEstimate(LoopEstimateSnapshot snapshot);
