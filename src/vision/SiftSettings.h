#pragma once

// Parameters forwarded to cv::SIFT::create().
struct SiftSettings
{
    int nFeatures = 2000;            // 0 = unlimited; capped by default so brute-force
                                     // matching and F-RANSAC scoring stay real-time-ish
    int nOctaveLayers = 3;          // nLayer
    double contrastThreshold = 0.04;
    double edgeThreshold = 10.0;
    double sigma = 1.6;             // Gaussian sigma applied at octave 0
};
