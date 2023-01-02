#ifndef _RECON_PARAMS_H_
#define _RECON_PARAMS_H_

struct ReconParams {
    // Image width, in pixels
    int nx{0};
    // Image height, in pixels
    int ny{0};
    // Number of slices, in pixels
    int nz{0};
    // Pixel width, in mm
    float dx{0.0f};
    // Pixel height, in mm
    float dy{0.0f};
    // Slice width (i.e. z dimension of voxel), in mm
    float dz{0.0f};
    // Number of iterations
    int num_iterations{0};
    // Number of ordered subsets per iteration
    int num_subsets{0};
    // Center of the reconstruction volume in the x dimension, mm
    float xcen{0.0f};
    // Center of the reconstruction volume in the y dimension, mm
    float ycen{0.0f};
    // Center of the reconstruction volume in the z dimension, mm
    float zcen{0.0f};
};

#endif // _RECON_PARAMS_H_