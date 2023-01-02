#include "utils.h"

#include <cmath>

void FillCylinder(float *img, int nx, int ny, int nz, float xcen, float ycen,
                  float radius, float val) {
    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            const float y = (-1.0f * (ny / 2) + iy);
            for (int ix = 0; ix < nx; ix++) {
                const float x = (-1.0f * (nx / 2) + ix);
                const float dist =
                    sqrtf((xcen - x) * (xcen - x) + (ycen - y) * (ycen - y));
                if (dist <= radius) {
                    img[(iz * ny + iy) * nx + ix] = val;
                }
            }
        }
    }
}
