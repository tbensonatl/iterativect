#ifndef _UTILS_H_
#define _UTILS_H_

// FillCylinder populates a cylindrical shape of the specified radius
// in img centered in-plane at (xcen, ycen). The cylinder is in a 3D
// volume with width, height, and depth of nx, ny, and nz, respectively.
// The voxels in the cylinder have value val and outside the cylinder
// have value 0.0f. This function is primarily used to generate a
// test volume that is then forward projected to test the projector
// implementation.
void FillCylinder(float *img, int nx, int ny, int nz, float xcen, float ycen,
                  float radius, float val);

#endif // _UTILS_H_