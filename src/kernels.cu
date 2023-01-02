#include <cstdio>

#include "helpers.h"
#include "kernels.h"

#include <cub/cub.cuh>

#define MIN_ALPHA_INTERSECTION_LENGTH (1.0e-2f)

static int idivup(int x, int y) { return (x + y - 1) / y; }

static inline __device__ float CalcAlpha(float v0, int index, float dv,
                                         float v1, float v2) {
    return (v0 + index * dv - v1) / (v2 - v1);
}

static inline __device__ float max3(float a, float b, float c) {
    const float tmp = a >= b ? a : b;
    return tmp >= c ? tmp : c;
}

static inline __device__ float min3(float a, float b, float c) {
    const float tmp = a <= b ? a : b;
    return tmp <= c ? tmp : c;
}

// For the ray-driven projectors, we use the terminology specified in:
// Sundermann, Erik & Jacobs, Filip & Christiaens, Mark & De Sutter, Bjorn &
// Lemahieu, Ignace. (1998). A Fast Algorithm to Calculate the Exact
// Radiological Path Through a Pixel Or Voxel Space. Journal of Computing
// and Information Technology. 6.
// For the primary forward and backprojector loop, we need to calculate:
//   alpha_min, alpha_max : min and max parameterized line values
//   {i,j,k}_{min, max} : min and max voxel indices in the x (i), y (j),
//     and z (k) dimensions
// The logic is shared for both projectors, so we provide a utility function
// to calculate all of the values.
struct RayDrivenBounds {
    float alpha_min;
    float alpha_max;
    float alpha_x_start;
    float alpha_y_start;
    float alpha_z_start;
    int num_intersections;
    int i_start;
    int j_start;
    int k_start;
};

static inline __device__ RayDrivenBounds CalcRayBounds(float3 src, float3 det,
                                                       int nx, int ny, int nz,
                                                       float dx, float dy,
                                                       float dz, float ycen,
                                                       float zcen) {
    const float x0 = -1.0f * (nx / 2) * dx;
    const float y0 = -1.0f * (ny / 2) * dy + ycen;
    const float z0 = -1.0f * (nz / 2) * dz + zcen;

    const float alpha_x0 = CalcAlpha(x0, 0, dx, src.x, det.x);
    const float alpha_xn = CalcAlpha(x0, nx - 1, dx, src.x, det.x);
    const float alpha_y0 = CalcAlpha(y0, 0, dy, src.y, det.y);
    const float alpha_yn = CalcAlpha(y0, ny - 1, dy, src.y, det.y);
    const float alpha_z0 = CalcAlpha(z0, 0, dz, src.z, det.z);
    const float alpha_zn = CalcAlpha(z0, nz - 1, dz, src.z, det.z);

    // Depending on the positioning of the source and detector relative
    // to the volume, the alpha values for the first and last voxel
    // along a given axis could be oriented min-to-max or max-to-min
    // in terms of alpha values.
    float alpha_xmin, alpha_xmax, alpha_ymin, alpha_ymax, alpha_zmin,
        alpha_zmax;
    if (alpha_x0 < alpha_xn) {
        alpha_xmin = alpha_x0;
        alpha_xmax = alpha_xn;
    } else {
        alpha_xmin = alpha_xn;
        alpha_xmax = alpha_x0;
    }
    if (alpha_y0 < alpha_yn) {
        alpha_ymin = alpha_y0;
        alpha_ymax = alpha_yn;
    } else {
        alpha_ymin = alpha_yn;
        alpha_ymax = alpha_y0;
    }
    if (alpha_z0 < alpha_zn) {
        alpha_zmin = alpha_z0;
        alpha_zmax = alpha_zn;
    } else {
        alpha_zmin = alpha_zn;
        alpha_zmax = alpha_z0;
    }

    const float alpha_min = max3(alpha_xmin, alpha_ymin, alpha_zmin);
    const float alpha_max = min3(alpha_xmax, alpha_ymax, alpha_zmax);

    RayDrivenBounds bounds;
    bounds.alpha_min = alpha_min;
    bounds.alpha_max = alpha_max;

    // Early-exit if this line is not considered a valid intersection. It is
    // assumed that the caller will perform this same check upon return and not
    // use the remaining fields from the RayDrivenBounds struct
    if (bounds.alpha_max - bounds.alpha_min <= MIN_ALPHA_INTERSECTION_LENGTH) {
        return bounds;
    }

    int i_min, i_max, j_min, j_max, k_min, k_max;

    // See equations (11)-(18) of the above-referenced paper for the following
    if (src.x < det.x) {
        i_min = (alpha_min == alpha_xmin)
                    ? 1
                    : static_cast<int>(ceilf(
                          (src.x + alpha_min * (det.x - src.x) - x0) / dx));
        i_max = (alpha_max == alpha_xmax)
                    ? nx - 1
                    : static_cast<int>(floorf(
                          (src.x + alpha_max * (det.x - src.x) - x0) / dx));
    } else {
        i_max = (alpha_min == alpha_xmin)
                    ? nx - 2
                    : static_cast<int>(floorf(
                          (src.x + alpha_min * (det.x - src.x) - x0) / dx));
        i_min = (alpha_max == alpha_xmax)
                    ? 0
                    : static_cast<int>(ceilf(
                          (src.x + alpha_max * (det.x - src.x) - x0) / dx));
    }

    if (src.y < det.y) {
        j_min = (alpha_min == alpha_ymin)
                    ? 1
                    : static_cast<int>(ceilf(
                          (src.y + alpha_min * (det.y - src.y) - y0) / dy));
        j_max = (alpha_max == alpha_ymax)
                    ? ny - 1
                    : static_cast<int>(floorf(
                          (src.y + alpha_max * (det.y - src.y) - y0) / dy));
    } else {
        j_max = (alpha_min == alpha_ymin)
                    ? ny - 2
                    : static_cast<int>(floorf(
                          (src.y + alpha_min * (det.y - src.y) - y0) / dy));
        j_min = (alpha_max == alpha_ymax)
                    ? 0
                    : static_cast<int>(ceilf(
                          (src.y + alpha_max * (det.y - src.y) - y0) / dy));
    }

    if (src.z < det.z) {
        k_min = (alpha_min == alpha_zmin)
                    ? 1
                    : static_cast<int>(ceilf(
                          (src.z + alpha_min * (det.z - src.z) - z0) / dz));
        k_max = (alpha_max == alpha_zmax)
                    ? nz - 1
                    : static_cast<int>(floorf(
                          (src.z + alpha_max * (det.z - src.z) - z0) / dz));
    } else {
        k_max = (alpha_min == alpha_zmin)
                    ? nz - 2
                    : static_cast<int>(floorf(
                          (src.z + alpha_min * (det.z - src.z) - z0) / dz));
        k_min = (alpha_max == alpha_zmax)
                    ? 0
                    : static_cast<int>(ceilf(
                          (src.z + alpha_max * (det.z - src.z) - z0) / dz));
    }

    bounds.num_intersections =
        (i_max - i_min + 1) + (j_max - j_min + 1) + (k_max - k_min + 1);
    bounds.alpha_x_start = src.x < det.x
                               ? CalcAlpha(x0, i_min, dx, src.x, det.x)
                               : CalcAlpha(x0, i_max, dx, src.x, det.x);
    bounds.alpha_y_start = src.y < det.y
                               ? CalcAlpha(y0, j_min, dy, src.y, det.y)
                               : CalcAlpha(y0, j_max, dy, src.y, det.y);
    bounds.alpha_z_start = src.z < det.z
                               ? CalcAlpha(z0, k_min, dz, src.z, det.z)
                               : CalcAlpha(z0, k_max, dz, src.z, det.z);

    bounds.i_start = (src.x < det.x) ? i_min : i_max;
    bounds.j_start = (src.y < det.y) ? j_min : j_max;
    bounds.k_start = (src.z < det.z) ? k_min : k_max;

    return bounds;
}

static inline __device__ float3
GetSourceCoords(float source_phi, int view, float source_z,
                float source_to_iso_mm, const float3 *focal_spot_offsets) {
    float sinphi, cosphi;
    sincosf(source_phi + focal_spot_offsets[view].x, &sinphi, &cosphi);
    source_z += focal_spot_offsets[view].z;
    const float sid = source_to_iso_mm + focal_spot_offsets[view].y;
    return make_float3(-1.0f * sid * sinphi, sid * cosphi, source_z);
}

static inline __device__ float3 GetDetectorCoords(float source_phi, int view,
                                                  float source_z,
                                                  float source_to_iso_mm,
                                                  float3 det_orig, float3 src) {
    float sinphi, cosphi;
    sincosf(source_phi, &sinphi, &cosphi);
    float3 det = make_float3(det_orig.x * cosphi - det_orig.y * sinphi,
                             det_orig.x * sinphi + det_orig.y * cosphi,
                             source_z + det_orig.z);
    // Degenerate cases will prevent the below ray-following logic from
    // working, so perturb any destination point that is identical to
    // the source along any axis
    if (src.x == det.x) {
        det.x += 0.001f;
    }
    if (src.y == det.y) {
        det.y += 0.001f;
    }
    if (src.z == det.z) {
        det.z += 0.001f;
    }
    return det;
}

__global__ void RayDriven3rdGenBackProjKernel(float *image, float *err_norm,
                                              const float *errsino, int subset,
                                              bool do_weight,
                                              RayDriven3rdGenParams params) {

    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int view = params.num_subsets * blockIdx.z + subset;

    if (col >= params.num_cols || row >= params.num_rows ||
        view >= params.num_views) {
        return;
    }

    const float proj_val = errsino[view * params.num_rows * params.num_cols +
                                   row * params.num_cols + col];
    if (proj_val == 0.0f) {
        return;
    }

    const float phi = params.first_view_phi + view * params.del_phi;
    const float3 src = GetSourceCoords(
        phi, view, params.source_z_start + view * params.mm_per_view,
        params.source_to_iso_mm, params.focal_spot_offsets);

    const float3 det_orig = params.det_centers[row * params.num_cols + col];
    const float3 det = GetDetectorCoords(
        phi, view, params.source_z_start + view * params.mm_per_view,
        params.source_to_iso_mm, det_orig, src);

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    const RayDrivenBounds bounds =
        CalcRayBounds(src, det, nx, ny, nz, params.dx, params.dy, params.dz,
                      params.ycen, params.zcen);
    if (bounds.alpha_max - bounds.alpha_min <= MIN_ALPHA_INTERSECTION_LENGTH) {
        return;
    }

    const int num_intersections = bounds.num_intersections;
    float next_alpha_x = bounds.alpha_x_start;
    float next_alpha_y = bounds.alpha_y_start;
    float next_alpha_z = bounds.alpha_z_start;
    float alpha_current = bounds.alpha_min;
    const float dconv = sqrtf((src.x - det.x) * (src.x - det.x) +
                              (src.y - det.y) * (src.y - det.y) +
                              (src.z - det.z) * (src.z - det.z));
    const float weight = (bounds.alpha_max - bounds.alpha_min) * dconv;

    const float proj_val_weighted =
        (do_weight && weight > 0.0f) ? proj_val / weight : proj_val;
    if (weight > 0.0f && err_norm) {
        atomicAdd(err_norm, proj_val * proj_val / weight);
    }

    const float x_inc = params.dx / fabs(det.x - src.x);
    const float y_inc = params.dy / fabs(det.y - src.y);
    const float z_inc = params.dz / fabs(det.z - src.z);

    int i = bounds.i_start;
    int j = bounds.j_start;
    int k = bounds.k_start;

    const int i_inc = (src.x < det.x) ? 1 : -1;
    const int j_inc = (src.y < det.y) ? 1 : -1;
    const int k_inc = (src.z < det.z) ? 1 : -1;

    for (int count = 0; count < num_intersections; count++) {
        const int img_ind = (k * ny + j) * nx + i;
        if (next_alpha_x <= next_alpha_y && next_alpha_x < next_alpha_z) {
            const float len = (next_alpha_x - alpha_current) * dconv;
            atomicAdd(image + img_ind, proj_val_weighted * len);
            alpha_current = next_alpha_x;
            next_alpha_x += x_inc;
            i += i_inc;
            if (i < 0 || i >= nx) {
                break;
            }
        } else if (next_alpha_y < next_alpha_x &&
                   next_alpha_y <= next_alpha_z) {
            const float len = (next_alpha_y - alpha_current) * dconv;
            atomicAdd(image + img_ind, proj_val_weighted * len);
            alpha_current = next_alpha_y;
            next_alpha_y += y_inc;
            j += j_inc;
            if (j < 0 || j >= ny) {
                break;
            }
        } else {
            const float len = (next_alpha_z - alpha_current) * dconv;
            atomicAdd(image + img_ind, proj_val_weighted * len);
            alpha_current = next_alpha_z;
            next_alpha_z += z_inc;
            k += k_inc;
            if (k < 0 || k >= nz) {
                break;
            }
        }
    }
}

__global__ void RayDriven3rdGenForwardProjKernel(float *errsino,
                                                 const float *image, int subset,
                                                 RayDriven3rdGenParams params) {

    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int view = params.num_subsets * blockIdx.z + subset;

    if (col >= params.num_cols || row >= params.num_rows ||
        view >= params.num_views) {
        return;
    }

    const float phi = params.first_view_phi + view * params.del_phi;
    const float3 src = GetSourceCoords(
        phi, view, params.source_z_start + view * params.mm_per_view,
        params.source_to_iso_mm, params.focal_spot_offsets);

    const float3 det_orig = params.det_centers[row * params.num_cols + col];
    const float3 det = GetDetectorCoords(
        phi, view, params.source_z_start + view * params.mm_per_view,
        params.source_to_iso_mm, det_orig, src);

    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;
    const RayDrivenBounds bounds =
        CalcRayBounds(src, det, nx, ny, nz, params.dx, params.dy, params.dz,
                      params.ycen, params.zcen);
    if (bounds.alpha_max - bounds.alpha_min <= MIN_ALPHA_INTERSECTION_LENGTH) {
        return;
    }

    const int num_intersections = bounds.num_intersections;
    float next_alpha_x = bounds.alpha_x_start;
    float next_alpha_y = bounds.alpha_y_start;
    float next_alpha_z = bounds.alpha_z_start;
    float alpha_current = bounds.alpha_min;
    const float dconv = sqrtf((src.x - det.x) * (src.x - det.x) +
                              (src.y - det.y) * (src.y - det.y) +
                              (src.z - det.z) * (src.z - det.z));

    const float x_inc = params.dx / fabs(det.x - src.x);
    const float y_inc = params.dy / fabs(det.y - src.y);
    const float z_inc = params.dz / fabs(det.z - src.z);

    int i = bounds.i_start;
    int j = bounds.j_start;
    int k = bounds.k_start;
    const int i_inc = (src.x < det.x) ? 1 : -1;
    const int j_inc = (src.y < det.y) ? 1 : -1;
    const int k_inc = (src.z < det.z) ? 1 : -1;

    float accum = 0.0f;
    for (int count = 0; count < num_intersections; count++) {
        const float img_val = image[(k * ny + j) * nx + i];
        if (next_alpha_x <= next_alpha_y && next_alpha_x < next_alpha_z) {
            const float len = (next_alpha_x - alpha_current) * dconv;
            accum += len * img_val;
            alpha_current = next_alpha_x;
            next_alpha_x += x_inc;
            i += i_inc;
            if (i < 0 || i >= nx) {
                break;
            }
        } else if (next_alpha_y < next_alpha_x &&
                   next_alpha_y <= next_alpha_z) {
            const float len = (next_alpha_y - alpha_current) * dconv;
            accum += len * img_val;
            alpha_current = next_alpha_y;
            next_alpha_y += y_inc;
            j += j_inc;
            if (j < 0 || j >= ny) {
                break;
            }
        } else {
            const float len = (next_alpha_z - alpha_current) * dconv;
            accum += len * img_val;
            alpha_current = next_alpha_z;
            next_alpha_z += z_inc;
            k += k_inc;
            if (k < 0 || k >= nz) {
                break;
            }
        }
    }

    const int proj_ind =
        view * params.num_rows * params.num_cols + row * params.num_cols + col;
    if (params.proj_data) {
        const float p = params.rescale_intercept +
                        params.rescale_slope * params.proj_data[proj_ind];
        errsino[proj_ind] = p - accum;
    } else {
        errsino[proj_ind] = accum;
    }
}

void RayDriven3rdGenForwardProj(float *errsino, const float *image, int subset,
                                RayDriven3rdGenParams params,
                                cudaStream_t stream) {
    dim3 block(64, 1);
    dim3 grid(idivup(params.num_cols, block.x),
              idivup(params.num_rows, block.y),
              idivup(params.num_views, params.num_subsets));
    RayDriven3rdGenForwardProjKernel<<<grid, block, 0, stream>>>(
        errsino, image, subset, params);
    cudaChecked(cudaGetLastError());
}

void RayDriven3rdGenBackProj(float *image, float *err_norm,
                             const float *errsino, int subset, bool do_weight,
                             RayDriven3rdGenParams params,
                             cudaStream_t stream) {
    dim3 block(64, 1);
    dim3 grid(idivup(params.num_cols, block.x),
              idivup(params.num_rows, block.y),
              idivup(params.num_views, params.num_subsets));
    if (err_norm) {
        cudaChecked(cudaMemset(err_norm, 0, sizeof(float)));
    }
    RayDriven3rdGenBackProjKernel<<<grid, block, 0, stream>>>(
        image, err_norm, errsino, subset, do_weight, params);
    cudaChecked(cudaGetLastError());
}

__global__ void InvertArrayKernel(float *array, int N) {
    const int ind = blockDim.x * blockIdx.x + threadIdx.x;
    if (ind >= N) {
        return;
    }
    const float v = array[ind];
    array[ind] = (v != 0) ? 1.0f / v : 0.0f;
}

__global__ void FillArrayKernel(float *array, float val, int N) {
    const int ind = blockDim.x * blockIdx.x + threadIdx.x;
    if (ind >= N) {
        return;
    }
    array[ind] = val;
}

__global__ void MergeScalarWeightedImageUpdateKernel(float *img,
                                                     const float *update,
                                                     float scalingFactor,
                                                     int N) {
    const int ind = blockDim.x * blockIdx.x + threadIdx.x;
    if (ind >= N) {
        return;
    }
    const float newval = img[ind] + scalingFactor * update[ind];
    img[ind] = (newval > 0.0f) ? newval : 0.0f;
}

__global__ void
PopulateErrSinoWithProjDataKernel(float *errsino, int subset,
                                  RayDriven3rdGenParams params) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int view = params.num_subsets * blockIdx.z + subset;

    if (col >= params.num_cols || row >= params.num_rows ||
        view >= params.num_views) {
        return;
    }

    const int proj_ind =
        view * params.num_rows * params.num_cols + row * params.num_cols + col;
    errsino[proj_ind] = params.rescale_intercept +
                        params.rescale_slope * params.proj_data[proj_ind];
}

__global__ void ConvertMuToHounsfieldUnitsKernel(int16_t *hu, const float *mu,
                                                 float inv_mu_water,
                                                 int num_voxels) {
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= num_voxels) {
        return;
    }
    const float h = 1000.0f * mu[ind] * inv_mu_water - 1000.0f;
    hu[ind] = min(max(-32768.0f, roundf(h)), 32767.0f);
}

void InvertArray(float *array, int N, cudaStream_t stream) {
    const dim3 block(256, 1, 1);
    const dim3 grid(idivup(N, block.x), 1, 1);
    InvertArrayKernel<<<grid, block, 0, stream>>>(array, N);
    cudaChecked(cudaGetLastError());
}

void FillArray(float *array, float val, int N, cudaStream_t stream) {
    const dim3 block(256, 1, 1);
    const dim3 grid(idivup(N, block.x), 1, 1);
    FillArrayKernel<<<grid, block, 0, stream>>>(array, val, N);
    cudaChecked(cudaGetLastError());
}

void MergeScalarWeightedImageUpdate(float *dev_img, const float *dev_update,
                                    float scalingFactor, int numVoxels,
                                    cudaStream_t stream) {
    const dim3 block(256, 1, 1);
    const dim3 grid(idivup(numVoxels, block.x), 1, 1);
    MergeScalarWeightedImageUpdateKernel<<<grid, block, 0, stream>>>(
        dev_img, dev_update, scalingFactor, numVoxels);
    cudaChecked(cudaGetLastError());
}

size_t GetColumnSumMaxWorkBufSize(const float *colsums, float *max_colsum,
                                  int num_voxels, cudaStream_t stream) {
    size_t workbuf_size_bytes{0};
    cub::DeviceReduce::Max(nullptr, workbuf_size_bytes, colsums, max_colsum,
                           num_voxels, stream);
    cudaChecked(cudaGetLastError());
    return workbuf_size_bytes;
}

void CalcColumnSumMax(const float *colsums, float *max_colsum, void *workbuf,
                      size_t workbuf_size, int num_voxels,
                      cudaStream_t stream) {
    cub::DeviceReduce::Max(workbuf, workbuf_size, colsums, max_colsum,
                           num_voxels, stream);
    cudaChecked(cudaGetLastError());
}

void PopulateErrSinoWithProjData(float *errsino, int subset,
                                 RayDriven3rdGenParams params,
                                 cudaStream_t stream) {
    dim3 block(32, 4);
    dim3 grid(idivup(params.num_cols, block.x),
              idivup(params.num_rows, block.y),
              idivup(params.num_views, params.num_subsets));
    PopulateErrSinoWithProjDataKernel<<<grid, block, 0, stream>>>(
        errsino, subset, params);
    cudaChecked(cudaGetLastError());
}

void ConvertMuToHounsfieldUnits(int16_t *hu, const float *mu, float mu_water,
                                int num_voxels, cudaStream_t stream) {
    assert(mu_water > 0.0f);
    const dim3 block(256, 1, 1);
    const dim3 grid(idivup(num_voxels, block.x), 1, 1);
    ConvertMuToHounsfieldUnitsKernel<<<grid, block, 0, stream>>>(
        hu, mu, 1.0f / mu_water, num_voxels);
    cudaChecked(cudaGetLastError());
}