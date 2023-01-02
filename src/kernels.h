#include <cstdint>

#include <vector_types.h>

struct RayDriven3rdGenParams {
    float source_to_iso_mm;
    float first_view_phi;
    float del_phi;
    float source_z_start;
    float mm_per_view;
    int num_rows;
    int num_cols;
    int num_views;
    int num_subsets;
    int nx;
    int ny;
    int nz;
    float dx;
    float dy;
    float dz;
    float xcen;
    float ycen;
    float zcen;
    float rescale_intercept;
    float rescale_slope;
    const uint16_t *proj_data;
    const float3 *det_centers;
    const float3 *focal_spot_offsets;
};

// If params.proj_data is nullptr, then errsino contains just the forward
// projection values. If proj is specified, errsino contains the error sinogram
// (i.e. b - Ax where b is proj and Ax is the forward projection).
// The form that computes just the forward projection (i.e. when proj
// is nullptr) is primarily to support debugging or other uses where
// the error sinogram is not needed.
void RayDriven3rdGenForwardProj(float *errsino, const float *image, int subset,
                                RayDriven3rdGenParams params,
                                cudaStream_t stream);

// If do_weight is true, then the projection values will be weighted by
// the inverse row sums during backprojection. This is the R in the
// update equation: x^{k+1} = x^{k} + CA^TR(b - Ax^{k}). Otherwise, this
// function just computes A^Ty where y is proj.
void RayDriven3rdGenBackProj(float *image, float *err_norm, const float *proj,
                             int subset, bool do_weight,
                             RayDriven3rdGenParams params, cudaStream_t stream);

// The reduction used in CalcColumnSumMax requires some scratch space,
// so get the required size via this function and allocate that work
// buffer prior to calling CalcColumnSumMax
size_t GetColumnSumMaxWorkBufSize(const float *colsums, float *max_colsum,
                                  int num_voxels, cudaStream_t stream);

void CalcColumnSumMax(const float *colsums, float *max_colsum, void *workbuf,
                      size_t workbuf_size, int num_voxels, cudaStream_t stream);

void InvertArray(float *array, int N, cudaStream_t stream);

void FillArray(float *array, float val, int N, cudaStream_t stream);

// Fill the projections of errsino needed for the specified subset
// with converted-to-float projection data. This is only needed for
// the initial error sinogram that has value b = b - Ax for x = 0
// due to initializing the image with zeros.
void PopulateErrSinoWithProjData(float *errsino, int subset,
                                 RayDriven3rdGenParams params,
                                 cudaStream_t stream);

// Apply the image update in dev_update to dev_image using a scaling
// factor of scalingFactor. The scaling factor is effectively a step
// size for the gradient accumulated in dev_update.
void MergeScalarWeightedImageUpdate(float *dev_img, const float *dev_update,
                                    float scalingFactor, int numVoxels,
                                    cudaStream_t stream);

// Convert the reconstructed linear attenuation coefficients, mu, into
// Hounsfield units, hu, using mu_water as the linear attenuation coefficient
// of water. The output is rounded and clamped to int16 values.
void ConvertMuToHounsfieldUnits(int16_t *hu, const float *mu, float mu_water,
                                int num_voxels, cudaStream_t stream);