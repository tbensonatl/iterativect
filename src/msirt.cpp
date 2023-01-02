#include <cmath>
#include <vector>

#include "helpers.h"
#include "kernels.h"
#include "msirt.h"
#include "system_geom.h"

#include <cuda_runtime_api.h>

size_t MSIRT::GetDeviceMemoryRequirement(const DataSet3rdGen &data_set,
                                         const ReconParams &recon_params) {
    const size_t num_voxels =
        recon_params.nx * recon_params.ny * recon_params.nz;
    const size_t num_detector_cells =
        data_set.geom.num_detector_channels * data_set.geom.num_detector_rows;
    const size_t num_projections = data_set.num_projections;
    const size_t num_dexels = num_detector_cells * num_projections;
    float dummy = 0.0f;
    // This helper function only needs the dummy pointers because it needs
    // to know the type of the input and output data in order to compute
    // the necessary work buffer size.
    const size_t workbuf_size =
        GetColumnSumMaxWorkBufSize(&dummy, &dummy, num_voxels, 0);
    return sizeof(float) * num_voxels +          // m_dev_image
           sizeof(float) * num_voxels +          // m_dev_update
           sizeof(uint16_t) * num_dexels +       // m_dev_proj
           sizeof(float) * num_dexels +          // m_dev_errsino
           sizeof(float3) * num_projections +    // m_dev_focal_spot_offsets
           sizeof(float3) * num_detector_cells + // m_dev_det_centers
           sizeof(float) * recon_params.num_subsets + // m_dev_inv_max_colsums
           sizeof(float) +                            // m_dev_errnorm
           workbuf_size;
}

void MSIRT::Reconstruct(int16_t *volume, const DataSet3rdGen &data_set,
                        const ReconParams &recon_params) {
    m_recon_params = recon_params;

    const int num_voxels =
        m_recon_params.nx * m_recon_params.ny * m_recon_params.nz;
    const int num_det_rows = data_set.geom.num_detector_rows;
    const int num_det_cols = data_set.geom.num_detector_channels;
    const int num_detector_cells = num_det_rows * num_det_cols;
    const int num_projections = data_set.num_projections;
    cudaChecked(cudaMalloc((void **)&m_dev_image, sizeof(float) * num_voxels));
    cudaChecked(
        cudaMalloc((void **)&m_dev_errsino,
                   sizeof(float) * num_projections * num_detector_cells));
    cudaChecked(
        cudaMalloc((void **)&m_dev_proj,
                   sizeof(uint16_t) * num_projections * num_detector_cells));
    cudaChecked(cudaMalloc((void **)&m_dev_focal_spot_offsets,
                           sizeof(float3) * num_projections));
    cudaChecked(cudaMalloc((void **)&m_dev_det_centers,
                           sizeof(float3) * num_detector_cells));
    cudaChecked(cudaMalloc((void **)&m_dev_inv_max_colsums,
                           sizeof(float) * m_recon_params.num_subsets));
    cudaChecked(cudaMalloc((void **)&m_dev_update, sizeof(float) * num_voxels));
    cudaChecked(cudaMalloc((void **)&m_dev_errnorm, sizeof(float)));

    cudaChecked(cudaStreamCreate(&m_stream));

    m_dev_reduction_workbuf_size = GetColumnSumMaxWorkBufSize(
        m_dev_image, m_dev_inv_max_colsums, num_voxels, m_stream);
    cudaChecked(cudaStreamSynchronize(m_stream));
    cudaChecked(
        cudaMalloc(&m_dev_reduction_workbuf, m_dev_reduction_workbuf_size));

    std::vector<float3> detector_coords;
    GetDetectorCoords3rdGen(data_set.geom, detector_coords);

    // The host memory is not pinned, so we do not bother with async transfers.
    // The workflow of a CT iterative recon is that a large batch of data is
    // moved to the GPU and then processed for a long period of time, so faster
    // and overlapped transfers are less critical than many other applications.
    // If we were reconstructing a large series of 2D sinograms into slices, for
    // example, then optimizing the device-to-host and host-to-device transfers
    // would be more important.
    cudaChecked(
        cudaMemcpy(m_dev_focal_spot_offsets, data_set.focal_spot_offsets.get(),
                   sizeof(float3) * num_projections, cudaMemcpyHostToDevice));
    cudaChecked(cudaMemcpy(m_dev_det_centers, detector_coords.data(),
                           sizeof(float3) * num_detector_cells,
                           cudaMemcpyHostToDevice));
    cudaChecked(
        cudaMemcpy(m_dev_proj, data_set.data.get(),
                   sizeof(uint16_t) * num_projections * num_detector_cells,
                   cudaMemcpyHostToDevice));

    // The traditional SIRT algorithm is:
    //   x^{k+1} = x^{k} + CA^TR(b - Ax^{k})
    // where x^{k} is the k-th image/volume, A is the forward model, A^T is
    // A transposed, b is the original post-log-normalized projection data,
    // R is a diagonal matrix where r_{i,i} is the inverse of the i-th row
    // sum, and C is a diagonal matrix where c_{i,i} is the inverse of the
    // i-th column sum. The C matrix is a preconditioner, or a per-voxel
    // step-size / learning rate. The Modified SIRT (MSIRT) algorithm
    // replaces C with a scalar corresponding to the inverse of the largest
    // column sum. MSIRT still converges, but no longer requires volume-sized
    // weights to be stored for each subset. The PSIRT algorithm applies a
    // scalar factor of 1.0 < alpha < 2.0 where in practice alpha is close
    // to 2.0. PSIRT still converges based on knowledge of the bounds of
    // the minimum and maximum eigenvalues of the iteration matrix, but it
    // generally does so more quickly than MSIRT due to the larger step size.
    ComputeInverseColumnSums(data_set);

    const struct RayDriven3rdGenParams params = {
        .source_to_iso_mm = data_set.geom.source_to_iso_mm,
        .first_view_phi = data_set.source_phi[0],
        .del_phi = data_set.geom.del_phi,
        .source_z_start = data_set.source_z_offsets[0],
        .mm_per_view =
            data_set.source_z_offsets[1] - data_set.source_z_offsets[0],
        .num_rows = data_set.geom.num_detector_rows,
        .num_cols = data_set.geom.num_detector_channels,
        .num_views = data_set.num_projections,
        .num_subsets = m_recon_params.num_subsets,
        .nx = m_recon_params.nx,
        .ny = m_recon_params.ny,
        .nz = m_recon_params.nz,
        .dx = m_recon_params.dx,
        .dy = m_recon_params.dy,
        .dz = m_recon_params.dz,
        .xcen = m_recon_params.xcen,
        .ycen = m_recon_params.ycen,
        .zcen = m_recon_params.zcen,
        .rescale_intercept = data_set.rescale_intercept,
        .rescale_slope = data_set.rescale_slope,
        .proj_data = m_dev_proj,
        .det_centers = m_dev_det_centers,
        .focal_spot_offsets = m_dev_focal_spot_offsets,
    };

    cudaEvent_t fp_start, fp_done, bp_start, bp_done;
    cudaChecked(cudaEventCreate(&fp_start));
    cudaChecked(cudaEventCreate(&fp_done));
    cudaChecked(cudaEventCreate(&bp_start));
    cudaChecked(cudaEventCreate(&bp_done));

    const float alpha = 1.99f;
    cudaChecked(
        cudaMemsetAsync(m_dev_image, 0, sizeof(float) * num_voxels, m_stream));
    for (int iter = 1; iter <= m_recon_params.num_iterations; iter++) {
        for (int subset = 0; subset < m_recon_params.num_subsets; subset++) {
            cudaChecked(cudaEventRecord(fp_start, m_stream));
            // The image is initialized to 0, so the first forward projection
            // will generate an output of all zeros and can be skipped
            if (iter > 1 || subset > 0) {
                // We do not need to zero out m_dev_errsino because each
                // value will be written (not accumulated) during forward
                // projection
                RayDriven3rdGenForwardProj(m_dev_errsino, m_dev_image, subset,
                                           params, m_stream);
            } else {
                // The error sinogram is b - Ax. When x is zero, i.e. during
                // the first iteration and subset, the error sinogram is just b.
                PopulateErrSinoWithProjData(m_dev_errsino, subset, params,
                                            m_stream);
            }
            cudaChecked(cudaEventRecord(fp_done, m_stream));

            cudaChecked(cudaMemsetAsync(m_dev_update, 0,
                                        sizeof(float) * num_voxels, m_stream));

            cudaChecked(cudaEventRecord(bp_start, m_stream));
            const bool do_weight = true;
            RayDriven3rdGenBackProj(m_dev_update, m_dev_errnorm, m_dev_errsino,
                                    subset, do_weight, params, m_stream);
            cudaChecked(cudaEventRecord(bp_done, m_stream));

            MergeScalarWeightedImageUpdate(m_dev_image, m_dev_update,
                                           alpha * m_inv_max_colsums[subset],
                                           num_voxels, m_stream);

            float errnorm = 0.0f;
            cudaChecked(cudaMemcpyAsync(&errnorm, m_dev_errnorm, sizeof(float),
                                        cudaMemcpyDeviceToHost, m_stream));
            cudaChecked(cudaStreamSynchronize(m_stream));
            float elapsed_fp_ms, elapsed_bp_ms;
            cudaChecked(
                cudaEventElapsedTime(&elapsed_fp_ms, fp_start, fp_done));
            cudaChecked(
                cudaEventElapsedTime(&elapsed_bp_ms, bp_start, bp_done));
            LOG("Partial error norm after iteration %d subset %d: %.6e "
                "[elapsed ms: fp %.3f, bp %.3f]",
                iter, subset + 1, sqrt(errnorm), elapsed_fp_ms, elapsed_bp_ms);
        }
    }

    ConvertMuToHounsfieldUnits(reinterpret_cast<int16_t *>(m_dev_update),
                               m_dev_image, data_set.mu_water, num_voxels,
                               m_stream);

    cudaChecked(cudaDeviceSynchronize());
    cudaChecked(cudaGetLastError());

    cudaChecked(cudaMemcpy(volume, m_dev_update, sizeof(int16_t) * num_voxels,
                           cudaMemcpyDeviceToHost));

    cudaChecked(cudaEventDestroy(fp_start));
    cudaChecked(cudaEventDestroy(fp_done));
    cudaChecked(cudaEventDestroy(bp_start));
    cudaChecked(cudaEventDestroy(bp_done));
}

void MSIRT::ComputeInverseColumnSums(const DataSet3rdGen &data_set) {
    const int num_dexels = data_set.num_projections *
                           data_set.geom.num_detector_rows *
                           data_set.geom.num_detector_channels;
    const int num_voxels =
        m_recon_params.nx * m_recon_params.ny * m_recon_params.nz;
    FillArray(m_dev_errsino, 1.0f, num_dexels, m_stream);

    const struct RayDriven3rdGenParams params = {
        .source_to_iso_mm = data_set.geom.source_to_iso_mm,
        .first_view_phi = data_set.source_phi[0],
        .del_phi = data_set.geom.del_phi,
        .source_z_start = data_set.source_z_offsets[0],
        .mm_per_view =
            data_set.source_z_offsets[1] - data_set.source_z_offsets[0],
        .num_rows = data_set.geom.num_detector_rows,
        .num_cols = data_set.geom.num_detector_channels,
        .num_views = data_set.num_projections,
        .num_subsets = m_recon_params.num_subsets,
        .nx = m_recon_params.nx,
        .ny = m_recon_params.ny,
        .nz = m_recon_params.nz,
        .dx = m_recon_params.dx,
        .dy = m_recon_params.dy,
        .dz = m_recon_params.dz,
        .xcen = m_recon_params.xcen,
        .ycen = m_recon_params.ycen,
        .zcen = m_recon_params.zcen,
        .rescale_intercept = data_set.rescale_intercept,
        .rescale_slope = data_set.rescale_slope,
        .proj_data = m_dev_proj,
        .det_centers = m_dev_det_centers,
        .focal_spot_offsets = m_dev_focal_spot_offsets,
    };

    // We do not want to weight the column sum values by the inverse row sums
    const bool do_weight = false;
    for (int s = 0; s < m_recon_params.num_subsets; s++) {
        cudaChecked(cudaMemsetAsync(m_dev_image, 0, sizeof(float) * num_voxels,
                                    m_stream));

        RayDriven3rdGenBackProj(m_dev_image, nullptr, m_dev_errsino, s,
                                do_weight, params, m_stream);

        CalcColumnSumMax(m_dev_image, m_dev_inv_max_colsums + s,
                         m_dev_reduction_workbuf, m_dev_reduction_workbuf_size,
                         num_voxels, 0);
    }

    InvertArray(m_dev_inv_max_colsums, m_recon_params.num_subsets, m_stream);
    m_inv_max_colsums.resize(m_recon_params.num_subsets);
    cudaChecked(cudaMemcpyAsync(m_inv_max_colsums.data(), m_dev_inv_max_colsums,
                                sizeof(float) * m_recon_params.num_subsets,
                                cudaMemcpyDeviceToHost, m_stream));
    cudaChecked(cudaStreamSynchronize(m_stream));
}

MSIRT::~MSIRT() {
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_image);
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_update);
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_errsino);
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_errnorm);
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_proj);
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_focal_spot_offsets);
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_det_centers);
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_inv_max_colsums);
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_reduction_workbuf);
    cudaStreamDestroy(m_stream);
    m_stream = 0;
}