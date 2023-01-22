#ifndef _MSIRT_H_
#define _MSIRT_H_

#include <vector>

#include "data_set.h"
#include "recon_params.h"

#include <driver_types.h>
#include <vector_types.h>

// An implementation of Modified Simultaneous Iterative Reconstruction
// Technique (MSIRT) with support for ordered subsets.
class MSIRT {
  public:
    MSIRT() {}
    virtual ~MSIRT();

    void Reconstruct(int16_t *volume, const DataSet3rdGen &data_set,
                     const ReconParams &recon_params);

    // Returns the number of bytes of device memory needed to reconstruct
    // data_set using recon_params
    size_t GetDeviceMemoryRequirement(const DataSet3rdGen &data_set,
                                      const ReconParams &recon_params);

    // Do not allow copies via copy xtor or assignment due to the device
    // memory allocations
    MSIRT(const MSIRT &) = delete;
    MSIRT &operator=(const MSIRT &) = delete;

    // We could support move operations in the future if desired by
    // copying the device pointers to the new object and nulling them
    // out in the old object
    MSIRT(const MSIRT &&) = delete;
    MSIRT &operator=(const MSIRT &&) = delete;

  private:
    void ComputeInverseColumnSums(const DataSet3rdGen &data_set);

    float *m_dev_image{nullptr};
    float *m_dev_update{nullptr};
    float *m_dev_errsino{nullptr};
    uint16_t *m_dev_proj{nullptr};
    float *m_dev_inv_max_colsums{nullptr};
    float *m_dev_errnorm{nullptr};
    size_t m_dev_reduction_workbuf_size{0};
    void *m_dev_reduction_workbuf{nullptr};
    float3 *m_dev_focal_spot_offsets{nullptr};
    float3 *m_dev_det_centers{nullptr};
    cudaStream_t m_stream{0};

    std::vector<float> m_inv_max_colsums;

    ReconParams m_recon_params;
};

#endif // _MSIRT_H_