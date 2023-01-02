#ifndef _DATA_SET_H_
#define _DATA_SET_H_

#include <cstdint>
#include <memory>

#include "system_geom.h"

#include <vector_types.h>

struct DataSet3rdGen {
    SystemGeometry3rdGen geom;
    // Angle of the focal center / source per projection
    std::unique_ptr<float[]> source_phi;
    // Axial/z position of the focal center / source per projection
    std::unique_ptr<float[]> source_z_offsets;
    // If the focal spot is not the same as the detector focal center
    // (e.g. in the case of a Siemens flying focal spot), then the
    // focal_spot_offsets array should be populated with the
    // angular, radial, and axial offsets, in that order.
    std::unique_ptr<float3[]> focal_spot_offsets;
    // The data in DICOM-CT-PD format is stored in uint16 format with
    // scalar slope/intercept values used to convert the data to
    // post-log p-values. We apply this conversion on-the-fly in
    // the projectors to reduce device memory requirements by
    // storing the original projection data as uint16 values rather
    // than floats. For now, we assume that the slope/intercept
    // values in every view are close enough to one another so that
    // we do not need to store them as per-view values, but that
    // assumption is verified during data read. If some data sets
    // vary the values per projection, then we can store the
    // per-projection values.
    float rescale_slope;
    float rescale_intercept;
    // The linear attenuation coefficent of water. This is energy
    // dependent and thus depends on the particular scan energy
    // and pre-processing / beam hardening correction. The reconstructed
    // linear attenuation coefficient values (mu) are converted to
    // Hounsfield units via HU = 1000 * (mu - mu_water) / mu_water
    float mu_water;
    std::unique_ptr<uint16_t[]> data;
    int num_projections{0};

    bool focal_spot_is_detector_focal_center{false};
};

#endif // _DATA_SET_H_