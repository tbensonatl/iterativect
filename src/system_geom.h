#ifndef _SYSTEM_GEOM_H_
#define _SYSTEM_GEOM_H_

#include <vector>

#include <vector_types.h>

struct SystemGeometry3rdGen {
    float source_to_iso_mm{0.0f};
    float source_to_detector_mm{0.0f};
    float detector_center_channel{0.0f};
    float detector_center_row{0.0f};
    float detector_channel_width{0.0f};
    // width in the z direction
    float detector_row_width{0.0f};
    // Angular increment per projection, in radians
    float del_phi{0.0f};
    int num_detector_rows{0};
    int num_detector_channels{0};
    int num_views_per_rotation{0};
};

void GetDetectorCoords3rdGen(const SystemGeometry3rdGen &geom,
                             std::vector<float3> &detector_coords);

#endif // _SYSTEM_GEOM_H_