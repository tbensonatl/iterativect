#include "system_geom.h"

#include <cmath>
#include <iostream>

#include <vector_functions.h>

void GetDetectorCoords3rdGen(const SystemGeometry3rdGen &geom,
                             std::vector<float3> &detector_coords) {

    const float del_gamma = static_cast<float>(geom.detector_channel_width) /
                            geom.source_to_detector_mm;
    const float chan_offset =
        0.5f * (geom.num_detector_channels - 1) - geom.detector_center_channel;
    const float start_gamma =
        (-0.5f * (geom.num_detector_channels - 1) + chan_offset) * del_gamma;
    detector_coords.resize(geom.num_detector_channels * geom.num_detector_rows);
    const float det_z_width = geom.detector_row_width;
    const float row_offset =
        geom.detector_center_row - 0.5f * (geom.num_detector_rows - 1);
    const float z_coord0 =
        (0.5f * (geom.num_detector_rows - 1) + row_offset) * det_z_width;
    for (int c = 0; c < geom.num_detector_channels; c++) {
        const float gamma = start_gamma + c * del_gamma;
        float sin_gamma, cos_gamma;
        sincosf(gamma, &sin_gamma, &cos_gamma);
        detector_coords[c] = make_float3(
            geom.source_to_detector_mm * sin_gamma,
            geom.source_to_iso_mm - geom.source_to_detector_mm * cos_gamma,
            z_coord0);
    }
    for (int r = 1; r < geom.num_detector_rows; r++) {
        const float z_coord = z_coord0 - r * det_z_width;
        for (int c = 0; c < geom.num_detector_channels; c++) {
            detector_coords[r * geom.num_detector_channels + c] = make_float3(
                detector_coords[c].x, detector_coords[c].y, z_coord);
        }
    }
}
