#include "dicom_tcia_loader.h"
#include "helpers.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string_view>

#include <dcmtk/dcmdata/dctk.h>

#include <vector_functions.h>

namespace fs = std::filesystem;

DicomTCIALoader::DicomTCIALoader(const char *data_dictionary_filename) {
    // DCMTK uses a global DICOM data dictionary
    DcmDataDictionary &dict = dcmDataDict.wrlock();
    dict.loadDictionary(data_dictionary_filename);
    dcmDataDict.wrunlock();
}

static bool ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           (str.compare(str.size() - suffix.size(), suffix.size(), suffix) ==
            0);
}

template <typename T>
void DcmReadTag(const std::string &filename, DcmDataset *data_set, DcmTag tag,
                T &val) {
    auto check_status = [filename](DcmTagKey tag, OFCondition status) {
        if (!status.good()) {
            const std::string tagStr = "(" + std::to_string(tag.getGroup()) +
                                       "," + std::to_string(tag.getElement()) +
                                       ")";
            throw std::runtime_error("failed to load tag " + tagStr + " from " +
                                     filename + ": " + status.text());
        }
    };

    if constexpr (std::is_same_v<T, float>) {
        check_status(tag, data_set->findAndGetFloat32(tag, val));
    } else if constexpr (std::is_same_v<T, double>) {
        check_status(tag, data_set->findAndGetFloat64(tag, val));
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        check_status(tag, data_set->findAndGetUint16(tag, val));
    } else if constexpr (std::is_same_v<T, std::string>) {
        OFString str;
        check_status(tag, data_set->findAndGetOFString(tag, str));
        val = std::string(str.begin(), str.end());
    } else if constexpr (std::is_same_v<T, std::vector<float>>) {
        const float *ptr = nullptr;
        unsigned long count = 0;
        check_status(tag, data_set->findAndGetFloat32Array(tag, ptr, &count));
        if (count > 0) {
            val.insert(val.end(), ptr, ptr + count);
        }
    } else if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
        const uint16_t *ptr = nullptr;
        unsigned long count = 0;
        check_status(tag, data_set->findAndGetUint16Array(tag, ptr, &count));
        if (count > 0) {
            val.insert(val.end(), ptr, ptr + count);
        }
    }
}

struct DicomProjection {
    double rescale_intercept{0.0};
    double rescale_slope{0.0};
    double mu_water{0.0};
    float angular_position{0.0f};
    float axial_position{0.0f};
    float detector_transverse_mm{0.0f};
    float detector_axial_mm{0.0f};
    float source_to_isocenter_mm{0.0f};
    float source_to_detector_mm{0.0f};
    float detecter_center_row{0.0f};
    float detector_center_col{0.0f};
    float ffs_angular_shift{0.0f};
    float ffs_axial_shift{0.0f};
    float ffs_radial_shift{0.0f};
    uint16_t num_detector_rows{0};
    uint16_t num_detector_cols{0};
    uint16_t num_proj_per_rotation{0};
    int proj_ind{0};
    std::string flying_focal_spot_mode;
    std::unique_ptr<uint16_t[]> data;
};

// See the DICOM-CTPD user manual at:
//   https://wiki.cancerimagingarchive.net/download/attachments/52758026/DICOM-CT-PD%20User%20Manual_Version%203.pdf?api=v2
// For the meaning of the various DICOM-CTPD tags. Here, we use the Attribute
// Names for the fields as the enum name.
enum class CTPDAttr {
    RescaleIntercept,
    RescaleSlope,
    DetectorFocalCenterAngularPosition,
    DetectorFocalCenterAxialPosition,
    DetectorFocalCenterRadialDistance,
    ConstantRadialDistance,
    DetectorElementTransverseSpacing,
    DetectorElementAxialSpacing,
    NumberofDetectorRows,
    NumberofDetectorColumns,
    FlyingFocalSpotMode,
    NumberofSourceAngularSteps,
    SourceAngularPositionShift,
    SourceAxialPositionShift,
    SourceRadialDistanceShift,
    DetectorCentralElement,
    Rows,
    Columns,
    InstanceNumber,
    WaterAttenuationCoefficient,
    PixelData
};

static const std::map<CTPDAttr, DcmTagKey> s_attr = {
    {CTPDAttr::RescaleIntercept, DcmTagKey(0x0028, 0x1052)},
    {CTPDAttr::RescaleSlope, DcmTagKey(0x0028, 0x1053)},
    {CTPDAttr::DetectorFocalCenterAngularPosition, DcmTagKey(0x7031, 0x1001)},
    {CTPDAttr::DetectorFocalCenterAxialPosition, DcmTagKey(0x7031, 0x1002)},
    {CTPDAttr::DetectorFocalCenterRadialDistance, DcmTagKey(0x7031, 0x1003)},
    {CTPDAttr::ConstantRadialDistance, DcmTagKey(0x7031, 0x1031)},
    {CTPDAttr::DetectorElementTransverseSpacing, DcmTagKey(0x7029, 0x1002)},
    {CTPDAttr::DetectorElementAxialSpacing, DcmTagKey(0x7029, 0x1006)},
    {CTPDAttr::NumberofDetectorRows, DcmTagKey(0x7029, 0x1010)},
    {CTPDAttr::NumberofDetectorColumns, DcmTagKey(0x7029, 0x1011)},
    {CTPDAttr::FlyingFocalSpotMode, DcmTagKey(0x7033, 0x100E)},
    {CTPDAttr::NumberofSourceAngularSteps, DcmTagKey(0x7033, 0x1013)},
    {CTPDAttr::SourceAngularPositionShift, DcmTagKey(0x7033, 0x100B)},
    {CTPDAttr::SourceAxialPositionShift, DcmTagKey(0x7033, 0x100C)},
    {CTPDAttr::SourceRadialDistanceShift, DcmTagKey(0x7033, 0x100D)},
    {CTPDAttr::DetectorCentralElement, DcmTagKey(0x7031, 0x1033)},
    {CTPDAttr::Rows, DcmTagKey(0x0028, 0x0010)},
    {CTPDAttr::Columns, DcmTagKey(0x0028, 0x0011)},
    {CTPDAttr::InstanceNumber, DcmTagKey(0x0020, 0x0013)},
    {CTPDAttr::WaterAttenuationCoefficient, DcmTagKey(0x7041, 0x1001)},
    {CTPDAttr::PixelData, DcmTagKey(0x7fe0, 0x0010)}};

DataSet3rdGen DicomTCIALoader::LoadDataSet(const std::string &dir_name) {
    std::map<int, DicomProjection> proj_map;
    int max_proj_ind = -1;
    LOG("Reading data from %s", dir_name.c_str());
    std::vector<float> tmp_detector_row;
    double rescale_slope = 0.0, rescale_intercept = 0.0;
    for (const auto &entry : fs::directory_iterator(dir_name)) {
        const std::string &filename = entry.path().string();
        if (!ends_with(filename, ".dcm")) {
            continue;
        }
        DicomProjection proj;
        DcmFileFormat fmt;
        OFCondition status = fmt.loadFile(filename.c_str());
        if (!status.good()) {
            throw std::runtime_error("failed to read " + filename);
        }
        DcmDataset *data_set = fmt.getDataset();
        if (!data_set) {
            throw std::runtime_error("failed to get dataset from " + filename);
        }

        DcmReadTag(filename, data_set, s_attr.at(CTPDAttr::RescaleIntercept),
                   proj.rescale_intercept);
        DcmReadTag(filename, data_set, s_attr.at(CTPDAttr::RescaleSlope),
                   proj.rescale_slope);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::WaterAttenuationCoefficient),
                   proj.mu_water);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::DetectorFocalCenterAngularPosition),
                   proj.angular_position);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::DetectorFocalCenterAxialPosition),
                   proj.axial_position);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::DetectorFocalCenterRadialDistance),
                   proj.source_to_isocenter_mm);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::ConstantRadialDistance),
                   proj.source_to_detector_mm);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::DetectorElementTransverseSpacing),
                   proj.detector_transverse_mm);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::DetectorElementAxialSpacing),
                   proj.detector_axial_mm);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::NumberofDetectorRows),
                   proj.num_detector_rows);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::NumberofDetectorColumns),
                   proj.num_detector_cols);
        DcmReadTag(filename, data_set, s_attr.at(CTPDAttr::FlyingFocalSpotMode),
                   proj.flying_focal_spot_mode);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::NumberofSourceAngularSteps),
                   proj.num_proj_per_rotation);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::SourceAngularPositionShift),
                   proj.ffs_angular_shift);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::SourceAxialPositionShift),
                   proj.ffs_axial_shift);
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::SourceRadialDistanceShift),
                   proj.ffs_radial_shift);
        std::vector<float> center;
        DcmReadTag(filename, data_set,
                   s_attr.at(CTPDAttr::DetectorCentralElement), center);
        if (center.size() == 2) {
            // The detector center coordinates are 1-based indices, so
            // convert to 0-based indices
            proj.detector_center_col = center[0] - 1.0f;
            proj.detecter_center_row = center[1] - 1.0f;
        } else {
            throw std::runtime_error(filename +
                                     ": unable to find detector center coords");
        }

        if (max_proj_ind < 0) {
            rescale_intercept = proj.rescale_intercept;
            rescale_slope = proj.rescale_slope;
        } else {
            if (fabs(rescale_intercept - proj.rescale_intercept) > 1.0e-6 ||
                fabs(rescale_slope - proj.rescale_slope) > 1.0e-6) {
                throw std::runtime_error("rescale slope/intercept varies by "
                                         "view; do not yet support this case");
            }
        }
        uint16_t rows, cols;
        DcmReadTag(filename, data_set, s_attr.at(CTPDAttr::Rows), rows);
        DcmReadTag(filename, data_set, s_attr.at(CTPDAttr::Columns), cols);

        // The Siemens data all seems to be transposed relative to the pixel
        // data ordering specified in the file format document
        bool do_transpose = false;
        if (rows == proj.num_detector_cols && cols == proj.num_detector_rows) {
            do_transpose = true;
            rows = proj.num_detector_rows;
            cols = proj.num_detector_cols;
        }

        std::string instance_number;
        DcmReadTag(filename, data_set, s_attr.at(CTPDAttr::InstanceNumber),
                   instance_number);
        proj.proj_ind = std::stoi(instance_number);
        if (proj.proj_ind < 0) {
            throw std::runtime_error("invalid projection number " +
                                     instance_number);
        }

        if (rows != proj.num_detector_rows) {
            throw std::runtime_error("incorrect detector row count: expected " +
                                     std::to_string(proj.num_detector_rows) +
                                     ", got " + std::to_string(rows));
        }
        if (cols != proj.num_detector_cols) {
            throw std::runtime_error(
                "incorrect detector channel count: expected " +
                std::to_string(proj.num_detector_cols) + ", got " +
                std::to_string(cols));
        }

        std::vector<uint16_t> vals;
        vals.reserve(static_cast<uint32_t>(rows) * static_cast<uint32_t>(cols));
        DcmReadTag(filename, data_set, s_attr.at(CTPDAttr::PixelData), vals);
        proj.data = std::make_unique<uint16_t[]>(vals.size());
        const int num_detector_cells = rows * cols;
        if (do_transpose) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    proj.data[r * cols + c] = vals[c * rows + r];
                }
            }
        } else {
            memcpy(proj.data.get(), vals.data(),
                   sizeof(uint16_t) * num_detector_cells);
        }

        if (proj.proj_ind > max_proj_ind) {
            max_proj_ind = proj.proj_ind;
        }

        proj_map[proj.proj_ind] = std::move(proj);
    }

    if (max_proj_ind != static_cast<int>(proj_map.size())) {
        throw std::runtime_error(
            "projection count/index mismatch: max ind is " +
            std::to_string(max_proj_ind) + ", but num projections is " +
            std::to_string(proj_map.size()));
    }

    if (proj_map.size() == 0) {
        throw std::runtime_error("found no projections");
    }

    const auto &template_proj = proj_map.cbegin()->second;
    const int nrows = template_proj.num_detector_rows;
    const int ncols = template_proj.num_detector_cols;
    const size_t num_detector_cells = nrows * ncols;
    const size_t max_proj_to_use = std::min(
        std::numeric_limits<int>::max() / num_detector_cells, proj_map.size());

    if (max_proj_to_use < proj_map.size()) {
        LOG("Pruning from %zu to %zu projections to maintain manageable size",
            proj_map.size(), max_proj_to_use);
    }

    std::unique_ptr<uint16_t[]> data =
        std::make_unique<uint16_t[]>(max_proj_to_use * num_detector_cells);
    std::unique_ptr<float[]> source_phi =
        std::make_unique<float[]>(max_proj_to_use);
    std::unique_ptr<float[]> source_z_offsets =
        std::make_unique<float[]>(max_proj_to_use);
    std::unique_ptr<float3[]> ffs_offsets =
        std::make_unique<float3[]>(max_proj_to_use);
    for (const auto &entry : proj_map) {
        if (static_cast<size_t>(entry.second.proj_ind) > max_proj_to_use) {
            continue;
        }
        const size_t ind = entry.second.proj_ind - 1;
        memcpy(data.get() + ind * num_detector_cells, entry.second.data.get(),
               sizeof(uint16_t) * num_detector_cells);
        source_phi[ind] = entry.second.angular_position;
        source_z_offsets[ind] = entry.second.axial_position;
        ffs_offsets[ind] = make_float3(entry.second.ffs_angular_shift,
                                       entry.second.ffs_radial_shift,
                                       entry.second.ffs_axial_shift);
    }

    const float del_phi =
        (max_proj_to_use > 1) ? source_phi[1] - source_phi[0] : 0.0f;
    return DataSet3rdGen{
        .geom =
            SystemGeometry3rdGen{
                .source_to_iso_mm = template_proj.source_to_isocenter_mm,
                .source_to_detector_mm = template_proj.source_to_detector_mm,
                .detector_center_channel = template_proj.detector_center_col,
                .detector_center_row = template_proj.detecter_center_row,
                .detector_channel_width = template_proj.detector_transverse_mm,
                .detector_row_width = template_proj.detector_axial_mm,
                .del_phi = del_phi,
                .num_detector_rows = template_proj.num_detector_rows,
                .num_detector_channels = template_proj.num_detector_cols,
                .num_views_per_rotation = template_proj.num_proj_per_rotation,
            },
        .source_phi = std::move(source_phi),
        .source_z_offsets = std::move(source_z_offsets),
        .focal_spot_offsets = std::move(ffs_offsets),
        .rescale_slope = static_cast<float>(rescale_slope),
        .rescale_intercept = static_cast<float>(rescale_intercept),
        .mu_water = static_cast<float>(template_proj.mu_water),
        .data = std::move(data),
        .num_projections = static_cast<int>(max_proj_to_use),
    };
}