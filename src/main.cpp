#include "dicom_tcia_loader.h"
#include "helpers.h"
#include "kernels.h"
#include "msirt.h"
#include "system_geom.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>

#include <vector_functions.h>

namespace po = boost::program_options;

template <typename T>
static int WriteVolumetoFile(const std::string &filename, const T *volume,
                             int num_voxels) {
    int rc = EXIT_SUCCESS;
    std::ofstream outfile(filename.c_str(), std::ios::out | std::ios::binary);
    LOG("Writing output to %s", filename.c_str());
    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char *>(volume),
                      sizeof(T) * num_voxels);
        if (outfile.bad()) {
            LOG_ERR("Failed to write image to %s", filename.c_str());
            rc = EXIT_FAILURE;
        }
        outfile.close();
    } else {
        LOG_ERR("Failed to open %s for writing", filename.c_str());
        rc = EXIT_FAILURE;
    }
    return rc;
}

int main(int argc, char **argv) {
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "help message")(
        "dz", po::value<float>()->default_value(1.0f),
        "Reconstruction slice width in the z dimension (mm)")(
        "fov", po::value<float>()->default_value(500.0f),
        "Reconstruction field of view (mm)")(
        "num-iter", po::value<int>()->default_value(1),
        "Number of full reconstruction iterations")(
        "num-subsets", po::value<int>()->default_value(64),
        "Number of ordered subsets iterations")(
        "nx", po::value<int>()->default_value(512),
        "Reconstruction image width (x dimension)")(
        "ny", po::value<int>()->default_value(512),
        "Reconstruction image height (y dimension)")(
        "dicom-dict",
        po::value<std::string>()->default_value("assets/dicom_data_dict.txt"),
        "Path to DICOM data dictionary")(
        "proj-dir", po::value<std::string>(),
        "Path to projection data (i.e. DICOM projection data)")(
        "gpu", po::value<int>()->default_value(0),
        "Enables GPU-based backprojection on the specified device")(
        "output-file,o",
        po::value<std::string>()->default_value("/tmp/vol.bin"),
        "Output filename for reconstructed volume");
    po::positional_options_description pos;
    pos.add("proj-dir", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(desc)
                      .positional(pos)
                      .run(),
                  vm);
        po::notify(vm);
    } catch (po::error &e) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (argc == 1 || vm.count("help")) {
        std::cout << "Usage: " << argv[0] << " [options] <proj-dir>\n";
        std::cout << desc;
        return EXIT_SUCCESS;
    }

    if (vm.count("proj-dir") == 0) {
        std::cout << "Usage: " << argv[0] << " [options] <proj-dir>\n";
        std::cout << desc;
        return EXIT_FAILURE;
    }

    const int device_id = vm["gpu"].as<int>();
    int num_devices = 0;
    cudaChecked(cudaGetDeviceCount(&num_devices));
    if (device_id < 0 || device_id >= num_devices) {
        LOG_ERR("Invalid GPU device ID %d; have %d available devices.",
                device_id, num_devices);
        exit(EXIT_FAILURE);
    }

    cudaChecked(cudaSetDevice(device_id));
    LOG("Using CUDA device ID %d", device_id);

    DicomTCIALoader loader(vm["dicom-dict"].as<std::string>().c_str());
    const std::string proj_dir = vm["proj-dir"].as<std::string>();

    const auto t1 = std::chrono::steady_clock::now();
    DataSet3rdGen data_set = loader.LoadDataSet(proj_dir);
    const auto t2 = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    LOG("Loaded %d %dx%d projections from %s [%.3f ms]",
        data_set.num_projections, data_set.geom.num_detector_channels,
        data_set.geom.num_detector_rows, proj_dir.c_str(),
        static_cast<double>(elapsed) / 1.0e3);

    const int nx = vm["nx"].as<int>();
    const int ny = vm["ny"].as<int>();
    const float fov = vm["fov"].as<float>();
    const float dx = fov / nx;
    const float dy = fov / ny;
    const float dz = vm["dz"].as<float>();

    // Add some padding beyond the first and last source z position.
    // There is insufficient data in these regions to reconstruction
    // a quality image, but the conebeam will have illuminated part
    // of the object in this range and the forward model will need to
    // account for that acquired data.
    const float zpad = 20.0f;
    const int nz = static_cast<int>(
        ceilf((data_set.source_z_offsets[data_set.num_projections - 1] -
               data_set.source_z_offsets[0] + 2 * zpad) /
              dz));
    const float zcen =
        0.5f * (data_set.source_z_offsets[data_set.num_projections - 1] +
                data_set.source_z_offsets[0]);

    const int num_iterations = vm["num-iter"].as<int>();
    const int num_subsets = vm["num-subsets"].as<int>();

    LOG("Reconstructing a %dx%dx%d volume using %d iterations with %d "
        "subsets",
        nx, ny, nz, num_iterations, num_subsets);

    ReconParams recon_params = {.nx = nx,
                                .ny = ny,
                                .nz = nz,
                                .dx = dx,
                                .dy = dy,
                                .dz = dz,
                                .num_iterations = num_iterations,
                                .num_subsets = num_subsets,
                                .xcen = 0.0f,
                                .ycen = 0.0f,
                                .zcen = zcen};

    std::unique_ptr<int16_t[]> volume =
        std::make_unique<int16_t[]>(nx * ny * nz);

    MSIRT recon;
    const size_t mem_req =
        recon.GetDeviceMemoryRequirement(data_set, recon_params);
    LOG("Reconstrution requires %.1f MiB of device memory",
        static_cast<double>(mem_req) / 1024.0 / 1024.0);

    const auto t3 = std::chrono::steady_clock::now();
    recon.Reconstruct(volume.get(), data_set, recon_params);
    const auto t4 = std::chrono::steady_clock::now();
    const auto elapsed_recon =
        std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    LOG("Reconstruction took %.3f seconds",
        static_cast<double>(elapsed_recon) / 1.0e6);

    std::string output_file = vm["output-file"].as<std::string>();
    return WriteVolumetoFile(output_file, volume.get(), nx * ny * nz);
}