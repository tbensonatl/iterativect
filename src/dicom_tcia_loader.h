#ifndef _DICOM_TCIA_LOADER_H_
#define _DICOM_TCIA_LOADER_H_

#include <string_view>

#include "data_set.h"

class DicomTCIALoader {
  public:
    DicomTCIALoader(const char *data_dictionary_filename);
    DataSet3rdGen LoadDataSet(const std::string &dir_name);
};

#endif // _DICOM_TCIA_LOADER_H_