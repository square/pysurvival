#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iterator>

#include "survival_forest_data.h"
#include "survival_forest_utility.h"

namespace ranger {

  Data::Data() :
      num_rows(0), num_rows_rounded(0), num_cols(0), snp_data(0), num_cols_no_snp(0), externalData(true), index_data(0), max_num_unique_values(
          0), order_snps(false) {
  }

  size_t Data::getVariableID(const std::string& variable_name) const {
    auto it = std::find(variable_names.cbegin(), variable_names.cend(), variable_name);
    if (it == variable_names.cend()) {
      throw std::runtime_error("Variable " + variable_name + " not found.");
    }
    return (std::distance(variable_names.cbegin(), it));
  }

  void Data::getAllValues(std::vector<double>& all_values, std::vector<size_t>& sampleIDs, size_t varID) const {

    // All values for varID (no duplicates) for given sampleIDs
    if (getUnpermutedVarID(varID) < num_cols_no_snp) {

      all_values.reserve(sampleIDs.size());
      for (size_t i = 0; i < sampleIDs.size(); ++i) {
        all_values.push_back(get(sampleIDs[i], varID));
      }
      std::sort(all_values.begin(), all_values.end());
      all_values.erase(std::unique(all_values.begin(), all_values.end()), all_values.end());
    } else {
      // If GWA data just use 0, 1, 2
      all_values = std::vector<double>( { 0, 1, 2 });
    }
  }

  void Data::getMinMaxValues(double& min, double&max, std::vector<size_t>& sampleIDs, size_t varID) const {
    if (sampleIDs.size() > 0) {
      min = get(sampleIDs[0], varID);
      max = min;
    }
    for (size_t i = 1; i < sampleIDs.size(); ++i) {
      double value = get(sampleIDs[i], varID);
      if (value < min) {
        min = value;
      }
      if (value > max) {
        max = value;
      }
    }
  }

  void Data::sort() {

    // Reserve memory
    index_data.resize(num_cols_no_snp * num_rows);

    // For all columns, get unique values and save index for each observation
    for (size_t col = 0; col < num_cols_no_snp; ++col) {

      // Get all unique values
      std::vector<double> unique_values(num_rows);
      for (size_t row = 0; row < num_rows; ++row) {
        unique_values[row] = get(row, col);
      }
      std::sort(unique_values.begin(), unique_values.end());
      unique_values.erase(unique(unique_values.begin(), unique_values.end()), unique_values.end());

      // Get index of unique value
      for (size_t row = 0; row < num_rows; ++row) {
        size_t idx = std::lower_bound(unique_values.begin(), unique_values.end(), get(row, col)) - unique_values.begin();
        index_data[col * num_rows + row] = idx;
      }

      // Save unique values
      unique_data_values.push_back(unique_values);
      if (unique_values.size() > max_num_unique_values) {
        max_num_unique_values = unique_values.size();
      }
    }
  }


} // namespace ranger

