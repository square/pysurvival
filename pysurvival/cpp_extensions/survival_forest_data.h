#ifndef SURVIVAL_FOREST_DATA_H_
#define SURVIVAL_FOREST_DATA_H_

#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <utility>

#include "survival_forest_globals.h"

namespace ranger {

	class Data {
	public:
		Data();

		Data(const Data&) = delete;

		Data(std::vector<double> data, std::vector<std::string> variable_names, size_t num_rows, size_t num_cols) :
		  data { std::move(data) } {
			this->variable_names = variable_names;
			this->num_rows = num_rows;
			this->num_cols = num_cols;
			this->num_cols_no_snp = num_cols;
		}

		Data& operator=(const Data&) = delete;

		virtual ~Data() = default;

		double get(size_t row, size_t col) const  {
			// Use permuted data for corrected impurity importance
			if (col >= num_cols) {
				col = getUnpermutedVarID(col);
				row = getPermutedSampleID(row);
			}

			if (col < num_cols_no_snp) {
				return data[col * num_rows + row];
			}
		};

		size_t getVariableID(const std::string& variable_name) const;

		void reserveMemory()  {
			data.resize(num_cols * num_rows);
		};

		void set(size_t col, size_t row, double value, bool& error) {
			data[col * num_rows + row] = value;
		}
		;

		void getAllValues(std::vector<double>& all_values, std::vector<size_t>& sampleIDs, size_t varID) const;

		void getMinMaxValues(double& min, double&max, std::vector<size_t>& sampleIDs, size_t varID) const;

		size_t getIndex(size_t row, size_t col) const {
			// Use permuted data for corrected impurity importance
			if (col >= num_cols) {
				col = getUnpermutedVarID(col);
				row = getPermutedSampleID(row);
			}

			if (col < num_cols_no_snp) {
				return index_data[col * num_rows + row];
			}
		}

		void loadData(std::vector <std::vector<double> > Input_Data, std::vector<std::string> variable_names){
			this->variable_names = variable_names;
			this->num_rows = Input_Data.size();
			this->num_cols = variable_names.size();
			this->num_cols_no_snp = this->num_cols;

			size_t row;
			size_t column ;
			bool error = false;
			reserveMemory();
			for(row = 0; row < this->num_rows; row++){

				for(column = 0; column < this->num_cols; column++){
						set(column, row, Input_Data[row][column], error);
				}

			}

		}


		double getUniqueDataValue(size_t varID, size_t index) const {
			// Use permuted data for corrected impurity importance
			if (varID >= num_cols) {
				varID = getUnpermutedVarID(varID);
			}

			if (varID < num_cols_no_snp) {
				return unique_data_values[varID][index];
			} else {
				// For GWAS data the index is the value
				return (index);
			}
		}

		size_t getNumUniqueDataValues(size_t varID) const {
			// Use permuted data for corrected impurity importance
			if (varID >= num_cols) {
				varID = getUnpermutedVarID(varID);
			}

			if (varID < num_cols_no_snp) {
				return unique_data_values[varID].size();
			} else {
				// For GWAS data 0,1,2
				return (3);
			}
		}

		void sort();

		const std::vector<std::string>& getVariableNames() const {
			return variable_names;
		}
		size_t getNumCols() const {
			return num_cols;
		}
		size_t getNumRows() const {
			return num_rows;
		}

		size_t getMaxNumUniqueValues() const {
			if (snp_data == 0 || max_num_unique_values > 3) {
				// If no snp data or one variable with more than 3 unique values, return that value
				return max_num_unique_values;
			} else {
				// If snp data and no variable with more than 3 unique values, return 3
				return 3;
			}
		}

		const std::vector<size_t>& getNoSplitVariables() const noexcept {
			return no_split_variables;
		}

		void addNoSplitVariable(size_t varID) {
			no_split_variables.push_back(varID);
			std::sort(no_split_variables.begin(), no_split_variables.end());
		}

		std::vector<bool>& getIsOrderedVariable() noexcept {
			return is_ordered_variable;
		}

		void setIsOrderedVariable(const std::vector<std::string>& unordered_variable_names) {
			is_ordered_variable.resize(num_cols, true);
			for (auto& variable_name : unordered_variable_names) {
				size_t varID = getVariableID(variable_name);
				is_ordered_variable[varID] = false;
			}
		}

		void setIsOrderedVariable(std::vector<bool>& is_ordered_variable) {
			this->is_ordered_variable = is_ordered_variable;
		}

		bool isOrderedVariable(size_t varID) const {
			// Use permuted data for corrected impurity importance
			if (varID >= num_cols) {
				varID = getUnpermutedVarID(varID);
			}
			return is_ordered_variable[varID];
		}

		void permuteSampleIDs(std::mt19937_64 random_number_generator) {
			permuted_sampleIDs.resize(num_rows);
			std::iota(permuted_sampleIDs.begin(), permuted_sampleIDs.end(), 0);
			std::shuffle(permuted_sampleIDs.begin(), permuted_sampleIDs.end(), random_number_generator);
		}

		size_t getPermutedSampleID(size_t sampleID) const {
			return permuted_sampleIDs[sampleID];
		}

		size_t getUnpermutedVarID(size_t varID) const {
			if (varID >= num_cols) {
				varID -= num_cols;

				for (auto& skip : no_split_variables) {
					if (varID >= skip) {
						++varID;
					}
				}
			}
			return varID;
		}


	private:
		std::vector<double> data;

	protected:
		std::vector<std::string> variable_names;
		size_t num_rows;
		size_t num_rows_rounded;
		size_t num_cols;

		unsigned char* snp_data;
		size_t num_cols_no_snp;

		bool externalData;

		std::vector<size_t> index_data;
		std::vector<std::vector<double>> unique_data_values;
		size_t max_num_unique_values;

		// Variable to not split at (only dependent_varID for non-survival trees)
		std::vector<size_t> no_split_variables;

		// For each varID true if ordered
		std::vector<bool> is_ordered_variable;

		// Permuted samples for corrected impurity importance
		std::vector<size_t> permuted_sampleIDs;

		// Order of 0/1/2 for ordered splitting
		std::vector<std::vector<size_t>> snp_order;
		bool order_snps;
	};

} // namespace ranger

#endif /* SURVIVAL_FOREST_DATA_H_ */
