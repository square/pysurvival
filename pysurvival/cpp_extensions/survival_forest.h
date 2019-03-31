#ifndef SURVIVAL_FOREST_H_
#define SURVIVAL_FOREST_H_

#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include <memory>
#ifndef OLD_WIN_R_BUILD
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#endif

#include "survival_forest_globals.h"
#include "survival_forest_tree.h"
#include "survival_forest_data.h"

namespace ranger {

  class Forest {
  public:
    Forest();

    Forest(const Forest&) = delete;
    Forest& operator=(const Forest&) = delete;

    virtual ~Forest() = default;

    // Init from c++ main or Rcpp from R
    void initR(std::string dependent_variable_name, std::unique_ptr<Data> input_data, uint mtry, uint num_trees,
        std::ostream* verbose_out, uint seed, uint num_threads, ImportanceMode importance_mode, uint min_node_size,
        std::vector<std::vector<double>>& split_select_weights,
        const std::vector<std::string>& always_split_variable_names, std::string status_variable_name,
        bool prediction_mode, bool sample_with_replacement, const std::vector<std::string>& unordered_variable_names,
        bool memory_saving_splitting, SplitRule splitrule, std::vector<double>& case_weights,
        std::vector<std::vector<size_t>>& manual_inbag, bool predict_all, bool keep_inbag,
        std::vector<double>& sample_fraction, double alpha, double minprop, bool holdout, PredictionType prediction_type,
        uint num_random_splits, bool order_snps, uint max_depth);
    void init(std::string dependent_variable_name, MemoryMode memory_mode, std::unique_ptr<Data> input_data, uint mtry,
        std::string output_prefix, uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode,
        uint min_node_size, std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
        const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
        bool predict_all, std::vector<double>& sample_fraction, double alpha, double minprop, bool holdout,
        PredictionType prediction_type, uint num_random_splits, bool order_snps, uint max_depth);
    virtual void initInternal(std::string status_variable_name) = 0;

    // Grow or predict
    void run(bool verbose, bool compute_oob_error);

    std::vector<std::vector<std::vector<size_t>>> getChildNodeIDs() {
      std::vector<std::vector<std::vector<size_t>>> result;
      for (auto& tree : trees) {
        result.push_back(tree->getChildNodeIDs());
      }
      return result;
    }
    std::vector<std::vector<size_t>> getSplitVarIDs() {
      std::vector<std::vector<size_t>> result;
      for (auto& tree : trees) {
        result.push_back(tree->getSplitVarIDs());
      }
      return result;
    }
    std::vector<std::vector<double>> getSplitValues() {
      std::vector<std::vector<double>> result;
      for (auto& tree : trees) {
        result.push_back(tree->getSplitValues());
      }
      return result;
    }
    const std::vector<double>& getVariableImportance() const {
      return variable_importance;
    }
    double getOverallPredictionError() const {
      return overall_prediction_error;
    }
    const std::vector<std::vector<std::vector<double>>>& getPredictions() const {
      return predictions;
    }
    size_t getDependentVarId() const {
      return dependent_varID;
    }
    size_t getNumTrees() const {
      return num_trees;
    }
    uint getMtry() const {
      return mtry;
    }
    uint getMinNodeSize() const {
      return min_node_size;
    }
    size_t getNumIndependentVariables() const {
      return num_independent_variables;
    }

    const std::vector<bool>& getIsOrderedVariable() const {
      return data->getIsOrderedVariable();
    }

    std::vector<std::vector<size_t>> getInbagCounts() const {
      std::vector<std::vector<size_t>> result;
      for (auto& tree : trees) {
        result.push_back(tree->getInbagCounts());
      }
      return result;
    }


  protected:
    void grow();
    virtual void growInternal() = 0;

    // Predict using existing tree from file and data as prediction data
    void predict();
    virtual void allocatePredictMemory() = 0;
    virtual void predictInternal(size_t sample_idx) = 0;

    void computePredictionError();
    virtual void computePredictionErrorInternal() = 0;

    void computePermutationImportance();

    // Multithreading methods for growing/prediction/importance, called by each thread
    void growTreesInThread(uint thread_idx, std::vector<double>* variable_importance);
    void predictTreesInThread(uint thread_idx, const Data* prediction_data, bool oob_prediction);
    void predictInternalInThread(uint thread_idx);
    void computeTreePermutationImportanceInThread(uint thread_idx, std::vector<double>& importance,
        std::vector<double>& variance);


    // Set split select weights and variables to be always considered for splitting
    void setSplitWeightVector(std::vector<std::vector<double>>& split_select_weights);
    void setAlwaysSplitVariables(const std::vector<std::string>& always_split_variable_names);

    // Show progress every few seconds
  #ifdef OLD_WIN_R_BUILD
    void showProgress(std::string operation, clock_t start_time, clock_t& lap_time);
  #else
    void showProgress(std::string operation, size_t max_progress);
  #endif

    // Verbose output stream, cout if verbose==true, logfile if not
    std::ostream* verbose_out;

    size_t num_trees;
    uint mtry;
    uint min_node_size;
    size_t num_variables;
    size_t num_independent_variables;
    uint seed;
    size_t dependent_varID;
    size_t num_samples;
    bool prediction_mode;
    MemoryMode memory_mode;
    bool sample_with_replacement;
    bool memory_saving_splitting;
    SplitRule splitrule;
    bool predict_all;
    bool keep_inbag;
    std::vector<double> sample_fraction;
    bool holdout;
    PredictionType prediction_type;
    uint num_random_splits;
    uint max_depth;
    bool verbose;

    // MAXSTAT splitrule
    double alpha;
    double minprop;

    // Multithreading
    uint num_threads;
    std::vector<uint> thread_ranges;
  #ifndef OLD_WIN_R_BUILD
    std::mutex mutex;
    std::condition_variable condition_variable;
  #endif

    std::vector<std::unique_ptr<Tree>> trees;
    std::unique_ptr<Data> data;

    std::vector<std::vector<std::vector<double>>> predictions;
    double overall_prediction_error;

    // Weight vector for selecting possible split variables, one weight between 0 (never select) and 1 (always select) for each variable
    // Deterministic variables are always selected
    std::vector<size_t> deterministic_varIDs;
    std::vector<size_t> split_select_varIDs;
    std::vector<std::vector<double>> split_select_weights;

    // Bootstrap weights
    std::vector<double> case_weights;

    // Pre-selected bootstrap samples (per tree)
    std::vector<std::vector<size_t>> manual_inbag;

    // Random number generator
    std::mt19937_64 random_number_generator;

    std::string output_prefix;
    ImportanceMode importance_mode;

    // Variable importance for all variables in forest
    std::vector<double> variable_importance;

    // Computation progress (finished trees)
    size_t progress;
  #ifdef R_BUILD
    size_t aborted_threads;
    bool aborted;
  #endif
  };

class ForestSurvival: public Forest {
public:
  ForestSurvival() = default;

  ForestSurvival(const ForestSurvival&) = delete;
  ForestSurvival& operator=(const ForestSurvival&) = delete;

  virtual ~ForestSurvival() override = default;

  void loadForest(size_t dependent_varID, size_t num_trees,
      std::vector<std::vector<std::vector<size_t>> >& forest_child_nodeIDs,
      std::vector<std::vector<size_t>>& forest_split_varIDs, std::vector<std::vector<double>>& forest_split_values,
      size_t status_varID, std::vector<std::vector<std::vector<double>> >& forest_chf,
      std::vector<double>& unique_timepoints, std::vector<bool>& is_ordered_variable);

  std::vector<std::vector<std::vector<double>>> getChf() const;

  size_t getStatusVarId() const {
    return status_varID;
  }
  const std::vector<double>& getUniqueTimepoints() const {
    return unique_timepoints;
  }

private:
  void initInternal(std::string status_variable_name) override;
  void growInternal() override;
  void allocatePredictMemory() override;
  void predictInternal(size_t sample_idx) override;
  void computePredictionErrorInternal() override;

  size_t status_varID;
  std::vector<double> unique_timepoints;
  std::vector<size_t> response_timepointIDs;

private:
  const std::vector<double>& getTreePrediction(size_t tree_idx, size_t sample_idx) const;
  size_t getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const;
};


  class SurvivalForestModel  {
    public:

      // Initialization
      SurvivalForestModel();
          virtual ~SurvivalForestModel();

          // Attributes
      size_t num_independent_variables;
      uint mtry;
      uint min_node_size;
      std::vector<double> variable_importance;
      double overall_prediction_error; 
      size_t dependent_varID;
      size_t status_varID;
      size_t num_trees;
      std::vector<std::vector<std::vector<size_t>> > child_nodeIDs;
      std::vector<std::vector<size_t>> split_varIDs; 
      std::vector<std::vector<double>> split_values; 
      std::vector<bool> is_ordered;
      std::vector<std::vector<std::vector<double>> > chf;
      std::vector<double> unique_timepoints;
      vector<string> variable_names;

          // Methods
      void fit( std::vector <std::vector<double> > input_data, 
         std::string dependent_variable_name, std::string status_variable_name, 
         std::vector<std::string> variable_names, std::vector<double>& case_weights, 
         uint mtry, uint num_trees, uint min_node_size, double alpha, double minprop, 
         uint num_random_splits, uint max_depth, bool sample_with_replacement, 
         double sample_fraction_value, int importance_mode_r, int splitrule_r, 
         int prediction_type_r, bool verbose, int seed, int num_threads, bool save_memory);

      std::vector<std::vector<std::vector<double>>> predict( std::vector <std::vector<double> > input_data, 
         std::string dependent_variable_name, std::string status_variable_name, 
         std::vector<std::string> variable_names, std::vector<double>& case_weights, 
         uint mtry, uint num_trees, uint min_node_size, double alpha, double minprop, 
         uint num_random_splits, uint max_depth, bool sample_with_replacement, 
         double sample_fraction_value, int importance_mode_r, int splitrule_r, 
         int prediction_type_r, bool verbose, int seed, int num_threads, bool save_memory);

      std::vector<std::vector<double> > predict_survival( std::vector <std::vector<double> > input_data, int num_threads);

      std::vector<std::vector<double> > predict_hazard( std::vector <std::vector<double> > input_data, int num_threads);

      std::vector<double> predict_risk( std::vector <std::vector<double> > input_data, int num_threads);

    };


} // namespace ranger

#endif /* SURVIVAL_FOREST_H_ */
