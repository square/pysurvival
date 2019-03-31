#ifndef SURVIVAL_FOREST_TREE_H_
#define SURVIVAL_FOREST_TREE_H_

#include <vector>
#include <random>
#include <iostream>
#include <stdexcept>

#include "survival_forest_globals.h"
#include "survival_forest_data.h"

namespace ranger {

  class Tree {
  public:
    Tree();

    // Create from loaded forest
    Tree(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
        std::vector<double>& split_values);

    virtual ~Tree() = default;

    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;

    void init(const Data* data, uint mtry, size_t dependent_varID, size_t num_samples, uint seed,
        std::vector<size_t>* deterministic_varIDs, std::vector<size_t>* split_select_varIDs,
        std::vector<double>* split_select_weights, ImportanceMode importance_mode, uint min_node_size,
        bool sample_with_replacement, bool memory_saving_splitting, SplitRule splitrule,
        std::vector<double>* case_weights, std::vector<size_t>* manual_inbag, bool keep_inbag,
        std::vector<double>* sample_fraction, double alpha, double minprop, bool holdout, uint num_random_splits,
        uint max_depth);

    virtual void allocateMemory() = 0;

    void grow(std::vector<double>* variable_importance);

    void predict(const Data* prediction_data, bool oob_prediction);

    void computePermutationImportance(std::vector<double>& forest_importance, std::vector<double>& forest_variance);

    const std::vector<std::vector<size_t>>& getChildNodeIDs() const {
      return child_nodeIDs;
    }
    const std::vector<double>& getSplitValues() const {
      return split_values;
    }
    const std::vector<size_t>& getSplitVarIDs() const {
      return split_varIDs;
    }

    const std::vector<size_t>& getOobSampleIDs() const {
      return oob_sampleIDs;
    }
    size_t getNumSamplesOob() const {
      return num_samples_oob;
    }

    const std::vector<size_t>& getInbagCounts() const {
      return inbag_counts;
    }

  protected:
    void createPossibleSplitVarSubset(std::vector<size_t>& result);

    bool splitNode(size_t nodeID);
    virtual bool splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) = 0;

    void createEmptyNode();
    virtual void createEmptyNodeInternal() = 0;

    size_t dropDownSamplePermuted(size_t permuted_varID, size_t sampleID, size_t permuted_sampleID);
    void permuteAndPredictOobSamples(size_t permuted_varID, std::vector<size_t>& permutations);

    virtual double computePredictionAccuracyInternal() = 0;

    void bootstrap();
    void bootstrapWithoutReplacement();

    void bootstrapWeighted();
    void bootstrapWithoutReplacementWeighted();

    virtual void bootstrapClassWise();
    virtual void bootstrapWithoutReplacementClassWise();

    void setManualInbag();

    virtual void cleanUpInternal() = 0;

    size_t dependent_varID;
    uint mtry;

    // Number of samples (all samples, not only inbag for this tree)
    size_t num_samples;

    // Number of OOB samples
    size_t num_samples_oob;

    // Minimum node size to split, like in original RF nodes of smaller size can be produced
    uint min_node_size;

    // Weight vector for selecting possible split variables, one weight between 0 (never select) and 1 (always select) for each variable
    // Deterministic variables are always selected
    const std::vector<size_t>* deterministic_varIDs;
    const std::vector<size_t>* split_select_varIDs;
    const std::vector<double>* split_select_weights;

    // Bootstrap weights
    const std::vector<double>* case_weights;

    // Pre-selected bootstrap samples
    const std::vector<size_t>* manual_inbag;

    // Splitting variable for each node
    std::vector<size_t> split_varIDs;

    // Value to split at for each node, for now only binary split
    // For terminal nodes the prediction value is saved here
    std::vector<double> split_values;

    // Vector of left and right child node IDs, 0 for no child
    std::vector<std::vector<size_t>> child_nodeIDs;

    // For each node a vector with IDs of samples in node
    std::vector<std::vector<size_t>> sampleIDs;

    // IDs of OOB individuals, sorted
    std::vector<size_t> oob_sampleIDs;

    // Holdout mode
    bool holdout;

    // Inbag counts
    bool keep_inbag;
    std::vector<size_t> inbag_counts;

    // Random number generator
    std::mt19937_64 random_number_generator;

    // Pointer to original data
    const Data* data;

    // Variable importance for all variables
    std::vector<double>* variable_importance;
    ImportanceMode importance_mode;

    // When growing here the OOB set is used
    // Terminal nodeIDs for prediction samples
    std::vector<size_t> prediction_terminal_nodeIDs;

    bool sample_with_replacement;
    const std::vector<double>* sample_fraction;

    bool memory_saving_splitting;
    SplitRule splitrule;
    double alpha;
    double minprop;
    uint num_random_splits;
    uint max_depth;
    uint depth;
    size_t last_left_nodeID;
  };



class TreeSurvival: public Tree {
public:
  TreeSurvival(std::vector<double>* unique_timepoints, size_t status_varID, std::vector<size_t>* response_timepointIDs);

  // Create from loaded forest
  TreeSurvival(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
      std::vector<double>& split_values, std::vector<std::vector<double>> chf, std::vector<double>* unique_timepoints,
      std::vector<size_t>* response_timepointIDs);

  TreeSurvival(const TreeSurvival&) = delete;
  TreeSurvival& operator=(const TreeSurvival&) = delete;

  virtual ~TreeSurvival() override = default;

  void allocateMemory() override;

  void computePermutationImportanceInternal(std::vector<std::vector<size_t>>* permutations);

  const std::vector<std::vector<double>>& getChf() const {
    return chf;
  }

  const std::vector<double>& getPrediction(size_t sampleID) const {
    size_t terminal_nodeID = prediction_terminal_nodeIDs[sampleID];
    return chf[terminal_nodeID];
  }

  size_t getPredictionTerminalNodeID(size_t sampleID) const {
    return prediction_terminal_nodeIDs[sampleID];
  }

private:

  void createEmptyNodeInternal() override;
  void computeSurvival(size_t nodeID);
  double computePredictionAccuracyInternal() override;

  bool splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) override;

  bool findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs);
  bool findBestSplitMaxstat(size_t nodeID, std::vector<size_t>& possible_split_varIDs);

  void findBestSplitValueLogRank(size_t nodeID, size_t varID, std::vector<double>& possible_split_values,
      double& best_value, size_t& best_varID, double& best_logrank);
  void findBestSplitValueLogRankUnordered(size_t nodeID, size_t varID, std::vector<double>& factor_levels,
      double& best_value, size_t& best_varID, double& best_logrank);
  void findBestSplitValueAUC(size_t nodeID, size_t varID, double& best_value, size_t& best_varID, double& best_auc);

  void computeDeathCounts(size_t nodeID);
  void computeChildDeathCounts(size_t nodeID, size_t varID, std::vector<double>& possible_split_values,
      std::vector<size_t>& num_samples_right_child, std::vector<size_t>& num_samples_at_risk_right_child,
      std::vector<size_t>& num_deaths_right_child, size_t num_splits);

  void computeAucSplit(double time_k, double time_l, double status_k, double status_l, double value_k, double value_l,
      size_t num_splits, std::vector<double>& possible_split_values, std::vector<double>& num_count,
      std::vector<double>& num_total);

  void findBestSplitValueLogRank(size_t nodeID, size_t varID, double& best_value, size_t& best_varID,
      double& best_logrank);
  void findBestSplitValueLogRankUnordered(size_t nodeID, size_t varID, double& best_value, size_t& best_varID,
      double& best_logrank);

  bool findBestSplitExtraTrees(size_t nodeID, std::vector<size_t>& possible_split_varIDs);
  void findBestSplitValueExtraTrees(size_t nodeID, size_t varID, double& best_value, size_t& best_varID,
      double& best_logrank);
  void findBestSplitValueExtraTreesUnordered(size_t nodeID, size_t varID, double& best_value, size_t& best_varID,
      double& best_logrank);

  void addImpurityImportance(size_t nodeID, size_t varID, double decrease);

  void cleanUpInternal() override {
    num_deaths.clear();
    num_deaths.shrink_to_fit();
    num_samples_at_risk.clear();
    num_samples_at_risk.shrink_to_fit();
  }

  size_t status_varID;

  // Unique time points for all individuals (not only this bootstrap), sorted
  const std::vector<double>* unique_timepoints;
  size_t num_timepoints;
  const std::vector<size_t>* response_timepointIDs;

  // For all terminal nodes CHF for all unique timepoints. For other nodes empty vector.
  std::vector<std::vector<double>> chf;

  // Fields to save to while tree growing
  std::vector<size_t> num_deaths;
  std::vector<size_t> num_samples_at_risk;
};



} // namespace ranger

#endif /* TREE_H_ */
