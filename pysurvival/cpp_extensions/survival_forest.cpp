#include <math.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <ctime>
#include <functional>
#include <set>
#include <cmath>
#include <vector>
#include <sstream>
#include <memory>
#include <utility>
#include <random>     // mt19937 and uniform_int_distribution
#include <thread>      //thread
#include <iostream> 
#ifndef OLD_WIN_R_BUILD
#include <thread>
#include <chrono>
#endif


using namespace std;

#include "survival_forest_globals.h"
#include "survival_forest_utility.h"
#include "survival_forest_data.h"
#include "survival_forest_tree.h"
#include "survival_forest.h"


namespace ranger {

Forest::Forest() :
    verbose_out(0), num_trees(DEFAULT_NUM_TREE), mtry(0), min_node_size(0), num_variables(0), num_independent_variables(
        0), seed(0), dependent_varID(0), num_samples(0), prediction_mode(false), memory_mode(MEM_DOUBLE), sample_with_replacement(
        true), memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), predict_all(false), keep_inbag(false), sample_fraction(
        { 1 }), holdout(false), prediction_type(DEFAULT_PREDICTIONTYPE), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(
        DEFAULT_MAXDEPTH), alpha(DEFAULT_ALPHA), minprop(DEFAULT_MINPROP), num_threads(DEFAULT_NUM_THREADS), data { }, overall_prediction_error(
    NAN), importance_mode(DEFAULT_IMPORTANCE_MODE), progress(0) {
}




void Forest::initR(std::string dependent_variable_name, std::unique_ptr<Data> input_data, uint mtry, uint num_trees,
    std::ostream* verbose_out, uint seed, uint num_threads, ImportanceMode importance_mode, uint min_node_size,
    std::vector<std::vector<double>>& split_select_weights, const std::vector<std::string>& always_split_variable_names,
    std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
    const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
    std::vector<double>& case_weights, std::vector<std::vector<size_t>>& manual_inbag, bool predict_all,
    bool keep_inbag, std::vector<double>& sample_fraction, double alpha, double minprop, bool holdout,
    PredictionType prediction_type, uint num_random_splits, bool order_snps, uint max_depth) {

  this->verbose_out = verbose_out;

  // Call other init function
  init(dependent_variable_name, MEM_DOUBLE, std::move(input_data), mtry, "", num_trees, seed, num_threads,
      importance_mode, min_node_size, status_variable_name, prediction_mode, sample_with_replacement,
      unordered_variable_names, memory_saving_splitting, splitrule, predict_all, sample_fraction, alpha, minprop,
      holdout, prediction_type, num_random_splits, order_snps, max_depth);

  // Set variables to be always considered for splitting
  if (!always_split_variable_names.empty()) {
    setAlwaysSplitVariables(always_split_variable_names);
  }

  // Set split select weights
  if (!split_select_weights.empty()) {
    setSplitWeightVector(split_select_weights);
  }

  // Set case weights
  if (!case_weights.empty()) {
    if (case_weights.size() != num_samples) {
      throw std::runtime_error("Number of case weights not equal to number of samples.");
    }
    this->case_weights = case_weights;
  }

  // Set manual inbag
  if (!manual_inbag.empty()) {
    this->manual_inbag = manual_inbag;
  }

  // Keep inbag counts
  this->keep_inbag = keep_inbag;
}

void Forest::init(std::string dependent_variable_name, MemoryMode memory_mode, std::unique_ptr<Data> input_data,
    uint mtry, std::string output_prefix, uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode,
    uint min_node_size, std::string status_variable_name, bool prediction_mode, bool sample_with_replacement,
    const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
    bool predict_all, std::vector<double>& sample_fraction, double alpha, double minprop, bool holdout,
    PredictionType prediction_type, uint num_random_splits, bool order_snps, uint max_depth) {

  // Initialize data with memmode
  this->data = std::move(input_data);

  // Initialize random number generator and set seed
  if (seed == 0) {
    std::random_device random_device;
    random_number_generator.seed(random_device());
  } else {
    random_number_generator.seed(seed);
  }

  // Set number of threads
  if (num_threads == DEFAULT_NUM_THREADS) {
#ifdef OLD_WIN_R_BUILD
    this->num_threads = 1;
#else
    this->num_threads = std::thread::hardware_concurrency();
#endif
  } else {
    this->num_threads = num_threads;
  }

  // Set member variables
  this->num_trees = num_trees;
  this->mtry = mtry;
  this->seed = seed;
  this->output_prefix = output_prefix;
  this->importance_mode = importance_mode;
  this->min_node_size = min_node_size;
  this->memory_mode = memory_mode;
  this->prediction_mode = prediction_mode;
  this->sample_with_replacement = sample_with_replacement;
  this->memory_saving_splitting = memory_saving_splitting;
  this->splitrule = splitrule;
  this->predict_all = predict_all;
  this->sample_fraction = sample_fraction;
  this->holdout = holdout;
  this->alpha = alpha;
  this->minprop = minprop;
  this->prediction_type = prediction_type;
  this->num_random_splits = num_random_splits;
  this->max_depth = max_depth;

  // Set number of samples and variables
  num_samples = data->getNumRows();
  num_variables = data->getNumCols();

  // Convert dependent variable name to ID
  if (!prediction_mode && !dependent_variable_name.empty()) {
    dependent_varID = data->getVariableID(dependent_variable_name);
  }

  // Set unordered factor variables
  if (!prediction_mode) {
    data->setIsOrderedVariable(unordered_variable_names);
  }

  data->addNoSplitVariable(dependent_varID);

  initInternal(status_variable_name);

  num_independent_variables = num_variables - data->getNoSplitVariables().size();

  // Init split select weights
  split_select_weights.push_back(std::vector<double>());

  // Init manual inbag
  manual_inbag.push_back(std::vector<size_t>());

  // Check if mtry is in valid range
  if (this->mtry > num_variables - 1) {
    throw std::runtime_error("mtry can not be larger than number of variables in data.");
  }

  // Check if any observations samples
  if ((size_t) num_samples * sample_fraction[0] < 1) {
    throw std::runtime_error("sample_fraction too small, no observations sampled.");
  }

  // Permute samples for corrected Gini importance
  if (importance_mode == IMP_GINI_CORRECTED) {
    data->permuteSampleIDs(random_number_generator);
  }

  // Order SNP levels if in "order" splitting
  // if (!prediction_mode && order_snps) {
  //   data->orderSnpLevels(dependent_variable_name, (importance_mode == IMP_GINI_CORRECTED));
  // }
}

void Forest::run(bool verbose, bool compute_oob_error) {

	this->verbose = verbose;

  if (prediction_mode) {
    if (verbose && verbose_out) {
      *verbose_out << "Predicting .." << std::endl;
    }
    predict();
  } else {
    if (verbose && verbose_out) {
      *verbose_out << "Growing trees .." << std::endl;
    }

    grow();

    if (verbose && verbose_out) {
      *verbose_out << "Computing prediction error .." << std::endl;
    }

    if (compute_oob_error) {
      computePredictionError();
    }

    if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW || importance_mode == IMP_PERM_RAW) {
      if (verbose && verbose_out) {
        *verbose_out << "Computing permutation variable importance .." << std::endl;
      }
      computePermutationImportance();
    }
  }
}





void Forest::grow() {

  // Create thread ranges
  equalSplit(thread_ranges, 0, num_trees - 1, num_threads);

  // Call special grow functions of subclasses. There trees must be created.
  growInternal();

  // Init trees, create a seed for each tree, based on main seed
  std::uniform_int_distribution<uint> udist;
  for (size_t i = 0; i < num_trees; ++i) {
    uint tree_seed;
    if (seed == 0) {
      tree_seed = udist(random_number_generator);
    } else {
      tree_seed = (i + 1) * seed;
    }

    // Get split select weights for tree
    std::vector<double>* tree_split_select_weights;
    if (split_select_weights.size() > 1) {
      tree_split_select_weights = &split_select_weights[i];
    } else {
      tree_split_select_weights = &split_select_weights[0];
    }

    // Get inbag counts for tree
    std::vector<size_t>* tree_manual_inbag;
    if (manual_inbag.size() > 1) {
      tree_manual_inbag = &manual_inbag[i];
    } else {
      tree_manual_inbag = &manual_inbag[0];
    }

    trees[i]->init(data.get(), mtry, dependent_varID, num_samples, tree_seed, &deterministic_varIDs,
        &split_select_varIDs, tree_split_select_weights, importance_mode, min_node_size, sample_with_replacement,
        memory_saving_splitting, splitrule, &case_weights, tree_manual_inbag, keep_inbag, &sample_fraction, alpha,
        minprop, holdout, num_random_splits, max_depth);
  }

// Init variable importance
  variable_importance.resize(num_independent_variables, 0);

// Grow trees in multiple threads
#ifdef OLD_WIN_R_BUILD
  progress = 0;
  clock_t start_time = clock();
  clock_t lap_time = clock();
  for (size_t i = 0; i < num_trees; ++i) {
    trees[i]->grow(&variable_importance);
    progress++;
    showProgress("Growing trees..", start_time, lap_time);
  }
#else
  progress = 0;
#ifdef R_BUILD
  aborted = false;
  aborted_threads = 0;
#endif

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

// Initailize importance per thread
  std::vector<std::vector<double>> variable_importance_threads(num_threads);

  for (uint i = 0; i < num_threads; ++i) {
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
      variable_importance_threads[i].resize(num_independent_variables, 0);
    }
    threads.emplace_back(&Forest::growTreesInThread, this, i, &(variable_importance_threads[i]));
  }
  showProgress("Growing trees..", num_trees);
  for (auto &thread : threads) {
    thread.join();
  }

#ifdef R_BUILD
  if (aborted_threads > 0) {
    throw std::runtime_error("User interrupt.");
  }
#endif

  // Sum thread importances
  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    variable_importance.resize(num_independent_variables, 0);
    for (size_t i = 0; i < num_independent_variables; ++i) {
      for (uint j = 0; j < num_threads; ++j) {
        variable_importance[i] += variable_importance_threads[j][i];
      }
    }
    variable_importance_threads.clear();
  }

#endif

// Divide importance by number of trees
  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    for (auto& v : variable_importance) {
      v /= num_trees;
    }
  }
}

void Forest::predict() {

// Predict trees in multiple threads and join the threads with the main thread
#ifdef OLD_WIN_R_BUILD
  progress = 0;
  clock_t start_time = clock();
  clock_t lap_time = clock();
  for (size_t i = 0; i < num_trees; ++i) {
    trees[i]->predict(data.get(), false);
    progress++;
    showProgress("Predicting..", start_time, lap_time);
  }

  // For all samples get tree predictions
  allocatePredictMemory();
  for (size_t sample_idx = 0; sample_idx < data->getNumRows(); ++sample_idx) {
    predictInternal(sample_idx);
  }
#else
  progress = 0;
#ifdef R_BUILD
  aborted = false;
  aborted_threads = 0;
#endif

  // Predict
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (uint i = 0; i < num_threads; ++i) {
    threads.emplace_back(&Forest::predictTreesInThread, this, i, data.get(), false);
  }
  showProgress("Predicting..", num_trees);
  for (auto &thread : threads) {
    thread.join();
  }

  // Aggregate predictions
  allocatePredictMemory();
  threads.clear();
  threads.reserve(num_threads);
  progress = 0;
  for (uint i = 0; i < num_threads; ++i) {
    threads.emplace_back(&Forest::predictInternalInThread, this, i);
  }
  showProgress("Aggregating predictions..", num_samples);
  for (auto &thread : threads) {
    thread.join();
  }

#ifdef R_BUILD
  if (aborted_threads > 0) {
    throw std::runtime_error("User interrupt.");
  }
#endif
#endif
}

void Forest::computePredictionError() {

// Predict trees in multiple threads
#ifdef OLD_WIN_R_BUILD
  progress = 0;
  clock_t start_time = clock();
  clock_t lap_time = clock();
  for (size_t i = 0; i < num_trees; ++i) {
    trees[i]->predict(data.get(), true);
    progress++;
    showProgress("Predicting..", start_time, lap_time);
  }
#else
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  progress = 0;
  for (uint i = 0; i < num_threads; ++i) {
    threads.emplace_back(&Forest::predictTreesInThread, this, i, data.get(), true);
  }
  showProgress("Computing prediction error..", num_trees);
  for (auto &thread : threads) {
    thread.join();
  }

#ifdef R_BUILD
  if (aborted_threads > 0) {
    throw std::runtime_error("User interrupt.");
  }
#endif
#endif

  // Call special function for subclasses
  computePredictionErrorInternal();
}

void Forest::computePermutationImportance() {

// Compute tree permutation importance in multiple threads
#ifdef OLD_WIN_R_BUILD
  progress = 0;
  clock_t start_time = clock();
  clock_t lap_time = clock();

// Initailize importance and variance
  variable_importance.resize(num_independent_variables, 0);
  std::vector<double> variance;
  if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
    variance.resize(num_independent_variables, 0);
  }

// Compute importance
  for (size_t i = 0; i < num_trees; ++i) {
    trees[i]->computePermutationImportance(variable_importance, variance);
    progress++;
    showProgress("Computing permutation importance..", start_time, lap_time);
  }
#else
  progress = 0;
#ifdef R_BUILD
  aborted = false;
  aborted_threads = 0;
#endif

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

// Initailize importance and variance
  std::vector<std::vector<double>> variable_importance_threads(num_threads);
  std::vector<std::vector<double>> variance_threads(num_threads);

// Compute importance
  for (uint i = 0; i < num_threads; ++i) {
    variable_importance_threads[i].resize(num_independent_variables, 0);
    if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
      variance_threads[i].resize(num_independent_variables, 0);
    }
    threads.emplace_back(&Forest::computeTreePermutationImportanceInThread, this, i,
        std::ref(variable_importance_threads[i]), std::ref(variance_threads[i]));
  }
  showProgress("Computing permutation importance..", num_trees);
  for (auto &thread : threads) {
    thread.join();
  }

#ifdef R_BUILD
  if (aborted_threads > 0) {
    throw std::runtime_error("User interrupt.");
  }
#endif

// Sum thread importances
  variable_importance.resize(num_independent_variables, 0);
  for (size_t i = 0; i < num_independent_variables; ++i) {
    for (uint j = 0; j < num_threads; ++j) {
      variable_importance[i] += variable_importance_threads[j][i];
    }
  }
  variable_importance_threads.clear();

// Sum thread variances
  std::vector<double> variance(num_independent_variables, 0);
  if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
    for (size_t i = 0; i < num_independent_variables; ++i) {
      for (uint j = 0; j < num_threads; ++j) {
        variance[i] += variance_threads[j][i];
      }
    }
    variance_threads.clear();
  }
#endif

  for (size_t i = 0; i < variable_importance.size(); ++i) {
    variable_importance[i] /= num_trees;

    // Normalize by variance for scaled permutation importance
    if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
      if (variance[i] != 0) {
        variance[i] = variance[i] / num_trees - variable_importance[i] * variable_importance[i];
        variable_importance[i] /= sqrt(variance[i] / num_trees);
      }
    }
  }
}

#ifndef OLD_WIN_R_BUILD
void Forest::growTreesInThread(uint thread_idx, std::vector<double>* variable_importance) {
  if (thread_ranges.size() > thread_idx + 1) {
    for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
      trees[i]->grow(variable_importance);

      // Check for user interrupt
#ifdef R_BUILD
      if (aborted) {
        std::unique_lock<std::mutex> lock(mutex);
        ++aborted_threads;
        condition_variable.notify_one();
        return;
      }
#endif

      // Increase progress by 1 tree
      std::unique_lock<std::mutex> lock(mutex);
      ++progress;
      condition_variable.notify_one();
    }
  }
}

void Forest::predictTreesInThread(uint thread_idx, const Data* prediction_data, bool oob_prediction) {
  if (thread_ranges.size() > thread_idx + 1) {
    for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
      trees[i]->predict(prediction_data, oob_prediction);

      // Check for user interrupt
#ifdef R_BUILD
      if (aborted) {
        std::unique_lock<std::mutex> lock(mutex);
        ++aborted_threads;
        condition_variable.notify_one();
        return;
      }
#endif

      // Increase progress by 1 tree
      std::unique_lock<std::mutex> lock(mutex);
      ++progress;
      condition_variable.notify_one();
    }
  }
}

void Forest::predictInternalInThread(uint thread_idx) {
  // Create thread ranges
  std::vector<uint> predict_ranges;
  equalSplit(predict_ranges, 0, num_samples - 1, num_threads);

  if (predict_ranges.size() > thread_idx + 1) {
    for (size_t i = predict_ranges[thread_idx]; i < predict_ranges[thread_idx + 1]; ++i) {
      predictInternal(i);

      // Check for user interrupt
#ifdef R_BUILD
      if (aborted) {
        std::unique_lock<std::mutex> lock(mutex);
        ++aborted_threads;
        condition_variable.notify_one();
        return;
      }
#endif

      // Increase progress by 1 tree
      std::unique_lock<std::mutex> lock(mutex);
      ++progress;
      condition_variable.notify_one();
    }
  }
}

void Forest::computeTreePermutationImportanceInThread(uint thread_idx, std::vector<double>& importance,
    std::vector<double>& variance) {
  if (thread_ranges.size() > thread_idx + 1) {
    for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
      trees[i]->computePermutationImportance(importance, variance);

      // Check for user interrupt
#ifdef R_BUILD
      if (aborted) {
        std::unique_lock<std::mutex> lock(mutex);
        ++aborted_threads;
        condition_variable.notify_one();
        return;
      }
#endif

      // Increase progress by 1 tree
      std::unique_lock<std::mutex> lock(mutex);
      ++progress;
      condition_variable.notify_one();
    }
  }
}
#endif



void Forest::setSplitWeightVector(std::vector<std::vector<double>>& split_select_weights) {

// Size should be 1 x num_independent_variables or num_trees x num_independent_variables
  if (split_select_weights.size() != 1 && split_select_weights.size() != num_trees) {
    throw std::runtime_error("Size of split select weights not equal to 1 or number of trees.");
  }

// Reserve space
  size_t num_weights = num_independent_variables;
  if (importance_mode == IMP_GINI_CORRECTED) {
    num_weights = 2 * num_independent_variables;
  }
  if (split_select_weights.size() == 1) {
    this->split_select_weights[0].resize(num_weights);
  } else {
    this->split_select_weights.clear();
    this->split_select_weights.resize(num_trees, std::vector<double>(num_weights));
  }
  this->split_select_varIDs.resize(num_weights);
  deterministic_varIDs.reserve(num_weights);

  // Split up in deterministic and weighted variables, ignore zero weights
  size_t num_zero_weights = 0;
  for (size_t i = 0; i < split_select_weights.size(); ++i) {

    // Size should be 1 x num_independent_variables or num_trees x num_independent_variables
    if (split_select_weights[i].size() != num_independent_variables) {
      throw std::runtime_error("Number of split select weights not equal to number of independent variables.");
    }

    for (size_t j = 0; j < split_select_weights[i].size(); ++j) {
      double weight = split_select_weights[i][j];

      if (i == 0) {
        size_t varID = j;
        for (auto& skip : data->getNoSplitVariables()) {
          if (varID >= skip) {
            ++varID;
          }
        }

        if (weight == 1) {
          deterministic_varIDs.push_back(varID);
        } else if (weight < 1 && weight > 0) {
          this->split_select_varIDs[j] = varID;
          this->split_select_weights[i][j] = weight;
        } else if (weight == 0) {
          ++num_zero_weights;
        } else if (weight < 0 || weight > 1) {
          throw std::runtime_error("One or more split select weights not in range [0,1].");
        }

      } else {
        if (weight < 1 && weight > 0) {
          this->split_select_weights[i][j] = weight;
        } else if (weight < 0 || weight > 1) {
          throw std::runtime_error("One or more split select weights not in range [0,1].");
        }
      }
    }

    // Copy weights for corrected impurity importance
    if (importance_mode == IMP_GINI_CORRECTED) {
      std::vector<double>* sw = &(this->split_select_weights[i]);
      std::copy_n(sw->begin(), num_independent_variables, sw->begin() + num_independent_variables);

      for (size_t k = 0; k < num_independent_variables; ++k) {
        split_select_varIDs[num_independent_variables + k] = num_variables + k;
      }

      size_t num_deterministic_varIDs = deterministic_varIDs.size();
      for (size_t k = 0; k < num_deterministic_varIDs; ++k) {
        size_t varID = deterministic_varIDs[k];
        for (auto& skip : data->getNoSplitVariables()) {
          if (varID >= skip) {
            --varID;
          }
        }
        deterministic_varIDs.push_back(varID + num_variables);
      }
    }
  }

  if (num_weights - deterministic_varIDs.size() - num_zero_weights < mtry) {
    throw std::runtime_error("Too many zeros or ones in split select weights. Need at least mtry variables to split at.");
  }
}

void Forest::setAlwaysSplitVariables(const std::vector<std::string>& always_split_variable_names) {

  deterministic_varIDs.reserve(num_independent_variables);

  for (auto& variable_name : always_split_variable_names) {
    size_t varID = data->getVariableID(variable_name);
    deterministic_varIDs.push_back(varID);
  }

  if (deterministic_varIDs.size() + this->mtry > num_independent_variables) {
    throw std::runtime_error(
        "Number of variables to be always considered for splitting plus mtry cannot be larger than number of independent variables.");
  }

  // Also add variables for corrected impurity importance
  if (importance_mode == IMP_GINI_CORRECTED) {
    size_t num_deterministic_varIDs = deterministic_varIDs.size();
    for (size_t k = 0; k < num_deterministic_varIDs; ++k) {
      size_t varID = deterministic_varIDs[k];
      for (auto& skip : data->getNoSplitVariables()) {
        if (varID >= skip) {
          --varID;
        }
      }
      deterministic_varIDs.push_back(varID + num_variables);
    }
  }
}

#ifdef OLD_WIN_R_BUILD
void Forest::showProgress(std::string operation, clock_t start_time, clock_t& lap_time) {

// Check for user interrupt
  if (checkInterrupt()) {
    throw std::runtime_error("User interrupt.");
  }

  double elapsed_time = (clock() - lap_time) / CLOCKS_PER_SEC;
  if (elapsed_time > STATUS_INTERVAL) {
    double relative_progress = (double) progress / (double) num_trees;
    double time_from_start = (clock() - start_time) / CLOCKS_PER_SEC;
    uint remaining_time = (1 / relative_progress - 1) * time_from_start;
    if (verbose_out) {
      *verbose_out << operation << " Progress: " << round(100 * relative_progress)
      << "%. Estimated remaining time: " << beautifyTime(remaining_time) << "." << std::endl;
    }
    lap_time = clock();
  }
}
#else
void Forest::showProgress(std::string operation, size_t max_progress) {
  using std::chrono::steady_clock;
  using std::chrono::duration_cast;
  using std::chrono::seconds;

  steady_clock::time_point start_time = steady_clock::now();
  steady_clock::time_point last_time = steady_clock::now();
  std::unique_lock<std::mutex> lock(mutex);

// Wait for message from threads and show output if enough time elapsed
  while (progress < max_progress) {
    condition_variable.wait(lock);
    seconds elapsed_time = duration_cast<seconds>(steady_clock::now() - last_time);

    // Check for user interrupt
#ifdef R_BUILD
    if (!aborted && checkInterrupt()) {
      aborted = true;
    }
    if (aborted && aborted_threads >= num_threads) {
      return;
    }
#endif

    if (progress > 0 && elapsed_time.count() > STATUS_INTERVAL) {
      double relative_progress = (double) progress / (double) max_progress;
      seconds time_from_start = duration_cast<seconds>(steady_clock::now() - start_time);
      uint remaining_time = (1 / relative_progress - 1) * time_from_start.count();
      // if (verbose_out) {
      //   *verbose_out << operation << " Progress: " << round(100 * relative_progress) << "%. Estimated remaining time: "
      //       << beautifyTime(remaining_time) << "." << std::endl;
      // }
      if (this->verbose) {
        *verbose_out << operation << " Progress: " << round(100 * relative_progress) << "%. Estimated remaining time: "
            << beautifyTime(remaining_time) << "." << std::endl;
      }
      last_time = steady_clock::now();
    }
  }
}
#endif




void ForestSurvival::loadForest(size_t dependent_varID, size_t num_trees,
    std::vector<std::vector<std::vector<size_t>> >& forest_child_nodeIDs,
    std::vector<std::vector<size_t>>& forest_split_varIDs, std::vector<std::vector<double>>& forest_split_values,
    size_t status_varID, std::vector<std::vector<std::vector<double>> >& forest_chf,
    std::vector<double>& unique_timepoints, std::vector<bool>& is_ordered_variable) {

  this->dependent_varID = dependent_varID;
  this->status_varID = status_varID;
  this->num_trees = num_trees;
  this->unique_timepoints = unique_timepoints;
  data->setIsOrderedVariable(is_ordered_variable);

  // Create trees
  trees.reserve(num_trees);
  for (size_t i = 0; i < num_trees; ++i) {
    trees.push_back(
        make_unique<TreeSurvival>(forest_child_nodeIDs[i], forest_split_varIDs[i], forest_split_values[i],
            forest_chf[i], &this->unique_timepoints, &response_timepointIDs));
  }

  // Create thread ranges
  equalSplit(thread_ranges, 0, num_trees - 1, num_threads);
}

std::vector<std::vector<std::vector<double>>> ForestSurvival::getChf() const {
  std::vector<std::vector<std::vector<double>>> result;
  result.reserve(num_trees);
  for (const auto& tree : trees) {
    const auto& temp = dynamic_cast<const TreeSurvival&>(*tree);
    result.push_back(temp.getChf());
  }
  return result;
}

void ForestSurvival::initInternal(std::string status_variable_name) {

  // Convert status variable name to ID
  if (!prediction_mode && !status_variable_name.empty()) {
    status_varID = data->getVariableID(status_variable_name);
  }

  data->addNoSplitVariable(status_varID);

  // If mtry not set, use floored square root of number of independent variables.
  if (mtry == 0) {
    unsigned long temp = ceil(sqrt((double) (num_variables - 2)));
    mtry = std::max((unsigned long) 1, temp);
  }

  // Set minimal node size
  if (min_node_size == 0) {
    min_node_size = DEFAULT_MIN_NODE_SIZE_SURVIVAL;
  }

  // Create unique timepoints
  std::set<double> unique_timepoint_set;
  for (size_t i = 0; i < num_samples; ++i) {
    unique_timepoint_set.insert(data->get(i, dependent_varID));
  }
  unique_timepoints.reserve(unique_timepoint_set.size());
  for (auto& t : unique_timepoint_set) {
    unique_timepoints.push_back(t);
  }

  // Create response_timepointIDs
  if (!prediction_mode) {
    for (size_t i = 0; i < num_samples; ++i) {
      double value = data->get(i, dependent_varID);

      // If timepoint is already in unique_timepoints, use ID. Else create a new one.
      uint timepointID = find(unique_timepoints.begin(), unique_timepoints.end(), value) - unique_timepoints.begin();
      response_timepointIDs.push_back(timepointID);
    }
  }

  // Sort data if extratrees and not memory saving mode
  if (splitrule == EXTRATREES && !memory_saving_splitting) {
    data->sort();
  }
}

void ForestSurvival::growInternal() {
  trees.reserve(num_trees);
  for (size_t i = 0; i < num_trees; ++i) {
    trees.push_back(make_unique<TreeSurvival>(&unique_timepoints, status_varID, &response_timepointIDs));
  }
}

void ForestSurvival::allocatePredictMemory() {
  size_t num_prediction_samples = data->getNumRows();
  size_t num_timepoints = unique_timepoints.size();
  if (predict_all) {
    predictions = std::vector<std::vector<std::vector<double>>>(num_prediction_samples,
        std::vector<std::vector<double>>(num_timepoints, std::vector<double>(num_trees, 0)));
  } else if (prediction_type == TERMINALNODES) {
    predictions = std::vector<std::vector<std::vector<double>>>(1,
        std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_trees, 0)));
  } else {
    predictions = std::vector<std::vector<std::vector<double>>>(1,
        std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_timepoints, 0)));
  }
}

void ForestSurvival::predictInternal(size_t sample_idx) {
  // For each timepoint sum over trees
  if (predict_all) {
    for (size_t j = 0; j < unique_timepoints.size(); ++j) {
      for (size_t k = 0; k < num_trees; ++k) {
        predictions[sample_idx][j][k] = getTreePrediction(k, sample_idx)[j];
      }
    }
  } else if (prediction_type == TERMINALNODES) {
    for (size_t k = 0; k < num_trees; ++k) {
      predictions[0][sample_idx][k] = getTreePredictionTerminalNodeID(k, sample_idx);
    }
  } else {
    for (size_t j = 0; j < unique_timepoints.size(); ++j) {
      double sample_time_prediction = 0;
      for (size_t k = 0; k < num_trees; ++k) {
        sample_time_prediction += getTreePrediction(k, sample_idx)[j];
      }
      predictions[0][sample_idx][j] = sample_time_prediction / num_trees;
    }
  }
}

void ForestSurvival::computePredictionErrorInternal() {

  size_t num_timepoints = unique_timepoints.size();

  // For each sample sum over trees where sample is OOB
  std::vector<size_t> samples_oob_count;
  samples_oob_count.resize(num_samples, 0);
  predictions = std::vector<std::vector<std::vector<double>>>(1,
      std::vector<std::vector<double>>(num_samples, std::vector<double>(num_timepoints, 0)));

  for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
    for (size_t sample_idx = 0; sample_idx < trees[tree_idx]->getNumSamplesOob(); ++sample_idx) {
      size_t sampleID = trees[tree_idx]->getOobSampleIDs()[sample_idx];
      std::vector<double> tree_sample_chf = getTreePrediction(tree_idx, sample_idx);

      for (size_t time_idx = 0; time_idx < tree_sample_chf.size(); ++time_idx) {
        predictions[0][sampleID][time_idx] += tree_sample_chf[time_idx];
      }
      ++samples_oob_count[sampleID];
    }
  }

  // Divide sample predictions by number of trees where sample is oob and compute summed chf for samples
  std::vector<double> sum_chf;
  sum_chf.reserve(predictions[0].size());
  std::vector<size_t> oob_sampleIDs;
  oob_sampleIDs.reserve(predictions[0].size());
  for (size_t i = 0; i < predictions[0].size(); ++i) {
    if (samples_oob_count[i] > 0) {
      double sum = 0;
      for (size_t j = 0; j < predictions[0][i].size(); ++j) {
        predictions[0][i][j] /= samples_oob_count[i];
        sum += predictions[0][i][j];
      }
      sum_chf.push_back(sum);
      oob_sampleIDs.push_back(i);
    }
  }

  // Use all samples which are OOB at least once
  overall_prediction_error = 1 - computeConcordanceIndex(*data, sum_chf, dependent_varID, status_varID, oob_sampleIDs);
}


const std::vector<double>& ForestSurvival::getTreePrediction(size_t tree_idx, size_t sample_idx) const {
  const auto& tree = dynamic_cast<const TreeSurvival&>(*trees[tree_idx]);
  return tree.getPrediction(sample_idx);
}

size_t ForestSurvival::getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const {
  const auto& tree = dynamic_cast<const TreeSurvival&>(*trees[tree_idx]);
  return tree.getPredictionTerminalNodeID(sample_idx);
}


  SurvivalForestModel::SurvivalForestModel() {
  }

  SurvivalForestModel::~SurvivalForestModel() {
  }

  void SurvivalForestModel::fit( std::vector <std::vector<double> > input_data, 
     std::string dependent_variable_name, std::string status_variable_name, 
     std::vector<std::string> variable_names, std::vector<double>& case_weights, 
     uint mtry, uint num_trees, uint min_node_size, double alpha, double minprop, 
     uint num_random_splits, uint max_depth, bool sample_with_replacement, 
     double sample_fraction_value, int importance_mode_r, int splitrule_r, 
     int prediction_type_r, bool verbose, int seed, int num_threads, bool save_memory){

    /**

    ##' @param sample.fraction Fraction of observations to sample. Default is 1 for sampling with replacement and 0.632 for sampling without replacement.
    ##' @param case.weights Weights for sampling of training observations. Observations with larger weights will be selected with higher probability in the bootstrap (or subsampled) samples for the trees.
    ##' @param split.select.weights Numeric vector with weights between 0 and 1, representing the probability to select variables for splitting. Alternatively, a list of size num.trees, containing split select weight vectors for each tree can be used.  
    ##' @param always.split.variables Character vector with variable names to be always selected in addition to the \code{mtry} variables tried for splitting.
    ##' @param keep.inbag Save how often observations are in-bag in each tree. 
    ##' @param inbag Manually set observations per tree. List of size num.trees, containing inbag counts for each observation. Can be used for stratified sampling.
    ##' @param holdout Hold-out mode. Hold-out all samples with case weight 0 and use these for variable importance and prediction error.
    ##' @param save.memory Use memory saving (but slower) splitting mode. No effect for survival and GWAS data. Warning: This option slows down the tree growing, use only if you encounter memory problems.
    */



      /* ------------ Initializing variables that won't be used ------------ */

      // probability to select variables for splitting.
      std::vector<std::vector<double>> split_select_weights;
      split_select_weights.clear();

      // variable names to be always selected
      std::vector<std::string> always_split_variable_names;
      bool use_always_split_variable_names;
      always_split_variable_names.clear();

      // Dealing with unordered factor covariates-> all features are numerical so it doesn't apply
      // bool use_unordered_variable_names=false;
      std::vector<std::string> unordered_variable_names;
      unordered_variable_names.clear();

      // Manually set observations per tree. 
      std::vector<std::vector<size_t>> inbag;
      inbag.clear();
      // Save how often observations are in-bag in each tree.
      bool keep_inbag=false;

      // Related to GenABEL GWA data so it doesn't apply
      bool order_snps = false;

      // Hold-out mode. 
      bool holdout = false;

      // Predicting everything
      bool predict_all=false;
      bool prediction_mode=false;


      /* ------------ Initializing variables that will be used ------------ */

      //Verbose
      std::ostream* verbose_out;
      verbose_out =  new std::stringstream; //&std::cout; //

      // Seed
      uint seed_value;
      if(seed<=0){
        seed_value = (uint)(clock()); 
      } else {
        seed_value = seed;
      }

      // Adjusting the number of threads to be <= number of cores
      uint max_num_threads = (uint) thread::hardware_concurrency();
      if ((num_threads < 0) | (num_threads >= max_num_threads)){
        num_threads = max_num_threads; 
      } else if(num_threads == 0){
            num_threads = 1; 
        }

      // Parameters
      ImportanceMode importance_mode = (ImportanceMode) importance_mode_r;
      SplitRule splitrule = (SplitRule) splitrule_r;
      PredictionType prediction_type = (PredictionType) prediction_type_r;
      std::vector<double> sample_fraction;
      sample_fraction.push_back(sample_fraction_value);

      // Data
      std::unique_ptr<Data> data { };
      size_t num_rows = input_data.size();
      size_t num_cols = input_data[0].size();
      data = make_unique<Data>();
      data->loadData(input_data, variable_names);

      // Forest object - Initializing the forest
      std::unique_ptr<Forest> forest { };
      forest = make_unique<ForestSurvival>();
      forest->initR(dependent_variable_name, std::move(data), mtry, num_trees, verbose_out, seed_value, num_threads,
          importance_mode, min_node_size, split_select_weights, always_split_variable_names, status_variable_name,
          prediction_mode, sample_with_replacement, unordered_variable_names, save_memory, splitrule, case_weights,
          inbag, predict_all, keep_inbag, sample_fraction, alpha, minprop, holdout, prediction_type, num_random_splits, 
          order_snps, max_depth);

      // Run the model
      forest->run(verbose, true);

        // Saving the attributes
      auto& temp = dynamic_cast<ForestSurvival&>(*forest);
        this->num_trees = forest->getNumTrees();
      this->num_independent_variables = forest->getNumIndependentVariables();
      this->unique_timepoints = temp.getUniqueTimepoints();
      this->mtry = forest->getMtry();
      this->min_node_size = forest->getMinNodeSize();
      this->variable_importance = forest->getVariableImportance();
      this->overall_prediction_error = forest->getOverallPredictionError(); 
      this->dependent_varID   = forest->getDependentVarId();
      this->child_nodeIDs     = forest->getChildNodeIDs();
      this->split_varIDs      = forest->getSplitVarIDs();
      this->split_values      = forest->getSplitValues();
      this->is_ordered        = forest->getIsOrderedVariable();
      this->status_varID      = temp.getStatusVarId();
      this->chf               = temp.getChf();
      this->variable_names    = variable_names ;
  }



  std::vector<std::vector<std::vector<double>>> SurvivalForestModel::predict( std::vector <std::vector<double> > input_data, 
     std::string dependent_variable_name, std::string status_variable_name, 
     std::vector<std::string> variable_names, std::vector<double>& case_weights, 
     uint mtry, uint num_trees, uint min_node_size, double alpha, double minprop, 
     uint num_random_splits, uint max_depth, bool sample_with_replacement, 
     double sample_fraction_value, int importance_mode_r, int splitrule_r, 
     int prediction_type_r, bool verbose, int seed, int num_threads, bool save_memory){

    /* ------------ Initializing variables that won't be used ------------ */

    // probability to select variables for splitting.
    std::vector<std::vector<double>> split_select_weights;
    split_select_weights.clear();

    // variable names to be always selected
    std::vector<std::string> always_split_variable_names;
    bool use_always_split_variable_names;
    always_split_variable_names.clear();

    // Dealing with unordered factor covariates-> all features are numerical so it doesn't apply
    // bool use_unordered_variable_names=false;
    std::vector<std::string> unordered_variable_names;
    unordered_variable_names.clear();

    // Manually set observations per tree. 
    std::vector<std::vector<size_t>> inbag;
    inbag.clear();
    // Save how often observations are in-bag in each tree.
    bool keep_inbag=false;

    // Related to GenABEL GWA data so it doesn't apply
    bool order_snps = false;

    // Hold-out mode. 
    bool holdout = false;


    /* ------------ Initializing variables that will be used ------------ */

    // Predicting everything
    bool predict_all=true;
    bool prediction_mode=true;
    prediction_type_r = 1;

    //Verbose
    std::ostream* verbose_out;
    verbose_out =  new std::stringstream; //&std::cout; //

    // Seed
    uint seed_value;
    if(seed<=0){
      seed_value = (uint)(clock()); 
    } else {
      seed_value = seed;
    }

    // Adjusting the number of threads to be <= number of cores
    uint max_num_threads = (uint) thread::hardware_concurrency();
    if ((num_threads < 0) | (num_threads >= max_num_threads)){
      num_threads = max_num_threads; 
    } else if(num_threads == 0){
          num_threads = 1; 
      }

    // Parameters
    ImportanceMode importance_mode = (ImportanceMode) importance_mode_r;
    SplitRule splitrule = (SplitRule) splitrule_r;
    PredictionType prediction_type = (PredictionType) prediction_type_r;
    std::vector<double> sample_fraction;
    sample_fraction.push_back(sample_fraction_value);

    // Data
    std::unique_ptr<Data> data { };
    size_t num_rows = input_data.size();
    size_t num_cols = input_data[0].size();
    data = make_unique<Data>();
    data->loadData(input_data, variable_names);

    // Forest object - Initializing the forest
    std::unique_ptr<Forest> forest { };
    forest = make_unique<ForestSurvival>();
    forest->initR(dependent_variable_name, std::move(data), mtry, num_trees, verbose_out, seed_value, num_threads,
        importance_mode, min_node_size, split_select_weights, always_split_variable_names, status_variable_name,
        prediction_mode, sample_with_replacement, unordered_variable_names, save_memory, splitrule, case_weights,
        inbag, predict_all, keep_inbag, sample_fraction, alpha, minprop, holdout, prediction_type, num_random_splits, 
        order_snps, max_depth);

    // Initializing the prediction
    size_t dependent_varID = this->dependent_varID; 
    std::vector<std::vector<std::vector<size_t>> > child_nodeIDs = this-> child_nodeIDs;
    std::vector<std::vector<size_t>> split_varIDs = this->split_varIDs; 
    std::vector<std::vector<double>> split_values = this->split_values; 
    std::vector<bool> is_ordered = this->is_ordered ;
    size_t status_varID = this->status_varID ;
    std::vector<std::vector<std::vector<double>> > chf = this->chf;
    std::vector<double> unique_timepoints = this->unique_timepoints;
    auto& temp = dynamic_cast<ForestSurvival&>(*forest);
    temp.loadForest(dependent_varID, num_trees, child_nodeIDs, split_varIDs, split_values, status_varID, chf,
        unique_timepoints, is_ordered);

    // Run the model
      forest->run(false, false);

        const std::vector<std::vector<std::vector<double>>>& predictions = forest->getPredictions();
        return predictions;

}



  std::vector<std::vector<double> > SurvivalForestModel::predict_survival( std::vector <std::vector<double> > input_data, int num_threads){

    std::vector<std::string> variable_names = this->variable_names;
     std::string dependent_variable_name = this->variable_names[this->dependent_varID];
     std::string status_variable_name = this->variable_names[this->status_varID];
     std::vector<double> case_weights;
     for (int i = 0; i < input_data.size(); ++i){
      case_weights.push_back(0.);
     }
     
     uint mtry = 0; //this->mtry; 
     uint num_trees= this->num_trees;  
     uint min_node_size= 0; //this->min_node_size; 
     double alpha = 0; //0.05; 
     double minprop = 0.1;
     uint num_random_splits = 1;
     uint max_depth = 0; //1;
     bool sample_with_replacement = true; 
     double sample_fraction_value = 1; 
     int importance_mode_r = 0;//1;
     int splitrule_r= 1; //5; 
     int prediction_type_r=1; 
     bool verbose=false;
     int seed = 1;
     bool save_memory = false;

    const std::vector<std::vector<std::vector<double>>> predictions = this->predict(input_data,
      dependent_variable_name, status_variable_name, variable_names, case_weights, 
       mtry, num_trees, min_node_size, alpha, minprop, 
       num_random_splits, max_depth, sample_with_replacement, 
       sample_fraction_value, importance_mode_r, splitrule_r, 
       prediction_type_r, verbose, seed, num_threads, save_memory);

    vector<double> results_vector;
    vector< vector<double> > results;

    results_vector.resize(predictions[0].size(), 0.);
    results.resize(predictions.size(), results_vector );

    for (size_t b = 0; b < this->num_trees; ++b){ //num trees

      for (size_t i = 0; i < predictions.size(); ++i){ // num of samples
    
        for (size_t j = 0; j < predictions[0].size(); ++j){ // num of samples

          results[i][j] += exp(-predictions[i][j][b])/this->num_trees;
        }
      }
    }

    return results;
  }



  std::vector<std::vector<double> > SurvivalForestModel::predict_hazard( std::vector <std::vector<double> > input_data, int num_threads){


    std::vector<std::string> variable_names = this->variable_names;
     std::string dependent_variable_name = this->variable_names[this->dependent_varID];
     std::string status_variable_name = this->variable_names[this->status_varID];
     std::vector<double> case_weights;
     for (int i = 0; i < input_data.size(); ++i){
      case_weights.push_back(1./input_data.size());
     }
     
     uint mtry = this->mtry; 
     uint num_trees= this->num_trees;  
     uint min_node_size= this->min_node_size; 
     double alpha = 0.05; 
     double minprop = 0.1;
     uint num_random_splits = 1;
     uint max_depth = 1;
     bool sample_with_replacement = false; 
     double sample_fraction_value = 1; 
     int importance_mode_r = 1;
     int splitrule_r=5; 
     int prediction_type_r=1; 
     bool verbose=false;
     int seed = 1;
     bool save_memory = false;

    const std::vector<std::vector<std::vector<double>>> predictions = this->predict(input_data,
      dependent_variable_name, status_variable_name, variable_names, case_weights, 
       mtry, num_trees, min_node_size, alpha, minprop, 
       num_random_splits, max_depth, sample_with_replacement, 
       sample_fraction_value, importance_mode_r, splitrule_r, 
       prediction_type_r, verbose, seed, num_threads, save_memory);
      vector<double> results_vector;
      vector< vector<double> > results;

      results_vector.resize(predictions[0].size(), 0.);
      results.resize(predictions.size(), results_vector );

      for (size_t b = 0; b < this->num_trees; ++b){ //num trees

        for (size_t i = 0; i < predictions.size(); ++i){ // num of samples
        
          results[i][0] += predictions[i][0][b]/this->num_trees;

          for (size_t j = 1; j < predictions[0].size(); ++j){ // num of samples
            results[i][j] += (predictions[i][j][b]-predictions[i][j-1][b])/this->num_trees;
          }
        }
      }

    return results;
  }


  std::vector<double> SurvivalForestModel::predict_risk( std::vector <std::vector<double> > input_data, int num_threads){


    std::vector<std::string> variable_names = this->variable_names;
     std::string dependent_variable_name = this->variable_names[this->dependent_varID];
     std::string status_variable_name = this->variable_names[this->status_varID];
     std::vector<double> case_weights;
     for (int i = 0; i < input_data.size(); ++i){
      case_weights.push_back(1./input_data.size());
     }
     
     uint mtry = this->mtry; 
     uint num_trees= this->num_trees;  
     uint min_node_size= this->min_node_size; 
     double alpha = 0.05; 
     double minprop = 0.1;
     uint num_random_splits = 1;
     uint max_depth = 1;
     bool sample_with_replacement = false; 
     double sample_fraction_value = 1; 
     int importance_mode_r = 1;
     int splitrule_r=5; 
     int prediction_type_r=1; 
     bool verbose=false;
     int seed = 1;
     bool save_memory = false;

    const std::vector<std::vector<std::vector<double>>> predictions = this->predict(input_data,
      dependent_variable_name, status_variable_name, variable_names, case_weights, 
       mtry, num_trees, min_node_size, alpha, minprop, 
       num_random_splits, max_depth, sample_with_replacement, 
       sample_fraction_value, importance_mode_r, splitrule_r, 
       prediction_type_r, verbose, seed, num_threads, save_memory);

      vector<double> results;

      results.resize(predictions.size(), 0. );

    for (size_t b = 0; b < this->num_trees; ++b){ //num trees

      for (size_t i = 0; i < predictions.size(); ++i){ // num of samples
    
        for (size_t j = 0; j < predictions[0].size(); ++j){ // num of samples

          results[i] += predictions[i][j][b]/this->num_trees;
        }
      }
    }

    return results;
  }



} // namespace ranger
