#%%cython --a  
# distutils: language = c++

# Importing cython and C++ 
#---------------------------
import cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp cimport bool
from libc.stdint cimport uint32_t as uint

# Importing C++ object SurvivalForestModel from SurvivalForestModel.h
#-----------------------------------------------------------
cdef extern from "survival_forest.h" namespace "ranger":

    cdef cppclass SurvivalForestModel:
        SurvivalForestModel() except+;

        # Attributes
        size_t num_independent_variables;
        uint mtry;
        uint min_node_size;
        vector[double]& variable_importance;
        double overall_prediction_error; 
        size_t dependent_varID;
        size_t status_varID;
        size_t num_trees;
        vector[vector[vector[size_t]] ] child_nodeIDs;
        vector[vector[size_t]] split_varIDs; 
        vector[vector[double]] split_values; 
        vector[bool] is_ordered;
        vector[vector[vector[double]] ] chf;
        vector[double] unique_timepoints;
        vector[string] variable_names;

        # Methods
        void fit( vector [vector[double] ] input_data, 
             string dependent_variable_name, string status_variable_name, 
             vector[string] variable_names, vector[double]& case_weights, 
             uint mtry, uint num_trees, uint min_node_size, double alpha, double minprop, 
             uint num_random_splits, uint max_depth, bool sample_with_replacement, 
             double sample_fraction_value, int importance_mode_r, int splitrule_r, 
             int prediction_type_r, bool verbose, int seed, int num_threads, bool save_memory);

        vector[vector[double] ] predict_survival( vector [vector[double] ] input_data, int num_threads);

        vector[vector[double] ] predict_hazard( vector [vector[double] ] input_data, int num_threads);

        vector[double] predict_risk( vector [vector[double] ] input_data, int num_threads);




cdef class _SurvivalForestModel :

    cdef:
        SurvivalForestModel *thisptr # hold a C++ instance which we're wrapping

    def __init__(self):
        self.thisptr = new SurvivalForestModel()

    def __dealloc__(self):
        del self.thisptr

    # Times
    @property
    def unique_timepoints(self):
        return self.thisptr.unique_timepoints

    @unique_timepoints.setter
    def unique_timepoints(self, unique_timepoints):
        self.thisptr.unique_timepoints = unique_timepoints

    # num_trees
    @property
    def num_trees(self):
        return self.thisptr.num_trees

    @num_trees.setter
    def num_trees(self, num_trees):
        self.thisptr.num_trees = num_trees

    # variable_importance
    @property
    def variable_importance(self):
        return self.thisptr.variable_importance

    @variable_importance.setter
    def variable_importance(self, variable_importance):
        self.thisptr.variable_importance = variable_importance

    # num_independent_variables
    @property
    def num_independent_variables(self):
        return self.thisptr.num_independent_variables

    @num_independent_variables.setter
    def num_independent_variables(self, num_independent_variables):
        self.thisptr.num_independent_variables = num_independent_variables

    # mtry
    @property
    def mtry(self):
        return self.thisptr.mtry

    @mtry.setter
    def mtry(self, mtry):
        self.thisptr.mtry = mtry

    # min_node_size
    @property
    def min_node_size(self):
        return self.thisptr.min_node_size

    @min_node_size.setter
    def min_node_size(self, min_node_size):
        self.thisptr.min_node_size = min_node_size

    # overall_prediction_error
    @property
    def overall_prediction_error(self):
        return self.thisptr.overall_prediction_error

    @overall_prediction_error.setter
    def overall_prediction_error(self, overall_prediction_error):
        self.thisptr.overall_prediction_error = overall_prediction_error

    # dependent_varID
    @property
    def dependent_varID(self):
        return self.thisptr.dependent_varID

    @dependent_varID.setter
    def dependent_varID(self, dependent_varID):
        self.thisptr.dependent_varID = dependent_varID

    # status_varID
    @property
    def status_varID(self):
        return self.thisptr.status_varID

    @status_varID.setter
    def status_varID(self, status_varID):
        self.thisptr.status_varID = status_varID

    # child_nodeIDs
    @property
    def child_nodeIDs(self):
        return self.thisptr.child_nodeIDs

    @child_nodeIDs.setter
    def child_nodeIDs(self, child_nodeIDs):
        self.thisptr.child_nodeIDs = child_nodeIDs

    # split_varIDs
    @property
    def split_varIDs(self):
        return self.thisptr.split_varIDs

    @split_varIDs.setter
    def split_varIDs(self, split_varIDs):
        self.thisptr.split_varIDs = split_varIDs

    # split_values
    @property
    def split_values(self):
        return self.thisptr.split_values

    @split_values.setter
    def split_values(self, split_values):
        self.thisptr.split_values = split_values

    # is_ordered
    @property
    def is_ordered(self):
        return self.thisptr.is_ordered

    @is_ordered.setter
    def is_ordered(self, is_ordered):
        self.thisptr.is_ordered = is_ordered

    # chf
    @property
    def chf(self):
        return self.thisptr.chf

    @chf.setter
    def chf(self, chf):
        self.thisptr.chf = chf

    # variable_names
    @property
    def variable_names(self):
        return self.thisptr.variable_names

    @variable_names.setter
    def variable_names(self, variable_names):
        self.thisptr.variable_names = variable_names

    # -------------------------- Fitting the model -------------------------- #
    cpdef void fit(self, vector[vector[double] ] input_data, 
        vector[string] all_data_features, vector[double]& case_weights,
        uint num_trees, uint mtry, uint min_node_size, uint max_depth,
        double alpha, double minprop, uint num_random_splits, 
        double sample_fraction_value, int importance_mode_r, int splitrule_r, 
        bool verbose, int seed, int num_threads, bool save_memory):
        """ Fitting the C++ model and saving the attributes """

        # Declaring the variables
        cdef:
            string dependent_variable_name = all_data_features[0];
            string status_variable_name = all_data_features[1];
            int prediction_type_r = 1;
            bool with_replacement=True;

        # Sending the variables to C++ object to be trained
        self.thisptr.fit( input_data, dependent_variable_name, 
            status_variable_name, all_data_features, #variable_names, 
            case_weights, mtry, num_trees, min_node_size, alpha, minprop, 
            num_random_splits, max_depth, with_replacement, #sample_with_replacement, 
            sample_fraction_value, importance_mode_r, splitrule_r, 
            prediction_type_r, verbose, seed, num_threads, save_memory)


    # ---------------------------- Predicting ------------------------------ #
    cpdef vector[vector[double]] predict_survival(self, vector[vector[double]] input_data, 
        int num_threads= -1):
        """ Predicting the hazard functions """

        return self.thisptr.predict_survival(input_data, num_threads)


    cpdef vector[vector[double]] predict_hazard(self, vector[vector[double]] input_data, 
        int num_threads= -1):
        """ Predicting the hazard functions """

        return self.thisptr.predict_hazard(input_data, num_threads)


    cpdef vector[double] predict_risk(self, vector[vector[double]] input_data, 
        int num_threads= -1):
        """ Predicting the risk values """

        return self.thisptr.predict_risk(input_data, num_threads)
