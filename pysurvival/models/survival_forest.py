from __future__ import absolute_import
import warnings
import numpy as np
import pandas as pd
import os
import copy
from sklearn.preprocessing import StandardScaler
from pysurvival import utils
from pysurvival.models import BaseModel
from pysurvival.models._survival_forest import _SurvivalForestModel
from pysurvival import PYTHON_VERSION

# Available Splitting
SPLITTING_RULES = {'logrank': 1, 'maxstat': 4, 'extratrees': 5}


class BaseSurvivalForest(BaseModel):
    """
    A random survival forest base class.

    Parameters
    ----------

    * num_trees : int (default=10)
        The number of trees in the forest.

    * splitrule: int (default=0)
        Splitting rule used to build trees:  
            - 1, "logrank" yields the RandomSurvivalForest
            - 4, "maxstat" yields the ConditionalSurvivalForest
            - 5, "extratrees" yields the ExtraSurvivalTrees

    """

    def __init__(self, splitrule="Logrank", num_trees=10):

        # Checking the format of num_trees
        if not (isinstance(num_trees, int) or isinstance(num_trees, float)):
            error = '{} is not a valid value for "num_trees" '
            error += 'as "num_trees" is a positive integer'.format(num_trees)
            raise ValueError(error)
        if num_trees <= 0:
            error = '{} is not a valid value for "num_trees" '
            error += 'as "num_trees" is a positive integer'.format(num_trees)
            raise ValueError(error)
        self.num_trees = num_trees

        # Checking the format of splitrule
        if SPLITTING_RULES.get(splitrule.lower()) is None:
            error = '{} is not a valid splitrule method. Choose between '
            error += '"' + '", "'.join(SPLITTING_RULES) + '"'
            error = error.format(splitrule)
            raise ValueError(error)
        self.splitrule = splitrule

        # Initializing the inner model
        self.model = _SurvivalForestModel()

        # Initializing the elements from BaseModel
        super(BaseSurvivalForest, self).__init__(auto_scaler=False)

    def save(self, path_file):
        """ Save the model parameters of the model and compress them into 
            a zip file
        """

        # Ensuring the file has the proper name
        folder_name = os.path.dirname(path_file) + '/'
        file_name = os.path.basename(path_file)

        # Checking if the folder is accessible
        if not os.access(folder_name, os.W_OK):
            error_msg = '{} is not an accessible directory.'.format(folder_name)
            raise OSError(error_msg)

        # Delete the C++ object before saving
        del self.model

        # Saving the model
        super(BaseSurvivalForest, self).save(path_file)

        # Re-introduce the C++ object
        self.model = _SurvivalForestModel()
        self.load_properties()

    def load(self, path_file):
        """ Load the model parameters from a zip file into a C++ external
            model 
        """

        # Loading the model
        super(BaseSurvivalForest, self).load(path_file)

        # Re-introduce the C++ object
        self.model = _SurvivalForestModel()
        self.load_properties()

    def fit(self, X, T, E, max_features='sqrt', max_depth=5,
            min_node_size=10, num_threads=-1, weights=None,
            sample_size_pct=0.63, alpha=0.5, minprop=0.1,
            num_random_splits=100, importance_mode='impurity_corrected',
            seed=None, save_memory=False):
        """
        Arguments:
        ---------
        * X : array-like, shape=(n_samples, n_features)
            The input samples.

        * T : array-like, shape = [n_samples] 
            The target values describing when the event of interest or censoring
            occurred

        * E : array-like, shape = [n_samples] 
            The Event indicator array such that E = 1. if the event occurred
            E = 0. if censoring occurred

        * max_features : int, float or string, optional (default="all")
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "sqrt", then `max_features=sqrt(n_features)` 
            - If "log2", then `max_features=log2(n_features)`.

        * min_node_size : int(default=10)
            The minimum number of samples required to be at a leaf node

        * num_threads: int (Default: -1)
            The number of jobs to run in parallel for both fit and predict. 
            If -1, then the number of jobs is set to the number of cores.

        * weights: array-like, shape = [n_samples] (default=None)
            Weights for sampling of training observations. 
            Observations with larger weights will be selected with 
            higher probability in the bootstrap

        * sample_size_pct: double (default = 0.63)
            Percentage of original samples used in each tree building

        * alpha: float
            For "maxstat" splitrule: Significance threshold to allow splitting.

        * minprop: float
            For "maxstat" splitrule: Lower quantile of covariate 
            distribution to be considered for splitting.

        * num_random_splits: int (default=100)
            For "extratrees" splitrule, it is the Number of random splits 
            to consider for each candidate splitting variable.

        * importance_mode:  (default=impurity_corrected)
            Variable importance mode. Here are the 2 options:
            - `impurity` or `impurity_corrected`: 
                it's the unbiased heterogeneity reduction developed 
                by Sandri & Zuccolotto (2008)
            - "permutation" it's unnormalized as recommended by Nicodemus et al.
            - "normalized_permutation" it's normalized version of the 
                permutation importance computations by Breiman et al.

        * `seed`: int (default=None) -- 
            seed used by the random number generator. If None, the current 
            timestamp converted in UNIX is used.

        * save_memory:  bool (default=False) --
            Use memory saving splitting mode. This will slow down the model 
            training. So, only set to `True` if you encounter memory problems.


        Example:
        --------

        #### 1 - Importing packages
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        from sklearn.model_selection import train_test_split
        from pysurvival.models.simulations import SimulationModel
        from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
        from pysurvival.utils.metrics import concordance_index
        from pysurvival.utils.display import integrated_brier_score
        #%matplotlib inline # To use with Jupyter notebooks

        #### 2 - Generating the dataset from a Exponential parametric model
        # Initializing the simulation model
        sim = SimulationModel( survival_distribution = 'exponential',  
                               risk_type = 'linear',
                               censored_parameter = 1, 
                               alpha = 3)

        # Generating N random samples 
        N = 1000
        dataset = sim.generate_data(num_samples = N, num_features=4)

        # Showing a few data-points 
        dataset.head(2)

        #### 3 - Creating the modeling dataset
        # Defining the features
        features = sim.features

        # Building training and testing sets #
        index_train, index_test = train_test_split( range(N), test_size = 0.2)
        data_train = dataset.loc[index_train].reset_index( drop = True )
        data_test  = dataset.loc[index_test].reset_index( drop = True )

        # Creating the X, T and E input
        X_train, X_test = data_train[features], data_test[features]
        T_train, T_test = data_train['time'].values, data_test['time'].values
        E_train, E_test = data_train['event'].values, data_test['event'].values


        #### 4 - Creating an instance of the model and fitting the data.
        # Building the model
        csf = ConditionalSurvivalForestModel(num_trees=200) 
        csf.fit(X_train, T_train, E_train, 
                max_features="sqrt", max_depth=5, min_node_size=20,
                alpha = 0.05, minprop=0.1)


        #### 5 - Cross Validation / Model Performances
        c_index = concordance_index(l_mtlr, X_test, T_test, E_test) #0.81
        print('C-index: {:.2f}'.format(c_index))

        ibs = integrated_brier_score(l_mtlr, X_test, T_test, E_test, t_max=30, 
                    figure_size=(20, 6.5) )
        """

        # Collecting features names
        N, self.num_variables = X.shape
        if isinstance(X, pd.DataFrame):
            features = X.columns.tolist()
        else:
            features = ['x_{}'.format(i) for i in range(self.num_variables)]
        all_data_features = ["time", "event"] + features

        # Transforming the strings into bytes
        all_data_features = utils.as_bytes(all_data_features,
                                           python_version=PYTHON_VERSION)

        # Checking the format of the data 
        X, T, E = utils.check_data(X, T, E)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            T = T.reshape(1, -1)
            E = E.reshape(1, -1)
        input_data = np.c_[T, E, X]

        # Number of trees
        num_trees = self.num_trees

        # Seed
        if seed is None:
            seed = 0

        # sample_size_pct
        if not isinstance(sample_size_pct, float):
            error = "Error: Invalid value for sample_size_pct, "
            error += "please provide a value that is > 0 and <= 1."
            raise ValueError(error)

        if (sample_size_pct <= 0 or sample_size_pct > 1):
            error = "Error: Invalid value for sample_size_pct, "
            error += "please provide a value that is > 0 and <= 1."
            raise ValueError(error)

            # Split Rule
        if self.splitrule.lower() == 'logrank':
            split_mode = 1
            alpha = 0
            minprop = 0
            num_random_splits = 1

        elif self.splitrule.lower() == "maxstat":
            split_mode = 4
            num_random_splits = 1

            # Maxstat splitting
            if not isinstance(alpha, float):
                error = "Error: Invalid value for alpha, "
                error += "please provide a value that is > 0 and < 1."
                raise ValueError(error)

            if (alpha <= 0 or alpha >= 1):
                error = "Error: Invalid value for alpha, "
                error += "please provide a value between 0 and 1."
                raise ValueError(error)

            if not isinstance(minprop, float):
                error = "Error: Invalid value for minprop, "
                error += "please provide a value between 0 and 0.5"
                raise ValueError(error)

            if (minprop < 0 or minprop > 0.5):
                error = "Error: Invalid value for minprop, "
                error += "please provide a value between 0 and 0.5"
                raise ValueError(error)

        elif self.splitrule.lower() == 'extratrees':
            split_mode = 5
            alpha = 0
            minprop = 0

        # Number of variables to possibly split at in each node
        self.max_features = max_features
        if isinstance(self.max_features, str):

            if self.max_features.lower() == 'sqrt':
                num_variables_to_use = int(np.sqrt(self.num_variables))

            elif 'log' in self.max_features.lower():
                num_variables_to_use = int(np.log(self.num_variables))

            elif self.max_features.lower() == 'all':
                num_variables_to_use = self.num_variables

            else:
                raise ValueError("Unknown max features option")

        elif isinstance(self.max_features, float) or \
                isinstance(self.max_features, int):

            if 0 < self.max_features < 1:
                num_variables_to_use = int(self.num_variables * self.max_features)

            elif self.max_features >= 1:
                num_variables_to_use = min(self.num_variables, self.max_features)
                if self.max_features > self.num_variables:
                    msg = "max features value is greater than the number of "
                    msg += "variables ({num_variables}) of the input X. "
                    msg += "So it was set to {num_variables}."
                    msg = msg.format(num_variables=self.num_variables)
                    warnings.warn(msg, UserWarning)

            elif self.max_features <= 0:
                raise ValueError("max features is a positive value")

        else:
            raise ValueError("Unknown max features option")

        # Defining importance mode
        if 'permutation' in importance_mode.lower():

            if 'scaled' in importance_mode.lower() or \
                    'normalized' in importance_mode.lower():
                importance_mode = 2
            else:
                importance_mode = 3

        elif 'impurity' in importance_mode.lower():
            importance_mode = 5

        else:
            error = "{} is not a valid importance mode".format(importance_mode)
            raise ValueError(error)

        # Weights
        if weights is None:
            case_weights = [1. / N] * N
        else:
            case_weights = utils.check_data(weights)

            if abs(sum(case_weights) - 1.) >= 1e-4:
                raise Exception("The sum of the weights needs to be equal to 1.")

            if len(case_weights) != N:
                raise Exception("weights length needs to be {} ".format(N))

        # Fitting the model using the C++ object
        verbose = True
        self.model.fit(input_data, all_data_features, case_weights,
                       num_trees, num_variables_to_use, min_node_size, max_depth,
                       alpha, minprop, num_random_splits, sample_size_pct,
                       importance_mode, split_mode, verbose, seed, num_threads,
                       save_memory)

        # Saving the attributes
        self.save_properties()
        self.get_time_buckets()

        # Extracting the Variable Importance
        self.variable_importance = {}
        for i, value in enumerate(self.variable_importance_):
            self.variable_importance[features[i]] = value

        # Saving the importance in a dataframe
        self.variable_importance_table = pd.DataFrame(
            data={'feature': list(self.variable_importance.keys()),
                  'importance': list(self.variable_importance.values())
                  },
            columns=['feature', 'importance']). \
            sort_values('importance', ascending=0).reset_index(drop=True)
        importance = self.variable_importance_table['importance'].values
        importance = np.maximum(importance, 0.)
        sum_imp = sum(importance) * 1.
        self.variable_importance_table['pct_importance'] = importance / sum_imp

        return self

    def _predict(self, x, t=None, **kwargs):
        num_threads = kwargs.pop("num_threads", -1)

        # Checking if the data has the right format
        X = utils.check_data(x)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        T = np.array([1.] * X.shape[0])
        E = np.array([1.] * X.shape[0])
        input_data = np.c_[T, E, X]

        # Loading the attributes of the model
        self.load_properties()

        # Computing Survival
        survival = np.array(self.model.predict_survival(input_data, num_threads))

        # Computing hazard
        hazard = np.array(self.model.predict_hazard(input_data, num_threads))

        # Computing density
        density = hazard * survival

        if t is None:
            return hazard, density, survival
        else:
            min_index = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_index)
            return hazard[:, index], density[:, index], survival[:, index]

    def predict_risk(self, x, **kwargs):
        num_threads = kwargs.pop("num_threads", -1)

        # Checking if the data has the right format
        X = utils.check_data(x)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        T = np.array([1.] * X.shape[0])
        E = np.array([1.] * X.shape[0])
        input_data = np.c_[T, E, X]

        # Loading the attributes of the model
        self.load_properties()

        # Computing risk
        risk = self.model.predict_risk(input_data, num_threads)
        return np.array(risk)

    def save_properties(self):
        """ Loading the properties of the model """

        self.times = self.model.unique_timepoints
        self.num_trees = self.model.num_trees
        self.chf = self.model.chf
        self.is_ordered = self.model.is_ordered
        self.split_varIDs = self.model.split_varIDs
        self.split_values = self.model.split_values
        self.child_nodeIDs = self.model.child_nodeIDs
        self.status_varID = self.model.status_varID
        self.overall_prediction_error = self.model.overall_prediction_error
        self.dependent_varID = self.model.dependent_varID
        self.min_node_size = self.model.min_node_size
        self.variable_importance_ = self.model.variable_importance
        self.mtry = self.model.mtry
        self.num_independent_variables = self.model.num_independent_variables
        self.variable_names = self.model.variable_names

    def load_properties(self):
        """ Loading the properties of the model """

        self.model.unique_timepoints = self.times
        self.model.num_trees = self.num_trees
        self.model.chf = self.chf
        self.model.is_ordered = self.is_ordered
        self.model.split_varIDs = self.split_varIDs
        self.model.split_values = self.split_values
        self.model.child_nodeIDs = self.child_nodeIDs
        self.model.status_varID = self.status_varID
        self.model.overall_prediction_error = self.overall_prediction_error
        self.model.dependent_varID = self.dependent_varID
        self.model.min_node_size = self.min_node_size
        self.model.variable_importance = self.variable_importance_
        self.model.mtry = self.mtry
        self.model.num_independent_variables = self.num_independent_variables
        self.model.variable_names = self.variable_names


class RandomSurvivalForestModel(BaseSurvivalForest):

    def __init__(self, num_trees=10):
        super(RandomSurvivalForestModel, self).__init__("logrank", num_trees)

    def fit(self, X, T, E, max_features='sqrt', max_depth=5,
            min_node_size=10, num_threads=-1, weights=None,
            sample_size_pct=0.63, alpha=0.5, minprop=0.1,
            num_random_splits=100, importance_mode='normalized_permutation',
            seed=None, save_memory=False):
        return super(RandomSurvivalForestModel, self).fit(X=X, T=T, E=E,
                                                          max_features=max_features, max_depth=max_depth,
                                                          weights=weights,
                                                          min_node_size=min_node_size, num_threads=num_threads,
                                                          sample_size_pct=sample_size_pct, seed=seed,
                                                          save_memory=save_memory, importance_mode=importance_mode)


class ExtraSurvivalTreesModel(BaseSurvivalForest):

    def __init__(self, num_trees=10):
        super(ExtraSurvivalTreesModel, self).__init__("extratrees", num_trees)

    def fit(self, X, T, E, max_features='sqrt', max_depth=5,
            min_node_size=10, num_threads=-1, weights=None,
            sample_size_pct=0.63, alpha=0.5, minprop=0.1,
            num_random_splits=100, importance_mode='normalized_permutation',
            seed=None, save_memory=False):
        return super(ExtraSurvivalTreesModel, self).fit(X=X, T=T, E=E,
                                                        max_features=max_features, max_depth=max_depth, weights=weights,
                                                        min_node_size=min_node_size, num_threads=num_threads,
                                                        sample_size_pct=sample_size_pct, seed=seed,
                                                        num_random_splits=num_random_splits, save_memory=save_memory,
                                                        importance_mode=importance_mode)


class ConditionalSurvivalForestModel(BaseSurvivalForest):

    def __init__(self, num_trees=10):
        super(ConditionalSurvivalForestModel, self).__init__("maxstat", num_trees)

    def fit(self, X, T, E, max_features='sqrt', max_depth=5,
            min_node_size=10, num_threads=-1, weights=None,
            sample_size_pct=0.63, alpha=0.5, minprop=0.1,
            num_random_splits=100, importance_mode='normalized_permutation',
            seed=None, save_memory=False):
        return super(ConditionalSurvivalForestModel, self).fit(X=X, T=T, E=E,
                                                               max_features=max_features, max_depth=max_depth,
                                                               min_node_size=min_node_size, num_threads=num_threads,
                                                               weights=weights, sample_size_pct=sample_size_pct,
                                                               alpha=alpha, minprop=minprop,
                                                               importance_mode=importance_mode,
                                                               seed=seed, save_memory=save_memory)
