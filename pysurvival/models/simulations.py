from __future__ import absolute_import
import numpy as np
import pandas as pd
import random
import scipy
import copy
from pysurvival import utils
from pysurvival.models import BaseModel

# %matplotlib inline

# List of Survival Distributions
DISTRIBUTIONS = ['Exponential',
                 'Weibull',
                 'Gompertz',
                 'Log-Logistic',
                 'Log-Normal', ]

# List of risk types
RISK_TYPES = ['Linear', 'Square', 'Gaussian']


class SimulationModel(BaseModel):
    """ 
    A general framework for simulating right-censored survival data 
    for proportional hazards models by incorporating 
        * a baseline hazard function from a known survival distribution, 
        * a set of covariates. 
    
    The framework is based on "Generating Survival Times to Simulate 
    Cox Proportional Hazards Models"
    https://www.ncbi.nlm.nih.gov/pubmed/22763916

    The formula for the different survival times and functions, and hazard
    functions can be found at :
    http://data.princeton.edu/pop509/ParametricSurvival.pdf

    Parameters:
    -----------

    * survival_distribution: string (default = 'exponential')
        Defines a known survival distribution. The available options are:
            - Exponential
            - Weibull
            - Gompertz
            - Log-Logistic
            - Log-Normal
        
    * risk_type: string (default='linear')
        Defines the type of risk function. The available options are:
            - Linear
            - Square
            - Gaussian
        
    * alpha: double (default = 1.) 
         the scale parameter

    * beta: double (default = 1.)
         the shape parameter
         
    * bins: int (default=100)
        the number of bins of the time axis

    * censored_parameter: double (default = 1.)
         coefficient used to calculate the censored distribution. This
         distribution is a normal such that N(loc=censored_parameter, scale=5)
         
    * risk_parameter: double (default = 1.)
        Scaling coefficient of the risk score such that:
            - linear: r(x) = exp(<x, W>) 
            - square: r(x) = exp(risk_parameter*(<x, W>)^2)
            - gaussian: r(x) = exp( exp(-(<x, W>)^2/risk_parameter) )  
        <.,.> is the dot product                  
    """

    def __init__(self, survival_distribution='exponential',
                 risk_type='linear', censored_parameter=1., alpha=1, beta=1.,
                 bins=100, risk_parameter=1.):

        # Saving the attributes
        self.censored_parameter = censored_parameter
        self.alpha = alpha
        self.beta = beta
        self.risk_parameter = risk_parameter
        self.bins = bins
        self.features = []

        # Checking risk_type
        if any([risk_type.lower() == r.lower() for r in RISK_TYPES]):
            self.risk_type = risk_type
        else:
            error = "{} isn't a valid risk type. "
            error += "Only {} are currently available."
            error = error.format(risk_type, ", ".join(RISK_TYPES))
            raise NotImplementedError(error)

        # Checking distribution
        if any([survival_distribution.lower() == d.lower() \
                for d in DISTRIBUTIONS]):
            self.survival_distribution = survival_distribution
        else:
            error = "{} isn't a valid survival distribution. "
            error += "Only {} are currently available."
            error = error.format(survival_distribution, ", ".join(DISTRIBUTIONS))
            raise NotImplementedError(error)

        # Initializing the elements from BaseModel
        super(SimulationModel, self).__init__(auto_scaler=True)

    @staticmethod
    def random_data(N):
        """ 
        Generating a array of size N from a random distribution -- the available 
        distributions are:
            * binomial,
            * chisquare,
            * exponential, 
            * gamma, 
            * normal, 
            * uniform 
            * laplace 
        """

        index = np.random.binomial(n=4, p=0.5)
        distributions = {
            'binomial_a': np.random.binomial(n=20, p=0.6, size=N),
            'binomial_b': np.random.binomial(n=200, p=0.6, size=N),
            'chisquare': np.random.chisquare(df=10, size=N),
            'exponential_a': np.random.exponential(scale=0.1, size=N),
            'exponential_b': np.random.exponential(scale=0.01, size=N),
            'gamma': np.random.gamma(shape=2., scale=2., size=N),
            'normal_a': np.random.normal(loc=-1.0, scale=5.0, size=N),
            'normal_b': np.random.normal(loc=10.0, scale=10.0, size=N),
            'uniform_a': np.random.uniform(low=-2.0, high=10.0, size=N),
            'uniform_b': np.random.uniform(low=-20.0, high=100.0, size=N),
            'laplace': np.random.laplace(loc=0.0, scale=1.0, size=N)
        }

        list_distributions = copy.deepcopy(list(distributions.keys()))
        random.shuffle(list_distributions)
        key = list_distributions[index]
        return key, distributions[key]

    def time_function(self, BX):
        """ 
        Calculating the survival times based on the given distribution
        T = H^(-1)( -log(U)/risk_score ), where:
            * H is the cumulative baseline hazard function 
                (H^(-1) is the inverse function)
            * U is a random variable uniform - Uni[0,1].

        The method is inspired by https://gist.github.com/jcrudy/10481743
        """

        # Calculating scale coefficient using the features
        num_samples = BX.shape[0]
        lambda_exp_BX = np.exp(BX) * self.alpha
        lambda_exp_BX = lambda_exp_BX.flatten()

        # Generating random uniform variables
        U = np.random.uniform(0, 1, num_samples)

        # Exponential 
        if self.survival_distribution.lower().startswith('exp'):
            self.survival_distribution = 'Exponential'
            return - np.log(U) / (lambda_exp_BX)

        # Weibull 
        elif self.survival_distribution.lower().startswith('wei'):
            self.survival_distribution = 'Weibull'
            return np.power(- np.log(U) / (lambda_exp_BX), 1. / self.beta)

        # Gompertz 
        elif self.survival_distribution.lower().startswith('gom'):
            self.survival_distribution = 'Gompertz'
            return (1. / self.beta) * \
                   np.log(1 - self.beta * np.log(U) / (lambda_exp_BX))

        # Log-Logistic 
        elif 'logistic' in self.survival_distribution.lower():
            self.survival_distribution = 'Log-Logistic'
            return np.power(U / (1. - U), 1. / self.beta) / (lambda_exp_BX)

        # Log-Normal
        elif 'normal' in self.survival_distribution.lower():
            self.survival_distribution = 'Log-Normal'
            W = np.random.normal(0, 1, num_samples)
            return lambda_exp_BX * np.exp(self.beta * W)

    def hazard_function(self, t, BX):
        """ Calculating the hazard function based on the given distribution """

        # Calculating scale coefficient using the features
        _lambda = self.alpha * np.exp(BX)

        # Exponential 
        if self.survival_distribution.lower().startswith('exp'):
            return np.repeat(_lambda, len(t))

            # Weibull
        elif self.survival_distribution.lower().startswith('wei'):
            return _lambda * self.beta * np.power(t, self.beta - 1)

            # Gompertz
        elif self.survival_distribution.lower().startswith('gom'):
            return _lambda * np.exp(self.beta * t)

        # Log-Logistic 
        elif self.survival_distribution.lower().endswith('logistic'):
            numerator = _lambda * self.beta * np.power((_lambda * t), self.beta - 1)
            denominator = (1 + np.power((_lambda * t), self.beta))
            return numerator / denominator

        # Log-Normal
        elif self.survival_distribution.lower().endswith('normal'):
            arg_normal = (np.log(t) - np.log(_lambda)) / self.beta
            numerator = (1. / (t * self.beta)) * scipy.stats.norm.pdf(arg_normal)
            denominator = 1. - scipy.stats.norm.cdf(arg_normal)
            return numerator / denominator

    def survival_function(self, t, BX):
        """ 
        Calculating the survival function based on the given 
        distribution 
        """

        # Calculating scale coefficient using the features
        _lambda = self.alpha * np.exp(BX)

        # Exponential 
        if self.survival_distribution.lower().startswith('exp'):
            return np.exp(-t * _lambda)

        # Weibull 
        elif self.survival_distribution.lower().startswith('wei'):
            return np.exp(-np.power(t, self.beta) * _lambda)

        # Gompertz 
        elif self.survival_distribution.lower().startswith('gom'):
            return np.exp(-_lambda / self.beta * (np.exp(self.beta * t) - 1))

        # Log-Logistic 
        elif self.survival_distribution.lower().endswith('logistic'):
            return 1. / (1. + np.power(_lambda * t, self.beta))

        # Log-Normal
        elif self.survival_distribution.lower().endswith('normal'):
            arg_cdf = (np.log(t) - np.log(_lambda)) / self.beta
            return 1. - scipy.stats.norm.cdf(arg_cdf)

    def risk_function(self, x_std):
        """ Calculating the risk function based on the given risk type """

        # Dot product
        risk = np.dot(x_std, self.feature_weights)

        # Choosing the type of risk
        if self.risk_type.lower() == 'linear':
            return risk.reshape(-1, 1)

        elif self.risk_type.lower() == 'square':
            risk = np.square(risk * self.risk_parameter)


        elif self.risk_type.lower() == 'gaussian':
            risk = np.square(risk)
            risk = np.exp(- risk * self.risk_parameter)

        return risk.reshape(-1, 1)

    def generate_data(self, num_samples=100, num_features=3,
                      feature_weights=None):
        """ 
        Generating a dataset of simulated survival times from a given 
        distribution through the hazard function using the Cox model  
        
        Parameters:
        -----------
        * `num_samples`: **int** *(default=100)* --
            Number of samples to generate

        * `num_features`: **int** *(default=3)* --
            Number of features to generate

        * `feature_weights`: **array-like** *(default=None)* -- 
            list of the coefficients of the underlying Cox-Model. 
            The features linked to each coefficient are generated 
            from random distribution from the following list:

            * binomial
            * chisquare
            * exponential
            * gamma
            * normal
            * uniform
            * laplace

            If None then feature_weights = [1.]*num_features

        Returns:
        --------
        * dataset: pandas.DataFrame
            dataset of simulated survival times, event status and features


        Example:
        --------
        from pysurvival.models.simulations import SimulationModel

        # Initializing the simulation model
        sim = SimulationModel( survival_distribution = 'gompertz',  
                               risk_type = 'linear',
                               censored_parameter = 5.0, 
                               alpha = 0.01, 
                               beta = 5., )

        # Generating N Random samples
        N = 1000
        dataset = sim.generate_data(num_samples = N, num_features=5)

        # Showing a few data-points
        dataset.head()
        """

        # Data parameters
        self.num_variables = num_features
        if feature_weights is None:
            self.feature_weights = [1.] * self.num_variables
            feature_weights = self.feature_weights

        else:
            feature_weights = utils.check_data(feature_weights)
            if num_features != len(feature_weights):
                error = "The length of feature_weights ({}) "
                error += "and num_features ({}) are not the same."
                error = error.format(len(feature_weights), num_features)
                raise ValueError(error)
            self.feature_weights = feature_weights

        # Generating random features
        # Creating the features
        X = np.zeros((num_samples, self.num_variables))
        columns = []
        for i in range(self.num_variables):
            key, X[:, i] = self.random_data(num_samples)
            columns.append('x_' + str(i + 1))
        X_std = self.scaler.fit_transform(X)
        BX = self.risk_function(X_std)

        # Building the survival times
        T = self.time_function(BX)
        C = np.random.normal(loc=self.censored_parameter,
                             scale=5, size=num_samples)
        C = np.maximum(C, 0.)
        time = np.minimum(T, C)
        E = 1. * (T == time)

        # Building dataset
        self.features = columns
        self.dataset = pd.DataFrame(data=np.c_[X, time, E],
                                    columns=columns + ['time', 'event'])

        # Building the time axis and time buckets
        self.times = np.linspace(0., max(self.dataset['time']), self.bins)
        self.get_time_buckets()

        # Building baseline functions
        self.baseline_hazard = self.hazard_function(self.times, 0)
        self.baseline_survival = self.survival_function(self.times, 0)

        # Printing summary message
        message_to_print = "Number of data-points: {} - Number of events: {}"
        print(message_to_print.format(num_samples, sum(E)))

        return self.dataset

    def _predict(self, x, t=None, **kwargs):
        """ 
        Predicting the hazard, density and survival functions
        
        Parameters:
        -----------
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
                x should not be standardized before, the model
                will take care of it
            * t: float (default=None)
                Time at which hazard, density and survival functions
                should be calculated. If None, the method returns 
                the functions for all times t. 
        """

        # Convert x into the right format
        x = utils.check_data(x)

        # Scaling the dataset
        if x.ndim == 1:
            x = self.scaler.transform(x.reshape(1, -1))

        elif x.ndim == 2:
            x = self.scaler.transform(x)

        else:
            # Ensuring x has 2 dimensions
            if x.ndim == 1:
                x = np.reshape(x, (1, -1))

        # Calculating risk_score, hazard, density and survival 
        BX = self.risk_function(x)
        hazard = self.hazard_function(self.times, BX.reshape(-1, 1))
        survival = self.survival_function(self.times, BX.reshape(-1, 1))
        density = (hazard * survival)

        if t is None:
            return hazard, density, survival
        else:
            min_abs_value = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_abs_value)
            return hazard[:, index], density[:, index], survival[:, index]

    def predict_risk(self, x, **kwargs):
        """
        Predicting the risk score function
        
        Parameters:
        -----------
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
                x should not be standardized before, the model
                will take care of it
        """

        # Convert x into the right format
        x = utils.check_data(x)

        # Scaling the dataset
        if x.ndim == 1:
            x = self.scaler.transform(x.reshape(1, -1))

        elif x.ndim == 2:
            x = self.scaler.transform(x)

        else:
            # Ensuring x has 2 dimensions
            if x.ndim == 1:
                x = np.reshape(x, (1, -1))

        # Calculating risk_score
        risk_score = self.risk_function(x)

        return risk_score
