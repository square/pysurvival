from __future__ import absolute_import
import numpy as np
import scipy
import torch
from pysurvival.models import BaseModel
from pysurvival import utils, HAS_GPU
from pysurvival.utils import optimization as opt
from pysurvival.utils import neural_networks as nn


class BaseParametricModel(BaseModel):
    """ Base class for all the Parametric estimators:
        ---------------------------------------------

        Parametric models are special cases of the Cox proportional hazard
        model where is is assumed that the baseline hazard has a specific 
        parametric form.
        
        The BaseParametricModel object provides the necessary framework to use
        the properties of parametric models. It should not be used on its own.

        Parameters
        ----------
        * bins: int (default=100)
             Number of subdivisions of the time axis 

        * auto_scaler: boolean (default=True)
            Determines whether a sklearn scaler should be automatically applied
    """

    def __init__(self, bins=100, auto_scaler=True):

        # Saving the attributes
        self.loss_values = []
        self.bins = bins

        # Initializing the elements from BaseModel
        super(BaseParametricModel, self).__init__(auto_scaler)

    def get_hazard_survival(self, model, x, t):
        raise NotImplementedError()

    def get_times(self, T, is_min_time_zero=True, extra_pct_time=0.1):
        """ Building the time axis (self.times) as well as the time intervals 
            ( all the [ t(k-1), t(k) ) in the time axis.
        """

        # Setting the min_time and max_time
        max_time = max(T)
        if is_min_time_zero:
            min_time = 0.
        else:
            min_time = min(T)

        # Setting optional extra percentage time
        if 0. <= extra_pct_time <= 1.:
            p = extra_pct_time
        else:
            raise Exception("extra_pct_time has to be between [0, 1].")

            # Building time points and time buckets
        self.times = np.linspace(min_time, max_time * (1. + p), self.bins)
        self.get_time_buckets()
        self.nb_times = len(self.time_buckets)

    def loss_function(self, model, X, T, E, l2_reg):
        """ Computes the loss function of any Parametric models. 
            All the operations have been vectorized to ensure optimal speed
        """

        # Adapting the optimization for the use of GPU
        # if HAS_GPU:
        #     X = X.cuda(async=True)
        #     T = T.cuda(async=True)
        #     E = E.cuda(async=True)
        #     model = model.cuda()   

        # Hazard & Survival calculations
        hazard, Survival = self.get_hazard_survival(model, X, T)

        # Loss function calculations
        hazard = torch.max(hazard, torch.FloatTensor([1e-6]))
        Survival = torch.max(Survival, torch.FloatTensor([1e-6]))
        loss = - torch.sum(E * torch.log(hazard) + torch.log(Survival))

        # Adding the regularized loss
        for w in model.parameters():
            loss += l2_reg * torch.sum(w * w) / 2.

        return loss

    def fit(self, X, T, E, init_method='glorot_uniform', optimizer='adam',
            lr=1e-4, num_epochs=1000, l2_reg=1e-2, verbose=True,
            is_min_time_zero=True, extra_pct_time=0.1):
        """ 
        Fit the estimator based on the given parameters.

        Parameters:
        -----------
        * `X` : **array-like**, *shape=(n_samples, n_features)* --
            The input samples.

        * `T` : **array-like** -- 
            The target values describing when the event of interest or censoring
            occurred.

        * `E` : **array-like** --
            The values that indicate if the event of interest occurred i.e.: 
            E[i]=1 corresponds to an event, and E[i] = 0 means censoring, 
            for all i.

        * `init_method` : **str** *(default = 'glorot_uniform')* -- 
            Initialization method to use. Here are the possible options:

            * `glorot_uniform`:  Glorot/Xavier uniform initializer 
            * `he_uniform`:  He uniform variance scaling initializer 
            * `uniform`: Initializing tensors with uniform (-1, 1) distribution
            * `glorot_normal`: Glorot normal initializer,
            * `he_normal`: He normal initializer.
            * `normal`: Initializing tensors with standard normal distribution
            * `ones`: Initializing tensors to 1
            * `zeros`: Initializing tensors to 0
            * `orthogonal`: Initializing tensors with a orthogonal matrix,

        * `optimizer`:  **str** *(default = 'adam')* -- 
            iterative method for optimizing a differentiable objective function.
            Here are the possible options:

            - `adadelta`
            - `adagrad`
            - `adam`
            - `adamax`
            - `rmsprop`
            - `sparseadam`
            - `sgd`

        * `lr`: **float** *(default=1e-4)* -- 
            learning rate used in the optimization

        * `num_epochs`: **int** *(default=1000)* -- 
            The number of iterations in the optimization

        * `l2_reg`: **float** *(default=1e-4)* -- 
            L2 regularization parameter for the model coefficients

        * `verbose`: **bool** *(default=True)* -- 
            Whether or not producing detailed logging about the modeling

        * `extra_pct_time`: **float** *(default=0.1)* -- 
            Providing an extra fraction of time in the time axis

        * `is_min_time_zero`: **bool** *(default=True)* -- 
            Whether the the time axis starts at 0

        Returns:
        --------
        * self : object


        Example:
        --------

        #### 1 - Importing packages
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        from sklearn.model_selection import train_test_split
        from pysurvival.models.simulations import SimulationModel
        from pysurvival.models.parametric import GompertzModel
        from pysurvival.utils.metrics import concordance_index
        from pysurvival.utils.display import integrated_brier_score
        #%matplotlib inline  # To use with Jupyter notebooks

        #### 2 - Generating the dataset from a Gompertz parametric model
        # Initializing the simulation model
        sim = SimulationModel( survival_distribution = 'Gompertz',  
                               risk_type = 'linear',
                               censored_parameter = 10.0, 
                               alpha = .01, beta = 3.0 )

        # Generating N random samples 
        N = 1000
        dataset = sim.generate_data(num_samples = N, num_features = 3)

        # Showing a few data-points 
        time_column = 'time'
        event_column = 'event'
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

        #### 4 - Creating an instance of the Gompertz model and fitting the data
        # Building the model
        gomp_model = GompertzModel() 
        gomp_model.fit(X_train, T_train, E_train, lr=1e-2, init_method='zeros',
            optimizer ='adam', l2_reg = 1e-3, num_epochs=2000)

        #### 5 - Cross Validation / Model Performances
        c_index = concordance_index(gomp_model, X_test, T_test, E_test) #0.8
        print('C-index: {:.2f}'.format(c_index))

        ibs = integrated_brier_score(gomp_model, X_test, T_test, E_test, 
            t_max=30, figure_size=(20, 6.5) )

        """

        # Checking data format (i.e.: transforming into numpy array)
        X, T, E = utils.check_data(X, T, E)
        T = np.maximum(T, 1e-6)
        self.get_times(T, is_min_time_zero, extra_pct_time)

        # Extracting data parameters
        nb_units, self.num_vars = X.shape
        input_shape = self.num_vars

        # Scaling data 
        if self.auto_scaler:
            X = self.scaler.fit_transform(X)

            # Does the model need a parameter called Beta
        is_beta_used = True
        init_alpha = 1.
        if self.name == 'ExponentialModel':
            is_beta_used = False
        if self.name == 'GompertzModel':
            init_alpha = 1000.

        # Initializing the model
        model = nn.ParametricNet(input_shape, init_method, init_alpha,
                                 is_beta_used)

        # Trasnforming the inputs into tensors
        X = torch.FloatTensor(X)
        T = torch.FloatTensor(T.reshape(-1, 1))
        E = torch.FloatTensor(E.reshape(-1, 1))

        # Performing order 1 optimization
        model, loss_values = opt.optimize(self.loss_function, model, optimizer,
                                          lr, num_epochs, verbose, X=X, T=T, E=E, l2_reg=l2_reg)

        # Saving attributes
        self.model = model.eval()
        self.loss_values = loss_values

        # Calculating the AIC
        self.aic = 2 * self.loss_values[-1]
        self.aic -= 2 * (self.num_vars + 1 + is_beta_used * 1. - 1)

        return self

    def _predict(self, x, t=None, **kwargs):
        """ 
        Predicting the hazard, density and survival functions
        
        Arguments:
        ----------
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
                x should not be standardized before, the model
                will take care of it
            * t: float (default=None)
                Time at which hazard, density and survival functions
                should be calculated. If None, the method returns 
                the functions for all times t. 
        """

        # Scaling the data
        if self.auto_scaler:
            if x.ndim == 1:
                x = self.scaler.transform(x.reshape(1, -1))
            elif x.ndim == 2:
                x = self.scaler.transform(x)
        else:
            # Ensuring x has 2 dimensions
            if x.ndim == 1:
                x = np.reshape(x, (1, -1))

        # Transforming into pytorch objects
        x = torch.FloatTensor(x)
        times = torch.FloatTensor(self.times.flatten())

        # Calculating hazard, density, Survival
        hazard, Survival = self.get_hazard_survival(self.model, x, times)
        density = hazard * Survival

        # Transforming into numpy objects
        hazard = hazard.data.numpy()
        density = density.data.numpy()
        Survival = Survival.data.numpy()

        # Returning the full functions of just one time point
        if t is None:
            return hazard, density, Survival
        else:
            min_abs_value = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_abs_value)
            return hazard[:, index], density[:, index], Survival[:, index]


class ExponentialModel(BaseParametricModel):
    """ 
    ExponentialModel:
    -----------------
        
    The exponential distribution is the simplest and most
    important distribution in survival studies. Being independent
    of prior information, it is known as a "lack of
    memory" distribution requiring that the present age of the
    living organism does not influence its future survival.
    (Application of Parametric Models to a Survival Analysis of
    Hemodialysis Patients)
    """

    def get_hazard_survival(self, model, x, t):
        """ Computing the hazard and Survival functions. """

        # Computing the score
        score = model(x).reshape(-1, 1)

        # Computing hazard and Survival
        hazard = score
        Survival = torch.exp(-hazard * t)

        return hazard, Survival


class WeibullModel(BaseParametricModel):
    """
    WeibullModel:
    -------------
    
    The Weibull distribution is a generalized form of the exponential
    distribution and is de facto more flexible than the exponential model. 
    It is a two-parameter model (alpha and beta):
        * alpha is the location parameter
        * beta is the shape parameter 
    """

    def get_hazard_survival(self, model, x, t):
        """ Computing the hazard and Survival functions. """

        # Computing the score
        score = model(x).reshape(-1, 1)

        # Extracting beta
        beta = list(model.parameters())[-1]

        # Computing hazard and Survival
        hazard = beta * score * torch.pow(t, beta - 1)
        Survival = torch.exp(- score * torch.pow(t, beta))

        return hazard, Survival


class GompertzModel(BaseParametricModel):
    """
    GompertzModel:
    --------------

    The Gompertz distribution is a continuous probability distribution,  
    that has an exponentially increasing failure rate, and is often 
    applied to analyze survival data.
    
    """

    def get_hazard_survival(self, model, x, t):
        """ Computing the hazard and Survival functions. """

        # Computing the score
        score = model(x).reshape(-1, 1)

        # Extracting beta
        beta = list(model.parameters())[-1]

        # Computing hazard and Survival
        hazard = score * torch.exp(beta * t)
        Survival = torch.exp(-score / beta * (torch.exp(beta * t) - 1))

        return hazard, Survival


class LogLogisticModel(BaseParametricModel):
    """
    LogLogisticModel:
    ----------------
    
    As the name suggests, the log-logistic distribution is the distribution 
    of a variable whose logarithm has the logistic distribution. 
    The log-logistic distribution is often used to model random lifetimes.
    (http://www.randomservices.org/random/special/LogLogistic.html)
    
    """

    def get_hazard_survival(self, model, x, t):
        """ Computing the hazard and Survival functions. """

        # Computing the score
        score = model(x).reshape(-1, 1)

        # Extracting beta
        beta = list(model.parameters())[-1]

        # Computing hazard and Survival
        hazard = score * beta * torch.pow(t, beta - 1)
        hazard = hazard / (1 + score * torch.pow(t, beta))
        Survival = 1. / (1. + torch.pow(score * t, beta))

        return hazard, Survival


class LogNormalModel(BaseParametricModel):
    """
    LogNormalModel:
    ---------------
    
    The lognormal distribution is used to model continuous random quantities 
    when the distribution is believed to be skewed, such as lifetime variables
    (http://www.randomservices.org/random/special/LogNormal.html)
    
    """

    def get_hazard_survival(self, model, x, t):
        """ Computing the hazard and Survival functions. """

        # Computing the score
        score = model(x).reshape(-1, 1)

        # Extracting beta
        beta = list(model.parameters())[-1]

        # Initializing the Normal distribution
        from torch.distributions.normal import Normal
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        # Computing hazard and Survival
        hazard = (torch.log(t) - torch.log(score)) / (np.sqrt(2) * beta)
        Survival = 1. - m.cdf((torch.log(t) - torch.log(score)) / (np.sqrt(2) * beta))
        hazard = hazard * (torch.log(t) - torch.log(score)) / (np.sqrt(2) * beta)
        hazard = torch.exp(-hazard / 2.)
        hazard = hazard / (np.sqrt(2 * np.pi) * Survival * (t * beta))

        hazard = torch.max(hazard, torch.FloatTensor([1e-6]))
        Survival = torch.max(Survival, torch.FloatTensor([1e-6]))
        return hazard, Survival
