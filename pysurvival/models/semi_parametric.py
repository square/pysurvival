from __future__ import absolute_import
import torch
import numpy as np
import pandas as pd
import scipy
import copy
from pysurvival import HAS_GPU
from pysurvival import utils
from pysurvival.utils import neural_networks as nn
from pysurvival.utils import optimization as opt
from pysurvival.models import BaseModel
from pysurvival.models._coxph import _CoxPHModel
from pysurvival.models._coxph import _baseline_functions


class CoxPHModel(BaseModel):
    """ Cox proportional hazards model:
        -------------------------------
        The purpose of the model is to evaluate simultaneously 
        the effect of several factors on survival. 
        In other words, it allows us to examine how specified factors 
        influence the rate of a particular event happening 
        at a particular point in time. 
        
        The Cox model is expressed by the hazard function h(t)
        (the risk of dying at time t. )
        It can be estimated as follow:
            h(t, x)=h_0(t)*exp(<x, W>)

        Then the Survival function can be calculated as follow:
            H(t, x) = cumsum( h(t, x) )
            S(t, x) = exp( -H(t, x) )

        Reference:
            * http://www.sthda.com/english/wiki/cox-proportional-hazards-model
    """

    def get_summary(self, alpha=0.95, precision=3):
        """ Providing the summary of the regression results:
                * standard errors
                * z-score 
                * p-value
        """

        # Flattening the coef 
        W_flat = self.weights.flatten()

        # calculating standard error 
        self.std_err = np.sqrt(self.inv_Hessian.diagonal()) / self.std_scale

        # Confidence Intervals 
        alpha = scipy.stats.norm.ppf((1. + alpha) / 2.)
        lower_ci = np.round(W_flat - alpha * self.std_err, precision)
        upper_ci = np.round(W_flat + alpha * self.std_err, precision)
        z = np.round(W_flat / self.std_err, precision)
        p_values = np.round(scipy.stats.chi2.sf(np.square(z), 1), precision)
        W = np.round(W_flat, precision)
        std_err = np.round(self.std_err, precision)

        # Creating summary
        df = np.c_[self.variables, W, std_err,
                   lower_ci, upper_ci, z, p_values]
        df = pd.DataFrame(data=df,
                          columns=['variables', 'coef', 'std. err',
                                   'lower_ci', 'upper_ci',
                                   'z', 'p_values'])
        self.summary = df

        return df

    def fit(self, X, T, E, init_method='glorot_normal', lr=1e-2,
            max_iter=100, l2_reg=1e-2, alpha=0.95,
            tol=1e-3, verbose=True):
        """
        Fitting a proportional hazards regression model using
        the Efron's approximation method to take into account tied times.
        
        As the Hessian matrix of the log-likelihood can be 
        calculated without too much effort, the model parameters are 
        computed using the Newton_Raphson Optimization scheme:
                W_new = W_old - lr*<Hessian^(-1), gradient>
        
        Arguments:
        ---------
        * `X` : **array-like**, *shape=(n_samples, n_features)* --
            The input samples.

        * `T` : **array-like** -- 
            The target values describing when the event of interest or 
            censoring occurred.

        * `E` : **array-like** --
            The values that indicate if the event of interest occurred 
            i.e.: E[i]=1 corresponds to an event, and E[i] = 0 means censoring, 
            for all i.

        * `init_method` : **str** *(default = 'glorot_uniform')* -- 
            Initialization method to use. Here are the possible options:

            * `glorot_uniform`: Glorot/Xavier uniform initializer
            * `he_uniform`: He uniform variance scaling initializer
            * `uniform`: Initializing tensors with uniform (-1, 1) distribution
            * `glorot_normal`: Glorot normal initializer,
            * `he_normal`: He normal initializer.
            * `normal`: Initializing tensors with standard normal distribution
            * `ones`: Initializing tensors to 1
            * `zeros`: Initializing tensors to 0
            * `orthogonal`: Initializing tensors with a orthogonal matrix,
            
        * `lr`: **float** *(default=1e-4)* -- 
            learning rate used in the optimization

        * `max_iter`: **int** *(default=100)* -- 
            The maximum number of iterations in the Newton optimization

        * `l2_reg`: **float** *(default=1e-4)* -- 
            L2 regularization parameter for the model coefficients

        * `alpha`: **float** *(default=0.95)* -- 
            Confidence interval

        * `tol`: **float** *(default=1e-3)* -- 
            Tolerance for stopping criteria

        * `verbose`: **bool** *(default=True)* -- 
            Whether or not producing detailed logging about the modeling
 
        Example:
        --------

        #### 1 - Importing packages
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        from sklearn.model_selection import train_test_split
        from pysurvival.models.simulations import SimulationModel
        from pysurvival.models.semi_parametric import CoxPHModel
        from pysurvival.utils.metrics import concordance_index
        from pysurvival.utils.display import integrated_brier_score
        #%pylab inline  # To use with Jupyter notebooks


        #### 2 - Generating the dataset from a Log-Logistic parametric model
        # Initializing the simulation model
        sim = SimulationModel( survival_distribution = 'log-logistic',  
                               risk_type = 'linear',
                               censored_parameter = 10.1, 
                               alpha = 0.1, beta=1.2 )

        # Generating N random samples 
        N = 1000
        dataset = sim.generate_data(num_samples = N, num_features = 3)

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


        #### 4 - Creating an instance of the Cox PH model and fitting the data.
        # Building the model
        coxph = CoxPHModel()
        coxph.fit(X_train, T_train, E_train, lr=0.5, l2_reg=1e-2, 
            init_method='zeros')


        #### 5 - Cross Validation / Model Performances
        c_index = concordance_index(coxph, X_test, T_test, E_test) #0.92
        print('C-index: {:.2f}'.format(c_index))

        ibs = integrated_brier_score(coxph, X_test, T_test, E_test, t_max=10, 
                    figure_size=(20, 6.5) )

        References:
        -----------
        * https://en.wikipedia.org/wiki/Proportional_hazards_model#Tied_times
        * Efron, Bradley (1974). "The Efficiency of Cox's Likelihood 
          Function for Censored Data". Journal of the American Statistical 
          Association. 72 (359): 557-565. 
        """

        # Collecting features names
        N, self.num_vars = X.shape
        if isinstance(X, pd.DataFrame):
            self.variables = X.columns.tolist()
        else:
            self.variables = ['x_{}'.format(i) for i in range(self.num_vars)]

        # Checking the format of the data 
        X, T, E = utils.check_data(X, T, E)
        order = np.argsort(-T)
        T = T[order]
        E = E[order]
        X = self.scaler.fit_transform(X[order, :])
        self.std_scale = np.sqrt(self.scaler.var_)

        # Initializing the model
        self.model = _CoxPHModel()

        # Creating the time axis
        self.model.get_times(T, E)

        # Initializing the parameters 
        W = np.zeros(self.num_vars)
        W = opt.initialization(init_method, W, False).flatten()
        W = W.astype(np.float64)

        # Optimizing to find best parameters 
        epsilon = 1e-9
        self.model.newton_optimization(X, T, E, W, lr, l2_reg, tol, epsilon,
                                       max_iter, verbose)

        # Saving the Cython attributes in the Python object
        self.weights = np.array(self.model.W)
        self.loss = self.model.loss
        self.times = np.array(self.model.times)
        self.gradient = np.array(self.model.gradient)
        self.Hessian = np.array(self.model.Hessian)
        self.inv_Hessian = np.array(self.model.inv_Hessian)
        self.loss_values = np.array(self.model.loss_values)
        self.grad2_values = np.array(self.model.grad2_values)

        # Computing baseline functions
        score = np.exp(np.dot(X, self.weights))
        baselines = _baseline_functions(score, T, E)

        # Saving the Cython attributes in the Python object
        self.baseline_hazard = np.array(baselines[1])
        self.baseline_survival = np.array(baselines[2])
        del self.model
        self.get_time_buckets()

        # Calculating summary 
        self.get_summary(alpha)

        return self

    def _predict(self, x, t=None, **kwargs):
        """ 
        Predicting the hazard, density and survival functions
        
        Arguments:
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

        # Sacling the dataset
        if x.ndim == 1:
            x = self.scaler.transform(x.reshape(1, -1))
        elif x.ndim == 2:
            x = self.scaler.transform(x)

        # Calculating risk_score, hazard, density and survival 
        phi = np.exp(np.dot(x, self.weights))
        hazard = self.baseline_hazard * phi.reshape(-1, 1)
        survival = np.power(self.baseline_survival, phi.reshape(-1, 1))
        density = hazard * survival
        if t is None:
            return hazard, density, survival
        else:
            min_index = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_index)
            return hazard[:, index], density[:, index], survival[:, index]

    def predict_risk(self, x, use_log=False):
        """
        Predicting the risk score functions
        
        Arguments:
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

        # Calculating risk_score
        risk_score = np.exp(np.dot(x, self.weights))
        if not use_log:
            risk_score = np.exp(risk_score)

        return risk_score


class NonLinearCoxPHModel(BaseModel):
    """ NonLinear Cox Proportional Hazard model (NeuralCoxPH)
        
        The original Cox Proportional Hazard model, was first introduced 
        by  David R Cox in `Regression models and life-tables`.

        The NonLinear CoxPH model was popularized by Katzman et al.
        in `DeepSurv: Personalized Treatment Recommender System Using
        A Cox Proportional Hazards Deep Neural Network` by allowing the use of 
        Neural Networks within the original design. 
        This current adaptation of the model differs from DeepSurv 
        as it uses the Efron's method to take ties into account.

        Parameters
        ----------
            * structure: None or list of dictionaries
                Provides an MLP structure within the CoxPH
                If None, then the model becomes the Linear CoxPH
                ex: structure = [ {'activation': 'relu', 'num_units': 128}, 
                                  {'activation': 'tanh', 'num_units': 128}, ] 
                Here are the possible activation functions:
                    * Atan
                    * BentIdentity
                    * BipolarSigmoid
                    * CosReLU
                    * ELU
                    * Gaussian
                    * Hardtanh
                    * Identity
                    * InverseSqrt
                    * LeakyReLU
                    * LeCunTanh
                    * LogLog
                    * LogSigmoid
                    * ReLU
                    * SELU
                    * Sigmoid
                    * Sinc
                    * SinReLU
                    * Softmax
                    * Softplus
                    * Softsign
                    * Swish
                    * Tanh

            * auto_scaler: boolean (default=True)
                Determines whether a sklearn scaler should be automatically 
                applied             
    """

    def __init__(self, structure=None, auto_scaler=True):

        # Saving attributes
        self.structure = structure
        self.loss_values = []

        # Initializing the elements from BaseModel
        super(NonLinearCoxPHModel, self).__init__(auto_scaler)

    def risk_fail_matrix(self, T, E):
        """ Calculating the Risk, Fail matrices to calculate the loss 
            function by vectorizing all the quantities at stake
        """

        N = T.shape[0]
        Risk = np.zeros((self.nb_times, N))
        Fail = np.zeros((self.nb_times, N))

        for i in range(N):

            # At risk
            index_risk = np.argwhere(self.times <= T[i]).flatten()
            Risk[index_risk, i] = 1.

            # Failed
            if E[i] == 1:
                index_fail = np.argwhere(self.times == T[i])[0]
                Fail[index_fail, i] = 1.

        self.nb_fail_per_time = np.sum(Fail, axis=1).astype(int)
        return torch.FloatTensor(Risk), torch.FloatTensor(Fail)

    def efron_matrix(self):
        """ Computing the Efron Coefficient matrices to calculate the loss 
            function by vectorizing all the quantities at stake
        """

        max_nb_fails = int(max(self.nb_fail_per_time))
        Efron_coef = np.zeros((self.nb_times, max_nb_fails))
        Efron_one = np.zeros((self.nb_times, max_nb_fails))
        Efron_anti_one = np.ones((self.nb_times, max_nb_fails))

        for i, d in enumerate(self.nb_fail_per_time):
            if d > 0:
                Efron_coef[i, :d] = [h * 1.0 / d for h in range(d)]
                Efron_one[i, :d] = 1.
                Efron_anti_one[i, :d] = 0.

        Efron_coef = torch.FloatTensor(Efron_coef)
        Efron_one = torch.FloatTensor(Efron_one)
        Efron_anti_one = torch.FloatTensor(Efron_anti_one)
        return Efron_coef, Efron_one, Efron_anti_one

    def loss_function(self, model, X, Risk, Fail,
                      Efron_coef, Efron_one, Efron_anti_one, l2_reg):
        """ Efron's approximation loss function by vectorizing 
            all the quantities at stake
        """

        # Calculating the score
        pre_score = model(X)
        score = torch.reshape(torch.exp(pre_score), (-1, 1))
        max_nb_fails = Efron_coef.shape[1]

        # Numerator calculation
        log_score = torch.log(score)
        log_fail = torch.mm(Fail, log_score)
        numerator = torch.sum(log_fail)

        # Denominator calculation
        risk_score = torch.reshape(torch.mm(Risk, score), (-1, 1))
        risk_score = risk_score.repeat(1, max_nb_fails)

        fail_score = torch.reshape(torch.mm(Fail, score), (-1, 1))
        fail_score = fail_score.repeat(1, max_nb_fails)

        Efron_Fail = fail_score * Efron_coef
        Efron_Risk = risk_score * Efron_one
        log_efron = torch.log(Efron_Risk - Efron_Fail + Efron_anti_one)

        denominator = torch.sum(torch.sum(log_efron, dim=1))

        # Adding regularization
        loss = - (numerator - denominator)
        for w in model.parameters():
            loss += l2_reg * torch.sum(w * w) / 2.

        return loss

    def fit(self, X, T, E, init_method='glorot_uniform',
            optimizer='adam', lr=1e-4, num_epochs=1000,
            dropout=0.2, batch_normalization=False, bn_and_dropout=False,
            l2_reg=1e-5, verbose=True):
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

            * `glorot_uniform`: Glorot/Xavier uniform initializer
            * `he_uniform`: He uniform variance scaling initializer 
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

        * `dropout`: **float** *(default=0.5)* -- 
            Randomly sets a fraction rate of input units to 0 
            at each update during training time, which helps prevent overfitting.

        * `l2_reg`: **float** *(default=1e-4)* -- 
            L2 regularization parameter for the model coefficients

        * `batch_normalization`: **bool** *(default=True)* -- 
            Applying Batch Normalization or not

        * `bn_and_dropout`: **bool** *(default=False)* -- 
            Applying Batch Normalization and Dropout at the same time

        * `verbose`: **bool** *(default=True)* -- 
            Whether or not producing detailed logging about the modeling
                

        Example:
        --------

        #### 1 - Importing packages
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        from sklearn.model_selection import train_test_split
        from pysurvival.models.simulations import SimulationModel
        from pysurvival.models.semi_parametric import NonLinearCoxPHModel
        from pysurvival.utils.metrics import concordance_index
        from pysurvival.utils.display import integrated_brier_score
        #%matplotlib inline  # To use with Jupyter notebooks

        #### 2 - Generating the dataset from a nonlinear Weibull parametric model
        # Initializing the simulation model
        sim = SimulationModel( survival_distribution = 'weibull',  
                               risk_type = 'Gaussian',
                               censored_parameter = 2.1, 
                               alpha = 0.1, beta=3.2 )

        # Generating N random samples 
        N = 1000
        dataset = sim.generate_data(num_samples = N, num_features=3)

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


        #### 4 - Creating an instance of the NonLinear CoxPH model and fitting 
        # the data.

        # Defining the MLP structure. Here we will build a 1-hidden layer 
        # with 150 units and `BentIdentity` as its activation function
        structure = [ {'activation': 'BentIdentity', 'num_units': 150},  ]

        # Building the model
        nonlinear_coxph = NonLinearCoxPHModel(structure=structure) 
        nonlinear_coxph.fit(X_train, T_train, E_train, lr=1e-3, 
            init_method='xav_uniform')


        #### 5 - Cross Validation / Model Performances
        c_index = concordance_index(nonlinear_coxph, X_test, T_test, E_test)
        print('C-index: {:.2f}'.format(c_index))

        ibs = integrated_brier_score(nonlinear_coxph, X_test, T_test, E_test, 
            t_max=10, figure_size=(20, 6.5) )

        """

        # Checking data format (i.e.: transforming into numpy array)
        X, T, E = utils.check_data(X, T, E)

        # Extracting data parameters
        N, self.num_vars = X.shape
        input_shape = self.num_vars

        # Scaling data 
        if self.auto_scaler:
            X_original = self.scaler.fit_transform(X)

            # Sorting X, T, E in descending order according to T
        order = np.argsort(-T)
        T = T[order]
        E = E[order]
        X_original = X_original[order, :]
        self.times = np.unique(T[E.astype(bool)])
        self.nb_times = len(self.times)
        self.get_time_buckets()

        # Initializing the model
        model = nn.NeuralNet(input_shape, 1, self.structure,
                             init_method, dropout, batch_normalization,
                             bn_and_dropout)

        # Looping through the data to calculate the loss
        X = torch.FloatTensor(X_original)

        # Computing the Risk and Fail tensors
        Risk, Fail = self.risk_fail_matrix(T, E)
        Risk = torch.FloatTensor(Risk)
        Fail = torch.FloatTensor(Fail)

        # Computing Efron's matrices
        Efron_coef, Efron_one, Efron_anti_one = self.efron_matrix()
        Efron_coef = torch.FloatTensor(Efron_coef)
        Efron_one = torch.FloatTensor(Efron_one)
        Efron_anti_one = torch.FloatTensor(Efron_anti_one)

        # Performing order 1 optimization
        model, loss_values = opt.optimize(self.loss_function, model, optimizer,
                                          lr, num_epochs, verbose, X=X, Risk=Risk, Fail=Fail,
                                          Efron_coef=Efron_coef, Efron_one=Efron_one,
                                          Efron_anti_one=Efron_anti_one, l2_reg=l2_reg)

        # Saving attributes
        self.model = model.eval()
        self.loss_values = loss_values

        # Computing baseline functions
        x = X_original
        x = torch.FloatTensor(x)

        # Calculating risk_score
        score = np.exp(self.model(torch.FloatTensor(x)).data.numpy().flatten())
        baselines = _baseline_functions(score, T, E)

        # Saving the Cython attributes in the Python object
        self.times = np.array(baselines[0])
        self.baseline_hazard = np.array(baselines[1])
        self.baseline_survival = np.array(baselines[2])

        return self

    def _predict(self, x, t=None, **kwargs):
        """ 
        Predicting the hazard, density and survival functions
        
        Arguments:
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

        # Calculating risk_score, hazard, density and survival 
        score = self.model(torch.FloatTensor(x)).data.numpy().flatten()
        phi = np.exp(score)
        hazard = self.baseline_hazard * phi.reshape(-1, 1)
        survival = np.power(self.baseline_survival, phi.reshape(-1, 1))
        density = hazard * survival
        if t is None:
            return hazard, density, survival
        else:
            min_index = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_index)
            return hazard[:, index], density[:, index], survival[:, index]

    def predict_risk(self, x, use_log=False):
        """
        Predicting the risk score functions
        
        Arguments:
            * x: pd.Dataframe or np.ndarray or list
                x is the testing dataset containing the features
                x should not be standardized before, the model
                will take care of it
        """

        # Convert x into the right format
        x = utils.check_data(x)

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

        # Calculating risk_score
        score = self.model(x).data.numpy().flatten()
        if not use_log:
            score = np.exp(score)

        return score

    def __repr__(self):
        """ Representing the class object """

        if self.structure is None:
            super(NonLinearCoxPHModel, self).__repr__()
            return self.name

        else:
            S = len(self.structure)
            self.name = self.__class__.__name__
            empty = len(self.name)
            self.name += '( '
            for i, s in enumerate(self.structure):
                n = 'Layer({}): '.format(i + 1)
                activation = nn.activation_function(s['activation'],
                                                    return_text=True)
                n += 'activation = {}, '.format(s['activation'])
                n += 'num_units = {} '.format(s['num_units'])

                if i != S - 1:
                    self.name += n + '; \n'
                    self.name += empty * ' ' + '  '
                else:
                    self.name += n
            self.name += ')'
            return self.name
