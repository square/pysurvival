from __future__ import absolute_import
import torch
import numpy as np
import pandas as pd
import scipy
import os
import copy
from pysurvival import utils
from pysurvival.utils import optimization as opt
from pysurvival.models import BaseModel
from pysurvival.models._svm import _SVMModel

# Available Kernel functions
KERNELS = {'Linear': 0, 'Polynomial': 1, 'Gaussian': 2, 'Normal': 2,
           'Exponential': 3, 'Tanh': 4, 'Sigmoid': 5, 'Rational Quadratic': 6,
           'Inverse Multiquadratic': 7, 'Multiquadratic': 8}

REVERSE_KERNELS = {value: key for (key, value) in KERNELS.items()}


class SurvivalSVMModel(BaseModel):
    """ Survival Support Vector Machine model:
        --------------------------------------

        The purpose of the model is to help us look at Survival Analysis 
        as a Ranking Problem.
        Indeed, the idea behind formulating the survival problem as a ranking 
        problem is that in some applications, like clinical applications, 
        one is only interested in defining risks groups, and not the prediction 
        of the survival time, but in whether the unit has a high or low risk for 
        the event to occur. 

        The current implementation is based on the "Rank Support Vector Machines 
        (RankSVMs)" developed by Van Belle et al. This allows us to compute a 
        convex quadratic loss function, so that we can use the Newton 
        optimization to minimize it.

        References:
        * Fast Training of Support Vector Machines for Survival Analysis
          from Sebastian Posterl, Nassir Navab, and Amin Katouzian
          https://link.springer.com/chapter/10.1007/978-3-319-23525-7_15
        * An Efficient Training Algorithm for Kernel Survival Support Vector 
          Machines from Sebastian Posterl, Nassir Navab, and Amin Katouzian
          https://arxiv.org/abs/1611.07054
        * Support vector machines for survival analysis.
          Van Belle, V., Pelckmans, K., Suykens, J.A., Van Huffel, S.
          ftp://ftp.esat.kuleuven.be/sista/kpelckma/kp07-70.pdf


        Parameters:
        -----------
        * kernel: str (default="linear")
            The type of kernel used to fit the model. Here's the list
            of available kernels:
            
            * linear
            * polynomial
            * gaussian
            * exponential
            * tanh
            * sigmoid
            * rational_quadratic
            * inverse_multiquadratic
            * multiquadratic

        * scale: float (default=1)
            Scale parameter of the kernel function
            
        * offset: float (default=0)
            Offset parameter of the kernel function
            
        * degree: float (default=1)
            Degree parameter of the polynomial/kernel function

    """

    def __init__(self, kernel="linear", scale=1., offset=0., degree=1.,
                 auto_scaler=True):

        # Ensuring that the provided kernel is available
        valid_kernel = [key for key in KERNELS.keys() \
                        if kernel.lower().replace('_', ' ') in key.lower().replace('_', ' ')]

        if len(valid_kernel) == 0:
            raise NotImplementedError('{} is not a valid kernel function.'
                                      .format(kernel))
        else:
            kernel_type = KERNELS[valid_kernel[0]]
            kernel = valid_kernel[0]

        # Checking the kernel parameters
        if not (degree >= 0. and (isinstance(degree, float) or isinstance(degree, int))):
            error = "degree parameter is not valid. degree is a >= 0 value"
            raise ValueError(error)

        if not (isinstance(scale, float) or isinstance(scale, int)):
            error = "scale parameter is not valid."
            raise ValueError(error)

        if not (isinstance(offset, float) or isinstance(offset, int)):
            error = "offset parameter is not valid."
            raise ValueError(error)

        # Saving the attributes
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.scale = scale
        self.offset = offset
        self.degree = degree

        # Initializing the C++ object
        self.model = _SVMModel(self.kernel_type, self.scale, self.offset,
                               self.degree)

        # Initializing the elements from BaseModel
        super(SurvivalSVMModel, self).__init__(auto_scaler)

    def __repr__(self):
        """ Creates the representation of the Object """

        self.name = self.__class__.__name__
        if 'kernel' in self.name:
            self.name += "(kernel: '{}'".format(self.kernel) + ')'
        return self.name

    def save(self, path_file):
        """ Save the model paremeters of the model (.params) and Compress 
            them into a zip file
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
        super(SurvivalSVMModel, self).save(path_file)

        # Re-introduce the C++ object
        self.model = _SVMModel(self.kernel_type, self.scale, self.offset,
                               self.degree)
        self.load_properties()

    def load(self, path_file):
        """ Load the model parameters from a zip file into a C++ external
            model 
        """
        # Loading the model
        super(SurvivalSVMModel, self).load(path_file)

        # Re-introduce the C++ object
        self.model = _SVMModel(self.kernel_type, self.scale, self.offset,
                               self.degree)
        self.load_properties()

    def fit(self, X, T, E, with_bias=True, init_method='glorot_normal',
            lr=1e-2, max_iter=100, l2_reg=1e-4, tol=1e-3,
            verbose=True):
        """
        Fitting a Survival Support Vector Machine model.

        As the Hessian matrix of the log-likelihood can be 
        calculated without too much effort, the model parameters are 
        computed using the Newton_Raphson Optimization scheme:
                W_new = W_old - lr*<Hessian^(-1), gradient>

        Arguments:
        ---------
        
        * `X` : array-like, shape=(n_samples, n_features)
            The input samples.

        * `T` : array-like, shape = [n_samples] 
            The target values describing when the event of interest or censoring
            occurred

        * `E` : array-like, shape = [n_samples] 
            The Event indicator array such that E = 1. if the event occurred
            E = 0. if censoring occurred

        * `with_bias`: bool (default=True)
            Whether a bias should be added 

        * `init_method` : str (default = 'glorot_uniform')
            Initialization method to use. Here are the possible options:
                * 'glorot_uniform': Glorot/Xavier uniform initializer, 
                * 'he_uniform': He uniform variance scaling initializer
                * 'uniform': Initializing tensors with uniform (-1, 1) distribution
                * 'glorot_normal': Glorot normal initializer,
                * 'he_normal': He normal initializer.
                * 'normal': Initializing tensors with standard normal distribution
                * 'ones': Initializing tensors to 1
                * 'zeros': Initializing tensors to 0
                * 'orthogonal': Initializing tensors with a orthogonal matrix,

        * `lr`: float (default=1e-4)
            learning rate used in the optimization

        * `max_iter`: int (default=100)
            The maximum number of iterations in the Newton optimization

        * `l2_reg`: float (default=1e-4)
            L2 regularization parameter for the model coefficients

        * `alpha`: float (default=0.95)
            Confidence interval

        * `tol`: float (default=1e-3)
            Tolerance for stopping criteria

        * `verbose`: bool (default=True)
            Whether or not producing detailed logging about the modeling


        Example:
        --------

        #### 1 - Importing packages
        import numpy as np
        import pandas as pd
        from pysurvival.models.svm import LinearSVMModel
        from pysurvival.models.svm import KernelSVMModel
        from pysurvival.models.simulations import SimulationModel
        from pysurvival.utils.metrics import concordance_index
        from sklearn.model_selection import train_test_split
        from scipy.stats.stats import pearsonr   
        # %pylab inline # to use in jupyter notebooks

        #### 2 - Generating the dataset from the parametric model
        # Initializing the simulation model
        sim = SimulationModel( survival_distribution = 'Log-Logistic',  
                               risk_type = 'linear',
                               censored_parameter = 1.1, 
                               alpha = 1.5, beta = 4)

        # Generating N Random samples
        N = 1000
        dataset = sim.generate_data(num_samples = N, num_features = 4)

        #### 3 - Splitting the dataset into training and testing sets
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


        #### 4 - Creating an instance of the SVM model and fitting the data.
        svm_model = LinearSVMModel()
        svm_model = KernelSVMModel(kernel='Gaussian', scale=0.25)
        svm_model.fit(X_train, T_train, E_train, init_method='he_uniform', 
            with_bias = True, lr = 0.5,  tol = 1e-3,  l2_reg = 1e-3)

        #### 5 - Cross Validation / Model Performances
        c_index = concordance_index(svm_model, X_test, T_test, E_test) #0.93
        print('C-index: {:.2f}'.format(c_index))

        #### 6 - Comparing the model predictions to Actual risk score
        # Comparing risk scores
        svm_risks = svm_model.predict_risk(X_test)
        actual_risks = sim.predict_risk(X_test).flatten()
        print("corr={:.4f}, p_value={:.5f}".format(*pearsonr(svm_risks, 
            actual_risks)))# corr=-0.9992, p_value=0.00000

        """

        # Collecting features names
        N, self.num_vars = X.shape
        if isinstance(X, pd.DataFrame):
            self.variables = X.columns.tolist()
        else:
            self.variables = ['x_{}'.format(i) for i in range(self.num_vars)]

        # Adding a bias or not
        self.with_bias = with_bias
        if with_bias:
            self.variables += ['intercept']
        p = int(self.num_vars + 1. * with_bias)

        # Checking the format of the data 
        X, T, E = utils.check_data(X, T, E)

        if with_bias:
            # Adding the intercept
            X = np.c_[X, [1.] * N]
        X = self.scaler.fit_transform(X)

        # Initializing the parameters 
        if self.kernel_type == 0:
            W = np.zeros((p, 1))
        else:
            W = np.zeros((N, 1))
        W = opt.initialization(init_method, W, False).flatten()
        W = W.astype(np.float64)

        # Optimizing to find best parameters 
        self.model.newton_optimization(X, T, E, W, lr, l2_reg,
                                       tol, max_iter, verbose)
        self.save_properties()

        return self

    def save_properties(self):
        """ Loading the properties of the model """

        self.weights = np.array(self.model.W)
        self.Kernel_Matrix = np.array(self.model.Kernel_Matrix)
        self.kernel_type = self.model.kernel_type
        self.scale = self.model.scale
        self.offset = self.model.offset
        self.degree = self.model.degree
        self.loss = np.array(self.model.loss)
        self.inv_Hessian = np.array(self.model.inv_Hessian)
        self.loss_values = np.array(self.model.loss_values)
        self.grad2_values = np.array(self.model.grad2_values)
        self.internal_X = np.array(self.model.internal_X)

    def load_properties(self):
        """ Loading the properties of the model """

        self.model.W = self.weights
        self.model.Kernel_Matrix = self.Kernel_Matrix
        self.model.kernel_type = self.kernel_type
        self.model.scale = self.scale
        self.model.offset = self.offset
        self.model.degree = self.degree
        self.model.loss = self.loss
        self.model.inv_Hessian = self.inv_Hessian
        self.model.loss_values = self.loss_values
        self.model.grad2_values = self.grad2_values
        self.model.internal_X = self.internal_X
        self.kernel = REVERSE_KERNELS[self.kernel_type]

    def predict_risk(self, x, use_log=False):
        """ Predicts the Risk Score
        
            Parameter
            ----------
            * `x`, np.ndarray
                 array-like representing the datapoints

            * `use_log`: bool - (default=False)
                Applies the log function to the risk values

            Returns
            -------
            * `risk_score`, np.ndarray
                array-like representing the prediction of Risk Score function
        """

        # Ensuring that the C++ model has the fitted parameters
        self.load_properties()

        # Convert x into the right format
        x = utils.check_data(x)

        # Scaling the dataset
        if x.ndim == 1:
            if self.with_bias:
                x = np.r_[x, 1.]
            x = self.scaler.transform(x.reshape(1, -1))
        elif x.ndim == 2:
            n = x.shape[0]
            if self.with_bias:
                x = np.c_[x, [1.] * n]
            x = self.scaler.transform(x)

        # Calculating prdiction
        risk = np.exp(self.model.get_score(x))

        if use_log:
            return np.log(risk)
        else:
            return risk

    def predict_cumulative_hazard(self, *args, **kargs):
        raise NotImplementedError()

    def predict_cdf(self, *args, **kargs):
        raise NotImplementedError()

    def predict_survival(self, *args, **kargs):
        raise NotImplementedError()

    def predict_density(self, *args, **kargs):
        raise NotImplementedError()

    def predict_hazard(self, *args, **kargs):
        raise NotImplementedError()


class LinearSVMModel(SurvivalSVMModel):

    def __init__(self, auto_scaler=True):
        super(LinearSVMModel, self).__init__(kernel="linear", scale=1.,
                                             offset=0., degree=1., auto_scaler=True)


class KernelSVMModel(SurvivalSVMModel):

    def __init__(self, kernel="gaussian", scale=1., offset=0., degree=1.,
                 auto_scaler=True):
        if "linear" in kernel.lower():
            error = "To use a 'linear' svm model, create an instance of"
            error += "pysurvival.models.svm.LinearSVMModel"
            raise ValueError(error)

        super(KernelSVMModel, self).__init__(kernel=kernel, scale=scale,
                                             offset=offset, degree=degree, auto_scaler=auto_scaler)
