from __future__ import absolute_import
import os
import pandas as pd
import numpy as np
from scipy import stats
from pysurvival import utils
from pysurvival.models import BaseModel
from pysurvival.models._non_parametric import _KaplanMeierModel
from pysurvival.models._non_parametric import _KernelModel


class NonParametricModel(BaseModel):
    """ Non Parametric Model
        --------------------

        The Non Parametric Model object is tha base model for any Non Parametric 
        models in pysurvival such as :
        * Kaplan Meier model
        * Smooth Kaplan Meier model

        This object should not be used on its own.
    """

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

        # Delete the CPP object before saving
        del self.model

        # Saving the model
        super(NonParametricModel, self).save(path_file)

        # Re-introduce the C++ object
        if 'smooth' in self.name.lower():
            self.model = _KernelModel(self.bandwidth, self.kernel_type)
        else:
            self.model = _KaplanMeierModel()
        self.load_properties()

    def load(self, path_file):
        """ Load the model paremeters from a zip file into a C++ external
            model 
        """

        # Loading the model
        super(NonParametricModel, self).load(path_file)

        # Re-introduce the C++ object
        if 'smooth' in self.name.lower():
            self.model = _KernelModel(self.bandwidth, self.kernel_type)
        else:
            self.model = _KaplanMeierModel()
        self.load_properties()

    def fit(self, T, E, weights=None, alpha=0.95):
        """ Fitting the model according to the provided data.

        Parameters:
        -----------
        * `T` : **array-like** -- 
            The target values describing when the event of interest or censoring
            occurred.

        * `E` : **array-like** --
            The values that indicate if the event of interest occurred i.e.: E[i]=1
            corresponds to an event, and E[i] = 0 symbols censoring, for all i.

        * `weights` : **array-like** *(default = None)* -- 
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given a unit weight.

        * `alpha`: **float** *(default = 0.05)* --
            Significance level

        Returns:
        --------
        * self : object


        Example:
        --------

        # Importing modules
        import numpy as np
        from matplotlib import pyplot as plt
        from pysurvival.utils.display import display_non_parametric
        # %matplotlib inline #Uncomment when using Jupyter 

        # Generating random times and event indicators 
        T = np.round(np.abs(np.random.normal(10, 10, 1000)), 1)
        E = np.random.binomial(1, 0.3, 1000)

        # Initializing the KaplanMeierModel
        from pysurvival.models.non_parametric import KaplanMeierModel
        km_model = KaplanMeierModel()

        # Fitting the model 
        km_model.fit(T, E, alpha=0.95)

        # Displaying the survival function and confidence intervals
        display_non_parametric(km_model)

        # Initializing the SmoothKaplanMeierModel
        from pysurvival.models.non_parametric import SmoothKaplanMeierModel
        skm_model = SmoothKaplanMeierModel(bandwith=0.1, kernel='normal')

        # Fitting the model
        skm_model.fit(T, E)

        # Displaying the survival function and confidence intervals
        display_non_parametric(skm_model)
        """

        # Checking the format of the data 
        T, E = utils.check_data(T, E)

        # weighting
        if weights is None:
            weights = [1.] * T.shape[0]

        # Confidence Intervals
        z = stats.norm.ppf((1. - alpha) / 2.)

        # Building the Kaplan-Meier model
        survival = self.model.fit(T, E, weights, z)
        if sum(survival) <= 0.:
            mem_error = "The kernel matrix cannot fit in memory."
            mem_error += "You should use a bigger bandwidth b"
            raise MemoryError(mem_error)

        # Saving all properties
        self.save_properties()

        # Generating the Survival table
        if 'smooth' not in self.name.lower():
            self.get_survival_table()

    def predict_cumulative_hazard(self, *args, **kargs):
        raise NotImplementedError()

    def predict_risk(self, *args, **kargs):
        raise NotImplementedError()

    def predict_survival(self, x, t=None, **kwargs):
        """ Extracting the predicted survival probabilities at the individual 
            event times that have been used for fitting the model.
        """
        is_lagged = kwargs.pop("is_lagged", False)
        return self.model.predict_survival(t, is_lagged)

    def predict_survival_upper(self, x, t=None, **kwargs):
        """ Extracting the predicted survival CI upper probabilities at the 
            individual event times that have been used for fitting the model.
        """
        is_lagged = kwargs.pop("is_lagged", False)
        return self.model.predict_survival_upper(t, is_lagged)

    def predict_survival_lower(self, x, t=None, **kwargs):
        """ Extracting the predicted survival CI lower probabilities at the 
            individual event times that have been used for fitting the model.
        """
        is_lagged = kwargs.pop("is_lagged", False)
        return self.model.predict_survival_lower(t, is_lagged)

    def predict_density(self, x, t=None, **kwargs):
        """ Extracting the predicted density probabilities at the individual 
            event times that have been used for fitting the model.
        """
        is_lagged = kwargs.pop("is_lagged", False)
        return self.model.predict_density(t, is_lagged)

    def predict_hazard(self, x, t=None, **kwargs):
        """ Extracting the hazard function values at the individual 
            event times that have been used for fitting the model.
        """
        is_lagged = kwargs.pop("is_lagged", False)
        return self.model.predict_hazard(t, is_lagged)

    def save_properties(self):
        """ Saving the C++ attributes in the Python object """
        self.times = np.array(self.model.times)
        self.time_buckets = self.model.time_buckets
        self.survival = np.array(self.model.survival)
        self.hazard = np.array(self.model.hazard)
        self.cumulative_hazard = np.array(self.model.cumulative_hazard)

        if 'smooth' in self.name.lower():
            self.km_survival = np.array(self.model.km_survival)
            self.km_times = np.array(self.model.km_times)
            self.kernel_type = self.model.kernel_type
            self.bandwidth = self.model.b
            self.kernel_matrix = np.array(self.model.kernel_matrix)

        else:
            self.std_error = np.array(self.model.std_error)
            self.survival_ci_upper = np.array(self.model.survival_ci_upper)
            self.survival_ci_lower = np.array(self.model.survival_ci_lower)
            self.at_risk = np.array(self.model.at_risk)
            self.events = np.array(self.model.events)

    def load_properties(self):
        """ Loading Python object attributes into the C++  """
        self.model.times = self.times
        self.model.time_buckets = self.time_buckets
        self.model.survival = self.survival
        self.model.hazard = self.hazard
        self.model.cumulative_hazard = self.cumulative_hazard

        if 'smooth' in self.name.lower():
            self.model.kernel_type = self.kernel_type
            self.model.b = self.bandwidth
            self.model.kernel_matrix = self.kernel_matrix
            self.model.km_survival = self.km_survival
            self.model.km_times = self.km_times

        else:
            self.model.std_error = self.std_error
            self.model.survival_ci_upper = self.survival_ci_upper
            self.model.survival_ci_lower = self.survival_ci_lower
            self.model.at_risk = self.at_risk
            self.model.events = self.events

    def get_survival_table(self):
        """ Computing the survival table"""

        data = {'Time': self.times,
                'Number at risk': self.at_risk,
                'Number of events': self.events,
                'Survival': self.survival,
                'Survival - CI Lower': self.survival_ci_lower,
                'Survival - CI Upper': self.survival_ci_upper,
                }
        survival_df = pd.DataFrame(data,
                                   columns=['Time', 'Number at risk',
                                            'Number of events', 'Survival',
                                            'Survival - CI Lower',
                                            'Survival - CI Upper'])
        self.survival_table = survival_df
        return survival_df


class KaplanMeierModel(NonParametricModel):
    """ Kaplan-Meier Model
        ------------------
        
    The Kaplan-Meier estimator is a non-parametric statistic 
    used to estimate the survival function from lifetime data. 
    The estimator is named after Edward L. Kaplan and Paul Meier, 
    who each submitted similar manuscripts 
    to the Journal of the American Statistical Association.

    The estimator is given by:
        S(t) = (1.-d_1/r_1)*...*(1.-d_i/r_i)*...*(1.-d_n/r_n)
            t_i, a time when at least one event happened, 
            d_i, the number of events that happened at time t_i 
            r_i, the individuals known to survive at t_i.

    References:
    -----------
        https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator
    """

    def __init__(self):
        # Initializing the C++ object
        self.model = _KaplanMeierModel()

        # Saving the attributes
        self.__repr__()
        self.not_implemented_error = "{} does not have this method." \
            .format(self.name)


class SmoothKaplanMeierModel(NonParametricModel):
    """ SmoothKaplanMeierModel
        ------------------------
        Because the standard Kaplan-Meier estimator is a step function with jumps 
        located only at the uncensored observations, when many data are censored, 
        it can only have a few jumps with increasing sizes. Thus the accuracy of 
        the estimation might not be acceptable. 
        A Smooth estimator is a good alternative, since it is computed by 
        giving mass to all the data, including the censored observations. (1)
        
        The current implementation is based on Nonparametric density estimation 
        from censored data from W.J. Padgett and Diane T. McNichols (2)
        It was also inspired by CDF and survival function estimation with 
        infinite-order kernels from Berg and Politis(2)
        
        References:
        -----------
        * survPresmooth: An R Package for PreSmooth Estimation in Survival Analysis
          https://www.jstatsoft.org/article/view/v054i11 (1)
        * https://doi.org/10.1080/03610928408828780 (2)
        * https://projecteuclid.org/download/pdfview_1/euclid.ejs/1261671304 (3)

        Parameters:
        -----------
        * bandwidth: double (default=0.1)
             controls the degree of the smoothing. The smaller it is the closer
             to the original KM the function will be, but it will increase the 
             computation time. If it is very large, the resulting model will be 
             smoother than the estimator of KM, but it will stop being as accurate.
             
        * kernel: str (default='normal')
            defines the type of kernel the model will be using. 
            Here are the possible options:
                * uniform: f(x) = 0 if |x|<1 else f(x) = 0.5
                * epanechnikov: f(x) = 0 if |x|<=1 else f(x) = 0.75*(1. - x^2 )
                * normal: f(x) = exp( -x*x/2.) / sqrt( 2*pi )
                * biweight: f(x) = 0 if |x|<=1 else f(x)=(15./16)*(1.-x^2)^2
                * triweight: f(x) = 0 if |x|<=1 else f(x)=(35./32)*(1.-x^2)^3
                * Cosine:  f(x) = 0 if |x|<=1 else  f(x)=(pi/4.)*cos( pi*x/2. )  
    """

    def __init__(self, bandwidth=0.1, kernel='normal'):

        # Kernel function
        if kernel.lower() == 'uniform':
            kernel_type = 0  # Uniform kernel
            kernel = 'Uniform'

        elif kernel.lower().startswith("epa"):
            kernel_type = 1  # Epanechnikov kernel
            kernel = 'Epanechnikov'

        elif kernel.lower() == "normal":
            kernel_type = 2  # Normal kernel
            kernel = 'Normal'

        elif kernel.lower().startswith("bi"):
            kernel_type = 3  # Biweight kernel
            kernel = 'Biweight'

        elif kernel.lower().startswith("tri"):
            kernel_type = 4  # Triweight kernel
            kernel = 'Triweight'

        elif kernel.lower().startswith("cos"):
            kernel_type = 5  # Cosine kernel
            kernel = 'Cosine'

        else:
            raise NotImplementedError('{} is not a valid kernel function.'
                                      .format(kernel))

        # bandwidth
        if bandwidth <= 0.:
            raise ValueError('bandwidth has to be positive.')

        # Initializing the C++ object
        self.model = _KernelModel(bandwidth, kernel_type)

        # Saving the attributes
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kernel_type = kernel_type

        # Creating the representation of the object
        self.__repr__()
        self.not_implemented_error = "{} does not have this method." \
            .format(self.name)

    def __repr__(self):
        """ Creates the representation of the Object """
        self.name = "{}(bandwith={:.2f}, kernel='{}')"
        self.name = self.name.format(self.__class__.__name__,
                                     self.bandwidth, self.kernel)
        return self.name
