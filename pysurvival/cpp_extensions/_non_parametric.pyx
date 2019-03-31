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


# Importing C++ object KaplanMeierModel from non_parametric.h
#----------------------------------------------------------------
cdef extern from "non_parametric.h":

    cdef cppclass KaplanMeierModel:
        KaplanMeierModel() except+;

        # Attributes --  NonParametricModel
        vector[double] times;
        vector[double] survival, hazard, cumulative_hazard, std_error;
        vector[double] survival_ci_upper, survival_ci_lower;
        vector[double] at_risk, events;

        # Attributes -- KaplanMeierModel
        vector[ pair[double, double] ] time_buckets;

        # Methods - NonParametricModel
        double predict_survival(double t, bool is_lagged);
        double predict_density(double t, bool is_lagged);
        double predict_hazard(double t, bool is_lagged);
        double predict_survival_upper(double t, bool is_lagged);
        double predict_survival_lower(double t, bool is_lagged);

        # Methods -- KaplanMeierModel
        vector[double] fit(vector[double] T, vector[double] E, 
                            vector[double] weights, double z, bool ipcw);


cdef class _KaplanMeierModel :

    cdef:
        KaplanMeierModel *thisptr # hold a C++ instance which we're wrapping
        string name

    def __init__(self):
        self.thisptr = new KaplanMeierModel()

    def __dealloc__(self):
        del self.thisptr

    # Times
    @property
    def times(self):
        return self.thisptr.times

    @times.setter
    def times(self, times):
        self.thisptr.times = times

    # Time Buckets
    @property
    def time_buckets(self):
        return self.thisptr.time_buckets

    @time_buckets.setter
    def time_buckets(self, time_buckets):
        self.thisptr.time_buckets = time_buckets

    # Survival
    @property
    def survival(self):
        return self.thisptr.survival

    @survival.setter
    def survival(self, survival):
        self.thisptr.survival = survival

    # hazard
    @property
    def hazard(self):
        return self.thisptr.hazard

    @hazard.setter
    def hazard(self, hazard):
        self.thisptr.hazard = hazard        

    # cumulative_hazard
    @property
    def cumulative_hazard(self):
        return self.thisptr.cumulative_hazard

    @cumulative_hazard.setter
    def cumulative_hazard(self, cumulative_hazard):
        self.thisptr.cumulative_hazard = cumulative_hazard

    # std_error
    @property
    def std_error(self):
        return self.thisptr.std_error

    @std_error.setter
    def std_error(self, std_error):
        self.thisptr.std_error = std_error

    # survival_ci_upper
    @property
    def survival_ci_upper(self):
        return self.thisptr.survival_ci_upper

    @survival_ci_upper.setter
    def survival_ci_upper(self, survival_ci_upper):
        self.thisptr.survival_ci_upper = survival_ci_upper

    # survival_ci_lower
    @property
    def survival_ci_lower(self):
        return self.thisptr.survival_ci_lower

    @survival_ci_lower.setter
    def survival_ci_lower(self, survival_ci_lower):
        self.thisptr.survival_ci_lower = survival_ci_lower

    # at_risk
    @property
    def at_risk(self):
        return self.thisptr.at_risk

    @at_risk.setter
    def at_risk(self, at_risk):
        self.thisptr.at_risk = at_risk

    # events
    @property
    def events(self):
        return self.thisptr.events

    @events.setter
    def events(self, events):
        self.thisptr.events = events

    # name
    @property
    def name(self):
        return self.name

    def __repr__(self):
        """ Creates the representation of the Object """

        self.name = self.__class__.__name__
        return self.name


    cpdef vector[double] fit(self, vector[double] T, vector[double] E, 
                             vector[double] weights, double z):
        """ Building the Kaplan Meier Estimator. 
            This implementation is inspired by the source code for 
            statsmodels.duration.survfunc available in the package statsmodels
            http://www.statsmodels.org/
            This method assumes that T and E have been sorted in a descending 
            order according to T.
        """
        return self.thisptr.fit(T, E, weights, z, False)


    cpdef double predict_survival(self, double t, bool is_lagged=False ):
        """ Computing the Survival function at a time t """

        return self.thisptr.predict_survival(t, is_lagged)


    cpdef double predict_survival_upper(self, double t, bool is_lagged=False ):
        """ Computing the Survival CI upper function at a time t """

        return self.thisptr.predict_survival_upper(t, is_lagged)


    cpdef double predict_survival_lower(self, double t, bool is_lagged=False ):
        """ Computing the Survival CI lower function at a time t """

        return self.thisptr.predict_survival_lower(t, is_lagged)


    cpdef double predict_density(self, double t, bool is_lagged=False ):
        """ Computing the probability mass function at a time t """

        return self.thisptr.predict_density(t, is_lagged)


    cpdef double predict_hazard(self, double t, bool is_lagged=False ):
        """ Computing the hazard function at a time t """

        return self.thisptr.predict_hazard(t, is_lagged)



# Importing C++ object KernelModel from non_parametric.h
#-----------------------------------------------------------
cdef extern from "non_parametric.h":

    cdef cppclass KernelModel:
        KernelModel(double b , int kernel_type) except+;

        # Attributes --  NonParametricModel
        vector[double] times;
        vector[double] survival, hazard, cumulative_hazard, std_error;

        # Attributes -- KernelModel
        double b;
        int kernel_type;
        vector[ pair[double, double] ] time_buckets, time_intervals;
        vector[double] density, km_survival, km_times;
        vector[vector[double]] kernel_matrix;

        # Methods - NonParametricModel
        double predict_survival(double t, bool is_lagged);
        double predict_density(double t, bool is_lagged);
        double predict_hazard(double t, bool is_lagged);

        # Methods -- KaplanMeierModel
        vector[double] fit(vector[double] T, vector[double] E, 
                            vector[double] weights, double z);



cdef class _KernelModel :

    cdef:
        KernelModel *thisptr # hold a C++ instance which we're wrapping
        string name

    def __init__(self, double b , int kernel_type):
        self.thisptr = new KernelModel(b , kernel_type)

    def __dealloc__(self):
        del self.thisptr

    # Times
    @property
    def times(self):
        return self.thisptr.times

    @times.setter
    def times(self, times):
        self.thisptr.times = times

    # Time Buckets
    @property
    def time_buckets(self):
        return self.thisptr.time_buckets

    @time_buckets.setter
    def time_buckets(self, time_buckets):
        self.thisptr.time_buckets = time_buckets

    # Survival
    @property
    def survival(self):
        return self.thisptr.survival

    @survival.setter
    def survival(self, survival):
        self.thisptr.survival = survival

    # hazard
    @property
    def hazard(self):
        return self.thisptr.hazard

    @hazard.setter
    def hazard(self, hazard):
        self.thisptr.hazard = hazard        

    # cumulative_hazard
    @property
    def cumulative_hazard(self):
        return self.thisptr.cumulative_hazard

    @cumulative_hazard.setter
    def cumulative_hazard(self, cumulative_hazard):
        self.thisptr.cumulative_hazard = cumulative_hazard

    # density
    @property
    def density(self):
        return self.thisptr.density

    @density.setter
    def density(self, density):
        self.thisptr.density = density

    # km_survival
    @property
    def km_survival(self):
        return self.thisptr.km_survival

    @km_survival.setter
    def km_survival(self, km_survival):
        self.thisptr.km_survival = km_survival

    # km_times
    @property
    def km_times(self):
        return self.thisptr.km_times

    @km_times.setter
    def km_times(self, km_times):
        self.thisptr.km_times = km_times

    # kernel_matrix
    @property
    def kernel_matrix(self):
        return self.thisptr.kernel_matrix

    @kernel_matrix.setter
    def kernel_matrix(self, kernel_matrix):
        self.thisptr.kernel_matrix = kernel_matrix

    # bandwidth
    @property
    def b(self):
        return self.thisptr.b

    @b.setter
    def b(self, b):
        self.thisptr.b = b

    # kernel_type
    @property
    def kernel_type(self):
        return self.thisptr.kernel_type

    @kernel_type.setter
    def kernel_type(self, kernel_type):
        self.thisptr.kernel_type = kernel_type  

    # name
    @property
    def name(self):
        return self.name

    def __repr__(self):
        """ Creates the representation of the Object """

        self.name = self.__class__.__name__
        return self.name

    cpdef vector[double] fit(self, vector[double] T, vector[double] E, 
                             vector[double] weights, double z):
        """ Fitting the Non Parametric Kernel model """
        return self.thisptr.fit(T, E, weights, z)


    cpdef double predict_survival(self, double t, bool is_lagged=False):
        """ Computing the Survival function at a time t """
        return self.thisptr.predict_survival(t, is_lagged)


    cpdef double predict_density(self, double t, bool is_lagged=False ):
        """ Computing the probability mass function at a time t """
        return self.thisptr.predict_density(t, is_lagged)


    cpdef double predict_hazard(self, double t, bool is_lagged=False ):
        """ Computing the hazard function at a time t """

        return self.thisptr.predict_hazard(t, is_lagged)
