# %%cython --a  
# distutils: language = c++

# Importing Numpy
#-----------------
import numpy as np
cimport numpy as cnp

# Importing cython and C++ 
#---------------------------
import cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp cimport bool
from libcpp.algorithm cimport sort


# Importing C++ specific functions
#----------------------------------
cdef extern from "functions.h"  nogil:

    cdef vector[int] argsort(vector[double] v, bool descending);

    cdef vector[double] cumsum( vector[double] v );

    cdef map[ int, vector[double] ]  baseline_functions(vector[double] score, 
        vector[double] T, vector[double] E);



# Importing math functions
#--------------------------
cdef extern from "<math.h>" nogil:
    double fmax(double x, double y)
    double exp(double x)
    double sqrt(double x)
    double log(double x)
    double cos(double x)
    double fabs(double x)
    double M_PI
    double pow(double x, double y)
    double INFINITY
    double NAN

    
# -------------------------------- FUNCTIONS -------------------------------- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double norm(cnp.ndarray[double, ndim=1] x, bool with_sqrt=True ):
    cdef:
        double result = 0.
        size_t i, N = x.shape[0]
    for i in range(N):
        result += x[i]*x[i]
        
    if with_sqrt:
        return sqrt(result)
    else:
        return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef vector[double] reverse(vector[double] x):
    cdef:
        size_t i, N = x.size()
        vector[double] v
    for i in range(N):
        v.push_back(x[N-i-1])
    return v
        


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef map[ int, vector[double] ] _baseline_functions(  vector[double] score
                                                     , vector[double] T
                                                     , vector[double] E ):
    """ This method provides the calculations to estimate 
        the baseline survival function. This function assumes that X, T, E
        are sorted in a descending order.
        
        The formula used to calculate the baseline hazard is:
            h_0( T ) = |D(T)|/Sum( exp( <x_j, W> ), j in R(T) ) where:
                - T is a time of failure
                - |D(T)| is the number of failures at time T
                - R(T) is the set of at risk uites at time T 
        https://github.com/cran/survival/blob/master/R/basehaz.R
        http://www.utdallas.edu/~pkc022000/6390/SP06/NOTES/survival_week_5.pdf
        https://stats.stackexchange.com/questions/46532/cox-baseline-hazard
    """
    
    cdef:
        # Variables related to the loop
        size_t i, N = score.size()
        vector[int] order

        # Variables related to the risk score theta = exp(<x, W>)
        double theta_i
        double sum_theta_risk = 0.

        # Variables related to the time of failures
        size_t j, J
        int nb_fails = 0

        # Variables related to baseline function
        vector[double] times
        vector[double] baseline_hazard
        vector[double] baseline_survival
        vector[double] baseline_cumulative_hazard
        map[ int, vector[double] ] result
        
    # Looping through the data to calculate the baseline function
    for i in range(N):
        
        # Calculating the at risk variables
        sum_theta_risk += score[i]

        # Calculating the fail variables
        if E[i] == 1:
            nb_fails += 1

        if i < N-1 and T[i] == T[i+1]:
            continue

        if nb_fails == 0:
            continue
            
        baseline_hazard.push_back( nb_fails*1/sum_theta_risk )
        times.push_back(T[i])
        nb_fails = 0
        
    # Saving attributes
    baseline_hazard = reverse(baseline_hazard)
    baseline_cumulative_hazard = cumsum(baseline_hazard)
    J = baseline_cumulative_hazard.size()
    for j in range(J):
        baseline_survival.push_back( exp( - baseline_cumulative_hazard[j] ) )

    result[0] = reverse(times)
    result[1] = baseline_hazard
    result[2] = baseline_survival

    return result


# -------------------------------- CLASSES -------------------------------- #
cdef class _CoxPHModel :
    
    """ This object will help speed up the calculation of the Cox PH model.
        It will allow us to calculate the log_likelihood, its gradient 
        and hessian to perform the Newton-Raphson optimization.
    """
    
    # Attributes
    #-----------
    cdef public:

        # Model parameters 
        vector[double] times
        cnp.ndarray W
    
        # Log_likelihood
        double loss
        cnp.ndarray gradient
        cnp.ndarray Hessian, inv_Hessian
        vector[double] loss_values, grad2_values
        
    
    # Methods
    #--------    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    cpdef void get_times(self, cnp.ndarray[double, ndim=1] T
                             , cnp.ndarray[double, ndim=1] E ):
        """ Method used to calculate the vector of unique times of failures
            It assumes that T, E are sorted according to T in descending order
        """
        cdef:
            size_t i, N = T.shape[0]
            int nb_times = 0
            cnp.ndarray[double, ndim=1] temp_results = np.zeros(N)
            vector[double] times
            int nb_fails = 0
            
        for i in range(N):
            if E[i] == 1. :
                nb_fails += 1
                
            if i < N-1 and T[i] == T[i+1]:
                continue

            if nb_fails == 0:
                continue
            times.push_back(T[i])
            nb_fails = 0

        self.times = reverse(times)  


    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    cdef void get_derivatives(self, cnp.ndarray[double, ndim=2] X
                                  , cnp.ndarray[double, ndim=1] T
                                  , cnp.ndarray[double, ndim=1] E
                                  , cnp.ndarray[double, ndim=1] W):
        """ Method used to calculate the log_likelihood, its gradient 
            and hessian. 
            
            To take into account tied times, the log_likelihood 
            is defined using the Efron's approximation method.

            References:
            * https://en.wikipedia.org/wiki/Proportional_hazards_model#Tied_times
            * Efron, Bradley (1974). "The Efficiency of Cox's Likelihood 
            Function for Censored Data". Journal of the American Statistical 
            Association. 72 (359): 557–565. 
        """
        
        cdef:
            # Variables related to integers used in loops
            size_t i, j
            vector[int] order
            size_t N = X.shape[0]
            size_t P = X.shape[1]
            
            # Variables related to the risk score
            double theta_i = 0.
            cnp.ndarray[double, ndim=1] theta_x_i = np.zeros(P)
            cnp.ndarray[double, ndim=2] theta_x_x_i = np.zeros((P,P)) 

            # Variables related to the at risk units
            double sum_risk_theta = 0.
            cnp.ndarray[double, ndim=1] sum_risk_theta_x = np.zeros(P)
            cnp.ndarray[double, ndim=2] sum_risk_theta_x_x = np.zeros((P,P))

            # Variables related to the fail units
            int nb_fails = 0
            cnp.ndarray[double, ndim=1] sum_fail_x = np.zeros(P)
            double sum_fail_theta = 0.
            cnp.ndarray[double, ndim=1] sum_fail_theta_x = np.zeros(P)
            cnp.ndarray[double, ndim=2] sum_fail_theta_x_x = np.zeros((P,P))

            # Variables related to derivatives
            double phi_j, temp_loss, loss = 0.
            cnp.ndarray[double, ndim=1] Z_j, temp_gradient
            cnp.ndarray[double, ndim=1] gradient = np.zeros(P)
            cnp.ndarray[double, ndim=2] ZZ_j, temp_Hessian
            cnp.ndarray[double, ndim=2] Hessian = np.zeros((P,P))

        # Sorting X, T, E in descending order according to T
        order = argsort(T, True)
        T = T[order]
        E = E[order]
        X = X[order, :]

        # Looping through the data to calculate the derivatives
        for i in range(N):

            # Calculating theta_i = exp(<x_i, W>); theta_x_i = x_i*theta_i
            # and theta_x_x_i = <x_i, x_i.T>*theta_i
            x_i = X[i, :]
            theta_i = exp( np.dot(x_i, W) )
            theta_x_i = theta_i*x_i
            theta_x_x_i = np.multiply(theta_x_i.reshape(-1, 1), x_i.T)

            # Calculating the at risk variables
            sum_risk_theta += theta_i
            sum_risk_theta_x += theta_x_i
            sum_risk_theta_x_x += theta_x_x_i

            # Calculating the fail variables
            if E[i] == 1:
                sum_fail_x += x_i 
                sum_fail_theta += theta_i
                sum_fail_theta_x += theta_x_i
                sum_fail_theta_x_x += theta_x_x_i
                nb_fails += 1

            if i < N-1 and T[i] == T[i+1] :
                continue

            if nb_fails == 0:
                continue

            # Calculating temp derivatives
            temp_loss = 0.
            temp_gradient = np.zeros(P)
            temp_Hessian = np.zeros((P,P))
            for j in range(nb_fails):

                phi_j  = sum_risk_theta - sum_fail_theta*j*1./nb_fails
                Z_j    = sum_risk_theta_x - sum_fail_theta_x*j*1./nb_fails
                ZZ_j   = sum_risk_theta_x_x - sum_fail_theta_x_x*j*1./nb_fails

                temp_loss += log(phi_j)
                temp_gradient += Z_j/phi_j
                temp_Hessian += ZZ_j/phi_j
                temp_Hessian -= np.multiply(Z_j.reshape(-1, 1)/phi_j, Z_j.T/phi_j)

            loss += np.dot(sum_fail_x, W) - temp_loss
            gradient += sum_fail_x - temp_gradient
            Hessian -= temp_Hessian

            # Re-initializing the fail variables
            sum_fail_x = np.zeros(P)
            sum_fail_theta = 0
            sum_fail_theta_x = np.zeros(P)
            sum_fail_theta_x_x = np.zeros((P,P))
            nb_fails = 0
                
        # Saving attributes
        self.loss     = -loss
        self.gradient = -gradient
        self.Hessian  = -Hessian
        
        
       
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    cpdef void newton_optimization(self  , cnp.ndarray[double, ndim=2] X
                                         , cnp.ndarray[double, ndim=1] T
                                         , cnp.ndarray[double, ndim=1] E
                                         , cnp.ndarray[double, ndim=1] W 
                                         , double lr
                                         , double l2_reg
                                         , double tol = 1e-3
                                         , double epsilon =1e-9
                                         , int max_iter = 100
                                         , bool verbose = True):
        """ This method perfoms the Newton-Raphson optimization so as
            to calculate the best model parameters :
                W_new = W_old - lr*<Hessian^(-1), gradient>
            
            The optimization is conducted by a while loop whose condition is
            ||gradient||^2 < tol
        """

        cdef:
            # Variables related to the while loop/optimization
            size_t i, n_iter = 0
            str error = ""
            vector[double] loss_values 
            vector[double] grad2_values
                        
            # Variables related to the parameters
            cnp.ndarray[double, ndim=1] W_old, W_new
            size_t P = W.shape[0]
            
            # Variables related to the loss, gradient & Hessian
            double  diff_norm_W, loss , grad2
            cnp.ndarray[double, ndim=1] gradient 
            cnp.ndarray[double, ndim=2] Hessian   
            
        # Performing Newton-Raphson optimization
        grad2 = INFINITY
        W_old = W
        if verbose:
            print("Performing Newton-Raphson optimization")
        
        while grad2 > tol and n_iter < max_iter:

            # Getting derivatives
            self.get_derivatives(X, T, E, W_old)

            # Adjusting for regularization
            loss     = self.loss + l2_reg*norm(W_old)/2.
            gradient = self.gradient + l2_reg*W_old
            Hessian  = self.Hessian + l2_reg*np.eye(P)

            # Calculating the new W
            inv_Hessian = np.linalg.inv(Hessian+ epsilon*np.eye(P))
            W_new = W_old - lr*np.dot(inv_Hessian, gradient)

            # Stopping the optimization if gradient exploded
            diff_norm_W = norm(W_new - W_old)
            if np.isnan( diff_norm_W ) or np.isinf( diff_norm_W ):
                self.W = W_old
                error =  "The gradient exploded."
                error += "You should reduce the learning rate (lr) or "
                error += "try a different initialization."
                print(error)
                break
                
            # Moving to the next iteration
            W_old = W_new
            grad2 = norm(gradient)
            loss_values.push_back( loss )
            grad2_values.push_back( grad2 )
            n_iter += 1

            # Reducing size of learning rate 
            if diff_norm_W > 10:
                lr *= 0.9
            
            if verbose:
                iteration  = " * Iteration #{}"
                iteration += " - Loss = {:.3f}"
                iteration += " - ||grad||^2 = {:.5f}"
                print(iteration.format(n_iter, loss, grad2))           

        # Printing out final results
        if verbose and error == "":
            if n_iter == max_iter :
                print("Optimization reached max number of iterations.")
            else:
                print("Converged after {} iterations.".format(n_iter))
                
        # Saving attributes
        self.W = W_new
        self.inv_Hessian = inv_Hessian
        self.loss_values  = loss_values
        self.grad2_values = grad2_values
    
