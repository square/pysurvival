#%%cython --a  
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
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp cimport bool
from libc.stdlib cimport rand, srand, RAND_MAX

# Importing math functions
#--------------------------
cdef extern from "<math.h>" nogil:
    double fmax(double x, double y)
    double exp(double x)
    double sqrt(double x)
    double log(double x)
    double cos(double x)
    double fabs(double x)
    double tanh(double x)
    double M_PI
    double pow(double x, double y)
    double NAN
    
    
# -------------------------------- FUNCTIONS -------------------------------- #
cdef double tri_dot(  cnp.ndarray[double, ndim=1] x
                    , cnp.ndarray[double, ndim=2] A
                    , cnp.ndarray[double, ndim=1] y):
    """ Calculating <x, <A, y> > """
    return np.dot(x, np.dot(A, y))


cdef double sigmoid( double x):
    """ Calculating the sigmoid function """
    return 1./(1. + exp(-x) )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef double norm(vector[double] x, 
                 int d = 2,
                 bool with_sqrt=False ):
    """ Calculating norm 2 of a vector """
    cdef:
        double result = 0.
        size_t i, N = x.size()
        
    if d == 2:
        for i in range(N):
            result += x[i]*x[i]

        if with_sqrt:
            return sqrt(result)
        else:
            return result    
    
    elif d == 1:
        for i in range(N):
            result += fabs(x[i])
        return result
    
    
# --------------------------------- CLASSES --------------------------------- #
cdef class _SVMModel:
    """ This object will help speed up the calculation of the SurvivalSVM model.
        It will allow us to calculate the log_likelihood, its gradient 
        and hessian to perform the Newton-Raphson optimization.
    """
    
    # Attributes
    #-----------
    cdef :

        # Model parameters
        cnp.ndarray W, internal_X
        cnp.ndarray Kernel_Matrix
        int kernel_type
        double scale, offset, degree

        # Log_likelihood
        double loss
        cnp.ndarray gradient
        cnp.ndarray Hessian, inv_Hessian
        vector[double] loss_values, grad2_values

    # W
    @property
    def W(self):
        return self.W
    
    @W.setter
    def W(self, W):
        self.W = W

    # Kernel_Matrix
    @property
    def Kernel_Matrix(self):
        return self.Kernel_Matrix

    @Kernel_Matrix.setter
    def Kernel_Matrix(self, Kernel_Matrix):
        self.Kernel_Matrix = Kernel_Matrix

    # kernel_type
    @property
    def kernel_type(self):
        return self.kernel_type

    @kernel_type.setter
    def kernel_type(self, kernel_type):
        self.kernel_type = kernel_type

    # scale
    @property
    def scale(self):
        return self.scale

    @scale.setter
    def scale(self, scale):
        self.scale = scale

    # offset
    @property
    def offset(self):
        return self.offset

    @offset.setter
    def offset(self, offset):
        self.offset = offset

    # scale
    @property
    def scale(self):
        return self.scale

    @scale.setter
    def scale(self, scale):
        self.scale = scale

    # degree
    @property
    def degree(self):
        return self.degree

    @degree.setter
    def degree(self, degree):
        self.degree = degree

    # loss
    @property
    def loss(self):
        return self.loss

    @loss.setter
    def loss(self, loss):
        self.loss = loss

    # inv_Hessian
    @property
    def inv_Hessian(self):
        return self.inv_Hessian

    @inv_Hessian.setter
    def inv_Hessian(self, inv_Hessian):
        self.inv_Hessian = inv_Hessian

    # loss_values
    @property
    def loss_values(self):
        return self.loss_values

    @loss_values.setter
    def loss_values(self, loss_values):
        self.loss_values = loss_values

    # grad2_values
    @property
    def grad2_values(self):
        return self.grad2_values

    @grad2_values.setter
    def grad2_values(self, grad2_values):
        self.grad2_values = grad2_values

    # internal_X
    @property
    def internal_X(self):
        return self.internal_X

    @internal_X.setter
    def internal_X(self, internal_X):
        self.internal_X = internal_X
        
    # Methods
    #--------    
    def __init__(self, int kernel_type = 0
                     , double scale = 1.
                     , double offset=0.
                     , double degree=1):
        self.kernel_type= kernel_type 
        self.scale = scale  
        self.offset= offset 
        self.degree= degree        
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    cdef void get_derivatives(self , cnp.ndarray[double, ndim=2] X
                                   , cnp.ndarray[double, ndim=1] T
                                   , cnp.ndarray[double, ndim=1] E
                                   , cnp.ndarray[double, ndim=1] W
                                   , double l2_reg):    
        """ This method computes the Squared Hinge Loss function described in 
            Fast Training of Survival Support Vector Machine with Ranking 
            Constraints as well as its gradient and Hessian matrix.
        """
                                 
        cdef:
            # Variables related to the while loop/optimization
            int i, s, N = X.shape[0]
            
            # Variables related to derivatives computation
            cnp.ndarray[double, ndim=2] Hessian, AAK_matrix
            cnp.ndarray[double, ndim=2] AA_matrix = np.zeros((N, N))
            cnp.ndarray[double, ndim=1] gradient, A_vector  = np.zeros(N) 
            cnp.ndarray[double, ndim=1] K 
            double loss, l_plus, l_minus, m = 0.

        # Computing K for a linear/non-linear kernel
        if self.kernel_type != 0 :
            K = np.dot(self.Kernel_Matrix, W.reshape(-1,1)).flatten()
        else:
            K = np.dot(X, W)
            
        # Looping through all the pairs
        for i in range( N ):
            
            # Testing if the pair can be used for computation
            l_plus = 0.
            l_minus = 0.
            for s in range( N ):

                if i != s :
                    if T[s] > T[i] and K[s] < K[i] + 1 and E[i] == 1 :
                        l_plus  += 1.
                    if T[s] < T[i] and K[s] > K[i] - 1 and E[s] == 1 :
                        l_minus += 1.
                    AA_matrix[i,s] = -1

            # Calculating AA_matrix and A_vector
            AA_matrix[i, i] = l_minus + l_plus 
            A_vector[i] = l_minus - l_plus 
            m += l_minus

        # Computing loss, gradient and Hessian for non linear kernels
        if self.kernel_type != 0:
            AAK_matrix = np.dot(AA_matrix, self.Kernel_Matrix)
            loss = 0.5*tri_dot(W.T, self.Kernel_Matrix, W) 
            loss += (l2_reg/2.)*(m + tri_dot(W.T, self.Kernel_Matrix, \
                    ( np.dot(AAK_matrix, W) -2.*A_vector ))) 
            gradient = np.dot(self.Kernel_Matrix, W) 
            gradient += l2_reg*np.dot( self.Kernel_Matrix, \
                        ( np.dot(AAK_matrix, W) - A_vector ))
            Hessian  = self.Kernel_Matrix + l2_reg*np.dot( self.Kernel_Matrix, \
                        AAK_matrix )

        # Computing loss, gradient and Hessian for linear kernels
        else:
            AAK_vector = np.dot(AA_matrix, K)
            loss     = 0.5*np.dot(W.T,W) 
            loss    +=(l2_reg/2.)*(m + (tri_dot(W.T, X.T, \
                      (AAK_vector -2.*A_vector ))) )
            gradient = W + l2_reg*np.dot( X.T,  (AAK_vector  - A_vector ))
            Hessian  = np.identity(W.shape[0])
            Hessian  += l2_reg*np.dot(X.T, np.dot(AA_matrix, X))

        # Saving the attributes
        self.loss     = loss 
        self.gradient = gradient
        self.Hessian  = Hessian 

        
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
            str error = ''
            vector[double] loss_values 
            vector[double] grad2_values
                        
            # Variables related to the parameters
            cnp.ndarray[double, ndim=1] W_old, W_new
            size_t P = W.shape[0]
            
            # Variables related to the loss, gradient & Hessian
            double  diff_norm_W, loss , grad2
            cnp.ndarray[double, ndim=1] gradient 
            cnp.ndarray[double, ndim=2] Hessian   
            
        # Saving the X matrix
        self.internal_X = X
        
        if self.kernel_type != 0:
            self.get_kernel_matrix(X)
            
        # Performing Newton-Raphson optimization
        INFINITY = 10^8
        grad2 = INFINITY
        diff_norm_W = INFINITY
        W_old = W
        if verbose:
            print("Performing Newton-Raphson optimization: ")
        
        while grad2 > tol and n_iter < max_iter:

            # Getting derivatives
            self.get_derivatives(X, T, E, W_old, l2_reg)

            # Adjusting for regularization
            loss     = self.loss + l2_reg*norm(W_old)/2.
            gradient = self.gradient + l2_reg*W_old
            Hessian  = self.Hessian + l2_reg*np.eye(P)

            # Calculating the new W
            inv_Hessian = np.linalg.inv(Hessian) #+ epsilon*np.eye(P))
            W_new = W_old - np.dot(inv_Hessian, lr*gradient)

            # Stopping the optimization if gradient exploded
            diff_norm_W = norm(W_new - W_old, 1)
            if np.isnan( diff_norm_W ) or np.isinf( diff_norm_W ) :
                self.W = W_old
                error =  "The gradient exploded. "
                error += "You should reduce the learning rate (lr) or "
                error += "try a different initialization."
                print(error)
                break
                
            # Moving to the next iteration
            W_old = W_new
            grad2 = norm(gradient, True)
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
                iteration += " - ||diff_w|| = {:.5f}"
                print(iteration.format(n_iter, loss, grad2, diff_norm_W))           

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


    cdef double kernel_function(   self
                                 , cnp.ndarray[double, ndim=1] x
                                 , cnp.ndarray[double, ndim=1] y):
        """ Calculating one coefficient of the Kernel Matrix 
            KERNELS = { 'Linear': 0, 'Polynomial': 1, 'Gaussian':2, 'Normal':2,
                        'Exponential':3, 'Tanh':4, 'Sigmoid': 5, 
                        'Rational Quadratic':6, 'Inverse Multiquadratic': 7,  
                        'Multiquadratic': 8}
        """

        cdef:
            double scale = self.scale
            double offset = self.offset
            double degree = self.degree
            
        if self.kernel_type == 1: #"polynomial":
            return pow(scale*np.dot(x, y) + offset, degree)        

        elif self.kernel_type == 2: #"gaussian" 
            return exp( -0.5*fabs(scale)*norm(x-y, 2) )

        elif self.kernel_type == 3: #"exponential" :
            return exp( -fabs(scale)*norm(x-y, 1) )

        elif self.kernel_type == 4: #"tanh":
            return tanh(scale*np.dot(x, y) + offset)

        elif self.kernel_type == 5: #"sigmoid":
            return sigmoid(scale*np.dot(x, y)+ offset)

        elif self.kernel_type == 6: #"rational_quadratic" :
            return offset*1./( norm(x-y, 2) + offset )

        elif self.kernel_type == 7: #"inverse_multiquadratic" :
            return 1./sqrt( norm(x-y, 2) + offset )

        elif self.kernel_type == 8: #"multiquadratic" :
            return  sqrt( norm(x-y, 2) + offset )


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=2] get_kernel_matrix( self
            , cnp.ndarray[double, ndim=2] X):
        """ Calculating the Kernel Matrix of the features X, such that:
                K[i, j] = f(X[i, :], X[j, :])

            An exhaustive list of kernels can be found here:
            http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning\
            -applications/
        """

        cdef :
            size_t i, j, N = X.shape[0]
            cnp.ndarray[double, ndim=1] x_i, x_j
            cnp.ndarray[double, ndim=2] K = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1):
                x_i = X[i, :]
                x_j = X[j, :]
                K[i, j] = self.kernel_function(x_i, x_j)
                
        self.Kernel_Matrix = np.tril(K) + np.tril(K, -1).T      
        return self.Kernel_Matrix


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    cpdef cnp.ndarray[double, ndim=1] get_score(self, 
            cnp.ndarray[double, ndim=2] z):
        """ Calculating the risk score of the given data points """

        # Declaring the variables
        cdef:
            # Loops and sizes
            size_t i, j, M = z.shape[0]
            size_t N = self.internal_X.shape[0]
            cnp.ndarray[double, ndim=1] kernel_vector = np.zeros(N)
            cnp.ndarray[double, ndim=1] result = np.zeros(M)
            
        if self.kernel_type == 0:
            result = np.dot(z, self.W.reshape(-1,1) ).flatten()
        else:
            for j in range(M):
                for i in range(N):
                    kernel_vector[i] = self.kernel_function(self.internal_X[i,:]
                                                            , z[j, :])
                result[j] =  np.dot(kernel_vector, self.W)

        return result

