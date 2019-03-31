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


# Importing C++ objects from functions.h
#---------------------------------------
cdef extern from "functions.h":

    cdef map[ int, vector[double] ] baseline_functions(vector[double] score,
                                                       vector[double] T, 
                                                       vector[double] E);

    cdef vector[pair[double, double] ] get_time_buckets(vector[double] times );

    cdef int argmin_buckets(double x, vector[pair[double, double] ] buckets);


    cdef vector[double] logrankScores(vector[double] time, 
                                      vector[double] status);

cpdef vector[pair[double, double] ] _get_time_buckets(vector[double] times):
    return get_time_buckets(times)


cpdef int _argmin_buckets(double x, vector[pair[double, double] ] buckets):
    return argmin_buckets(x, buckets)
    
cpdef vector[double] _logrankScores(vector[double] time,  vector[double] status):
    return logrankScores(time, status)



