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


# Importing C++ functions from metrics.h
#----------------------------------------
cdef extern from "metrics.h":

	cdef map[int, double] concordance_index(vector[double] risk, 
		vector[double] T, vector[double] E, bool include_ties);

	cdef map[int, vector[double] ] brier_score(vector[vector[double]] Survival,
		vector[double] T, vector[double] E, double t_max, vector[double] times,
		vector[pair[double, double]] time_buckets, bool use_first);

	cdef map[int, vector[double] ] time_ROC(vector[double] risk, 
		vector[double] T, vector[double] E, double t);


cpdef map[int, double] _concordance_index(vector[double] risk, 
										 vector[double] T, 
										 vector[double] E, 
										 bool include_ties=True):
	# results[c_index] = C;
	# results[nb_pairs] = 2*weightedPairs;
	# results[nb_concordant_pairs] = 2*weightedConcPairs;
	return concordance_index(risk,  T,  E, include_ties)



cpdef map[int, vector[double] ] _brier_score(vector[vector[double]] Survival,
									vector[double] T, 
									vector[double] E, 
									double t_max,
									vector[double] times, 
									vector[pair[double, double]] time_buckets,
									bool use_mean_point):
	# results[times] = times_to_consider;
	# results[brier_scores] = bs; 
	return brier_score(Survival, T, E, t_max, times, time_buckets, use_mean_point)


cpdef map[int, vector[double] ] _timeROC(vector[double] risk, 
										 vector[double] T, 
										 vector[double] E, 
										 double t):
	# results[TP] = TP_vector;
	# results[FP] = FP_vector;
	return time_ROC(risk,  T, E, t)
