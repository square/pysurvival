#include <random>     // mt19937 and uniform_int_distribution
#include <algorithm>  // generate
#include <vector>     // vector
#include <iterator>   // begin, end, and ostream_iterator
#include <functional> // bind
#include <iostream>   // cout
#include <math.h>       /* exp */
#include <numeric>
#include <iomanip>
#include <map>
#include <utility>
#include <set>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <numeric>
#include <sstream>
#include <limits>
#include "non_parametric.h"
#include "functions.h"
#include "metrics.h"

using namespace std;



map<int, double> concordance_index(vector<double> risk, vector<double> T, 
								   vector<double> E, bool include_ties = true){

	/** Computing the C-index based on *On The C-Statistics For Evaluating Overall
		Adequacy Of Risk Prediction Procedures With Censored Survival Data* and
		*Estimating the Concordance Probability in a Survival Analysis
		with a Discrete Number of Risk Groups*

		This function assumes that risk, T and E have been sorted 
		in a descending order according to T.
	*/

	// Initializing 
	size_t i, j, N = risk.size();
	vector<double> weights, E_1, censored_survival;
	double C, w, weightedPairs, weightedConcPairs;
	map<int, double > results;
	KaplanMeierModel censored_km;
	int c_index = 0; //'c_index'
	int nb_pairs = 1; //'nb_pairs'
	int nb_concordant_pairs = 2; //'nb_concordant_pairs'

	// Creating the weights
	weights.resize(N, 1.);

	// 	Initializing/Fitting the KM model
	censored_survival = censored_km.fit(T, E, weights, 0., true);

	// Initializing temp variables used for the computation of the c-index
	weightedConcPairs=0.;/* weighted concordant pairs */
	weightedPairs=0. ; /* weighted pairs */

	// Looping through the data to calculate the c-index
	for (i = 0; i < N; ++i){

		// Only if i experienced an event
		if( E[i]==1){

			for (j = 0; j < N; ++j){

				if (j!=i){
					w  = censored_km.predict_survival(T[i], false);
					w *= censored_km.predict_survival(T[i], true);

					// count pairs 
					if( (T[i]<T[j]) | (T[j]==T[i] & E[j]==0) ){
						weightedPairs += 1./w;

						// concordant pairs
						if (risk[i] > risk[j]){
							weightedConcPairs += 1./w;
						}

						// pairs with equal predictions count 1/2 or nothing
						if ((risk[i] == risk[j]) & include_ties){
							weightedConcPairs += (1./w)/2.;
						}

					}
				}
			}
		}
	}

	// Saving results
	C = weightedConcPairs/weightedPairs ;
	C = fmax(C, 1. - C);
	results[c_index] = C;
	results[nb_pairs] = 2*weightedPairs;
	results[nb_concordant_pairs] = 2*weightedConcPairs;
	return results;
}



map<int, vector<double> > brier_score(vector<vector<double> > Survival,
								  	  vector<double> T, vector<double> E, 
								  	  double t_max, vector<double> times,
								      vector<pair<double, double> > time_buckets,
								      bool use_mean_point){

	/** Computing the Brier score at a given time t;
		it represents the average squared distances between 
		the observed survival status and the predicted
		survival probability.

		In the case of right censoring, it is necessary to adjust
		the score by weighting the squared distances to 
		avoid bias. It can be achieved by using 
		the inverse probability of censoring weights method (IPCW),
		(proposed by Graf et al. 1999; Gerds and Schumacher 2006)
		by using the estimator of the conditional survival function
		of the censoring times calculated using the Kaplan-Meier method,
		such that :
		BS(t) = 1/N*( W_1(t)*(Y_1(t) - S_1(t))^2 + ... + 
					  W_N(t)*(Y_N(t) - S_N(t))^2)

		This function assumes that risk, T and E have been sorted 
		in a descending order according to T.
	*/

	// Initializing variables 
	size_t i, j, M, N = Survival.size();
	double censored_s, bs, t, S;
	map<int, vector<double> > results;
	int n, times_ = 0; //'times'
	int brier_scores_ = 1; //'brier_scores'
	KaplanMeierModel censored_km;
	vector<double> weights_km, times_to_consider, brier_scores_values;

	// Creating the Censored KM model to get the IPCW
	weights_km.resize(N, 1.);
	censored_km = KaplanMeierModel();
	censored_km.fit(T, E, weights_km, 0., true);

	// Initializing/computing the brier score vector
	M = times.size();
	size_t Nt = time_buckets.size();
	for (j = 0; j < M; ++j){
		bs = 0.;

		// Extracting the time of interest
		t = times[j];	


		if(t<=t_max){

			n = argmin_buckets(t, time_buckets);

			// Looping through each unit
			for (i = 0; i < N; ++i){

				S = Survival[i][n];
				if(use_mean_point){
					if(n<N-1){
						S = (Survival[i][n] + Survival[i][n+1])/2. ;
					}
				}
				if( T[i] <= t){
					censored_s = censored_km.predict_survival(T[i], true);
					censored_s = fmax(censored_s, 1e-4);
					bs += E[i]*S*S/ censored_s /N;
				}
				else{
					censored_s = censored_km.predict_survival(t, false);
					censored_s = fmax(censored_s, 1e-4);
					bs += (1-S)*(1-S)/censored_s/N;
				}
			}

			// Saving the times and the brier scores
			times_to_consider.push_back(t);
			brier_scores_values.push_back(bs);
		}
	}

	// Results
	results[times_] = times_to_consider;
	results[brier_scores_] = brier_scores_values;
	return results;
}








map<int, vector<double> > time_ROC(vector<double> risk, vector<double> T,
								   vector<double> E, double t){

	/** Time-Dependent ROC Curve and AUC for Censored Survival Data 
	
		This fucntion calculates the Inverse Probability of Censoring 
		Weighting (IPCW) estimation of Cumulative/Dynamic time-dependent 
		ROC curve. 
		It assumes that risk, T, and E have been sorted in a descending 
		order according to T.
		
		Parameters:
		-----------
		* risk : array-like, shape = [n_samples]
			The risk scores for each unit.

		* T : array-like, shape = [n_samples] 
			The target values describing when the event of interest or censoring
			occured

		* E : array-like, shape = [n_samples] 
			The Event indicator array such that E = 1. if the event occured
			E = 0. if censoring occured
			
		* t: double
			The time at which we want to compute the time-dependent ROC curve.
	*/

	// Initializing 
	size_t i, j, N = T.size();	
	vector<double> weights, censored_Surv_KM; 
	vector<double> weighted_cases, TP_vector, weighted_controls, FP_vector;
	vector<int> order_risk;
	double den_TP_t, den_FP_t, S_censored;
	map<int, vector<double> > results;
	int TP = 0;// 'TP'
	int FP = 1;// 'FP'
	KaplanMeierModel censored_km;

	// Modeling the Censored KM for the IPCW 
	weights.resize(N, 1.);
	censored_km = KaplanMeierModel();
	censored_Surv_KM = censored_km.fit(T, E, weights, 0., true);

	// Sorting the risk by descending order
	order_risk = argsort(risk, true);
	den_TP_t = 0.;
	den_FP_t = 0.;
	weighted_cases.resize(N, 0.);
	S_censored = fmax(censored_km.predict_survival(t, false), 1e-4);
	weighted_controls.resize(N, 1./(N*S_censored) );

	for (j = 0; j < N; ++j){

		// Extracting the sorted index
		i = order_risk[j];

		// Calculating the True Positive Rates
		if ( (T[i] < t) and (E[i] == 1.) ){
			S_censored = fmax(censored_km.predict_survival(T[i], true), 1e-4);
			weighted_cases[j] = 1./(N*S_censored);
		}
		den_TP_t += weighted_cases[j];

		// Calculating the False Positive rates
		if ( T[i] <= t ){
			weighted_controls[j] = 0.;
		}
		den_FP_t += weighted_controls[j];
	}

	// Creating the finalized TP and FP
	TP_vector.resize(N+1, 0.);
	FP_vector.resize(N+1, 0.);
	for (i = 0; i < N; ++i){
		TP_vector[i+1] = TP_vector[i] + (weighted_cases[i])/den_TP_t;
		FP_vector[i+1] = FP_vector[i] + (weighted_controls[i])/den_FP_t;
	}
	results[TP] = TP_vector;
	results[FP] = FP_vector;
	return results;
}
