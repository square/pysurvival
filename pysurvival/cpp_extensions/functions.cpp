#include <algorithm>  // generate
#include <vector>     // vector
#include <iterator>   // begin, end, and ostream_iterator
#include <functional> // bind
#include <iostream>   // cout
#include <ctime> 	  //clock
#include <math.h>       /* exp */
#include <map>
#include <utility>
#include <set>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <numeric>
#include <limits>
#include <random>
#include <math.h>       /* fabs */
#include "functions.h"

using namespace std;


/* ------------------------------- Functions ------------------------------- */
vector<int>  argsort(vector<double> v, bool descending){
	//Initializing
    vector<int> idx;
    vector<pair<double, int> > a;
    size_t n = v.size();
    vector<double> temp_v;

    if(descending){
    	for (int i = 0; i < n; ++i){
    		temp_v.push_back(-v[i]);
    	}
    	v = temp_v;
    }

    for (int i = 0; i < n; ++i){
    	a.push_back(make_pair(v[i], i));
    }

    //sort indexes based on comparing values in v
    sort(a.begin(),a.end());
    for (int i = 0; i < n; ++i){
    	idx.push_back( a[i].second );
    }
    return idx;
}

vector<pair<double, double> > get_time_buckets(vector<double> times ){
    // Initializing the variables
    size_t N = times.size();
    vector<pair<double, double> > results;

    // Computing the time buckets
    for (int i = 0; i < N-1; ++i){
        results.push_back(make_pair(times[i], times[i+1]));
    }

    // Saving the attributes
    return results;
 }


long random_int(long low, long high){
	/** 
	 Random number generator that produces integer values according 
	 to a uniform discrete distribution in the interval [low, high] inclusive
	*/
	long seed_value = long(clock()); 
	mt19937 engine;
	engine.seed(seed_value);
	uniform_int_distribution<long> unif(low, high);
	return  unif(engine) ; 
}


double random_double(double low, double high){
	/** 
	 Random number generator that produces integer values according 
	 to a uniform discrete distribution in the interval [low, high] inclusive
	*/
	long seed_value = long(clock()); 
	mt19937 engine;
	engine.seed(seed_value);
	uniform_real_distribution<double> unif(low, high);
	return  unif(engine) ; 
}
 

size_t get_nb_unique_values(vector<double> x){
	x.erase( unique( x.begin(), x.end() ), x.end() );
	return x.size();
}


vector<double> reverse(vector<double> x){
	reverse(x.begin(),x.end());
	return x;
}


int argmin_buckets(double x, vector<pair<double, double> > buckets){
	size_t index_x = 0, J = buckets.size();
	double a, min_value = numeric_limits<double>::max();
	for (int j = 0; j < J; ++j){
		a = buckets[j].first;
		if(fabs(x-a)<= min_value){
			min_value = fabs(x-a);
			index_x = j;
		}
	}
	return  index_x;
}



vector<double> remove_duplicates(vector<double> x){
	x.erase( unique( x.begin(), x.end() ), x.end() );
	return x;
}


double max_function(vector<double> vector){
	double max = *max_element(vector.begin(), vector.end());
	return max;
}


vector<double> Mv_dot_product(vector<vector<double> > M, vector<double> v){
	size_t i, j, Ni, Nj;
	double dot;
	vector<double> results;

	Ni = M.size();
	Nj = M[0].size();
	results.resize(Nj, 0.);
	for (j = 0; j < Nj; ++j){
		dot = 0.;
		for (i = 0; i < Ni; ++i){
	 		dot += M[i][j]*v[i];
	 	}
	 	results[j] = dot;
	} 
	return results;
}


vector<double> cumsum( vector<double> v){
	double s=0.;
	size_t N = v.size();
	vector<double> results;
	results.resize(N, 0.);
	for (int i = 0; i < N; ++i){
		s += v[i];
		results[i] = s;
	}
	return results;
}


map< int, vector<double> > baseline_functions(vector<double> score,
											  vector<double> T, 
											  vector<double> E){
	/** This method provides the calculations to estimate 
		the baseline survival function. 
		
		The formula used to calculate the baseline hazard is:
			h_0( T ) = |D(T)|/Sum( exp( <x_j, W> ), j in R(T) ) where:
				- T is a time of failure
				- |D(T)| is the number of failures at time T
				- R(T) is the set of at risk uites at time T 
		https://github.com/cran/survival/blob/master/R/basehaz.R
		http://www.utdallas.edu/~pkc022000/6390/SP06/NOTES/survival_week_5.pdf
		https://stats.stackexchange.com/questions/46532/cox-baseline-hazard
	*/

	// Declaring variables
	size_t j, J, N = score.size();
	double sum_theta_risk = 0.;
	int nb_fails = 0;
	vector<double> baseline_hazard, baseline_survival, baseline_cumulative_hazard;
	map< int, vector<double> > results;

    // Ensuring that T and E are sorted in a descending order according to T.
    vector<double> times, T_temp, E_temp;
    vector<int> desc_index = argsort(T, true);
    int n;
    for (int i = 0; i < N; ++i){
        n = desc_index[i];
        T_temp.push_back(T[n]);
        E_temp.push_back(E[n]);
    }
    T = T_temp;
    E = E_temp;

	// Calculating the Baseline hazard function
    for (int i = 0; i < N; ++i){

		// Calculating the at risk variables
		sum_theta_risk += score[i];

		// Calculating the fail variables
		if (E[i] == 1){
			nb_fails += 1;
		}

		if (i < N-1 & T[i] == T[i+1]){
			continue;
		}

		if(nb_fails == 0){
			continue;
		}

		baseline_hazard.push_back( nb_fails*1./sum_theta_risk );
        times.push_back(T[i]);
		nb_fails = 0;
	}

	// Computing the Baseline survival function
	J = baseline_hazard.size();
	baseline_hazard = reverse(baseline_hazard);
	results[0] = reverse(times);
	results[1] = baseline_hazard;
	baseline_cumulative_hazard = cumsum(baseline_hazard);
	for (j = 0; j < J; ++j){
		baseline_survival.push_back( exp( - baseline_cumulative_hazard[j] ) );
	}
	results[2] = baseline_survival;
	return results;
}


std::vector<double> logrankScores(std::vector<double> time, std::vector<double> status) {
  size_t n = time.size();
  std::vector<double> scores(n);

  // Get order of timepoints
  std::vector<int> indices = argsort(time, false);

  // Compute scores
  double cumsum = 0;
  size_t last_unique = -1;
  for (size_t i = 0; i < n; ++i) {

    // Continue if next value is the same
    if (i < n - 1 && time[indices[i]] == time[indices[i + 1]]) {
      continue;
    }

    // Compute sum and scores for all non-unique values in a row
    for (size_t j = last_unique + 1; j <= i; ++j) {
      cumsum += status[indices[j]] / (n - i);
    }
    for (size_t j = last_unique + 1; j <= i; ++j) {
      scores[indices[j]] = status[indices[j]] - cumsum;
    }

    // Save last computed value
    last_unique = i;
  }

  return scores;
}
