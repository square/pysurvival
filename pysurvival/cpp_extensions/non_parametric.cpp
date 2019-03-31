#include <algorithm>  // generate
#include <vector>     // vector
#include <iterator>   // begin, end, and ostream_iterator
#include <functional> // bind
#include <iostream>   // cout
#include <ctime> 	  //clock
#include <math.h>       /* exp */
#include <map>
#include <stdexcept>
#include <utility>
#include <set>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <numeric>
#include <limits>
#include "non_parametric.h"
#include "functions.h"

using namespace std;


/* ------------------------- NonParametricModel ------------------------- */
NonParametricModel::NonParametricModel() {
}

NonParametricModel::~NonParametricModel() {
}

double NonParametricModel::predict_survival(double t, bool is_lagged=false){
	long index_t = argmin_buckets(t, this->time_buckets);
	if(is_lagged){
        if (index_t-1<0){
            index_t = 0;
        } else {
            index_t = index_t-1;
        }
	}
	return this->survival[index_t];
}

double NonParametricModel::predict_survival_upper(double t, bool is_lagged=false){
    long index_t = argmin_buckets(t, this->time_buckets);
    if(is_lagged){
        if (index_t-1<0){
            index_t = 0;
        } else {
            index_t = index_t-1;
        }
    }
    return this->survival_ci_upper[index_t];
}

double NonParametricModel::predict_survival_lower(double t, bool is_lagged=false){
    long index_t = argmin_buckets(t, this->time_buckets);
    if(is_lagged){
        if (index_t-1<0){
            index_t = 0;
        } else {
            index_t = index_t-1;
        }
    }
    return this->survival_ci_lower[index_t];
}


double NonParametricModel::predict_density(double t, bool is_lagged){
    long index_t = argmin_buckets(t, this->time_buckets);
    if(is_lagged){
        if (index_t-1<0){
            index_t = 0;
        } else {
            index_t = index_t-1;
        }
    }
    return this->survival[index_t]*this->hazard[index_t];
}

double NonParametricModel::predict_hazard(double t, bool is_lagged){
    long index_t = argmin_buckets(t, this->time_buckets);
    if(is_lagged){
        if (index_t-1<0){
            index_t = 0;
        } else {
            index_t = index_t-1;
        }
    }
    return this->hazard[index_t];
}


/* ------------------------- KaplanMeierModel ------------------------- */
KaplanMeierModel::KaplanMeierModel() {
}

KaplanMeierModel::~KaplanMeierModel() {
}


vector<double> KaplanMeierModel::fit(vector<double> T, vector<double> E,
                                     vector<double> weights, double z, bool ipcw){
    /** Building the Kaplan Meier Estimator. 

    This implementation is inspired by the source code for 
    statsmodels.duration.survfunc available in the package statsmodels
    http://www.statsmodels.org/
    */

    // Initialization
    size_t Nt, N = T.size();
    vector<double> at_risk, events, times, cens_times;
    vector<double> survival, hazard, std_error, cum_hazard;
    vector<double> survival_ci_upper, survival_ci_lower;
    double nb_at_risk = 0.;
    double nb_events = 0.;
    double risk_min=1e-8;
    double cum_hazard_new, cum_hazard_old, survival_new;
    double std_error_new, cum_std_error; 
    double hazard_new, survival_old;

    // Ensuring that T and E are sorted in a descending order according to T.
    vector<double> T_temp, E_temp, weights_temp;
    vector<int> desc_index = argsort(T, true);
    int n;
    for (int i = 0; i < N; ++i){
        n = desc_index[i];
        T_temp.push_back(T[n]);

        if(ipcw){
            E_temp.push_back(1.-E[n]);            
        } else{
            E_temp.push_back(E[n]);            
        }
        weights_temp.push_back(weights[n]);
    }
    T = T_temp;
    E = E_temp;
    weights = weights_temp;

    // Looping through the data to calculate Survival, hazard, 
    // Cumulative_Hazard and Variance 
    for (int i = 0; i < N; ++i){

        // Computing the at risk vector
        nb_at_risk += weights[i];

        // Computing the fail vector
        if(E[i] == 1){
            nb_events += weights[i];
        }

        if (i < N-1 & T[i] == T[i+1]){
            continue;
        } else{
            times.push_back(T[i]);
        }

        // Moving to next unique time
        at_risk.push_back( fmax(nb_at_risk, risk_min) );
        events.push_back( nb_events );
        nb_events = 0;
    }

    // Adding 0 if non present
    Nt = at_risk.size();
    if( T[N-1] != 0.){
        at_risk.push_back( at_risk[Nt-1] );
        events.push_back( 0 );
        times.push_back( 0 );
        Nt = at_risk.size();    
    }

    // Reversing the hazard, at_risk and events vectors
    at_risk = reverse(at_risk);
    events  = reverse(events);
    times   = reverse(times);

    // Calculating Cumulative_Hazard and Survival 
    cum_hazard_old = 0.;
    cum_std_error = 0.;
    survival_old = 1.;

    for (int j = 0; j < Nt; ++j){

        // Calculating hazard
        hazard_new = events[j]*1./at_risk[j];

        // Survival
        survival_new = 1.*survival_old*(1.-hazard_new);

        // Cumulative_Hazard
        cum_hazard_new = hazard_new + cum_hazard_old;

        // Variance
        if( at_risk[j] == events[j]){
            cum_std_error += 0.;
        } else{
            cum_std_error += hazard_new/(at_risk[j] - events[j]);
        }
        std_error_new = sqrt(cum_std_error)*survival_new;

        // Adding the new values to respective vectors
        hazard.push_back( hazard_new );        
        cum_hazard.push_back( cum_hazard_new );          
        survival.push_back( survival_new );
        std_error.push_back( std_error_new );
        survival_ci_upper.push_back(survival_new+z*std_error_new);
        survival_ci_lower.push_back(survival_new-z*std_error_new);

        // Moving to next iteration
        survival_old = survival_new;
        cum_hazard_old = cum_hazard_new;
    }

    // Printing message
    // string someString = "Model is built !";
    // printf("%s\n",someString.c_str());

    // Saving the attributes
    this->survival          = survival;
    this->hazard            = hazard;
    this->cumulative_hazard = cum_hazard;
    this->std_error         = std_error;
    this->times             = times;
    this->events            = events;
    this->at_risk           = at_risk;
    this->survival_ci_upper = survival_ci_upper;
    this->survival_ci_lower = survival_ci_lower;
    this->time_buckets = get_time_buckets(this->times);

    return survival;

}





/* ------------------------- KernelModel ------------------------- */
KernelModel::KernelModel(){
}

KernelModel::~KernelModel(){
}

void KernelModel::get_times(vector<double> T){
	/** Creating intervals by selecting a specific number of bins
        or with a step value that determines the difference between
        2 consecutives values 
    */

	// Initializing
	size_t n;
	vector<pair<double,double> > time_intervals;
    vector<double> times;
    double t1, t2, max_T = max_function(T);
    double step = this->b;

    t1 = 0.;
    t2 = 0.;
    n  = 0;
    while(t2 < max_T){
        t1 = step*n;
        t2 = step*(n+1);
        times.push_back(t1);
        time_intervals.push_back(make_pair(t1, t2));
        n += 1;
    }

    times.push_back( time_intervals.back().second );
    this->times = times;
    this->time_intervals = time_intervals;
}


vector<double> KernelModel::fit(vector<double> T, vector<double> E,
								vector<double> weights, double z = 0.){
	/** Fitting the Non Parametric Kernel model */

	// Initializing
    int size, i, j;
    vector<double> s_coefs, times, km_times, km_survival, kernel_vector;
    vector<vector<double> > kernel_matrix;
    int Nx, Nt = T.size();
    double x, b = this->b;
    int kernel = this->kernel_type;
    vector<double> survival, hazard, cumulative_hazard, survival_temp;
    double min_survival = 1e-8;
    double k_ = 0.;
    double bound = 1.;

    // Adjusting the value of the kernel bound
    if( kernel == 2){
    	bound = 5.;
    } 
    else if(kernel == 0){
    	bound = .9999;
    }
        
    // Defining time 
    this->get_times(T);
    times = this->times;
    Nx = times.size();
        
    // Building the Kaplan Meier model
    km_survival = this->km_model.fit(T, E, weights, 0., false);
    km_times = this->km_model.times;
    Nt = km_times.size();
    s_coefs.resize(Nt, 0.);

    for (i = 0; i < Nt-1; ++i){
    	s_coefs[i] = km_survival[i] - km_survival[i+1];
    }
    s_coefs[-1] = km_survival[-1];

    // check if the kernel matrix fits in memory
    size = this->times.size()*km_times.size();
    while(k_ < 10 & size/exp(k_*log(10)) > 1.){
    	k_+=1;
    }

    // Printing message
    if( k_ > 8){
        survival.push_back(0.);
        return survival;
    }

    // Calculating the kernel model
    kernel_vector.resize(Nx, 0.);
    kernel_matrix.resize(Nt, kernel_vector);
    for (j = 0; j < Nx; ++j){
    	for (i = 0; i < Nt; ++i){

    		x = (times[j] - km_times[i])/b;
    		if (fabs(x) <= bound){
    			
    			// Uniform kernel
    			if (kernel == 0){
    				kernel_matrix[i][j] = 0.5;
    			}

    			// Epanechnikov kernel
    			else if(kernel == 1){
    				kernel_matrix[i][j] = 0.75*(1. - pow(x, 2.) );
    			}

    			// Normal kernel
    			else if(kernel == 2){
    				kernel_matrix[i][j] = exp( -x*x/2.) / sqrt( 2* M_PI );
    			}

    			// Biweight kernel 
    			else if(kernel == 3){
    				kernel_matrix[i][j] = (15./16)* pow((1.-pow(x, 2.)), 2.);
    			}

    			// Triweight kernel
    			else if(kernel == 4){
    				kernel_matrix[i][j] = (35./32)*pow((1.-pow(x, 2.)), 3.);
    			}

    			// Cosine kernel
    			else if(kernel == 5){
    				kernel_matrix[i][j] = (M_PI/4.)*cos( M_PI*x/2. );
    			}
    		}
    	}
    }


    // Saving the attributes
    this->kernel_matrix = kernel_matrix;
    this->km_survival = km_survival;
    this->km_times = km_times;
    this->time_buckets = get_time_buckets(this->times);

    this->density = Mv_dot_product(this->kernel_matrix, s_coefs);
    size_t N = this->density.size();

    survival_temp = cumsum(this->density);
    for (auto s : survival_temp){
    	this->survival.push_back(1.-s);
    }

    for (int i = 0; i < N; ++i){
    	this->hazard.push_back( this->density[i]/max(this->survival[i], min_survival) );
    	this->cumulative_hazard.push_back(log(max(this->survival[i], min_survival)));
    }
    return this->survival;
}



