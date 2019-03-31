#ifndef NON_PARAMETRIC_H_
#define NON_PARAMETRIC_H_
#include <algorithm>  // generate
#include <vector>     // vector
#include <iterator>   // begin, end, and ostream_iterator
#include <functional> // bind
#include <iostream>   // cout
#include <ctime>      //clock
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
#include "functions.h"

using namespace std;

class NonParametricModel{

	public:

		// Initialization
		NonParametricModel();
  	    virtual ~NonParametricModel();

  		// Attributes
        vector<double> times;
        vector<pair<double, double> > time_buckets;
        vector<double> survival, hazard, cumulative_hazard, std_error;
        vector<double> survival_ci_upper, survival_ci_lower;

        // Methods
        double predict_survival(double t, bool is_lagged);
        double predict_survival_upper(double t, bool is_lagged);
        double predict_survival_lower(double t, bool is_lagged);
        double predict_density(double t, bool is_lagged);
        double predict_hazard(double t, bool is_lagged=false);
        vector<double> fit(vector<double> T, vector<double> E, 
                       vector<double> weights, double z);
};



class KaplanMeierModel: public NonParametricModel{

	public:

		// Initialization
		KaplanMeierModel();
  	    virtual ~KaplanMeierModel();

		// Attributes
        vector<double> at_risk, events;
        vector<double> fit(vector<double> T, vector<double> E, 
                       vector<double> weights, double z, bool ipcw);
};



class KernelModel: public NonParametricModel{

	public:

    		// Initialization
        KernelModel();
        KernelModel(double b , int kernel_type){
            this->b = b;
            this->kernel_type = kernel_type;  
            this->km_model = KaplanMeierModel();
        };
      	virtual ~KernelModel();

        // Attributes
        int kernel_type;
        KaplanMeierModel km_model;
        double b;
        vector<vector<double> > kernel_matrix;
        vector<pair<double,double> > time_intervals, time_buckets;
        vector<double> density, km_survival, km_times;

        // Methods
        vector<double> fit(vector<double> T, vector<double> E, 
                           vector<double> weights, double z);
        void get_times(vector<double> T);
};


#endif /* NON_PARAMETRIC_H_*/