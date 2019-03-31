#ifndef METRICS_H_
#define METRICS_H_
#include <algorithm>  // generate
#include <vector>     // vector
#include <iterator>   // begin, end, and ostream_iterator
#include <iostream>   // cout
#include <iomanip>
#include <map>
#include <utility>
#include <set>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <numeric>
#include <sstream>

using namespace std;


map<int, double> concordance_index(vector<double> risk, vector<double> T, 
								   vector<double> E, bool include_ties);

map<int, vector<double> > brier_score(vector<vector<double> > Survival,
								  	 vector<double> T, vector<double> E, 
								  	 double t_max, vector<double> times, 
								     vector<pair<double,double> > time_buckets,
								     bool use_mean_point);

map<int, vector<double> > time_ROC(vector<double> risk, vector<double> T,
								   vector<double> E, double t);

#endif /* METRICS_H_*/