#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

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

using namespace std;


/* ------------------------------- Functions ------------------------------- */
vector<int>  argsort(vector<double> v, bool descending);

vector<pair<double, double> > get_time_buckets(vector<double> times );

long random_int(long low, long high);

double random_double(double low, double high);
 
size_t get_nb_unique_values(vector<double> x);

vector<double> reverse(vector<double> x);

int argmin_buckets(double x, vector<pair<double, double> > buckets);

vector<double> remove_duplicates(vector<double> x);

double max_function(vector<double> vector);

vector<double> Mv_dot_product(vector<vector<double> > M, vector<double> v);

vector<double> cumsum( vector<double> v);

map< int, vector<double> > baseline_functions(vector<double> score,
											  vector<double> T, 
											  vector<double> E);

std::vector<double> logrankScores(std::vector<double> time, std::vector<double> status);

#endif /* FUNCTIONS_H_*/