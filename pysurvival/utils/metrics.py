from __future__ import absolute_import
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error
from pysurvival.models.non_parametric import KaplanMeierModel
from pysurvival.utils._metrics import _concordance_index
from pysurvival.utils._metrics import _brier_score, _timeROC
from pysurvival import utils


def concordance_index(model, X, T, E, include_ties = True, 
    additional_results=False, **kwargs):
    """ 
    Computing the C-index based on *On The C-Statistics For Evaluating Overall
    Adequacy Of Risk Prediction Procedures With Censored Survival Data* and
    *Estimating the Concordance Probability in a Survival Analysis
    with a Discrete Number of Risk Groups* and *Concordance for Survival 
    Time Data: Fixed and Time-Dependent Covariates and Possible Ties in
    Predictor and Time 

    Similarly to the AUC, C-index = 1 corresponds to the best model 
    prediction, and C-index = 0.5 represents a random prediction.

    Parameters:
    -----------
    * model : Pysurvival object
        Pysurvival model

    * X : array-like, shape=(n_samples, n_features)
        The input samples.

    * E : array-like, shape = [n_samples] 
        The Event indicator array such that E = 1. if the event occured
        E = 0. if censoring occured
    
    * include_ties: bool (default=True)
        Specifies whether ties in risk score are included in calculations

    * additional_results: bool (default=False)
        Specifies whether only the c-index should be returned (False)
        or if a dict of values should returned. the values are:
            - c-index
            - nb_pairs
            - nb_concordant_pairs

    Returns:
    --------
        * results: double or dict (if additional_results = True)
            - results is the c-index (double) if additional_results = False
            - results is dict if additional_results = True such that
                results[0] = C-index;
                results[1] = nb_pairs;
                results[2] = nb_concordant_pairs;
                    
    Example:
    --------


    """

    # Checking the format of the data 
    risk = model.predict_risk(X, **kwargs)
    risk, T, E = utils.check_data(risk, T, E)

    # Ordering risk, T and E in descending order according to T
    order = np.argsort(-T)
    risk = risk[order]
    T = T[order]
    E = E[order]

    # Calculating th c-index
    results = _concordance_index(risk, T, E, include_ties)

    if not additional_results:
        return results[0]
    else:
        return results


def c_index(model, X, T, E, include_ties = True, additional_results=False):
    return concordance_index(model, X, T, E, include_ties, additional_results)


def brier_score(model, X, T, E, t_max=None, use_mean_point=True, **kwargs):
    """ 
    Computing the Brier score at all times t such that t <= t_max;
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

    In terms of benchmarks, a useful model will have a Brier score below 
    0.25. Indeed, it is easy to see that if for all i in [1,N], 
    if S(t, xi) = 0.5, then BS(t) = 0.25.

    Parameters:
    -----------
    * model : Pysurvival object
        Pysurvival model

    * X : array-like, shape=(n_samples, n_features)
        The input samples.

    * T : array-like, shape = [n_samples] 
        The target values describing when the event of interest or censoring
        occured

    * E : array-like, shape = [n_samples] 
        The Event indicator array such that E = 1. if the event occured
        E = 0. if censoring occured
    
    * t_max: float 
        Maximal time for estimating the prediction error curves. 
        If missing the largest value of the response variable is used.

    Returns:
    --------
        * (times, brier_scores):tuple of arrays
            -times represents the time axis at which the brier scores were 
              computed
            - brier_scores represents the values of the brier scores
                    
    Example:
    --------


    """
    # Checking the format of the data 
    T, E = utils.check_data(T, E)

    # computing the Survival function
    Survival = model.predict_survival(X, None, **kwargs)

    # Extracting the time buckets
    times = model.times
    time_buckets = model.time_buckets

    # Ordering Survival, T and E in descending order according to T
    order = np.argsort(-T)
    Survival = Survival[order, :]
    T = T[order]
    E = E[order]

    if t_max is None or t_max <= 0.:
        t_max = max(T)

    # Calculating the brier scores at each t <= t_max
    results = _brier_score(Survival, T, E, t_max, times, time_buckets,
        use_mean_point)
    times = results[0] 
    brier_scores = results[1] 

    return (times, brier_scores)


def integrated_brier_score(model, X, T, E, t_max=None, use_mean_point=True):
    """ The Integrated Brier Score (IBS) provides an overall calculation of 
        the model performance at all available times.
    """

    # Computing the brier scores
    times, brier_scores = brier_score(model, X, T, E, t_max, use_mean_point)

    # Getting the proper value of t_max
    if t_max is None:
        t_max = max(times)
    else:
        t_max = min(t_max, max(times))

    # Computing the IBS
    ibs_value = np.trapz(brier_scores, times)/t_max 

    return ibs_value


def ibs(model, X, T, E, t_max=None, use_mean_point = True):
    return integrated_brier_score(model, X, T, E, t_max, use_mean_point)




def compare_to_actual(model, X, T, E, times = None, is_at_risk = False,  
    figsize=(16, 6), metrics = ['rmse', 'mean', 'median'], **kwargs):
    """
    Comparing the actual and predicted number of units at risk and units 
    experiencing an event at each time t.

    Parameters:
    -----------
    * model : pysurvival model
        The model that will be used for prediction

    * X : array-like, shape=(n_samples, n_features)
        The input samples.

    * T : array-like, shape = [n_samples] 
        The target values describing when the event of interest or censoring
        occured

    * E : array-like, shape = [n_samples] 
        The Event indicator array such that E = 1. if the event occured
        E = 0. if censoring occured

    * times: array-like, (default=None)
        A vector of timepoints.

    * is_at_risk: bool (default=True)
        Whether the function returns Expected number of units at risk
        or the Expected number of units experiencing the events.

    * figure_size: tuple of double (default= (16, 6))
        width, height in inches representing the size of the chart 

    * metrics: str or list of str (default='all')
        Indicates the performance metrics to compute:
            - if None, then no metric is computed
            - if str, then the metric is computed
            - if list of str, then the metrics are computed

        The available metrics are:
            - RMSE: root mean squared error
            - Mean Abs Error: mean absolute error
            - Median Abs Error: median absolute error

    Returns:
    --------
    * results: float or dict
        Performance metrics   

    """

    # Initializing the Kaplan-Meier model
    X, T, E = utils.check_data(X, T, E)
    kmf = KaplanMeierModel()
    kmf.fit(T, E)

    # Creating actual vs predicted
    N = T.shape[0]

    # Defining the time axis
    if times is None:
        times = kmf.times

    # Number of Expected number of units at risk
    # or the Expected number of units experiencing the events
    actual = []
    actual_upper = []
    actual_lower = []
    predicted = []
    if is_at_risk:
        model_predicted =  np.sum(model.predict_survival(X, **kwargs), 0)

        for t in times:
            min_index = [ abs(a_j_1-t) for (a_j_1, a_j) in model.time_buckets]
            index = np.argmin(min_index)
            actual.append(N*kmf.predict_survival(t))
            actual_upper.append(N*kmf.predict_survival_upper(t))
            actual_lower.append(N*kmf.predict_survival_lower(t))
            predicted.append( model_predicted[index] )

    else:
        model_predicted =  np.sum(model.predict_density(X, **kwargs), 0)

        for t in times:
            min_index = [ abs(a_j_1-t) for (a_j_1, a_j) in model.time_buckets]
            index = np.argmin(min_index)
            actual.append(N*kmf.predict_density(t))
            h = kmf.predict_hazard(t)
            actual_upper.append(N*kmf.predict_survival_upper(t)*h)
            actual_lower.append(N*kmf.predict_survival_lower(t)*h)
            predicted.append( model_predicted[index] )

    # Computing the performance metrics
    results = None
    if metrics is not None:

        # RMSE
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # Median Abs Error
        med_ae = median_absolute_error(actual, predicted)

        # Mean Abs Error
        mae = mean_absolute_error(actual, predicted)


        if isinstance(metrics, str) :

            # RMSE
            if 'rmse' in metrics.lower() or 'root' in metrics.lower():
                results = rmse
                # title += "\n"
                # title += "RMSE = {:.3f}".format(rmse)

            # Median Abs Error
            elif 'median' in metrics.lower() :
                results = med_ae
                # title += "\n"
                # title += "Median Abs Error = {:.3f}".format(med_ae)

            # Mean Abs Error
            elif 'mean' in metrics.lower() :
                results = mae
                # title += "\n"
                # title += "Mean Abs Error = {:.3f}".format(mae)

            else:
                raise NotImplementedError('{} is not a valid metric function.'
                .format(metrics))


        elif isinstance(metrics, list) or isinstance(metrics, numpy.ndarray) :
            results = {}

            # RMSE
            is_rmse = False
            if any( [ ('rmse' in m.lower() or 'root' in m.lower()) \
                for m in metrics ]):
                is_rmse = True
                results['root_mean_squared_error'] = rmse
                # title += "\n"
                # title += "RMSE = {:.3f}".format(rmse)

            # Median Abs Error
            is_med_ae = False
            if any( ['median' in m.lower() for m in metrics ]):
                is_med_ae = True
                results['median_absolute_error'] = rmse
                # title += "\n"
                # title += "Median Abs Error = {:.3f}".format(med_ae)

            # Mean Abs Error
            is_mae = False
            if any( ['mean' in m.lower() for m in metrics ]):
                is_mae = True
                results['mean_absolute_error'] = rmse
                # title += "\n"
                # title += "Mean Abs Error = {:.3f}".format(mae)

            if all([not is_mae, not is_rmse, not is_med_ae]):
                error = 'The provided metrics are not available.'
                raise NotImplementedError(error)
                
    return results