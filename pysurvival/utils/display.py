import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors
from matplotlib.patches import Rectangle
from pysurvival import utils
from pysurvival.utils import metrics
from pysurvival.utils.metrics import brier_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error
from pysurvival.models.non_parametric import KaplanMeierModel


def display_loss_values(model, figure_size=(18, 5)):
    """ Display the loss function values of the model fitting 

        Parameters:
        -----------
        * model : pysurvival model
            The model that will be used for prediction

        * figure_size: tuple of double (default= (18, 5))
            width, height in inches representing the size of the chart 
    """

    # Check that the model is not a Non-Parametric model
    if 'kaplan' in model.name.lower():
        error = "This function cannot only take as input a Non-Parametric model"
        raise NotImplementedError(error)

    if 'simulation' in model.name.lower():
        error = "This function cannot only take as input a simulation model"
        raise NotImplementedError(error)

    # Extracting the loss values
    loss_values = model.loss_values

    # Extracting the norm 2 of the gradient, if it exists
    grad2_values = model.__dict__.get('grad2_values')
    if grad2_values is None:
        order = 1
    else:
        order = 2

    # Displaying the loss values bsed on the type of optimization
    if order == 1:
        title = "Loss function values"
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot(loss_values, color='blue', label='Loss values')
        ax.set_xlabel('Number of epochs', fontsize=10)
        plt.legend(fontsize=10)
        plt.title(title, fontsize=10)
        plt.show()

    elif order == 2:

        # Initializing Chart 
        fig = plt.figure(figsize=figure_size)
        fig.suptitle('Loss function $l$ and $|| gradient ||_{L_{2}}$',
                     fontsize=12, fontweight='bold')

        # Plotting loss function
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('Loss function $l$')
        pl1 = ax1.plot(range(len(loss_values)), loss_values,
                       label='Loss function $l$', color='blue', linestyle='--')

        # Plotting ||grad|| values
        ax2 = ax1.twinx()
        pl2 = ax2.plot(range(len(grad2_values)), grad2_values,
                       label='$|| gradient ||_{L_{2}}$', color='red')
        ax2.set_ylabel('$|| gradient ||_{L_{2}}$')

        # added these three lines
        pl = pl1 + pl2
        labs = [l.get_label() for l in pl]
        ax1.legend(pl, labs, loc=1)

        # display chart
        plt.show()


def display_non_parametric(km_model, figure_size=(18, 5)):
    """ Plotting the survival function and its lower and upper bounds 

        Parameters:
        -----------
        * km_model : pysurvival Non-Parametric model
            The model that will be used for prediction

        * figure_size: tuple of double (default= (18, 5))
            width, height in inches representing the size of the chart 
    """

    # Check that the model is a Non-Parametric model
    if 'kaplan' not in km_model.name.lower():
        error = "This function can only take as input a Non-Parametric model"
        raise NotImplementedError(error)

    # Title of the chart
    if 'smooth' in km_model.name.lower():
        is_smoothed = True
        title = 'Smooth Kaplan-Meier Survival function'
    else:
        is_smoothed = False
        title = 'Kaplan-Meier Survival function'

    # Initializing the chart
    fig, ax = plt.subplots(figsize=figure_size)

    # Extracting times and survival function
    times, survival = km_model.times, km_model.survival

    # Plotting Survival
    plt.plot(times, survival, label=title,
             color='blue', lw=3)

    # Defining the x-axis and y-axis
    ax.set_xlabel('Time')
    ax.set_ylabel('S(t) Survival function')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, max(times) * 1.01])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:.1f}%'.format(v * 100) for v in vals])
    plt.title(title, fontsize=25)

    # Extracting times and survival function
    times, survival = km_model.times, km_model.survival

    if is_smoothed:

        # Display
        plt.plot(times, survival, label='Original Kaplan-Meier',
                 color='#f44141', ls='-.', lw=2.5)
        plt.legend(fontsize=15)
        plt.show()

    else:

        # Extracting CI
        survival_ci_upper = km_model.survival_ci_upper
        survival_ci_lower = km_model.survival_ci_lower

        # Plotting the Confidence Intervals
        plt.plot(times, survival_ci_upper,
                 color='red', alpha=0.1, ls='--')
        plt.plot(times, survival_ci_lower,
                 color='red', alpha=0.1, ls='--')

        # Filling the areas between the Survival and Confidence Intervals curves
        plt.fill_between(times, survival, survival_ci_lower,
                         label='Confidence Interval - lower', color='red', alpha=0.2)
        plt.fill_between(times, survival, survival_ci_upper,
                         label='Confidence Interval - upper', color='red', alpha=0.2)

        # Display
        plt.legend(fontsize=15)
        plt.show()


def display_baseline_simulations(sim_model, figure_size=(18, 5)):
    """ Display Simulation model baseline 

        Parameters:
        -----------
        * sim_model : pysurvival Simulations model
            The model that will be used for prediction

        * figure_size: tuple of double (default= (18, 5))
            width, height in inches representing the size of the chart 
    """

    # Check that the model is a Non-Parametric model
    if 'simulation' not in sim_model.name.lower():
        error = "This function can only take as input a Non-Parametric model"
        raise NotImplementedError(error)

    # Extracting parameters
    name_survival_distribution = sim_model.survival_distribution
    times = sim_model.times
    baseline_survival = sim_model.baseline_survival
    title = 'Base Survival function - ' + name_survival_distribution.title()

    # Display
    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(times, baseline_survival, color='blue', label='baseline')
    plt.legend()
    plt.title(title)
    plt.show()


def integrated_brier_score(model, X, T, E, t_max=None, use_mean_point=True,
                           figure_size=(20, 6.5)):
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
    ibs_value = np.trapz(brier_scores, times) / t_max

    # Displaying the Brier Scores at different t 
    title = 'Prediction error curve with IBS(t = {:.1f}) = {:.2f}'
    title = title.format(t_max, ibs_value)
    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(times, brier_scores, color='blue', lw=3)
    ax.set_xlim(-0.01, max(times))
    ax.axhline(y=0.25, ls='--', color='red')
    ax.text(0.90 * max(times), 0.235, '0.25 limit', fontsize=20, color='brown',
            fontweight='bold')
    plt.title(title, fontsize=20)
    plt.show()

    return ibs_value


def compare_to_actual(model, X, T, E, times=None, is_at_risk=False,
                      figure_size=(16, 6), metrics=['rmse', 'mean', 'median'], **kwargs):
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
        model_predicted = np.sum(model.predict_survival(X, **kwargs), 0)

        for t in times:
            min_index = [abs(a_j_1 - t) for (a_j_1, a_j) in model.time_buckets]
            index = np.argmin(min_index)
            actual.append(N * kmf.predict_survival(None, t))
            actual_upper.append(N * kmf.predict_survival_upper(None, t))
            actual_lower.append(N * kmf.predict_survival_lower(None, t))
            predicted.append(model_predicted[index])

    else:
        model_predicted = np.sum(model.predict_density(X, **kwargs), 0)

        for t in times:
            min_index = [abs(a_j_1 - t) for (a_j_1, a_j) in model.time_buckets]
            index = np.argmin(min_index)
            actual.append(N * kmf.predict_density(None, t))
            h = kmf.predict_hazard(None, t)
            actual_upper.append(N * kmf.predict_survival_upper(None, t) * h)
            actual_lower.append(N * kmf.predict_survival_lower(None, t) * h)
            predicted.append(model_predicted[index])

    # Computing the performance metrics
    results = None
    title = 'Actual vs Predicted'
    if metrics is not None:

        # RMSE
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # Median Abs Error
        med_ae = median_absolute_error(actual, predicted)

        # Mean Abs Error
        mae = mean_absolute_error(actual, predicted)

        if isinstance(metrics, str):

            # RMSE
            if 'rmse' in metrics.lower() or 'root' in metrics.lower():
                results = rmse
                title += "\n"
                title += "RMSE = {:.3f}".format(rmse)

            # Median Abs Error
            elif 'median' in metrics.lower():
                results = med_ae
                title += "\n"
                title += "Median Abs Error = {:.3f}".format(med_ae)

            # Mean Abs Error
            elif 'mean' in metrics.lower():
                results = mae
                title += "\n"
                title += "Mean Abs Error = {:.3f}".format(mae)

            else:
                raise NotImplementedError('{} is not a valid metric function.'
                                          .format(metrics))


        elif isinstance(metrics, list) or isinstance(metrics, numpy.ndarray):
            results = {}

            # RMSE
            is_rmse = False
            if any([('rmse' in m.lower() or 'root' in m.lower()) \
                    for m in metrics]):
                is_rmse = True
                results['root_mean_squared_error'] = rmse
                title += "\n"
                title += "RMSE = {:.3f}".format(rmse)

            # Median Abs Error
            is_med_ae = False
            if any(['median' in m.lower() for m in metrics]):
                is_med_ae = True
                results['median_absolute_error'] = med_ae
                title += "\n"
                title += "Median Abs Error = {:.3f}".format(med_ae)

            # Mean Abs Error
            is_mae = False
            if any(['mean' in m.lower() for m in metrics]):
                is_mae = True
                results['mean_absolute_error'] = mae
                title += "\n"
                title += "Mean Abs Error = {:.3f}".format(mae)

            if all([not is_mae, not is_rmse, not is_med_ae]):
                error = 'The provided metrics are not available.'
                raise NotImplementedError(error)

    # Plotting
    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(times, actual, color='red', label='Actual',
            alpha=0.8, lw=3)
    ax.plot(times, predicted, color='blue', label='Predicted',
            alpha=0.8, lw=3)
    plt.xlim(0, max(T))

    # Filling the areas between the Survival and Confidence Intervals curves
    plt.fill_between(times, actual, actual_lower,
                     label='Confidence Intervals - Lower', color='red', alpha=0.2)
    plt.fill_between(times, actual, actual_upper,
                     label='Confidence Intervals - Upper', color='red', alpha=0.2)

    # Finalizing the chart
    plt.title(title, fontsize=15)
    plt.legend(fontsize=15)
    plt.show()

    return results


def create_risk_groups(model, X, use_log=True, num_bins=50,
                       figure_size=(20, 8), **kwargs):
    """
    Computing and displaying the histogram of the risk scores of the given 
    model and test set X. If it is provided args, it will assign a color coding 
    to the scores that are below and above the given thresholds.

    Parameters:
    -----------
    
    * model : Pysurvival object
        Pysurvival model

    * X : array-like, shape=(n_samples, n_features)
        The input samples.
    
    * use_log: boolean (default=True)
        Whether applying the log function to the risk score
        
    * num_bins: int (default=50)
        The number of equal-width bins that will constitute the histogram
        
    * figure_size: tuple of double (default= (16, 6))
        width, height in inches representing the size of the chart 

    * kwargs: dict (optional)
        kwargs = low_risk = {'lower_bound': 0, 'upper_bound': 20, 'color': 'red'},
                 high_risk = {'lower_bound': 20, 'upper_bound': 120, 'color': 'blue'}
            that define the risk group
      
    """

    # Ensuring that the input data has the right format
    X = utils.check_data(X)

    # Computing the risk scores
    risk = model.predict_risk(X)
    if use_log:
        risk = np.log(risk)

    # Displaying simple histogram
    if len(kwargs) == 0:

        # Initializing the chart
        fig, ax1 = plt.subplots(figsize=figure_size)
        risk_groups = None

    # Applying any color coding
    else:
        # Initializing the results
        risk_groups = {}

        # Initializing the chart
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=figure_size)

        # Displaying simple histogram with risk groups
        nums_per_bins, bins, patches = ax2.hist(risk, bins=num_bins)
        ax2.set_title('Risk groups with colors', fontsize=15)

        # Number of group definitions
        num_group_def = len(kwargs.values())

        # Extracting the bounds values
        bounds = {}
        colors_ = {}
        indexes = {}
        group_names = []
        handles = []

        # we need to check that the boundaries match the bins
        is_not_valid = 0
        for group_name, group_def in kwargs.items():

            # by ensuring that the bounds are not outside
            # the bins values
            min_bin, max_bin = min(bins), max(bins)
            if (group_def['lower_bound'] < min_bin and \
                group_def['upper_bound'] < min_bin) or \
                    (group_def['lower_bound'] > max_bin and \
                     group_def['upper_bound'] > max_bin):
                is_not_valid += 1

            # Extracting the bounds
            bounds[group_name] = (group_def['lower_bound'],
                                  group_def['upper_bound'])

            # Extracting the colors
            colors_[group_name] = group_def['color']

            # Creating index placeholders
            indexes[group_name] = []
            group_names.append(group_name)
            color_indv = group_def['color']
            handles.append(Rectangle((0, 0), 1, 1, color=color_indv, ec="k"))

        if is_not_valid >= num_group_def:
            error_msg = "The boundaries definitions {} do not match"
            error_msg += ", the values of the risk scores."
            error_msg = error_msg.format(list(bounds.values()))
            raise ValueError(error_msg)

        # Assigning each rectangle/bin to its group definition
        # and color
        colored_patches = []
        bin_index = {}
        for i, bin_, patch_ in zip(range(num_bins), bins, patches):

            # Check if the bin belongs to this bound def
            for grp_name, bounds_ in bounds.items():

                if bounds_[0] <= bin_ < bounds_[-1]:
                    bin_index[i] = grp_name

                    # Extracting color
                    color_ = colors_[grp_name]
                    if color_ not in colors.CSS4_COLORS:
                        error_msg = '{} is not a valid color'
                        error_msg = error_msg.format(colors_[grp_name])
                        raise ValueError(error_msg)

                    patch_.set_facecolor(color_)

            # Saving the rectangles
            colored_patches.append(patch_)

        # Assigning each sample to its group
        risk_bins = np.minimum(np.digitize(risk, bins, True), num_bins - 1)
        for i, r in enumerate(risk_bins):
            # Extracting the right group_name
            group_name = bin_index[r]
            indexes[group_name].append(i)

    # Displaying the original distribution
    ax1.hist(risk, bins=num_bins, color='black', alpha=0.5)
    ax1.set_title('Risk Score Distribution', fontsize=15)

    # Show everything
    plt.show()

    # Returning results
    if risk_groups is not None:
        for group_name in group_names:
            result = (colors_[group_name], indexes[group_name])
            risk_groups[group_name] = result

    return risk_groups


def correlation_matrix(df, figure_size=(12, 8), text_fontsize=10):
    """ Takes dataframe and display the correlations between features """

    # Computing correlations
    corr = df.corr()

    # Display the correlations using heat map
    fig, ax1 = plt.subplots(figsize=figure_size)
    cmap = cm.get_cmap('RdYlBu', 20)  # 'rainbow', 20)
    cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap,
                     vmax=1, vmin=-1)

    # Add values in the cells
    for x in range(corr.shape[0]):
        for y in range(corr.shape[1]):

            if x == y:
                color = 'white'
            else:
                color = 'black'

            plt.text(x, y, '%.2f' % corr.values[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color=color,
                     fontsize=text_fontsize,
                     )

    # Reformat the x/y axis
    labels = df.columns
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=16)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=16)
    plt.xticks(rotation=80)

    # Add colorbar, specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=np.arange(-1.1, 1.1, 0.1))
    plt.show()
