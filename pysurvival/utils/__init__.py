import copy
import numpy as np
import pandas as pd
from pysurvival.utils._functions import _logrankScores
# %matplotlib inline for Jupyter notebooks 


def as_bytes(string_array, python_version=3):
    """ Transforming an array of string into an array of bytes in Python 3 
    """

    if python_version >= 3:
        results = []
        for s in string_array:
            # results.append( codecs.latin_1_encode(s)[0] )
            results.append( s.encode('utf-8') )

    else:
        results = string_array

    return results
    

def check_data(*args):
    """ Makes sure that the given inputs are numpy arrays, list or 
        pandas DataFrames.

        Parameters
        ----------
        * args : tuple of objects
                 Input object to check / convert.

        Returns
        -------
        * result : tuple of numpy arrays
                   The converted and validated arg.

        If the input isn't numpy arrays, list or pandas DataFrames, it will
        fail and ask to provide the valid format.
    """

    result = ()
    for i, arg in enumerate(args):

        if len(arg) == 0 :
            error = " The input is empty. "
            error += "Please provide at least 1 element in the array."
            raise IndexError(error)         

        else:

            if isinstance(arg, np.ndarray) :
                x = ( arg.astype(np.double),  )
            elif isinstance(arg, list):
                x = ( np.asarray(arg).astype(np.double),  )
            elif isinstance(arg, pd.Series):
                x = ( arg.values.astype(np.double),  )
            elif isinstance(arg, pd.DataFrame):
                x = ( arg.values.astype(np.double),  )
            else:
                
                error = "{arg} is not a valid data format. "
                error += "Only use 'list', 'np.ndarray', "
                error += "'pd.Series' "
                error += "'pd.DataFrame' ".format(arg=type(arg))
                raise TypeError(error)
            
            if np.sum( np.isnan(x) ) > 0. :
                error = "The #{} argument contains null values"
                error = error.format(i+1)
                raise ValueError(error)

            if len(args) > 1:
                result += x
            else:
                result = x[0]

    return result


def rank_scores(T, E):
    """ 
    Computing the ranks for each survival time T

        Parameters:
        -----------

        * T : array-like, shape = [n_samples] 
            The target values describing when the event of interest or censoring
            occured

        * E : array-like, shape = [n_samples] 
            The Event indicator array such that E = 1. if the event occured
            E = 0. if censoring occured

        Returns:
        --------
        * rank_scores : array-like
            ranks for each survival time T
    """
    T, E = check_data(T, E)
    return _logrankScores(T,E)


def save_model(model, path_file):
    """ Save the model and its parameters, and compress them into a zip file 

    Parameters:
    -----------
    * model : Pysurvival object
        Pysurvival model

    * path_file, str
        address of the file where the model will be loaded from 
    """

    model.save(path_file)




def load_model(path_file):
    """ Load the model and its parameters from a .zip file 

    Parameters:
    -----------
    * path_file, str
        address of the file where the model will be loaded from 

    Returns:
    --------
    * pysurvival_model : Pysurvival object
        Pysurvival model
    """

    # Initializing a base model
    from pysurvival.models import BaseModel
    base_model = BaseModel()

    # Temporary loading the model
    base_model.load(path_file)
    model_name = base_model.name

    # Loading the actual Pysurvival model - Kaplan-Meier
    if 'kaplanmeier' in model_name.lower():

        if 'smooth' in model_name.lower():
            from pysurvival.models.non_parametric import SmoothKaplanMeierModel
            pysurvival_model = SmoothKaplanMeierModel()

        else:
            from pysurvival.models.non_parametric import KaplanMeierModel
            pysurvival_model = KaplanMeierModel()


    elif 'linearmultitask' in model_name.lower():

        from pysurvival.models.multi_task import LinearMultiTaskModel
        pysurvival_model = LinearMultiTaskModel()


    elif 'neuralmultitask' in model_name.lower():

        from pysurvival.models.multi_task import NeuralMultiTaskModel
        structure = [ {'activation': 'relu', 'num_units': 128}, ]         
        pysurvival_model = NeuralMultiTaskModel(structure=structure)


    elif 'exponential' in model_name.lower():

        from pysurvival.models.parametric import ExponentialModel
        pysurvival_model = ExponentialModel()


    elif 'weibull' in model_name.lower():

        from pysurvival.models.parametric import WeibullModel
        pysurvival_model = WeibullModel()


    elif 'gompertz' in model_name.lower():

        from pysurvival.models.parametric import GompertzModel
        pysurvival_model = GompertzModel()


    elif 'loglogistic' in model_name.lower():

        from pysurvival.models.parametric import LogLogisticModel
        pysurvival_model = LogLogisticModel()


    elif 'lognormal' in model_name.lower():

        from pysurvival.models.parametric import LogNormalModel
        pysurvival_model = LogNormalModel()


    elif 'simulation' in model_name.lower():

        from pysurvival.models.simulations import SimulationModel
        pysurvival_model = SimulationModel()


    elif 'coxph' in model_name.lower():

        if 'nonlinear' in model_name.lower():
            from pysurvival.models.semi_parametric import NonLinearCoxPHModel
            pysurvival_model = NonLinearCoxPHModel()

        else:
            from pysurvival.models.semi_parametric import CoxPHModel
            pysurvival_model = CoxPHModel()


    elif 'random' in model_name.lower() and 'survival' in model_name.lower():

        from pysurvival.models.survival_forest import RandomSurvivalForestModel
        pysurvival_model = RandomSurvivalForestModel()


    elif 'extra' in model_name.lower() and 'survival' in model_name.lower():

        from pysurvival.models.survival_forest import ExtraSurvivalTreesModel
        pysurvival_model = ExtraSurvivalTreesModel()


    elif 'condi' in model_name.lower() and 'survival' in model_name.lower():

        from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
        pysurvival_model = ConditionalSurvivalForestModel()


    elif 'svm' in model_name.lower() :

        if 'linear' in model_name.lower():

            from pysurvival.models.svm import LinearSVMModel
            pysurvival_model = LinearSVMModel()

        elif 'kernel' in model_name.lower():

            from pysurvival.models.svm import KernelSVMModel
            pysurvival_model = KernelSVMModel()

    else:
        raise NotImplementedError('{} is not a valid pysurvival model.'
            .format(model_name))

    # Transferring the components
    pysurvival_model.__dict__.update(copy.deepcopy(base_model.__dict__))
    del base_model

    return pysurvival_model
