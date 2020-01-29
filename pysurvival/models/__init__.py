from __future__ import absolute_import
import copy
import tempfile
import pyarrow as pa
import os
import torch
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pysurvival import utils
from pysurvival.utils._functions import _get_time_buckets


class BaseModel(object):
    """ Base class for all estimators in pysurvival. It should not be used on
        its own.
    """

    def __init__(self, auto_scaler=True):

        # Creating a scikit-learner scaler
        self.auto_scaler = auto_scaler
        if self.auto_scaler:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # Creating a place holder for the time axis
        self.times = [0.]

        # Creating the model's name
        self.__repr__()

    def __repr__(self):
        """ Creates the representation of the Object """

        self.name = self.__class__.__name__
        return self.name

    def save(self, path_file):
        """ Save the model components: 
                * the paremeters of the model (parameters) 
                * the PyTorch model itself (model) if it exists
            And Compress them into a zip file

            Parameters
            ----------
            * path_file, str
                address of the file where the model will be saved
        """

        # Ensuring the file has the proper name
        folder_name = os.path.dirname(path_file) + '/'
        file_name = os.path.basename(path_file)
        if not file_name.endswith('.zip'):
            file_name += '.zip'

        # Checking if the folder is accessible
        if not os.access(folder_name, os.W_OK):
            error_msg = '{} is not an accessible directory.'.format(folder_name)
            raise OSError(error_msg)

        # Saving all the elements to save
        elements_to_save = []

        # Changing the format of scaler parameters if exist
        temp_scaler = copy.deepcopy(self.__dict__.get('scaler'))
        if temp_scaler is not None:
            self.__dict__['scaler'] = temp_scaler.__dict__

        # Saving the model parameters
        parameters_to_save = {}
        for k in self.__dict__:
            if k != 'model':
                parameters_to_save[k] = self.__dict__[k]

        # Serializing the parameters
        elements_to_save.append('parameters')
        with open('parameters', 'wb') as f:
            serialized_to_save = pa.serialize(parameters_to_save)
            f.write(serialized_to_save.to_buffer())

        # Saving the torch model if exists
        if 'model' in self.__dict__.keys():
            elements_to_save.append('model')
            torch.save(self.model, 'model')

            # Compressing the elements to save in zip
        full_path = folder_name + file_name
        print('Saving the model to disk as {}'.format(full_path))
        with zipfile.ZipFile(full_path, 'w') as myzip:
            for f in elements_to_save:
                myzip.write(f)

        # Erasing temp files
        for temp_file in elements_to_save:
            os.remove(temp_file)

        # Restore the scaler
        if temp_scaler is not None:
            self.scaler = StandardScaler()
            self.__dict__['scaler'] = copy.deepcopy(temp_scaler)

    def load(self, path_file):
        """ Load the model components from a .zip file: 
                * the parameters of the model (.params) 
                * the PyTorch model itself (.model) is exists

            Parameters
            ----------
            * path_file, str
                address of the file where the model will be loaded from 
        """

        # Ensuring the file has the proper name
        folder_name = os.path.dirname(path_file) + '/'
        file_name = os.path.basename(path_file)
        if not file_name.endswith('.zip'):
            file_name += '.zip'

        # Opening the '.zip' file 
        full_path = folder_name + file_name
        print('Loading the model from {}'.format(full_path))

        # Creating temp folder
        temp_folder = tempfile.mkdtemp() + '/'

        # Unzip files in temp folder
        with zipfile.ZipFile(path_file, 'r') as zip_ref:
            zip_ref.extractall(temp_folder)
            input_zip = zipfile.ZipFile(path_file)

        # Loading the files
        elements_to_load = []
        for file_name in input_zip.namelist():

            # Loading the parameters
            if 'parameters' in file_name.lower():
                content = input_zip.read('parameters')
                self.__dict__ = copy.deepcopy(pa.deserialize(content))
                elements_to_load.append(temp_folder + 'parameters')

                # If a scaler was available then load it too
                temp_scaler = copy.deepcopy(self.__dict__.get('scaler'))
                if temp_scaler is not None:
                    self.scaler = StandardScaler()
                    self.scaler.__dict__ = temp_scaler

            # Loading the PyTorch model
            if 'model' in file_name.lower():
                model = torch.load(temp_folder + 'model')
                self.model = model
                elements_to_load.append(temp_folder + 'model')

        # Erasing temp files
        for temp_file in elements_to_load:
            os.remove(temp_file)

    def get_time_buckets(self, extra_timepoint=False):
        """ Creating the time buckets based on the times axis such that
            for the k-th time bin is [ t(k-1), t(k) ] in the time axis.
        """

        # Checking if the time axis has already been created
        if self.times is None or len(self.times) <= 1:
            error = 'The time axis needs to be created before'
            error += ' using the method get_time_buckets.'
            raise AttributeError(error)

        # Creating the base time buckets
        time_buckets = _get_time_buckets(self.times)

        # Adding an additional element if specified
        if extra_timepoint:
            time_buckets += [(time_buckets[-1][1], time_buckets[-1][1] * 1.01)]
        self.time_buckets = time_buckets

    def _predict(self, x, t=None, **kwargs):
        raise NotImplementedError()

    def predict_hazard(self, x, t=None, **kwargs):
        """ Predicts the hazard function h(t, x)

            Parameters
            ----------
            * `x` : **array-like** *shape=(n_samples, n_features)* --
                array-like representing the datapoints. 
                x should not be standardized before, the model
                will take care of it

            * `t`: **double** *(default=None)* --
                 time at which the prediction should be performed. 
                 If None, then return the function for all available t.

            Returns
            -------
            * `hazard`: **numpy.ndarray** --
                array-like representing the prediction of the hazard function
        """

        # Checking if the data has the right format
        x = utils.check_data(x)

        # Calculating hazard, density, survival
        hazard, _, _ = self._predict(x, t, **kwargs)

        return hazard

    def predict_density(self, x, t=None, **kwargs):
        """ Predicts the density function d(t, x)

            Parameters
            ----------
            * `x` : **array-like** *shape=(n_samples, n_features)* --
                array-like representing the datapoints. 
                x should not be standardized before, the model
                will take care of it

            * `t`: **double** *(default=None)* --
                 time at which the prediction should be performed. 
                 If None, then return the function for all available t.

            Returns
            -------
            * `density`: **numpy.ndarray** --
                array-like representing the prediction of density function
        """

        # Checking if the data has the right format
        x = utils.check_data(x)

        # Calculating hazard, density, survival
        _, density, _ = self._predict(x, t, **kwargs)
        return density

    def predict_survival(self, x, t=None, **kwargs):
        """ Predicts the survival function S(t, x)

            Parameters
            ----------
            * `x` : **array-like** *shape=(n_samples, n_features)* --
                array-like representing the datapoints. 
                x should not be standardized before, the model
                will take care of it

            * `t`: **double** *(default=None)* --
                time at which the prediction should be performed. 
                If None, then return the function for all available t.

            Returns
            -------
            * `survival`: **numpy.ndarray** --
                array-like representing the prediction of the survival function
        """

        # Checking if the data has the right format
        x = utils.check_data(x)

        # Calculating hazard, density, survival
        _, _, survival = self._predict(x, t, **kwargs)
        return survival

    def predict_cdf(self, x, t=None, **kwargs):
        """ Predicts the cumulative density function F(t, x)

            Parameters
            ----------
            * `x` : **array-like** *shape=(n_samples, n_features)* --
                array-like representing the datapoints. 
                x should not be standardized before, the model
                will take care of it

            * `t`: **double** *(default=None)* --
                time at which the prediction should be performed. 
                If None, then return the function for all available t.

            Returns
            -------
            * `cdf`: **numpy.ndarray** --
                array-like representing the prediction of the cumulative 
                density function 
        """

        # Checking if the data has the right format
        x = utils.check_data(x)

        # Calculating survival and cdf
        survival = self.predict_survival(x, t, **kwargs)
        cdf = 1. - survival
        return cdf

    def predict_cumulative_hazard(self, x, t=None, **kwargs):
        """ Predicts the cumulative hazard function H(t, x)

            Parameters
            ----------
            * `x` : **array-like** *shape=(n_samples, n_features)* --
                array-like representing the datapoints. 
                x should not be standardized before, the model
                will take care of it

            * `t`: **double** *(default=None)* --
                time at which the prediction should be performed. 
                If None, then return the function for all available t.

            Returns
            -------
            * `cumulative_hazard`: **numpy.ndarray** --
                array-like representing the prediction of the cumulative_hazard
                function
        """

        # Checking if the data has the right format
        x = utils.check_data(x)

        # Calculating hazard/cumulative_hazard
        hazard = self.predict_hazard(x, t, **kwargs)
        cumulative_hazard = np.cumsum(hazard, 1)
        return cumulative_hazard

    def predict_risk(self, x, **kwargs):
        """ Predicts the Risk Score/Mortality function for all t,
            R(x) = sum( cumsum(hazard(t, x)) )
            According to Random survival forests from Ishwaran H et al
            https://arxiv.org/pdf/0811.1645.pdf

            Parameters
            ----------
            * `x` : **array-like** *shape=(n_samples, n_features)* --
                array-like representing the datapoints. 
                x should not be standardized before, the model
                will take care of it

            Returns
            -------
            * `risk_score`: **numpy.ndarray** --
                array-like representing the prediction of Risk Score function
        """

        # Checking if the data has the right format
        x = utils.check_data(x)

        # Calculating cumulative_hazard/risk
        cumulative_hazard = self.predict_cumulative_hazard(x, None, **kwargs)
        risk_score = np.sum(cumulative_hazard, 1)
        return risk_score
