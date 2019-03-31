from __future__ import absolute_import
import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split

try:
	CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/'
except:
	CURRENT_FOLDER = os.path.dirname(os.path.realpath('__file__')) + '/'


class Dataset(object):
	""" Helper object built to read/load tutorial datasets with ease."""

	def __init__(self, name):

		if 'maintenance' in name.lower() :
			self.name = 'maintenance'
			self.time_column = 'lifetime'
			self.event_column = 'broken'
			self.filename = CURRENT_FOLDER +'maintenance.csv'

		elif 'simple' in name.lower() or 'example' in name.lower()  :
			self.name = 'simple_example'
			self.time_column = 'time'
			self.event_column = 'event'
			self.filename = CURRENT_FOLDER +'simple_example.csv'

		elif 'credit' in name.lower() :
			self.name = 'credit_risk'
			self.time_column = 'duration'
			self.event_column = 'full_repaid'
			self.filename = CURRENT_FOLDER +'credit_risk.csv'

		elif 'employee' in name.lower() :
			self.name = 'employee_attrition'
			self.time_column = 'time_spend_company'
			self.event_column = 'left'
			self.filename = CURRENT_FOLDER +'employee_attrition.csv'

		elif 'churn' in name.lower() :
			self.name = 'churn'
			self.time_column = 'months_active'
			self.event_column = 'churn'
			self.filename = CURRENT_FOLDER +'churn.csv'


	def load(self):
		""" Loading the dataset """

		if 'maintenance' in self.filename.lower():
			sep = ";"
		else:
			sep = ","

		data = pd.read_csv(self.filename, sep=sep)
		return data


	def load_train_test(self, test_size = 0.3):
		""" Loading the dataset and returning X, T, E split between 
			training and testing 
		"""

		# Reading the dataset
		data = self.load()
		N = data.shape[0]

		# Extracting the features
		columns_to_exclude = [self.time_column, self.event_column]
		self.features = np.setdiff1d(data.columns, columns_to_exclude).tolist()
		features = self.features
		time = self.time_column
		event = self.event_column

		# Building training and testing sets #
		index_train, index_test = train_test_split( range(N), 
			test_size = test_size)
		data_train = data.loc[index_train].reset_index( drop = True )
		data_test  = data.loc[index_test].reset_index( drop = True )

		# Creating the X, T and E input
		X_train, X_test = data_train[features], data_test[features]
		T_train, T_test = data_train[time].values, data_test[time].values
		E_train, E_test = data_train[event].values, data_test[event].values

		return X_train, T_train, E_train, X_test, T_test, E_test


	def __repr__(self):
		return self.name
