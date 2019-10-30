#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:25:21 2019

@author: Shiv
"""

from xgboost import XGBRegressor, XGBRFRegressor
from yellowbrick.regressor import residuals_plot, prediction_error
from yellowbrick.features import rank1d, rank2d
from yellowbrick.model_selection import RFECV, ValidationCurve, LearningCurve, LearningCurve
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, SGDRegressor, PassiveAggressiveRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR, NuSVR
from warnings import filterwarnings
from matplotlib.pylab import rcParams
from pandas.plotting import register_matplotlib_converters
from seaborn import catplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import pandas_profiling
import statsmodels.api as sm
import sklearn.metrics as metrics
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
register_matplotlib_converters()
filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")
# rcParams['figure.figsize'] = 10, 8
np.random.seed(1000)
print("\nEnvironment is ready.")
