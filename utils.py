import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def best_forest_regressor(x_train, y_train, x_cv, y_cv):
    """
    Computes the best number based on the provided data for the min samples split, max depth and n estimators.
    These three will be further used for finding the best features based on forest regressor.
    Finally, a graph will be produced to show the mean squared error for each parameter.

    :param x_train: the train data sample of all the numeric independent features
    :param y_train: the target data sample for the train sample in a numeric format
    :param x_cv: the cross-validation sample of all the numeric independent features
    :param y_cv: the target data sample for the cross validation sample in a numeric format
    :return: best_min_split - best min sample split number to be used in the forest regressor parameters
             best_max_deph - best max depth number to be used in the forest regressor parameters
             best_n_estimators - best n estimators number to be used in the forest regressor parameters
    """

    min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 400, 500, 700, 800]
    max_depth_list = [2, 4, 8, 16, 32, 64, 128, None]
    n_estimators_list = [10, 50, 100, 500, 1000]

    all_mse_train = np.zeros(0)
    all_mse_cv = np.zeros(0)
    mse_list_train = []
    mse_list_cv = []
    diff_mse = np.zeros(0)
    best_min_split = 0
