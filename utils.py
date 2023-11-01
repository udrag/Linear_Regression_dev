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
    # First for loop to find the best min samples split
    for min_samples_split in min_samples_split_list:
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestRegressor(min_samples_split=min_samples_split,
                                      random_state=1234).fit(x_train, y_train)
        predictions_train = model.predict(x_train)  # The predicted values for the train dataset
        predictions_cv = model.predict(x_cv)  # The predicted values for the test dataset
        mse_train = mean_squared_error(predictions_train, y_train)
        mse_cv = mean_squared_error(predictions_cv, y_cv)
        mse_list_train.append(mse_train)
        mse_list_cv.append(mse_cv)
        all_mse_train = np.append(all_mse_train, mse_train)
        all_mse_cv = np.append(all_mse_cv, mse_cv)
        diff_mse = np.append(diff_mse, (mse_cv - mse_train))

    for i, (v, j, k) in enumerate(zip(diff_mse, all_mse_train, all_mse_cv)):
        if v < np.mean(diff_mse) and v > 0:
            if j < np.mean(all_mse_train) and k < np.mean(all_mse_cv):
                best_min_split = min_samples_split_list[i]
                break
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Train x Validation metrics')
    plt.xlabel('min_samples_split')
    plt.ylabel('mse')
    plt.xticks(ticks=range(len(min_samples_split_list)), labels=min_samples_split_list)
    plt.plot(mse_list_train)
    plt.plot(mse_list_cv)
    plt.legend(['Train', 'Validation'])
