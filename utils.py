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

    #Second for loop to find the best max_depth
    for max_depth in max_depth_list:
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestRegressor(max_depth=max_depth,
                                      random_state=1234).fit(x_train, y_train)
        predictions_train = model.predict(x_train)  ## The predicted values for the train dataset
        predictions_cv = model.predict(x_cv)  ## The predicted values for the test dataset
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
                best_max_deph = max_depth_list[i]
                break
    plt.subplot(1, 3, 2)
    plt.title('Train x Validation metrics')
    plt.xlabel('max_depth')
    plt.ylabel('mse')
    plt.xticks(ticks=range(len(max_depth_list)), labels=max_depth_list)
    plt.plot(mse_list_train)
    plt.plot(mse_list_cv)
    plt.legend(['Train', 'Validation'])

    #Third loop to find the best n estimators
    mse_list_train = []
    mse_list_cv = []
    all_mse_train = np.zeros(0)
    all_mse_cv = np.zeros(0)
    diff_mse = np.zeros(0)
    best_n_estimators = 0

    for n_estimators in n_estimators_list:
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      random_state=1234).fit(x_train, y_train)
        predictions_train = model.predict(x_train)  ## The predicted values for the train dataset
        predictions_cv = model.predict(x_cv)  ## The predicted values for the test dataset
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
                best_n_estimators = n_estimators_list[i]
                break
    plt.subplot(1, 3, 3)
    plt.title('Train x Validation metrics')
    plt.xlabel('max_depth')
    plt.ylabel('mse')
    plt.xticks(ticks=range(len(n_estimators_list)), labels=n_estimators_list)
    plt.plot(mse_list_train)
    plt.plot(mse_list_cv)
    plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    print(f"The selected best min splits is {best_min_split}, best max depth is {best_max_deph} "
          f"and the best n estimators is {best_n_estimators}.")

    return best_min_split, best_max_deph, best_n_estimators

def feature_importance(x_train, y_train, best_min_split, best_max_deph, best_n_estimators, threshold=0):
    """
    Computes the gini score for each feature and selects the best based on a threshold.

    :param x_train: the train data sample of all the numeric independent feature
    :param y_train: the target data sample for the train sample in a numeric format
    :param best_min_split: the number obtained from the best_forest_regressor function
    :param best_max_deph: the number obtained from the best_forest_regressor function
    :param best_n_estimators: the number obtained from the best_forest_regressor function
    :param threshold: gini score threshold above which the features will be selected
    :return: feature_importance - a list of selected features
    """
    dt_model = RandomForestRegressor(
        min_samples_split=best_min_split,
        max_depth=best_max_deph,
        n_estimators=best_n_estimators
    )
    dt_model.fit(x_train, y_train)
    feature_importance = pd.concat([
        pd.DataFrame(dt_model.feature_names_in_[dt_model.feature_importances_ > threshold], columns=['feature']),
        pd.DataFrame(dt_model.feature_importances_[dt_model.feature_importances_ > threshold], columns=['gini'])],
        axis=1).sort_values('gini', ascending=False)

    return feature_importance

def plot_features(x, y):
    """
    Visual assessment of the selected features on scatter plots.

    :param x: the values on the X-axis
    :param y: the values on the Y-axis
    :return: plt.show() function that shows the plots
    """
    aspect_ratio = (8,5)
    num_subplots = x.shape[1]
    num_columns = 2
    num_rows = (x.shape[1] + 1) // 2  # Calculate the number of rows based on the number of subplots

    # Calculate the size of each subplot based on the aspect ratio
    subplot_size = (5, 5)  # Default size
    if aspect_ratio[0] > aspect_ratio[1]:
        subplot_size = (subplot_size[0], subplot_size[0] * aspect_ratio[1] / aspect_ratio[0])
    else:
        subplot_size = (subplot_size[1] * aspect_ratio[0] / aspect_ratio[1], subplot_size[1])

    fig, axes = plt.subplots((x.shape[1]+1) // 2, 2, figsize=(num_columns * subplot_size[0], num_rows * subplot_size[1]), constrained_layout=True)
    # Flatten the axes array to access subplots one by one
    axes = axes.flatten()

    for i in range(x.shape[1]):
        axes[i].scatter(x[x.columns[:][i]], y, s=2)
        axes[i].set_xlabel(x.columns[i])
        axes[i].set_ylabel('Temperature')
    # Hide any remaining empty subplots (if any)
    for j in range(x.shape[1], len(axes)):
        fig.delaxes(axes[j])

    return plt.show()

def linear_regeression_feature_performance(x_train, y_train, x_cv, y_cv):
    """
    Computes the mean squared error for adding a feature one by one starting from the first one. Additionally,
    will transform the values of the feature through 4 polynomial degrees.

    :param x_train: the independent variables values of the training set
    :param y_train: the target values of training set
    :param x_cv: the independent variables values of the cross validation set
    :param y_cv: the target values of cross validation set
    :return:
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axes = axes.flatten()

    for d in range(1, 5):
        mse_train_list = []
        mse_cv_list = []

        for i in range(1, len(x_train.columns) + 1):
            standard = StandardScaler()
            train_std = standard.fit_transform(x_train.iloc[:, 0:i])

            polyn = PolynomialFeatures(degree=d, include_bias=False)
            train_std_poly = polyn.fit_transform(train_std)

            model = LinearRegression()
            model.fit(train_std_poly, y_train)

            pred_train = model.predict(train_std_poly)
            mse_train = mean_squared_error(y_train, pred_train)
            mse_train_list.append(mse_train)

            x_cv_std = standard.transform(x_cv.iloc[:, 0:i])
            x_cv_std_poly = polyn.transform(x_cv_std)
            pred_cv = model.predict(x_cv_std_poly)
            mse_cv = mean_squared_error(y_cv, pred_cv)
            mse_cv_list.append(mse_cv)
        print(f"The MSE of the {d} polynial degree for the cross validation set is {mse_cv}")
