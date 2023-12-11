import numpy as np
import pandas as pd
import math
import copy
import random

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import time
from datetime import datetime as dtime

from collections import defaultdict

random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)


########################################################
#           Section 1: Modelling functions             #
########################################################

class BestParam:

    @staticmethod
    def model_fit_and_predict(params, value, x_train, y_train, x_cv, y_cv, **kwargs):
        """
        Initialize a Random Forest Regressor with the predictions for train and cross validation sets.
        :param params: list of parameter names
        :param value: values of the parameters
        :param x_train: train sample
        :param y_train: target of the train sample
        :param x_cv: cross validation sample
        :param y_cv: target of the cross validation sample
        :param kwargs: supplimentary arguments
        :return: mse_train, mse_cv, mse_train / mse_cv, value
        """
        model = RandomForestRegressor(**params, **kwargs)
        model.fit(x_train, y_train)
        predictions_train = model.predict(x_train)
        predictions_cv = model.predict(x_cv)
        mse_train = mean_squared_error(y_train, predictions_train)
        mse_cv = mean_squared_error(y_cv, predictions_cv)
        return mse_train, mse_cv, mse_train / mse_cv, value

    @staticmethod
    def find_best_param_values(param_range, param_name, x_train, y_train, x_cv, y_cv, **kwargs):
        """
        Find the best parameters for Random Forest Regressor.
        :param param_range: range of parameter's values
        :param param_name: parameter's name
        :param x_train: train sample
        :param y_train: target of the train sample
        :param x_cv: cross validation sample
        :param y_cv: target of the cross validation sample
        :param kwargs: supplimentary arguments
        :return: results, best_param_value, min_distance_index
        """
        results = pd.DataFrame([BestParam.model_fit_and_predict(
            {param_name: param}, param, x_train, y_train, x_cv, y_cv, **kwargs)
            for param in param_range],
            columns=['mse_train', 'mse_cv', 'mse_prop', param_name])
        if results[results.iloc[:, 2] < 0.6].iloc[:, 2].nlargest(1).item() > 0.2:
            min_distance_index = results[results.iloc[:, 2] < 0.6].iloc[:, 2].nlargest(1).index[0]
        else:
            min_distance_index = \
                results[results.iloc[:, 2] < results[results.iloc[:, 2] < 0.6].iloc[:, 2].mean()].iloc[:, 2].nlargest(
                    1).index[0]
        best_param_value = results[param_name][min_distance_index]

        return results, best_param_value, min_distance_index

    @staticmethod
    def plot_results(results, best_param_value, min_distance_index, subplot, xlabel):
        """
        Plot the results of the mse of the parameters of the Random Forest Regressor.
        :param results: the values from find_best_param_values return variable
        :param best_param_value: best values for the parameters
        :param min_distance_index: the smallest index based on the data
        :param subplot: the number of subplots
        :param xlabel: the x-axis label
        """
        plt.subplot(1, 3, subplot)
        plt.title('Train x Validation Metrics')
        plt.xlabel(xlabel)
        plt.ylabel('mse')
        plt.xticks(ticks=range(len(results[xlabel])), labels=results[xlabel])
        plt.plot(results['mse_train'])
        plt.plot(results['mse_cv'])
        plt.vlines(min_distance_index, 0, results.loc[min_distance_index, ['mse_train', 'mse_cv']].max() * 1.1,
                   color='r', linestyle='--')
        plt.scatter(results[results[xlabel] == best_param_value].index[0], 0, color='skyblue', edgecolor='red',
                    linewidth=2)
        plt.legend(['Train', 'Validation'])

    @staticmethod
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
        # Transform independent variables
        standard = StandardScaler()
        train_std = standard.fit_transform(x_train)
        x_cv_std = standard.transform(x_cv)
        # Create a new figure
        plt.figure(figsize=(12, 4))
        # Set the parameters values and names
        param_ranges = [[2, 10, 30, 50, 100, 200, 300, 400],
                        [2, 4, 8, 16, 32, 64, 128],
                        [10, 25, 50, 100, 250, 500]]
        param_names = ['min_samples_split', 'max_depth', 'n_estimators']
        # Additional variables
        results_all = pd.DataFrame()
        best_param_values = []
        for i, (param_range, param_name) in enumerate(zip(param_ranges, param_names), 1):
            results, best_param_value, min_distance_index = BestParam.find_best_param_values(
                param_range, param_name, train_std, y_train, x_cv_std, y_cv, random_state=1234)
            results_all = pd.concat([results, results_all], axis=1)
            BestParam.plot_results(results, best_param_value, min_distance_index, i, param_name)
            best_param_values.append(best_param_value)

        plt.tight_layout()
        print(f"The selected best min splits is {best_param_values[0]}, best max depth is {best_param_values[1]} "
              f"and the best n estimators is {best_param_values[2]}.")
        return best_param_values[0], best_param_values[1], best_param_values[2]


def feature_importance(x_train, y_train, min_split, max_depth, n_estimators):
    """
    Computes the gini score for each feature using a Random Forest Regressor with the specified parameters.

    :param x_train: the train data sample of all the numeric independent features
    :param y_train: the target data sample for the train sample in numeric format
    :param min_split: the number of min_samples_split for the regressor
    :param max_depth: the number of max_depth for the regressor
    :param n_estimators: the number of n_estimators for the regressor
    :return: feature_importance - a DataFrame of selected features and their respective Gini scores
    """
    # Transform independent variables
    standard = StandardScaler()
    train_std = standard.fit_transform(x_train)
    # Add back column names
    train_std = pd.DataFrame(train_std, columns=x_train.columns)
    # Create and fit the model
    model = RandomForestRegressor(min_samples_split=min_split, max_depth=max_depth, n_estimators=n_estimators)
    model.fit(train_std, y_train)

    # Compute the feature importance
    significant_features_indices = model.feature_importances_ > 0
    features = pd.DataFrame(model.feature_names_in_[significant_features_indices], columns=['feature'])
    gini_scores = pd.DataFrame(model.feature_importances_[significant_features_indices], columns=['gini'])

    ft_importance = pd.concat([features, gini_scores], axis=1).sort_values('gini', ascending=False)
    ft_importance.reset_index(drop=True, inplace=True)
    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(y=ft_importance['feature'], width=ft_importance['gini'], color='skyblue')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    # Iterate over the features and Gini scores to add labels to the bars
    for i, v in enumerate(ft_importance['gini']):
        plt.text(v, i, " " + str(round(v, 4)), va='center')
    plt.gca().invert_yaxis()

    return ft_importance


def linear_regression_feature_performance(x_train, y_train, x_cv, y_cv, all_feature_importance, max_poly_degree=4,
                                          reduce_corr=True, corr_limit=0.7):
    """
    Computes the mean squared error for adding a feature one by one starting from the first one. Additionally,
    will transform the values of the feature through 4 polynomial degrees.

    Optionally, the function will remove the variables with higher correlation based on small gini importance computed previously in
    a Random Forest Regressor.

    :param x_train: the independent variables values of the training set
    :param y_train: the target values of training set
    :param x_cv: the independent variables values of the cross validation set
    :param y_cv: the target values of cross validation set
    :param all_feature_importance: dataframe of all features and corresponding gini value
    :param max_poly_degree: int of the maximum polynomial degree
    :param reduce_corr: Boolean True or False to compute and remove the specified highly correlated variables
    :param corr_limit: the lower correlation coefficient above which to remove the variables
    :return: plt.show()
    """
    cols = 2  # 2 columns of subplots
    rows = np.ceil(max_poly_degree / cols)  # determine the number of rows of subplots
    fig, axes = plt.subplots(int(rows), cols, figsize=(15, 15))
    axes = axes.ravel()  # flatten axes for easy iterating
    selected_features = defaultdict(int)
    all_mse = {}
    all_selected_features = {}

    for idx, degree in enumerate(range(1, max_poly_degree + 1)):
        all_columns = []
        # Reduce correlated features
        if reduce_corr:
            corr_matrix = x_train.corr().abs()
            mask = np.where((np.tri(*corr_matrix.shape)), True, False)
            corr_pairs = corr_matrix[mask].stack()
            results = pd.DataFrame(corr_pairs, columns=['correlation']).reset_index()
            results.columns = ['var_1', 'var_2', 'correlation']
            results = results[results['var_1'] != results['var_2']]
            results.drop_duplicates(inplace=True)
            results = results[results['correlation'] > corr_limit]
            temp_1 = pd.merge(all_feature_importance, results, left_on='feature', right_on='var_1', how='right')
            results = pd.merge(all_feature_importance, temp_1, left_on='feature', right_on='var_2', how='right')
            results.rename(columns={'gini_x': 'gini_var_2', 'gini_y': 'gini_var_1'}, inplace=True)
            results.drop(columns=['feature_x', 'feature_y'], inplace=True)
            mask_col = ['var_1', 'var_2', 'correlation', 'gini_var_1', 'gini_var_2']
            results = results[mask_col]
            results['combined'] = results.apply(lambda row: '-'.join(sorted([row['var_1'], row['var_2']])), axis=1)
            results = results.drop_duplicates(subset=['combined'])
            results = results.drop(columns=['combined'])
            results = results.sort_values(by='correlation', ascending=False).reset_index(drop=True)
            high_corr_drop = []
            for index, col in results.iterrows():
                if results['gini_var_1'][index] > results['gini_var_2'][index]:
                    high_corr_drop.append(results['var_2'][index])
                elif results['gini_var_1'][index] < results['gini_var_2'][index]:
                    high_corr_drop.append(results['var_1'][index])
            print(f'The removed columns are as follows: {high_corr_drop}')
            all_columns = list(all_feature_importance[~all_feature_importance['feature'].isin(high_corr_drop)]['feature'])
        else:
            all_columns = list(all_feature_importance['feature'])  # list of columns without the filtering

        # With the final all_features_importance defined we can proceed to selecting features
        print(f"Running for Polynomial degree = {degree}")
        mse_train_all = {}
        mse_cv_all = {}
        selected_columns = []
        min_mse = float('inf')
        # Continue selecting features until none are left
        while all_columns:
            mse_cv_list_remaining = {}  # define a new dictionary each time
            for i in all_columns:
                standard = StandardScaler()
                train_std = standard.fit_transform(pd.concat([x_train[selected_columns + [i]]], axis=1))

                polyn = PolynomialFeatures(degree=degree, include_bias=False)
                train_std_poly = polyn.fit_transform(train_std)

                model = LinearRegression()
                model.fit(train_std_poly, y_train)

                pred_train = model.predict(train_std_poly)
                mse_train = mean_squared_error(y_train, pred_train)

                x_cv_std = standard.transform(pd.concat([x_cv[selected_columns + [i]]], axis=1))
                x_cv_std_poly = polyn.transform(x_cv_std)
                pred_cv = model.predict(x_cv_std_poly)
                mse_cv = mean_squared_error(y_cv, pred_cv)

                mse_cv_list_remaining[i] = mse_cv
                mse_train_all[i] = mse_train
                mse_cv_all[i] = mse_cv

            min_key = min(mse_cv_list_remaining, key=lambda k: mse_cv_list_remaining[k])

            if mse_cv_list_remaining[min_key] * 1.01 < min_mse:
                min_mse = mse_cv_list_remaining[min_key]
                selected_columns.append(min_key)
                all_columns.remove(min_key)
                selected_features[min_key] += 1

                print(f"Added column {min_key} with the MSE: {min_mse:4f}")
            else:
                break
        all_selected_features[degree] = selected_columns
        all_mse[degree] = min(mse_cv_all.values())
        print('\nSelected features:', selected_columns, '\n')

        # Plot the MSE for train and cv samples
        mse_train_vals = [mse_train_all[i] for i in selected_columns]
        mse_cv_vals = [mse_cv_all[i] for i in selected_columns]

        axes[idx].plot(selected_columns, mse_train_vals, marker='o', label='Train MSE')
        axes[idx].plot(selected_columns, mse_cv_vals, marker='o', label='CV MSE')
        axes[idx].set_title(f'MSE for Polynomial degree = {degree}')
        axes[idx].set_xlabel('Features')
        axes[idx].set_ylabel('MSE')
        axes[idx].tick_params(axis='x', rotation=90)
        axes[idx].legend(loc='best')
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()
    selected_features_twice = [feature for feature, count in selected_features.items() if count >= 2]
    if len(selected_features_twice) < 2:
        selected_degree = min(all_mse, key=all_mse.get)
        selected_features = (
            all_feature_importance)[all_feature_importance['feature'].isin(all_selected_features[selected_degree])]
        print(
            f"Features selected with the minimum CV MSE of {selected_degree} are {all_selected_features[selected_degree]}")
    else:
        print("Features selected at least twice", selected_features_twice)
        selected_features = all_feature_importance[all_feature_importance['feature'].isin(selected_features_twice)]

    return selected_features, all_feature_importance


def linear_neural_regression(x_train, y_train, x_cv, y_cv, x_test, y_test, max_degree=4, epochs=300, verbose=0,
                             learning_rate=0.001):
    """
    Neural network with 4 Dense layers (64, 32, 15, 1) and each with a linear activator. The function runs several
    models in accordance with the indicated degree of polynomial. The model is selected by the smallest Mean Squared
    Error assessed via the sklearn library. A graph will show the mean squared error for each degree of polynomial.
    Finally, the function will also calculate the Mean Squared Error for the test set based on the performance of the
    selected model.

    :param x_train: the train data sample of all the numeric independent features
    :param y_train: the target data sample for the train sample in a numeric format
    :param x_cv: the cross-validation sample of all the numeric independent features
    :param y_cv: the target data sample for the cross validation sample in a numeric format
    :param x_test: the test sample of all the numeric independent features
    :param y_test: the target data sample for the test sample in a numeric format
    :param max_degree: the maximal polynomial degree to transform the variables
    :param epochs: an integer that constitutes the number of epochs each model will be run on, default 300
    :param verbose: an integer that suppresses the text notation of each step in the modelling phase
    :param learning_rate: the learning rate of the model. Default 0.001
    :return: selected_model - the selected model data
             standardscale  - scaling data to transform features
             polynomialft   - polynomial data to transform features
             pred_test      - predictions of the test sample
             mse_test       - the mean squared error on the test sample
    """
    cv_errors = np.zeros(0)
    train_errors = np.zeros(0)
    all_models = {}
    all_standard = {}
    all_poly = {}
    degrees = {}
    for i in range(1, max_degree + 1):
        standard = StandardScaler()
        train_std = standard.fit_transform(x_train)

        poly = PolynomialFeatures(degree=i, include_bias=False)
        train_std_poly = poly.fit_transform(train_std)

        model = tf.keras.models.Sequential()
        model.add(Dense(units=24, activation=LeakyReLU(alpha=0.2)))
        model.add(Dense(units=16, activation=LeakyReLU(alpha=0.2)))
        model.add(Dense(units=8, activation=LeakyReLU(alpha=0.2)))
        model.add(Dense(units=1, activation='linear'))
        model.compile(
            loss=tf.keras.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        )

        model.fit(train_std_poly, y_train, epochs=epochs, verbose=verbose)

        pred_train = model.predict(train_std_poly)
        mse_train = mean_squared_error(y_train, pred_train)
        train_errors = np.append(train_errors, mse_train)

        x_cv_std = standard.transform(x_cv)
        x_cv_std_poly = poly.transform(x_cv_std)
        pred_cv = model.predict(x_cv_std_poly)
        mse_cv = mean_squared_error(y_cv, pred_cv)
        cv_errors = np.append(cv_errors, mse_cv)
        # Append the model data
        all_models[str(i)] = model
        all_standard[str(i)] = standard
        all_poly[str(i)] = poly
        degrees[str(i)] = i
        print(f"Development of the model with polynomial degree of {i}\nThe MSE for the train set is {mse_train:4f} "
              f"and the Cross Validation is: {mse_cv:4f}")

    # Index the smallest CV mse value
    index_best_model = cv_errors.argmin() + 1

    x_values = range(1, max_degree + 1)

    plt.figure()
    plt.title('Train x Validation metrics')
    plt.xlabel('Polynomial degrees')
    plt.ylabel('MSE')
    plt.xticks(ticks=x_values, labels=x_values)
    plt.plot(x_values, train_errors)
    plt.plot(x_values, cv_errors)
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()

    # Assign the best model
    selected_model = all_models[str(index_best_model)]
    standard_selected = all_standard[str(index_best_model)]
    ploy_selected = all_poly[str(index_best_model)]
    # Transform test data
    x_test_std = standard_selected.transform(x_test)
    x_test_std_poly = ploy_selected.transform(x_test_std)
    pred_test = selected_model.predict(x_test_std_poly)
    mse_test = mean_squared_error(y_test, pred_test)

    print(f"The mean squared error of the selected model of {index_best_model} "
          f"polynomial degree on the test sample is {mse_test:4f}")
    return selected_model, standard_selected, ploy_selected, pred_test, mse_test


def linear_regression_ols(x_train, y_train, x_cv, y_cv, x_test, y_test, max_degree=5):
    """
    Computes a linear regression model based on the normal equation (OLS) method to select the coefficients for each
    feature. The features will be transformed using standard scaler and polynomial degree. The optimal model will be
    selected based on all the mean squared errors calculated from on the train and cross validation samples. A graph
    will be presented with the mean squared errors for each model.
    :param x_train: the train data sample of all the
    numeric independent features
    :param y_train: the target data sample for the train sample in a numeric format
    :param x_cv: the cross-validation sample of all the numeric independent features
    :param y_cv: the target data
    sample for the cross validation sample in a numeric format
    :param x_test: the test sample of all the numeric
    independent features
    :param y_test: the target data sample for the test sample in a numeric format
    :param max_degree: the maximal polynomial degree to transform the variables
    :return: selected_model, data standardscale, polynomialft, mse_test
    """
    cv_errors = np.zeros(0)
    train_errors = np.zeros(0)
    all_models = {}
    all_standard = {}
    all_poly = {}
    degrees = {}
    for i in range(1, max_degree + 1):
        # Fit-transform train data
        standard = StandardScaler()
        train_std = standard.fit_transform(x_train)
        polyn = PolynomialFeatures(degree=i, include_bias=False)
        train_std_poly = polyn.fit_transform(train_std)
        # Transform cross validation data
        cv_std = standard.transform(x_cv)
        cv_std_poly = polyn.transform(cv_std)

        # Initialize the model
        model_ols = LinearRegression()
        # Fit the model
        model_ols.fit(train_std_poly, y_train)
        # Predict the train values based on final w and b values
        pred_train = model_ols.predict(train_std_poly)
        # Calculate the mean squared error for the train sample
        mse_train = mean_squared_error(y_train, pred_train)
        # Predict the cross validation values based on final w and b values
        pred_cv = model_ols.predict(cv_std_poly)
        # Calculate the mean squared error for the cross validation sample
        mse_cv = mean_squared_error(y_cv, pred_cv)
        # Append the mse
        cv_errors = np.append(cv_errors, mse_cv)
        train_errors = np.append(train_errors, mse_train)
        # Append the model data
        all_models[str(i)] = model_ols
        all_standard[str(i)] = standard
        all_poly[str(i)] = polyn
        degrees[str(i)] = i
        print(
            f"The mean squared error for the polynomial degree of {i} on the train sample is: {mse_train:4f}, the mean "
            f"squared error on the cross"
            f"validation sample is: {mse_cv:4f}")
    min_value = np.min(np.concatenate([cv_errors, train_errors])) * 0.95
    max_value = min(cv_errors) * 4
    x_values = range(1, max_degree + 1)
    plt.figure()
    plt.title('Train x Validation metrics')
    plt.xlabel('Polynomial degrees')
    plt.ylabel('MSE')
    plt.xticks(ticks=x_values, labels=x_values)
    plt.ylim(min_value, max_value)
    plt.plot(x_values, train_errors)
    plt.plot(x_values, cv_errors)
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    # Index the smallest CV mse value
    index_best_model = cv_errors.argmin() + 1
    print(f"The selected degree of polynomial is {index_best_model}")
    # Assign the best model
    selected_model = all_models[str(index_best_model)]
    standard_selected = all_standard[str(index_best_model)]
    ploy_selected = all_poly[str(index_best_model)]
    # Transform test data
    test_std = standard_selected.transform(x_test)
    test_std_poly = ploy_selected.transform(test_std)
    # Predict the test values based on final w and b values
    pred_test = selected_model.predict(test_std_poly)
    # Calculate the mean squared error for the test sample
    mse_test = mean_squared_error(y_test, pred_test)
    print(f"The mean squared error of the selected model of {index_best_model} "
          f"polynomial degree on the test sample is {mse_test:4f}")

    return selected_model, standard_selected, ploy_selected, pred_test, mse_test


def linear_regression_gradient_descent(x_train, y_train, x_cv, y_cv, x_test, y_test, alpha, num_iters, poly_degree=1,
                                       last_errors=10,
                                       cost_decimals=4):
    """
    Performs linear regression with gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha.

    :param x_train: Train sample
    :param y_train: Train target values
    :param x_cv: Cross Validation sample
    :param y_cv: Cross Validation target values
    :param x_test: the test sample of all the numeric independent features
    :param y_test: the target data sample for the test sample in a numeric format
    :param alpha: Learning rate
    :param num_iters: number of iterations to run gradient descent
    :param poly_degree: The degree of polynomial to transform the variables
    :param last_errors: The number of previous values of the cost function to be similar
    :param cost_decimals: The number of decimals to compare the last value of the cost function and last_errors
    :return: w         - final weight coefficient
             b         - final bias coefficient
             pred_test - predictions of the test sample
             mse_test  - the mean squared error on the test sample
    """
    last_errors = np.negative(last_errors)
    # Fit-transform train data
    standard_gd = StandardScaler()
    train_std = standard_gd.fit_transform(x_train)
    poly_gd = PolynomialFeatures(degree=poly_degree, include_bias=False)
    train_std_poly = poly_gd.fit_transform(train_std)

    # An array to store cost J and w's at each iteration primarily for observing cost reduction during development
    J_history = []
    y_train = y_train.to_numpy().reshape((-1, 1))  # Reshape the target variable for the train sample
    y_cv = y_cv.to_numpy().reshape((-1, 1))  # Reshape the target variable for the cross validation sample
    y_test = y_test.to_numpy().reshape((-1, 1))  # Reshape the target variable for the test sample
    m, n = train_std_poly.shape  # (number of examples, number of features)
    w = np.zeros(n)
    b = 0.
    dj_dw = np.zeros((n,))
    dj_db = 0.
    cost = 0.0
    start = time.time()  # starting the timer

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        for j in range(m):
            err = (np.dot(train_std_poly[j], w) + b) - y_train[j]
            for f in range(n):
                dj_dw[f] = dj_dw[f] + err * train_std_poly[j, f]
            dj_db = dj_db + err
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        # Increase alpha by 10% if more than 5 minutes have passed
        elapsed = time.time() - start
        if elapsed > 240:  # 200 seconds = 5 minutes
            alpha *= 1.1
            start = time.time()  # restart the timer if alpha is increased

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        # Stop gradient decent when the 4 decimals rounded of the last 10 cost function equal to the last cost value
        if len(J_history) > 100:
            if (np.round(J_history[last_errors:-1], cost_decimals) == np.round(cost, cost_decimals)).all():
                break
            else:
                # Print cost at specified intervals
                if i % math.ceil(num_iters / 100) == 0:
                    print(f"Iteration {i}: Cost {J_history[-1]}")
        # Save cost J at each iteration
        for c in range(m):
            f_wb_i = np.dot(train_std_poly[c], w) + b
            cost = cost + (f_wb_i - y_train[c]) ** 2
        cost = cost / (2 * m)
        J_history.append(cost)

    # Predict the train values based on final w and b values
    pred_train = np.dot(train_std_poly, w) + b
    # Calculate the mean squared error for the train sample
    mse_train = mean_squared_error(y_train, pred_train)

    # Transform cross validation data
    cv_std = standard_gd.transform(x_cv)
    cv_std_poly = poly_gd.transform(cv_std)
    # Predict the cross validation values based on final w and b values
    pred_cv = np.dot(cv_std_poly, w) + b
    # Calculate the mean squared error for the cross validation sample
    mse_cv = mean_squared_error(y_cv, pred_cv)

    # Transform test data
    test_std = standard_gd.transform(x_test)
    test_std_poly = poly_gd.transform(test_std)
    # Predict the test values based on final w and b values
    pred_test = np.dot(test_std_poly, w) + b
    # Calculate the mean squared error for the test sample
    mse_test = mean_squared_error(y_test, pred_test)
    print(f"The mean squared error for the train sample is: {mse_train:4f}, the mean squared error for the cross "
          f"validation sample is: {mse_cv:4f} and for the test set is: {mse_test:4f}")

    return w, b, standard_gd, poly_gd, pred_test, mse_test


class Predict_gd:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def predict(self, x):
        predictions = np.dot(x, self.w) + self.b
        return predictions


def best_regression(all_mse, all_models, all_standardscaler, all_polyft):
    min_mse = min(all_mse, key=all_mse.get)
    selected_model = []
    standardization = []
    polynomial_transformation = []

    if min_mse != 'gradient':
        selected_model = all_models[min_mse]
        standardization = all_standardscaler[min_mse]
        polynomial_transformation = all_polyft[min_mse]
    if min_mse == 'gradient':
        selected_model = Predict_gd(w=all_models['gradient']['w'], b=all_models['gradient']['b'])
        standardization = all_standardscaler[min_mse]
        polynomial_transformation = all_polyft[min_mse]
    print(f"The best selected model is {min_mse}")
    return selected_model, standardization, polynomial_transformation


########################################################
#           Section 2: Polt functions                  #
########################################################

def plot_correlation_heatmap(x_train, y_train, cmap='coolwarm', linewidth=.5,
                             annot_kws=None):
    # Compute correlation and round off to 2 decimal places
    if annot_kws is None:
        annot_kws = {"size": 10}
    df = pd.concat([y_train, x_train], axis=1)
    corr = np.around(df.corr(), decimals=2)

    # Create a mask
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Plotting
    plt.figure(figsize=(15, 8))
    sns.heatmap(corr, mask=mask, linewidths=linewidth, cmap=cmap,
                annot=True, annot_kws=annot_kws, fmt=".2f")
    plt.title('Correlation of the selected variables', color='black')
    plt.show()


# noinspection PyIncorrectDocstring
def plot_actual_vs_predicted(y_test, *args, labels, observations=30):
    """
    Plot the values of the actual y and the predicted ones. The first argument always should be the actual values of y.
    :param *args: place any number of dataframe/numpay arrays to be plotted
    :param observations: integer representing the number of last observations to plot.
                         Default value of 30 last observations.
    :return: plt.show()
    """
    observations = np.negative(observations + 1)
    check_test = pd.DataFrame(y_test)
    for i, arg in enumerate(args):
        check_test = pd.concat(
            [check_test.reset_index(drop=True),
             pd.DataFrame(arg).reset_index(drop=True)
             ], axis=1)
    plt.figure(figsize=(15, 6))
    plt.plot(check_test.iloc[observations:-1, 0], label='Actual', marker='o')
    plt.xlabel('Prediction number')
    plt.ylabel('MSE')
    for i, _ in enumerate(args):
        plt.xticks()
        plt.plot(check_test.iloc[observations:-1, i + 1], label=f'Predicted {labels[i]}', marker='o')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
    return plt.show()


def plot_features(x, y):
    """
    Visual assessment of the selected features on scatter plots.

    :param x: the values on the X-axis
    :param y: the values on the Y-axis
    :return: plt.show() function that shows the plots
    """
    aspect_ratio = (8, 5)
    num_columns = 2
    num_rows = (x.shape[1] + 1) // 2  # Calculate the number of rows based on the number of subplots

    # Calculate the size of each subplot based on the aspect ratio
    subplot_size = (5, 5)  # Default size
    if aspect_ratio[0] > aspect_ratio[1]:
        subplot_size = (subplot_size[0], subplot_size[0] * aspect_ratio[1] / aspect_ratio[0])
    else:
        subplot_size = (subplot_size[1] * aspect_ratio[0] / aspect_ratio[1], subplot_size[1])

    fig, axes = plt.subplots((x.shape[1] + 1) // 2, 2,
                             figsize=(num_columns * subplot_size[0], num_rows * subplot_size[1]),
                             constrained_layout=True)
    # Flatten the axes array to access subplots one by one
    axes = axes.flatten()

    for i in range(x.shape[1]):
        axes[i].scatter(x[x.columns[:][i]], y, s=2)
        axes[i].set_xlabel(x.columns[i])
        axes[i].set_ylabel(y.name)
    # Hide any remaining empty subplots (if any)
    for j in range(x.shape[1], len(axes)):
        fig.delaxes(axes[j])

    return plt.show()


def plot_selected_features(selected_features, all_features):
    # Assign Gini values and feature names
    all_features = all_features[~all_features['feature'].isin(selected_features['feature'])]
    gini_values = pd.concat([selected_features['gini'], all_features['gini']]).round(4)
    all_features = pd.concat([selected_features, all_features])['feature']

    # Assign color based on whether the feature is selected or not
    colors = ['red' if feature in selected_features else 'blue' for feature in all_features]

    # Plot
    plt.figure(figsize=[10, 8])
    sns.barplot(x=gini_values, y=all_features, palette=colors)

    # Show values on the graph
    for index, value in enumerate(gini_values):
        plt.text(value, index, str(value))

    # The selected features are at the beginning of the list
    last_selected_feature_index = len(selected_features) - 0.5

    # Draw a horizontal line below the last selected feature
    plt.axhline(last_selected_feature_index, color='yellow', linestyle='--')

    plt.title('Gini values of Features')
    plt.xlabel('Gini values')
    plt.ylabel('Features')

    plt.show()
# %%
