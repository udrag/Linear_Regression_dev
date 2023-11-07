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
        
        x_positions = np.arange(len(x_train.columns))
        x_labels = x_train.columns

        min_value = min(mse_train_list + mse_cv_list) * 0.95
        max_value = min(mse_cv_list) * 2

        axes[d - 1].set_title(f'Train x Validation x Test for {d} degree')
        axes[d - 1].plot(x_positions, mse_train_list, label='Train')
        axes[d - 1].plot(x_positions, mse_cv_list, label='Validation')
        axes[d - 1].set_ylim(min_value, max_value)
        axes[d - 1].set_xticks(x_positions)
        axes[d - 1].set_xticklabels(x_labels, rotation=85)
        axes[d - 1].set_xlabel('Features added')
        axes[d - 1].set_ylabel('MSE')
        axes[d - 1].legend()

    # Hide any remaining empty subplots (if any)
    for j in range(4, len(axes)):
        fig.delaxes(axes[j])

    return plt.show()

def plot_actual_vs_predicted(y_test, *args, observations=30):
    """
    Plot the values of the actual y and the predicted ones. The first argument always should be the actual values of y.
    :param *args: place any number of dataframe/numpay arrays to be plotted
    :param observations: integer representing the number of last observations to plot.
                         Default value of 30 last observations.
    :return: plt.show()
    """
    observations = np.negative(observations+1)
    check_test = pd.DataFrame(y_test)
    for i, arg in enumerate(args):
        check_test = pd.concat(
            [check_test.reset_index(drop=True),
             pd.DataFrame(arg).reset_index(drop=True)
             ], axis=1)
    plt.figure(figsize=(15,8))
    plt.plot(check_test.iloc[observations:-1, 0], label='Actual', marker='o')
    plt.xlabel('Prediction number')
    plt.ylabel('MSE')
    for i, _ in enumerate(args):
        plt.xticks()
        plt.plot(check_test.iloc[observations:-1,i+1], label=f'Predicted {i+1}', marker='o')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
    return plt.show()

def gradient_descent(x_train, y_train, x_cv, y_cv, x_test, y_test, alpha, num_iters, poly_degree=1):
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
    :return: w - final weight coefficient
             b - final bias coefficient
             pred_test - predictions of the test sample
             J_history - the cost value at each itteration
    """

    # Fit-transform train data
    standard = StandardScaler()
    train_std = standard.fit_transform(x_train)
    polyn = PolynomialFeatures(degree=poly_degree, include_bias=False)
    train_std_poly = polyn.fit_transform(train_std)
    # Transform cross validation data
    cv_std = standard.transform(x_cv)
    cv_std_poly = polyn.transform(cv_std)
    # Transform test data
    test_std = standard.transform(x_test)
    test_std_poly = polyn.transform(test_std)
    # An array to store cost J and w's at each iteration primarily for observing cost reduction during development
    J_history = []

    y_train = y_train.to_numpy().reshape((-1,1)) # Reshape the target variable for the train sample
    y_cv = y_cv.to_numpy().reshape((-1,1)) # Reshape the target variable for the cross validation sample
    y_test = y_test.to_numpy().reshape((-1,1)) # Reshape the target variable for the test sample
    m, n = train_std_poly.shape   #(number of examples, number of features)
    w = np.zeros(n)
    b = 0.
    dj_dw = np.zeros((n,))
    dj_db = 0.
    cost = 0.0

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        for j in range(m):
            err = (np.dot(train_std_poly[j], w) + b) - y_train[j]
            for f in range(n):
                dj_dw[f] = dj_dw[f] + err * train_std_poly[j, f]
            dj_db = dj_db + err
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        # Stop gradient decent when the 4 decimals rounded of the last 10 cost function values equal to the current cost value
        if len(J_history) > 100:
            if (np.round(J_history[-10:-1],4) == np.round(cost, 4)).all() == True:
                break
            else:
                # Print cost at specified intervals
                if i% math.ceil(num_iters / 10) == 0:
                    print(f"Iteration {i}: Cost {J_history[-1]}")
        # Save cost J at each iteration
        for c in range(m):
            f_wb_i = np.dot(train_std_poly[c], w) + b
            cost = cost + (f_wb_i - y_train[c])**2
        cost = cost / (2 * m)
        J_history.append(cost)

    # Predict the train values based on final w and b values
    pred_train = np.dot(train_std_poly, w) + b
    # Calculate the mean squared error for the train sample
    mse_train = mean_squared_error(y_train, pred_train)
    # Predict the cross validation values based on final w and b values
    pred_cv = np.dot(cv_std_poly, w) + b
    # Calculate the mean squared error for the cross validation sample
    mse_cv = mean_squared_error(y_cv, pred_cv)
    # Predict the test values based on final w and b values
    pred_test = np.dot(test_std_poly, w) + b
    # Calculate the mean squared error for the test sample
    mse_test = mean_squared_error(y_test, pred_test)
    print(f"The mean squared error for the train sample is: {mse_train}, the mean squared error for the cross "
          f"validation sample is: {mse_cv} and for the test set is: {mse_test}")

    return w, b, pred_test, J_history

def linear_regression_ols(x_train, y_train, x_cv, y_cv, x_test, y_test, max_degree=5):
    """
    Computes a linear regression model based on the normal equation (OLS) method to select the coefficients for each feature. The features
    will be transformed using standard scaler and polynomial degree.
    The optimal model will be selected based on all the mean squared errors calculated from on the train and cross validation samples.
    A graph will be presented with the mean squared errors for each model.
    :param x_train: the train data sample of all the numeric independent features
    :param y_train: the target data sample for the train sample in a numeric format
    :param x_cv: the cross-validation sample of all the numeric independent features
    :param y_cv: the target data sample for the cross validation sample in a numeric format
    :param x_test: the test sample of all the numeric independent features
    :param y_test: the target data sample for the test sample in a numeric format
    :param max_degree: the maximal polynomial degree to transform the variables
    :return: selected_model - the selected model data
             standardscale  - scaling data to transform features
             polynomialft   - polynomial data to transform features
             mse_test       - the mean squared error on the test sample
    """
    cv_errors    = np.zeros(0)
    train_errors = np.zeros(0)
    all_models   = {}
    all_standard = {}
    all_poly     = {}
    degrees      = {}
    for i in range(1, max_degree + 1):
        # Fit-transform train data
        standard             = StandardScaler()
        train_std            = standard.fit_transform(x_train)
        polyn                = PolynomialFeatures(degree=i, include_bias=False)
        train_std_poly       = polyn.fit_transform(train_std)
        # Transform cross validation data
        cv_std               = standard.transform(x_cv)
        cv_std_poly          = polyn.transform(cv_std)

        # Initialize the model
        model_ols            = LinearRegression()
        # Fit the model
        model_ols.fit(train_std_poly, y_train)
        # Predict the train values based on final w and b values
        pred_train           = model_ols.predict(train_std_poly)
        # Calculate the mean squared error for the train sample
        mse_train            = mean_squared_error(y_train, pred_train)
        # Predict the cross validation values based on final w and b values
        pred_cv              = model_ols.predict(cv_std_poly)
        # Calculate the mean squared error for the cross validation sample
        mse_cv               = mean_squared_error(y_cv, pred_cv)
        # Append the mse
        cv_errors            = np.append(cv_errors, mse_cv)
        train_errors         = np.append(train_errors, mse_train)
        # Append the model data
        all_models[str(i)]   = model_ols
        all_standard[str(i)] = standard
        all_poly[str(i)]     = polyn
        degrees[str(i)]      = i
        print(f"The mean squared error for the polynomial degree of {i} on the train sample is: {mse_train}, the mean squared error on the cross "
              f"validation sample is: {mse_cv}")
    x_values = range(1, max_degree+1)
    plt.figure()
    plt.title('Train x Validation metrics')
    plt.xlabel('Polynomial degrees')
    plt.ylabel('MSE')
    plt.xticks(ticks=x_values, labels=x_values)
    plt.plot(x_values, train_errors)
    plt.plot(x_values, cv_errors)
    plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    # Index the smallest CV mse value
    diff_mse = cv_errors - train_errors
    for i, (v, j, k) in enumerate(zip(diff_mse, train_errors, cv_errors)):
        if v < np.min(diff_mse)*1.4 and v > 0 and k < np.min(cv_errors)*1.2:
            index_best_model = i + 1
    print(f"The selected degree of polynomial is {index_best_model}")
    # Assign the best model
    selected_model    = all_models[str(index_best_model)]
    standard_selected = all_standard[str(index_best_model)]
    ploy_selected     = all_poly[str(index_best_model)]
    # Transform test data
    test_std          = standard_selected.transform(x_test)
    test_std_poly     = ploy_selected.transform(test_std)
    # Predict the test values based on final w and b values
    pred_test         = selected_model.predict(test_std_poly)
    # Calculate the mean squared error for the test sample
    mse_test          = mean_squared_error(y_test, pred_test)
    print(f"The mean squared error of the selected model of {index_best_model + 1} polynomial degree on the test sample is {mse_test}")

    return selected_model, standard_selected, ploy_selected, pred_test, mse_test


def linear_regression_gradient_descent(x_train, y_train, x_cv, y_cv, x_test, y_test, alpha, num_iters, poly_degree=1):
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
    :return: w         - final weight coefficient
             b         - final bias coefficient
             pred_test - predictions of the test sample
             mse_test  - the mean squared error on the test sample
    """

    # Fit-transform train data
    standard_gd       = StandardScaler()
    train_std         = standard_gd.fit_transform(x_train)
    poly_gd           = PolynomialFeatures(degree=poly_degree, include_bias=False)
    train_std_poly    = poly_gd.fit_transform(train_std)

    # An array to store cost J and w's at each iteration primarily for observing cost reduction during development
    J_history = []
    y_train   = y_train.to_numpy().reshape((-1,1)) # Reshape the target variable for the train sample
    y_cv      = y_cv.to_numpy().reshape((-1,1)) # Reshape the target variable for the cross validation sample
    y_test    = y_test.to_numpy().reshape((-1,1)) # Reshape the target variable for the test sample
    m, n      = train_std_poly.shape   #(number of examples, number of features)
    w         = np.zeros(n)
    b         = 0.
    dj_dw     = np.zeros((n,))
    dj_db     = 0.
    cost      = 0.0
