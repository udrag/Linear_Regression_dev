# Linear_Neural_Networks
My first project after completing the Course Machine Learning Specialisation on Coursera. In this repository, you will find a Linear Neural Network model built to predict Bucharest's next day's average temperature.

In this project, you will find several classes and functions that build a predictor to estimate a quantitative target. This project aims to provide a framework able to work on any numerical data as long as the target value is positioned in the first column and there are no missing values.

The framework will be going through the following steps:

1. Split data into 3 sets: training, cross-validation and test.
     - Training will be used to find the coefficients.
     - Cross-validation will be used to assess and select the model.
     - Test will be used to assess the selected model. This will reduce the selection bias.
3. Find the best parameters for the Random Forest Regressor using train and cross-validation sets.
4. Computes a Random Forest Regressor to calculate the Gini values of the features.
5. Through a Linear Regression OLS method, the MSE is plotted for adding subsequently one by one the features to the model.
6. Scatter plotting the selected features concerning the target variable.
7. Build a neural network with Leaky ReLU (Leaky Rectified Linear Units) activation.
8. Build a linear regression using sklearn library (this method uses the OLS method or normal equation to find the coefficients).
9. Build a linear regression using gradient descent through partial derivatives to update the coefficient until convergence is reached.
10. Plot the predicted values using the test set from all three models vs the actual values from the test set.
11. Assign the model's, standardization and polynomial transformation information for later use.

Please keep in mind that although the aim is to automate the modeling process, manual input might be required. All functions allow manual input of parameters. 
