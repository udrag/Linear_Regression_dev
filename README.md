# Linear Regression builder

My first project after completing the Machine Learning Specialization on Coursera focuses on developing a framework that would take your data and produce the best prediction possible by the use of three Linear Regression methods:
   1. Deep Learning (Leaky ReLu)
   2. OLS regression using the normal equation
   3. Gradient Decent linear regression method (manual calculation)

Within this repository, you'll discover classes and functions forming a predictor for estimating quantitative targets. The objective is to offer a versatile framework applicable to any numerical dataset, provided the target value is in the first column, there are no missing values and all values are numerical.

The framework involves the following steps:

1. Divide data into three sets: training, cross-validation, and test.
   - Training identifies coefficients.
   - Cross-validation assesses and selects the model.
   - Testing validates the selected model, minimizing selection bias.
   
2. Identify optimal parameters for the Random Forest Regressor using training and cross-validation sets.

3. Utilize a Random Forest Regressor to compute Gini values for all features (independent variables).

4. Remove highly correlated variables and based on a Linear Regression OLS method select the most relevant variables based on the reduction of Mean Squared Error (MSE) of each variable sequentially.
   
5. Create scatter plots for selected features concerning the target variable.

6. Construct a neural network with Leaky Rectified Linear Units (ReLU) activation.

7. Develop linear regression models using the scikit-learn library, employing the OLS linear regression method.
   
8. Calculate the coefficients using the gradient descent linear regression method through partial derivatives until convergence.

9. Generate plots comparing predicted values from the test set for all three models against actual values.

10. Record model information, standardization, and polynomial transformation details for future use.

Despite the aim to automate modeling, certain functions allow manual parameter input. Manual input is facilitated across all functions.

The project relies on functions in the `utils.py` file, categorized into modeling and plot sections. Note that some modeling functions may include plots, considered part of the modeling functions.
