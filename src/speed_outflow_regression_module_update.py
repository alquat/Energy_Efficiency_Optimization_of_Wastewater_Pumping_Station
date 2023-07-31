import numpy as np
from sklearn.linear_model import LinearRegression

class LinearFunctionRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.predictor_ = None

    def fit(self, X, y, break_point):
        # Check that X is a 1D array or scalar
        if np.ndim(X) > 1:
            raise ValueError("X must be a 1D array or scalar")

        # Create a new feature matrix with a linearly increasing function
        X_linear = np.where(X >= break_point, X - break_point, -2.68284027721939).reshape(-1, 1)

        # Fit a linear regression model to the linear feature matrix
        reg = LinearRegression()
        reg.fit(X_linear, y)

        # Store the coefficients and intercept
        self.coef_ = reg.coef_
        self.intercept_ = reg.intercept_

        # Store the predictor function as a lambda expression
        self.predictor_ = lambda x: np.where(x >= break_point, self.intercept_ + self.coef_[0] * (x - break_point), -2.068284027721939)

    def predict(self, X):
        # Check that the model has been fit
        if self.predictor_ is None:
            raise RuntimeError("Model has not been fit")

        # Check that X is a 1D array or scalar
        if np.ndim(X) > 1:
            raise ValueError("X must be a 1D array or scalar")

        # Predict the output values using the predictor function
        return self.predictor_(X)
    
    def save(self, filename, break_point):
        # Save the coefficients, intercept and break_point to a file
        np.savez(filename, coef=self.coef_, intercept=self.intercept_, break_point=break_point)

    def load(self, filename):
        # Load the coefficients, intercept and break_point from a file
        data = np.load(filename,allow_pickle=True)
        self.coef_ = data['coef']
        self.intercept_ = data['intercept']
        self.break_point_ = data['break_point']
        self.r_squared_ = None  # R-squared value is not saved

        # Define the piecewise linear predictor function
        self.predictor_ = lambda x: np.where(x >= self.break_point_, self.intercept_ + self.coef_[0] * (x - self.break_point_), -2.068284027721939)

    def predict(self, X):
        # Check that the model has been fit
        if self.predictor_ is None:
            raise RuntimeError("Model has not been fit")

        # Check that X is a 1D array or scalar
        if np.ndim(X) > 1:
            raise ValueError("X must be a 1D array or scalar")

        # Predict the output values using the predictor function
        return self.predictor_(X)


