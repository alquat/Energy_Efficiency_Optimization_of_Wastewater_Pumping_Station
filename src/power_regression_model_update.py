import numpy as np
from sklearn.linear_model import LinearRegression

import numpy as np
from sklearn.linear_model import LinearRegression

class PowerQuadraticRegression():
        
    def fit(self, X, y, poly_degree):
        print("fitting new module")
        # Check that X is a 1D array
        if X.ndim != 1:
            raise ValueError("X must be a 1D array")

        # Perform polynomial fitting using polyfit
        coeffs = np.polyfit(X, y, deg= poly_degree)
        
        # Store the coefficients and intercept
        self.coef_ = coeffs[:-1]
        self.intercept_ = coeffs[-1]
        
        # Calculate the R-squared value
        y_pred = np.polyval(coeffs, X)
        ssr = np.sum((y_pred - y.mean())**2)
        sst = np.sum((y - y.mean())**2)
        self.r_squared_ = 1 - (ssr / sst)
        
        # Store the predictor function as a lambda expression
        self.predictor_ = lambda x: np.polyval(np.concatenate((self.coef_, [self.intercept_])), x)


    def save(self, filename):
        # Save the coefficients to a file
        np.savez(filename, coef=self.coef_, intercept=self.intercept_)

    def load(self, filename):
        # Load the coefficients from a file
        data = np.load(filename)
        self.coef_ = data['coef']
        self.intercept_ = data['intercept']
        self.r_squared_ = None  # R-squared value is not saved
        self.predictor_ = lambda x: self.intercept_ + self.coef_[0]*x + self.coef_[1]*x**2

    def predict(self, X):
        # Check that the model has been fit
        if self.predictor_ is None:
            raise RuntimeError("Model has not been fit")

        # Check that X is a 1D array or scalar
        if np.ndim(X) > 1:
            raise ValueError("X must be a 1D array or scalar")

        # Predict the output values using the predictor function
        return self.predictor_(X)




def regression_metrics(y_true, y_pred):
    # Compute the mean of the true values
    y_mean = np.mean(y_true)

    # Compute the sum of squares due to regression
    ssr = np.sum((y_pred - y_mean)**2)

    # Compute the sum of squares due to error
    sse = np.sum((y_true - y_pred)**2)

    # Compute the total sum of squares
    sst = np.sum((y_true - y_mean)**2)

    # Compute the R-squared value
    r_squared = ssr / sst

    # Compute the mean squared error (MSE)
    mse = sse / len(y_true)

    return r_squared, mse