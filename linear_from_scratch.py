import numpy as np


def standardize_features(X):
    """
    Standardize features to have mean 0 and standard deviation 1.
    
    X: input features
    Returns standardized X, mean, and std for later use
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def mse(y_true, y_pred):
    """Calculates the mean squared error between actual and predicted values."""
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    """Calculates the mean absolute error between actual and predicted values."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """Calculates the R-squared coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


class LinearRegression:
    """
    Base class for linear regression models.
    Provides shared prediction and scoring functionality.
    """
    
    def __init__(self, fit_intercept=True):
        """
        fit_intercept: whether to include an intercept term in the model
        """
        self.fit_intercept = fit_intercept
        self.coefficients_ = None
        self.intercept_ = None
    
    def predict(self, X):
        """
        Generate predictions for new data using the trained model.
        
        X: input features of shape (n_samples, n_features)
        """
        X = np.array(X)
        return X @ self.coefficients_ + self.intercept_
    
    def score(self, X, y):
        """
        Calculate the R-squared score to evaluate model performance.
        
        X: input features of shape (n_samples, n_features)
        y: true target values of shape (n_samples,)
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class LinearRegressionFromScratch(LinearRegression):
    """
    Linear regression using ordinary least squares method.
    Works with single or multiple input features.
    """
    
    def __init__(self, fit_intercept=True):
        """
        fit_intercept: whether to include an intercept term in the model
        """
        super().__init__(fit_intercept)
    
    def fit(self, X, y):
        """
        Train the model using the normal equation method.
        
        X: training features of shape (n_samples, n_features)
        y: target values of shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        try:
            self.coefficients_ = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            self.coefficients_ = np.linalg.pinv(X.T @ X) @ X.T @ y
        
        if self.fit_intercept:
            self.intercept_ = self.coefficients_[0]
            self.coefficients_ = self.coefficients_[1:]
        else:
            self.intercept_ = 0


class GradientDescentLinearRegression(LinearRegression):
    """
    Linear regression using gradient descent optimization.
    Supports stochastic, mini-batch, and batch gradient descent variants.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, 
                 tolerance=1e-6, fit_intercept=True, batch_size=None):
        """
        learning_rate: size of each step during optimization
        max_iterations: maximum number of gradient descent iterations
        tolerance: convergence threshold to stop training early
        fit_intercept: whether to include an intercept term
        batch_size: number of samples per batch (None = use all samples)
        """
        super().__init__(fit_intercept)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.cost_history_ = []
    
    def fit(self, X, y, verbose=False):
        """
        Train the model using iterative gradient descent.
        
        X: training features of shape (n_samples, n_features)
        y: target values of shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        self.coefficients_ = np.zeros(X.shape[1])
        n_samples = X.shape[0]
        batch_size = self.batch_size if self.batch_size is not None else n_samples
        
        indices = np.arange(n_samples)
        cost = 0
        
        for iteration in range(self.max_iterations):
            np.random.shuffle(indices)
            prev_coef = self.coefficients_.copy()
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                y_pred = X_batch @ self.coefficients_
                gradient = (2 / len(batch_indices)) * X_batch.T @ (y_pred - y_batch)
                
                self.coefficients_ = self.coefficients_ - self.learning_rate * gradient
            
            y_pred_full = X @ self.coefficients_
            cost = np.mean((y_pred_full - y) ** 2)
            self.cost_history_.append(cost)
            
            if np.linalg.norm(self.coefficients_ - prev_coef) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        if self.fit_intercept:
            self.intercept_ = self.coefficients_[0]
            self.coefficients_ = self.coefficients_[1:]
        else:
            self.intercept_ = 0
    
    def get_batch_type(self):
        """Get type of gradient descent based on batch size."""
        if self.batch_size is None:
            return "Batch (Full Dataset)"
        elif self.batch_size == 1:
            return "Stochastic (SGD)"
        else:
            return f"Mini-Batch ({self.batch_size} samples)"

