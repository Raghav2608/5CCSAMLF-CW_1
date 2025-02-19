import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class BaggedGradientBoosting:
    def __init__(self, n=10,sample_fraction=0.8,replace=True,seed=42, **gb_params):
        """
        :param n: Number of GBT models
        :param sample_fraction: Fraction of data for each model
        :param gb_params: Hyperparameters for XGBRegressor
        """
        
        self.n = n
        self.sample_fraction = sample_fraction
        self.models = [XGBRegressor(**gb_params) for _ in range(n)]
        self.replace = replace
        np.random.seed = seed
    
    def fit(self, X, y):
        self.samples = []
        #number of samples to train on 
        n_samples = int(self.sample_fraction * len(X))
        
        for model in self.models:
            indices = np.random.choice(len(X), size=n_samples, replace=self.replace)
            X_sample, y_sample = X[indices], y[indices]
            self.samples.append((X_sample, y_sample))
            model.fit(X_sample, y_sample)
    
    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        return np.mean(preds, axis=0)  # Averaging predictions


class AdaptiveBaggingGBT:
    def __init__(self, n=10,test_size=0.1, sample_fraction=0.8, alpha=1.0, **gb_params):
        """
        :param n: Number of GBT models
        :param sample_fraction: Fraction of data for each model (without replacement)
        :param alpha: Weighting sensitivity factor (higher = more aggressive weighting)
        :param gb_params: Hyperparameters for XGBRegressor
        """
        self.n = n
        self.sample_fraction = sample_fraction
        self.alpha = alpha  # Controls weight sensitivity
        self.models = [XGBRegressor(**gb_params) for _ in range(n)]
        self.weights = np.ones(n)  # Initialize equal weights
        self.test_size = test_size
    
    def fit(self, X, y):
        n_samples = int(self.sample_fraction * len(X))
        errors = []
        
        for i, model in enumerate(self.models):
            # Sample with replacement
            indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Split for validation
            X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=self.test_size)
            
            # Train model
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            
            # Compute MSE
            mse = mean_squared_error(y_val, y_pred_val)
            errors.append(mse)
        
        # Convert errors to adaptive weights using exponential function
        errors = np.array(errors)
        exp_weights = np.exp(-self.alpha * errors)  # Lower error → Higher weight
        self.weights = exp_weights / np.sum(exp_weights)  # Normalize
    
    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.weights, preds)  # Weighted sum
        return weighted_preds


