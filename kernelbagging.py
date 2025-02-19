import numpy as np
from sklearn.kernel_ridge import KernelRidge

#class for implementing Bagged Kernel Ridge 

class BaggedKernelRidge:
    """
        :param n: Number of Kernel Ridge models
        :param sample_fraction: Fraction of data for each model (with replacement)
        :param gb_params: Hyperparameters for KernelRidge
    """
    def __init__(self, n=10,sample_fraction=0.8, **gb_params):
        self.n_estimators = n
        self.sample_fraction = sample_fraction
        self.models = [KernelRidge(**gb_params) for _ in range(n)]
    
    def fit(self, X, y):
        self.samples = []
        n_samples = int(self.sample_fraction * len(X))
        for model in self.models:
            indices = np.random.choice(len(X), size=n_samples, replace=False)
            X_sample, y_sample = X[indices], y[indices]
            self.samples.append((X_sample, y_sample))
            model.fit(X_sample, y_sample)
    
    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        return np.mean(preds, axis=0)  # Averaging predictions


