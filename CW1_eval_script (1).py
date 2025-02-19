import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from GradientBagging import BaggedGradientBoosting
# Set seed
np.random.seed(123)

# Import training data
trn = pd.read_csv('CW1_train.csv')
X_tst = pd.read_csv('CW1_test.csv') # This does not include true outcomes (obviously)

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']

# One-hot encode categorical variables
trn = pd.get_dummies(trn, columns=categorical_cols, drop_first=True)

# Train your model (using a simple LM here as an example)
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']

model =  BaggedGradientBoosting(n=25,sample_fraction=1,seed=40,n_estimators=2000,max_depth=3,max_bin=100,reg_alpha=100,
                      booster="gbtree",tree_method="hist",grow_policy="lossguide",
                      eta=0.007,subsample=0.8)
model.fit(X_trn, y_trn)

# Test set predictions
yhat_lm = model.predict(X_tst)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission_K23098138.csv', index=False) # Please use your k-number here



