# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:21:42 2026

@author: thi_wi
"""
import numpy as np
from scipy import odr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from astroML.datasets import fetch_rrlyrae_combined

# 1. Load and Prepare Data
# data[0] is the matrix of colors, data[1] is the array of types (0 or 1)
X_raw, y = fetch_rrlyrae_combined()

# The columns in astroML's RR Lyrae dataset are ordered: 
# u-g, g-r, r-i, i-z
ug = X_raw[:, 0]
gr = X_raw[:, 1]
ri = X_raw[:, 2]
iz = X_raw[:, 3]

# Stack the features you want for the model
X = np.column_stack((ug, gr))
y = y.astype(int)

# 2. Simulate Observation Noise
# In real astronomy, each band has its own error. We'll add 0.05 mag noise.
np.random.seed(42)
X_error = np.full_like(X, 0.3)
X_noisy = X + np.random.normal(0, X_error)

# Split into simple Train/Test
train_idx = np.random.choice(len(X), 5000, replace=False)
test_idx = np.delete(np.arange(len(X)), train_idx)[:2000]

X_train, y_train = X_noisy[train_idx], y[train_idx]
X_test, y_test = X_noisy[test_idx], y[test_idx]
X_err_train = X_error[train_idx]

# 3. Random Forest (Standard Baseline)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# 4. ODR Classification (Your Scheme)
# For binary classification, we use a Logistic-style ODR function
def logistic_func(beta, x):
    # beta: [intercept, slope1, slope2]
    # x: shape (2, n) from ODR
    
    # beta[1:] has shape (2,)
    # np.dot(beta[1:], x) will result in a (n,) vector
    z = beta[0] + np.dot(beta[1:], x)
    
    # Use np.clip to prevent overflow in exp for very large/small z
    return 1 / (1 + np.exp(-np.clip(z, -100, 100)))

def fit_odr_classifier(X, y, X_err):
    # ODR needs a continuous Y, so we use the binary labels directly
    # In a full multinomial version, you would use the OVR/Softmax logic
    data_odr = odr.RealData(X.T, y, sx=X_err.T)
    model = odr.Model(logistic_func)
    # Starting guess: [intercept, slope_ug, slope_gr]
    my_odr = odr.ODR(data_odr, model, beta0=[0.0, 1.0, -1.0])
    return my_odr.run()

odr_res = fit_odr_classifier(X_train, y_train, X_err_train)
# Predict: Round the logistic output to 0 or 1
odr_probs = logistic_func(odr_res.beta, X_test.T)
odr_preds = (odr_probs > 0.5).astype(int)

# 5. Results
print("--- RR LYRAE CLASSIFICATION BENCHMARK ---")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"ODR Classifier Accuracy: {accuracy_score(y_test, odr_preds):.4f}")
print("-" * 40)
print(f"ODR Optimized Weights (Slopes): {odr_res.beta[1:]}")
