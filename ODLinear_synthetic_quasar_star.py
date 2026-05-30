# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:36:34 2026

@author: thi_wi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def generate_hard_case(n_samples=2000):
    np.random.seed(42)
    # Class 0: 'Stars' centered at (0, 0)
    # Class 1: 'Quasars' centered at (0.4, 0.4) - very close!
    y = np.random.randint(0, 2, n_samples)
    
    # Intrinsic (True) positions
    X_true = np.zeros((n_samples, 2))
    X_true[y == 0] = np.random.multivariate_normal([0, 0], [[0.1, 0.05], [0.05, 0.1]], sum(y==0))
    X_true[y == 1] = np.random.multivariate_normal([0.4, 0.4], [[0.1, 0.05], [0.05, 0.1]], sum(y==1))
    
    # ADD MASSIVE MEASUREMENT NOISE (The Attenuation Source)
    # 0.3 noise on a 0.4 separation is "The Danger Zone"
    X_err = 0.3
    X_noisy = X_true + np.random.normal(0, X_err, X_true.shape)
    
    return X_noisy, y, X_err, X_true

X_noisy, y, err_val, X_true = generate_hard_case()

# Train/Test Split
split = 1500
X_train, X_test = X_noisy[:split], X_noisy[split:]
y_train, y_test = y[:split], y[split:]
X_err_train = np.full_like(X_train, err_val)

# --- MODEL 1: RANDOM FOREST ---
rf = RandomForestClassifier(n_estimators=100, max_depth=5) # Shallow to prevent total chaos
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# --- MODEL 2: ODR CLASSIFIER ---
def logistic_func(beta, x):
    z = beta[0] + np.dot(beta[1:], x)
    return 1 / (1 + np.exp(-np.clip(z, -100, 100)))

data_odr = odr.RealData(X_train.T, y_train, sx=X_err_train.T)
model_odr = odr.Model(logistic_func)
my_odr = odr.ODR(data_odr, model_odr, beta0=[0.0, 1.0, 1.0])
res = my_odr.run()

odr_probs = logistic_func(res.beta, X_test.T)
odr_preds = (odr_probs > 0.5).astype(int)
odr_acc = accuracy_score(y_test, odr_preds)

print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"ODR Accuracy:           {odr_acc:.4f}")
print(f"ODR Optimized Weights:  {res.beta[1:]}")
