import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Ensure we can compare both
import ODLinear as old_lib
import ODLinear_fast as new_lib

# 1. SETUP DATA FIRST (Fixes the NameError)
np.random.seed(42)
n_samples = 30
x0 = np.random.multivariate_normal([1.0, 0.3], [[0.05, 0], [0, 0.01]], n_samples)
x1 = np.random.multivariate_normal([2.5, 0.8], [[0.1, 0], [0, 0.02]], n_samples)
x2 = np.random.multivariate_normal([4.0, 0.5], [[0.2, 0], [0, 0.05]], n_samples)

X_obs = np.vstack([x0, x1, x2])
y = np.concatenate([np.zeros(n_samples), np.ones(n_samples), np.full(n_samples, 2)])

X_err_vec = np.array([0.3, 0.15]) 
X_err_raw = np.tile(X_err_vec, (X_obs.shape[0], 1))

print("--- DIAGNOSTIC: OLD VS NEW MULTINOMIAL ---")

# 2. FIT OLD (The 97% Champion)
om = old_lib.OrthogonalDistanceMultinomialLogisticRegression(C=15.0)
om.fit(X_obs, y, X_err=X_err_raw)
print(f"[OLD CODE] Fit successful. Beta mean: {np.mean(om.beta):.6f}")

# 3. FIT NEW (Testing for Failure)
nm = new_lib.OrthogonalDistanceMultinomialLogisticRegression(C=15.0)

# Add this to your test_multinomial_versions.py after fitting 'nm'
print("\n--- ATTRIBUTE SEARCH (NEW CODE) ---")
attrs = [a for a in dir(nm) if not a.startswith('__')]
print("Available attributes:", attrs)

try:
    nm.fit(X_obs, y, X_err=X_err_raw)
    if hasattr(nm, 'beta'):
        print(f"[NEW CODE] Fit successful. Beta mean: {np.mean(nm.beta):.6f}")
    else:
        print("[NEW CODE] Fit method returned, but 'beta' attribute is missing!")
except Exception as e:
    print(f"[NEW CODE] FIT CRASHED: {e}")

# 4. RESULTS
y_pred_old = om.predict(X_obs)
acc_old = accuracy_score(y, y_pred_old)
print(f"\nOLD Accuracy: {acc_old:.4f}")

try:
    y_pred_new = nm.predict(X_obs)
    acc_new = accuracy_score(y, y_pred_new)
    print(f"NEW Accuracy: {acc_new:.4f}")
except Exception as e:
    print(f"NEW Predict Failed: {e}")