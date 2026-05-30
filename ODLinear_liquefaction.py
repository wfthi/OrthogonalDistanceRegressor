import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from ODLinear_fast import OrthogonalDistanceLogisticRegression
from sklearn.linear_model import LogisticRegression


# 1. Generate Synthetic Liquefaction Data
np.random.seed(10)
n_samples = 40
# 'True' Soil Density (Relative)
x_true = np.linspace(10, 50, n_samples)

# Transition: Soil usually liquefies below 30 SPT counts
# Probability of Liquefaction
p_true = expit(-(x_true - 30) * 0.5) 
y_obs = np.random.binomial(1, p_true)

# 2. Add Measurement Stress
# We simulate a "Borehole Error": 
# Several points near the boundary have high uncertainty
x_err = np.full(n_samples, 1.5) 
x_err[15:25] = 8.0  # Massive error near the transition zone

# Simulate the actual measurements (noisy X)
x_obs = x_true + np.random.normal(0, x_err)

# 3. Standard OLS Logistic Regression
ols = LogisticRegression(C=1e5)
ols.fit(x_obs.reshape(-1, 1), y_obs)

# 4. Your ODR Logistic Regression
C = 10.
odr_lr = OrthogonalDistanceLogisticRegression(C=C)
odr_lr.fit(x_obs, y_obs, X_err=x_err)

# 5. Visualization
x_plot = np.linspace(5, 55, 100)
y_ols = ols.predict_proba(x_plot.reshape(-1, 1))[:, 1]
y_odr = odr_lr.predict_proba(x_plot)[:, 1]

plt.figure(figsize=(10, 6))
plt.errorbar(x_obs, y_obs, xerr=x_err, fmt='o', color='gray', alpha=0.3, label='Measured SPT (with error bars)')
plt.scatter(x_obs, y_obs, c=y_obs, cmap='coolwarm', edgecolors='k', zorder=5)

plt.plot(x_plot, y_ols, 'r--', label='OLS Logistic (Ignores X error)')
plt.plot(x_plot, y_odr, 'k-', linewidth=2, label='ODR Logistic (Accounts for X error)')

plt.axvline(30, color='green', linestyle=':', label='True Physical Boundary')
plt.xlabel("Soil Strength (SPT Count)")
plt.ylabel("Liquefaction Probability")
plt.title(f"Stress Test: Soil Safety with Large Measurement Uncertainty - C={C}")
plt.grid()
plt.legend()
plt.show()

print(f"C= {C}")
print(f"OLS Intercept: {ols.intercept_[0]:.2f}, Coef: {ols.coef_[0][0]:.2f}")
print(f"ODR Intercept: {odr_lr.intercept_:.2f}, Coef: {odr_lr.coef_[0]:.2f}")
print(f"Uncertainties: {odr_lr.uncertainty_}")
print(f"Quasi/chi: {odr_lr.quasi_chisq_}")

# --- Stressing the C-Scaling Features ---

# 1. Add 'Malfunctioning Sensor' Outliers
x_outliers = np.array([12.0, 48.0])
y_outliers = np.array([0, 1])  
x_err_outliers = np.array([1.5, 1.5])

x_obs_polluted = np.append(x_obs, x_outliers)
y_obs_polluted = np.append(y_obs, y_outliers)
x_err_polluted = np.append(x_err, x_err_outliers)

# 2. Run t ODR with C-Scaling
# Using C=10.0 to rein in the coefficients
C = 20.
odr_model = OrthogonalDistanceLogisticRegression(
    C=C
)

# Fit will now handle the case where y_err is None
odr_model.fit(x_obs_polluted, y_obs_polluted, X_err=x_err_polluted)

# 3. Compare with OLS
ols_polluted = LogisticRegression(C=1e3) # High C for OLS to show the bias
ols_polluted.fit(x_obs_polluted.reshape(-1, 1), y_obs_polluted)

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_obs_polluted, y_obs_polluted, c=y_obs_polluted, cmap='coolwarm', edgecolors='k', alpha=0.6)
plt.scatter(x_outliers, y_outliers, color='red', marker='X', s=200, label='Extreme Outliers')

x_plot = np.linspace(5, 55, 100)
# Standardize OLS prediction call
plt.plot(x_plot, ols_polluted.predict_proba(x_plot.reshape(-1, 1))[:, 1], 'r--', label='OLS (Biased)')
plt.plot(x_plot, odr_model.predict_proba(x_plot)[:, 1], 'k-', linewidth=2, label='ODR')

plt.title(f"Stress Test: ODR vs. Data Pollution (C={C})")
plt.xlabel("Soil Strength (SPT)")
plt.ylabel("Liquefaction Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ------

print()
print("'Malfunctioning Sensor' Outliers")
print(f"C= {C}")
print(f"OLS Intercept: {ols_polluted.intercept_[0]:.2f}, Coef: {ols_polluted.coef_[0][0]:.2f}")
print(f"ODR Intercept: {odr_model.intercept_:.2f}, Coef: {odr_model.coef_[0]:.2f}")
print(f"Uncertainties: {odr_model.uncertainty_}")
print(f"Quasi/chi: {odr_model.quasi_chisq_}")
 