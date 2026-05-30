"""
Attenuation Bias (or regression dilution)

Attenuation Bias (or regression dilution) is most destructive when the noise 
in the features ($\sigma_x$) is comparable to the intrinsic width of the class 
distribution ($\sigma_{intrinsic}$).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import RealData, Model, ODR
from sklearn.linear_model import LinearRegression
from ODLinear_fast import *

# 1. Generate Data: y = x^5
np.random.seed(42)
x_true = np.linspace(1.5, 10, 30)
y_true = x_true**5

# Add measurement errors (Fixed standard deviations)
x_std, y_std = 0.2, 10.0
x_obs = x_true + np.random.normal(0, x_std, len(x_true))
y_obs = y_true + np.random.normal(0, y_std, len(y_true))

# Add INTRINSIC BIAS to foul the solver at the low end
# Simulating a sensor "floor" or offset for small values
x_obs[:5] += 0.8 

# 2. Log-Log Transformation for Linear Fitting
# ln(y) = 5 * ln(x) -> Slope should be 5
ln_x = np.log(x_obs)
ln_y = np.log(y_obs)

# Propagate errors to Log space: sigma_lnx approx sigma_x / x
sx_ln = x_std / x_obs
sy_ln = y_std / y_obs

# 3. Standard OLS Run (scikit-learn)
ols = LinearRegression()
ols.fit(ln_x.reshape(-1, 1), ln_y)
slope_ols = ols.coef_[0]

# 4. ODR Run (Using your methodology)
def linear_func(p, x):
    return p[0] * x + p[1]

# 4. ODR CODE: OrthogonalDistanceLinearRegression
# We use the C-scaling and the specific weighting logic of your class
odlr = OrthogonalDistanceLinearRegression(
    C=1e5, 
    maxit=100, 
    error_type='std', 
    verbose=False
)

# Fitting using your class's handling of X_err and y_err
odlr.fit(ln_x, ln_y, X_err=sx_ln, y_err=sy_ln, initial_guess=[5.0, 0.0])

# 5. Visualization
plt.figure(figsize=(10, 6))
plt.errorbar(ln_x, ln_y, xerr=sx_ln, yerr=sy_ln, fmt='o', alpha=0.5, label='Data (Noisy + Bias)')

# Predict using your class
x_fit = np.linspace(ln_x.min(), ln_x.max(), 100)
y_fit_odr = odlr.predict(x_fit)
y_fit_ols = ols.predict(x_fit.reshape(-1, 1))

plt.plot(x_fit, y_fit_ols, 'r--', label=f'OLS Slope: {ols.coef_[0]:.2f}')
plt.plot(x_fit, y_fit_odr, 'k-', label=f'ODR Slope: {odlr.coef_[0]:.2f}')
plt.plot(x_fit, 5.0 * x_fit + (np.mean(ln_y - 5.0 * ln_x)), 'g:', label='True Slope (5.0)')

plt.xlabel('ln(x)')
plt.ylabel('ln(y)')
plt.title('Stress Test: Recovering Power Law Slope using the ODR Class')
plt.grid()
plt.legend()
plt.show()

print(f"True Slope: 5.00")
print(f"OLS Slope:  {ols.coef_[0]:.4f}")
print(f"ODR Slope: {odlr.coef_[0]:.4f}")

# ----------------------------------------
