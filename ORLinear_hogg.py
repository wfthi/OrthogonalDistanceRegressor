# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:53:55 2026

@author: thi_wi

Hogg, Bovy & Lang (2010)

The first five points (indices 0–4) are famous "catastrophic outliers" that represent 
sensor failures or misidentified objects.

Heteroscedasticity refers to the circumstance in statistics where the variance
of errors or the variability of the dependent variable y is not constant across
all levels of the independent variables  x. This is in contrast to homoscedasticity, 
where the variance of the residuals or the spread of y values is constant.

"""
import numpy as np
from scipy import odr
import matplotlib.pyplot as plt

# 1. The Hogg Dataset (Test Data from Hogg 2010)
x = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
dy = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])
# Assume a uniform measurement error for X as per the Hogg paper (sigma_x = 0 or small)
dx = np.full_like(x, 5.0) 

# 2. Define Models (Linear vs 2nd Degree Polynomial)
def linear_model(beta, x):
    return beta[0] + beta[1] * x

def poly_2nd_model(beta, x):
    return beta[0] + beta[1] * x + beta[2] * x**2

def run_odr(x, y, dx, dy, model_func, beta0, robust=False):
    # Standard ODR setup
    data = odr.RealData(x, y, sx=dx, sy=dy)
    model = odr.Model(model_func)
    my_odr = odr.ODR(data, model, beta0=beta0)
    
    if robust:
        # Implementing your "Level 1" Robustness (Weight clipping)
        # First pass to find residuals
        initial_fit = my_odr.run()
        residuals = np.abs(initial_fit.eps) # eps is the residual in Y
        
        # 3-Sigma / MAD Clipping
        mad = np.median(np.abs(residuals - np.median(residuals)))
        threshold = 3.0 * 1.4826 * mad
        
        # De-weight outliers (set their error to infinity/very high)
        dy_robust = np.where(residuals > threshold, 1e10, dy)
        data_robust = odr.RealData(x, y, sx=dx, sy=dy_robust)
        my_odr = odr.ODR(data_robust, model, beta0=initial_fit.beta)
        
    return my_odr.run()

# 3. Execution
beta_lin = [10.0, 2.0]
beta_quad = [10.0, 2.0, 0.001]

out_std = run_odr(x, y, dx, dy, linear_model, beta_lin, robust=False)
out_rob = run_odr(x, y, dx, dy, linear_model, beta_lin, robust=True)
out_poly = run_odr(x, y, dx, dy, poly_2nd_model, beta_quad, robust=True)

# 4. Detailed Printing of Coefficients
print("--- HOGG DATASET ODR COEFFICIENTS ---")
print(f"{'Model Type':<25} | {'Intercept (b0)':<15} | {'Slope (b1)':<15} | {'Curvature (b2)':<15}")
print("-" * 75)

# Standard Linear: y = b0 + b1*x
print(f"{'Standard Linear':<25} | {out_std.beta[0]:15.4f} | {out_std.beta[1]:15.4f} | {'N/A':<15}")

# Robust Linear: y = b0 + b1*x (Outliers ignored)
print(f"{'Robust Linear':<25} | {out_rob.beta[0]:15.4f} | {out_rob.beta[1]:15.4f} | {'N/A':<15}")

# Robust 2nd Degree: y = b0 + b1*x + b2*x^2
print(f"{'Robust 2nd Degree Poly':<25} | {out_poly.beta[0]:15.4f} | {out_poly.beta[1]:15.4f} | {out_poly.beta[2]:15.6f}")

print("\n--- ANALYSIS ---")
diff_slope = ((out_rob.beta[1] - out_std.beta[1]) / out_std.beta[1]) * 100
print(f"Outlier Impact: The Robust slope is {abs(diff_slope):.2f}% {'steeper' if diff_slope > 0 else 'shallower'} than Standard ODR.")

# 5. Visualization
plt.figure(figsize=(10, 6))
plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='ko', alpha=0.5, label='Data (with errors)')
plt.scatter(x[:5], y[:5], color='red', s=100, facecolors='none', edgecolors='r', label='Hogg Outliers')

x_plot = np.linspace(0, 300, 100)
plt.plot(x_plot, linear_model(out_std.beta, x_plot), 'r--', label='Standard Linear ODR')
plt.plot(x_plot, linear_model(out_rob.beta, x_plot), 'g-', linewidth=2, label='Robust Linear ODR')
plt.plot(x_plot, poly_2nd_model(out_poly.beta, x_plot), 'b:', label='Robust 2nd Degree Poly')

plt.ylim(0, 700)
plt.legend()
plt.title("Hogg Dataset: Standard vs Robust ODR")
plt.show()

print(f"Standard Slope: {out_std.beta[1]:.4f}")
print(f"Robust Slope:   {out_rob.beta[1]:.4f}")