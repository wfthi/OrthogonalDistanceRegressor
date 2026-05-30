# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:19:23 2026

@author: thi_wi
"""
import numpy as np
from ODLinear_fast import OrthogonalDistanceLogisticRegressionOVR
from ODLinear_fast import OrthogonalDistanceMultinomialLogisticRegression

def run_entropy_comparison(model_ovr, model_multi, X_test, X_err_test):
    # 1. Calculate Robust Entropy for OVR
    # This uses the stable sigma-check mean probabilities
    h_ovr = model_ovr.calculate_entropy(X_test, X_err_test)
    
    # 2. Calculate Robust Entropy for Multinomial
    # This uses the new Softmax-based sigma-check mean
    h_multi = model_multi.calculate_entropy(X_test, X_err_test)
    
    # 3. Analyze results
    print(f"OVR Mean Entropy: {np.mean(h_ovr):.4f}")
    print(f"Multinomial Mean Entropy: {np.mean(h_multi):.4f}")
    
    return h_ovr, h_multi

# Execute test
h_ovr, h_multi = run_entropy_comparison(odr_ovr, odr_multi, X_sdss, X_err_sdss)
