# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:19:23 2026

@author: thi_wi
"""
import sys
import numpy as np

def report_entropy_vs_performance(y_true, y_pred, entropy_array, n_classes=2):
    """
    Screens the relationship between model 'confusion' (Entropy) 
    and actual classification errors.
    """
    h_max = np.log(n_classes)
    ratios = entropy_array / h_max
    
    # Define bins for the Uncertainty Ratio (0 to 1)
    bins = [0, 0.3, 0.7, 0.9, 1.0]
    labels = ['Low (Confident)', 'Mid (Ambiguous)', 'High (Boundary)', 'Max (Noise)']
    
    print("-" * 65)
    print(f"{'Entropy Tier':<20} | {'N Samples':<10} | {'FP':<5} | {'FN':<5} | {'Acc':<6}")
    print("-" * 65)
    
    for i in range(len(bins)-1):
        # Mask for the current entropy bin
        mask = (ratios >= bins[i]) & (ratios < bins[i+1])
        
        if not np.any(mask):
            continue
            
        y_t_bin = y_true[mask]
        y_p_bin = y_pred[mask]
        
        # Calculate Errors
        fp = np.sum((y_p_bin == 1) & (y_t_bin == 0))
        fn = np.sum((y_p_bin == 0) & (y_t_bin == 1))
        acc = np.mean(y_t_bin == y_p_bin)
        
        print(f"{labels[i]:<20} | {np.sum(mask):<10} | {fp:<5} | {fn:<5} | {acc:.1%}")
    
    print("-" * 65)


def run_entropy_comparison(model_ovr, model_multi, X_test, X_err_test, y_test):
    """
    Calculates robust entropy for both architectures and compares their 
    error profiles across automated entropy tiers.
    """
    # 1. Calculate Robust Entropy for OVR
    h_ovr = model_ovr.calculate_entropy(X_test, X_err_test)
    
    # 2. Calculate Robust Entropy for Multinomial
    h_multi = model_multi.calculate_entropy(X_test, X_err_test)
    
    # 3. Analyze global stats
    print(f"\n=== GLOBAL ENTROPY COMPARISON ===")
    print(f"OVR Mean Entropy:         {np.mean(h_ovr):.4f}")
    print(f"Multinomial Mean Entropy: {np.mean(h_multi):.4f}")
    print()

    # 4. Get predictions for error mapping
    preds_ovr = model_ovr.predict(X_test)
    preds_multi = model_multi.predict(X_test)
    
    # 5. Screen screen outputs (OVR uses n_classes=2, Multinomial adjusts if necessary)
    print("\n--- OVR PERFORMANCE BY ENTROPY TIER ---")
    report_entropy_vs_performance(y_test, preds_ovr, h_ovr, n_classes=2)
    
    print("\n--- MULTINOMIAL PERFORMANCE BY ENTROPY TIER ---")
    report_entropy_vs_performance(y_test, preds_multi, h_multi, n_classes=2)
    
    return h_ovr, h_multi


# =============================================================================
# INTERACTIVE WORKSPACE CHECK
# =============================================================================
# Look for your existing variables in the interactive console's user namespace (__main__)
import __main__

# In your main script, OVR is called 'odr' and Multinomial is 'odr_multi'
required_vars = {
    'model_ovr': 'odr', 
    'model_multi': 'odr_multi', 
    'X': 'X_test_scaled', 
    'E': 'E_test_scaled', 
    'y': 'y_test'
}

workspace_vars = {}
missing = []

for internal_name, workspace_name in required_vars.items():
    if hasattr(__main__, workspace_name):
        workspace_vars[internal_name] = getattr(__main__, workspace_name)
    else:
        missing.append(workspace_name)

if missing:
    print(f"\n[ERROR] Execution skipped: The following variables were not found in your console workspace: {missing}")
    print("--> Please execute your main SDSS script first so these variables exist in your Variable Explorer, then run this file.")
    sys.exit()
else:
    # Run comparison using variables directly from your active session
    h_ovr, h_multi = run_entropy_comparison(
        workspace_vars['model_ovr'], 
        workspace_vars['model_multi'], 
        workspace_vars['X'], 
        workspace_vars['E'], 
        workspace_vars['y']
    )
