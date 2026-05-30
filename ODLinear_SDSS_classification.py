# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:32:36 2026

The SDSS test evaluated an Orthogonal Distance Regression (ODR) classifier 
against standard Logistic Regression (LR) for distinguishing stars from galaxies
 using five-band ($u, g, r, i, z$) photometric data. By expanding the feature 
 space into 14-dimensional 2nd-degree polynomials, the model successfully 
 captured the non-linear curvature of the stellar locus, significantly reducing
 the class overlap (Mixing Index).While the standard LR maintained a slight edge
 in raw accuracy ($73\%$ vs. $70\%$), the ODR demonstrated a more physically 
 grounded boundary by prioritizing high-precision measurements and down-weighting
 noisy bands. In the high-precision tier, the ODR proved its stability, anchoring
 its classification to the $u-g$ break and curvature terms like $(u-g)^2$ rather
 than overfitting to the photometric noise present in both the training and test
 sets.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from astroML.datasets import fetch_imaging_sample
from ODLinear_fast import compute_mixing_index, OrthogonalDistanceLogisticRegressionOVR, OrthogonalDistanceMultinomialLogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import FunctionTransformer

def tune_no_error_baseline(X, y):
    # 1. Get the "Target" weights from OLS/LR
    # We use this to provide a 'warm start' to ODR
    lr = LogisticRegression(C=5.0, penalty='l2', solver='lbfgs')
    lr.fit(X, y)
    beta_start = np.concatenate([lr.coef_.flatten(), lr.intercept_])

    # 2. Define the Linear model for ODR
    def linear_func(p, x):
        # p[:-1] are weights, p[-1] is intercept
        return np.dot(x, p[:-1]) + p[-1]

    # 3. Setup ODR with NO errors (identity weights)
    # We increase 'maxit' and tighten 'tol' to ensure it doesn't give up
    data = odr.Data(X, y)
    model = odr.Model(linear_func)
    my_odr = odr.ODR(data, model, beta0=beta_start)
    
    # Force the solver to work harder
    my_odr.set_iprint(init=1, iter=1, final=1) # Debugging output
    my_odr.set_job(fit_type=2) # 2 = Ordinary Least Squares (ignores delta)
    
    output = my_odr.run()
    return output.beta

def report_entropy_vs_performance(y_true, y_pred, entropy_array, X_test, E_test, model, n_classes=2, n_draws=100):
    """
    Screens the relationship between Entropy, accuracy, and Monte Carlo 
    error to identify 'Spurious Confidence' caused by data noise.
    """
    h_max = np.log(n_classes)
    ratios = entropy_array / h_max
    
    # Define bins for the Uncertainty Ratio (0 to 1)
    bins = [0, 0.3, 0.7, 0.9, 1.05] # 1.05 to include edge cases
    labels = ['Low (Confident)', 'Mid (Ambiguous)', 'High (Boundary)', 'Max (Noise)']
    
    # Calculate the MC Error for the entire set once to save time
    _, mc_error_array = model.predict_proba_MC_error(X_test, E_test, Number_of_MC_iterations=n_draws)

    # If multicomial, mc_error_array might be (N, classes); we take the mean across classes 
    # since we noted the uncertainty is identical/symmetric for both probas.
    if mc_error_array.ndim > 1:
        mc_error_dist = np.mean(mc_error_array, axis=1)
    else:
        mc_error_dist = mc_error_array

    print("-" * 105)
    print(f"{'Entropy Tier':<20} | {'N':<6} | {'FP':<4} | {'FN':<4} | {'Acc':<6} | {'MC Min':<8} | {'MC Max':<8} | {'MC Avg':<8}")
    print("-" * 105)
    
    for i in range(len(bins)-1):
        mask = (ratios >= bins[i]) & (ratios < bins[i+1])
        
        if not np.any(mask):
            continue
            
        y_t_bin = y_true[mask]
        y_p_bin = y_pred[mask]
        mc_bin = mc_error_dist[mask]
        
        # Calculate Errors
        fp = np.sum((y_p_bin == 1) & (y_t_bin == 0))
        fn = np.sum((y_p_bin == 0) & (y_t_bin == 1))
        acc = np.mean(y_t_bin == y_p_bin)
        
        # MC Stats
        mc_min = np.min(mc_bin)
        mc_max = np.max(mc_bin)
        mc_avg = np.mean(mc_bin)
        
        print(f"{labels[i]:<20} | {np.sum(mask):<6} | {fp:<4} | {fn:<4} | {acc:<6.1%} | {mc_min:<8.4f} | {mc_max:<8.4f} | {mc_avg:<8.4f}")
    
    print("-" * 105)

def interpret_entropy(entropy_array, n_classes=2):
    """
    Automatically interprets the stability and confidence of ODR predictions.
    """
    # Calculate theoretical maximum entropy for the number of classes
    h_max = np.log(n_classes) 
    
    # Define thresholds based on the 'Fuzzy' logic of your code
    # Confident: < 30% of max entropy
    # Ambiguous: 30% - 85% of max entropy
    # High Conflict/Boundary: > 85% of max entropy
    
    results = []
    for h in entropy_array:
        ratio = h / h_max
        
        if ratio < 0.3:
            status = "Confident"
            action = "Accept Classification"
        elif ratio < 0.85:
            status = "Ambiguous"
            action = "Check Multi-wavelength Data"
        else:
            status = "Decision Boundary / Noise Dominant"
            action = "Exclude from Clean Sample"
            
        results.append({
            'Entropy': h,
            'Uncertainty_Ratio': ratio,
            'Status': status,
            'Recommendation': action
        })
        
    return results

def suggest_adaptive_c(X, Mixing_Index, use_errors=False):
    """
    Consolidated C-suggestion for the shrinkage formula: p_reg = p / (1 + p/C).
    
    Logic:
    - OLS: C acts as the primary regularizer; keep it tight to prevent overfitting.
    - ODR: C acts as a dynamic range ceiling; keep it high to let SNR weights 
           and measurement errors define the boundary.
    """
    n_samples, n_features = X.shape
    
    # 1. Base C calculated from sample density and mixing (0.1 to 0.4 typical)
    # Higher mixing requires a more conservative ceiling.
    base_c = 1.0 / (Mixing_Index + 0.1)
    
    # 2. Complexity Scaling (Standard across both models)
    complexity_factor = np.log10(n_samples) / n_features
    suggested_c = base_c * complexity_factor
    
    # 3. Noise-Aware Regime Shift
    if use_errors:
        # Relax the shrinkage to allow for 'Physical Regression'.
        # This prevents the 'Clipping' of high-weight features like SNR_iz.
        # Shift target from ~1.44 to ~10.0-15.0
        suggested_c *= 7.0  
    else:
        # Maintain the 'Mathematical Regularization' for OLS/No-Error runs.
        # We ensure it stays low enough to act as a proper L2-equivalent.
        suggested_c = suggested_c 
        
    return np.clip(suggested_c, 1.0, 25.0)

def get_governed_C(X, y, C_max=10.0, alpha=7.0, n_neighbors=30):
    """Dynamically adjusts C based on class overlap (Mixing Index)."""
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    
    neighbor_labels = y[indices[:, 1:]]
    mismatch_counts = np.sum(neighbor_labels != y[:, np.newaxis], axis=1)
    mixing_index = np.mean(mismatch_counts / n_neighbors)
    
    C_opt = C_max * np.exp(-alpha * mixing_index)
    return np.clip(C_opt, 0.05, C_max), mixing_index

def prepare_sdss_linear_odr_colors(n_samples=10000):
    """
    Prepares SDSS data with extinction-corrected linear colors.
    Returns 4 features: (u-g), (g-r), (r-i), (i-z).
    Returns a 'poly' equivalent for feature name extraction.
    """
    data = fetch_imaging_sample()
    objtype = data['type']
    
    s_mask = (objtype == 6); g_mask = (objtype == 3)
    stars = data[s_mask][:n_samples]
    galaxies = data[g_mask][:n_samples]
    
    mags = ['uRaw', 'gRaw', 'rRaw', 'iRaw', 'zRaw']
    errs = ['uErr', 'gErr', 'rErr', 'iErr', 'zErr']
    
    # SFD Extinction Coefficients
    ext_coeffs = np.array([4.239, 3.303, 2.285, 1.698, 1.263])
    r_coeff = 2.285 

    def get_corrected_data(subset):
        X_vals = np.column_stack([subset[m] for m in mags])
        E_vals = np.column_stack([subset[e] for e in errs])
        
        # Apply extinction correction
        r_ext = subset['rExtSFD']
        extinction_vectors = np.outer(r_ext / r_coeff, ext_coeffs)
        X_corr = X_vals - extinction_vectors
        
        return X_corr, E_vals

    X_s, E_s = get_corrected_data(stars)
    X_g, E_g = get_corrected_data(galaxies)
    
    X_raw = np.vstack([X_s, X_g])
    E_raw = np.vstack([E_s, E_g])
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])

    # 1. Linear Colors: (u-g), (g-r), (r-i), (i-z)
    X_linear = np.column_stack([X_raw[:, i] - X_raw[:, i+1] for i in range(4)])
    E_linear = np.column_stack([np.sqrt(E_raw[:, i]**2 + E_raw[:, i+1]**2) for i in range(4)])

    # 2. Create a "Identity" Poly equivalent
    # This allows feature_names = poly.get_feature_names_out(...) to still work.
    poly_linear = FunctionTransformer(lambda x: x, validate=True)
    poly_linear.get_feature_names_out = lambda input_features=None: np.array(input_features)

    return train_test_split(X_linear, y, E_linear, test_size=0.3, random_state=42, stratify=y), poly_linear

def prepare_sdss_linear_odr(n_samples=10000):
    """
    Prepares SDSS data with extinction-corrected linear colors AND Color SNR.
    Returns 8 features: (u-g), (g-r), (r-i), (i-z), SNR_ug, SNR_gr, SNR_ri, SNR_iz.
    """
    data = fetch_imaging_sample()
    objtype = data['type']
    
    s_mask = (objtype == 6); g_mask = (objtype == 3)
    stars = data[s_mask][:n_samples]
    galaxies = data[g_mask][:n_samples]
    
    mags = ['uRaw', 'gRaw', 'rRaw', 'iRaw', 'zRaw']
    errs = ['uErr', 'gErr', 'rErr', 'iErr', 'zErr']
    
    ext_coeffs = np.array([4.239, 3.303, 2.285, 1.698, 1.263])
    r_coeff = 2.285 

    def get_corrected_data(subset):
        X_vals = np.column_stack([subset[m] for m in mags])
        E_vals = np.column_stack([subset[e] for e in errs])
        r_ext = subset['rExtSFD']
        extinction_vectors = np.outer(r_ext / r_coeff, ext_coeffs)
        X_corr = X_vals - extinction_vectors
        return X_corr, E_vals

    X_s, E_s = get_corrected_data(stars)
    X_g, E_g = get_corrected_data(galaxies)
    
    X_raw = np.vstack([X_s, X_g])
    E_raw = np.vstack([E_s, E_g])
    y = np.concatenate([np.zeros(len(X_s)), np.ones(len(X_g))])

    # 1. Compute Linear Colors
    X_colors = np.column_stack([X_raw[:, i] - X_raw[:, i+1] for i in range(4)])
    E_colors = np.column_stack([np.sqrt(E_raw[:, i]**2 + E_raw[:, i+1]**2) for i in range(4)])

    # 2. Compute SNR for each band: SNR = 1.0857 / mag_err
    # This is the standard conversion from mag error to fractional flux error
    SNR_bands = 1.0857 / E_raw
    
    # 3. Compute Quadrature SNR for each color (Flux Ratio SNR)
    # SNR_color = 1 / sqrt( (1/SNR1)^2 + (1/SNR2)^2 )
    # Which simplifies back to 1.0857 / E_color
    X_snr = 1.0857 / E_colors
    
    # 4. Combine: [Colors, SNRs]
    X_final = np.hstack([X_colors, X_snr])
    
    # The error in a color measurement remains E_colors. 
    # For the SNR features themselves, we treat them as observed 'constants' (low error)
    # to avoid recursive error propagation in the ODR solver.
    E_final = np.hstack([E_colors, np.zeros_like(X_snr) + 1e-5])

    # Update names for the Identity Poly
    names = ['u-g', 'g-r', 'r-i', 'i-z', 'SNR_ug', 'SNR_gr', 'SNR_ri', 'SNR_iz']
    poly_linear = FunctionTransformer(lambda x: x, validate=True)
    poly_linear.get_feature_names_out = lambda input_features=None: np.array(names)

    return train_test_split(X_final, y, E_final, test_size=0.3, random_state=42, stratify=y), poly_linear

def prepare_sdss_polynomial_odr(n_samples=10000):
    """
    Prepares SDSS data with extinction-corrected u,g,r,i,z bands 
    and 2nd degree polynomial terms.
    """
    data = fetch_imaging_sample()
    objtype = data['type']
    
    s_mask = (objtype == 6); g_mask = (objtype == 3)
    stars = data[s_mask][:n_samples]
    galaxies = data[g_mask][:n_samples]
    
    mags = ['uRaw', 'gRaw', 'rRaw', 'iRaw', 'zRaw']
    errs = ['uErr', 'gErr', 'rErr', 'iErr', 'zErr']
    
    # SDSS Extinction Coefficients (R_x) for u, g, r, i, z
    # Based on Schlafly & Finkbeiner (2011) assuming R_V = 3.1
    ext_coeffs = np.array([4.239, 3.303, 2.285, 1.698, 1.263])
    r_coeff = 2.285 # R_r for normalization

    def get_corrected_data(subset):
        # 1. Extract raw magnitudes and errors
        X_vals = np.column_stack([subset[m] for m in mags])
        E_vals = np.column_stack([subset[e] for e in errs])
        
        # 2. Apply extinction correction: mag_corr = mag_raw - (R_x * rExtSFD / R_r)
        # subset['rExtSFD'] is the r-band extinction value from the dust map
        r_ext = subset['rExtSFD']
        extinction_vectors = np.outer(r_ext / r_coeff, ext_coeffs)
        X_corr = X_vals - extinction_vectors
        
        return X_corr, E_vals

    X_s, E_s = get_corrected_data(stars)
    X_g, E_g = get_corrected_data(galaxies)
    
    X_raw = np.vstack([X_s, X_g])
    E_raw = np.vstack([E_s, E_g])
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])

    # 1. Linear Colors: (u-g), (g-r), (r-i), (i-z) using CORRECTED magnitudes
    colors = np.column_stack([X_raw[:, i] - X_raw[:, i+1] for i in range(4)])
    color_errs = np.column_stack([np.sqrt(E_raw[:, i]**2 + E_raw[:, i+1]**2) for i in range(4)])

    # 2. Polynomial Expansion
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(colors)
    
    # 3. Error Propagation for Polynomial Terms
    E_poly = np.zeros_like(X_poly)
    n_colors = colors.shape[1]
    eps_floor = 1e-6 
    
    E_poly[:, :n_colors] = color_errs + eps_floor # Linear terms
    
    curr = n_colors
    for i in range(n_colors):
        for j in range(i, n_colors):
            if i == j: 
                E_poly[:, curr] = np.abs(2 * colors[:, i] * color_errs[:, i])
            else: 
                E_poly[:, curr] = np.sqrt((colors[:, j] * color_errs[:, i])**2 + 
                                          (colors[:, i] * color_errs[:, j])**2)
            E_poly[:, curr] += eps_floor
            curr += 1

    return train_test_split(X_poly, y, E_poly, test_size=0.3, random_state=42, stratify=y), poly

def precision_tiered_report(X_test, y_test, E_test, odr_model, lr_model):
    """
    Evaluates model performance across different photometric quality tiers.
    Includes safeguards against empty slices and nan errors.
    """
    # 1. Calculate noise proxy (mean error across features)
    noise_proxy = np.mean(E_test, axis=1)
    
    # 2. Define thresholds using the median of the current batch
    # This guarantees a 50/50 split even if the global noise level is low.
    threshold = np.median(noise_proxy)
    
    # Use <= and > to ensure every point is in exactly one bucket
    high_precision_mask = noise_proxy <= threshold
    low_precision_mask = ~high_precision_mask
    
    # 3. Helper to safely calculate accuracy
    def safe_accuracy(y_true, y_pred, mask):
        if np.sum(mask) == 0:
            return 0.0
        return accuracy_score(y_true[mask], y_pred[mask])

    # ODR Predictions (Standard)
    odr_preds = odr_model.predict(X_test)
    
    # LR Predictions
    lr_preds = lr_model.predict(X_test)
    
    # 4. Calculate Tiered Accuracies
    odr_acc_high = safe_accuracy(y_test, odr_preds, high_precision_mask)
    lr_acc_high = safe_accuracy(y_test, lr_preds, high_precision_mask)
    
    odr_acc_low = safe_accuracy(y_test, odr_preds, low_precision_mask)
    lr_acc_low = safe_accuracy(y_test, lr_preds, low_precision_mask)
    
    # 5. Print the Report
    print("\n=== Precision-Tiered Performance ===")
    print(f"Noise Threshold (Median): {threshold:.4f}")
    
    print(f"High-Precision Tier (N={np.sum(high_precision_mask)}):")
    print(f"  > ODR Accuracy: {odr_acc_high:.4f}")
    print(f"  > LR Accuracy:  {lr_acc_high:.4f}")
    
    if np.sum(low_precision_mask) > 0:
        print(f"\nLow-Precision Tier (N={np.sum(low_precision_mask)}):")
        print(f"  > ODR Accuracy: {odr_acc_low:.4f}")
        print(f"  > LR Accuracy:  {lr_acc_low:.4f}")
    else:
        print("\nLow-Precision Tier (N=0): No data available in this tier.")
    
    return high_precision_mask

def plot_precision_confusion_matrices(X_test, y_test, hp_mask, odr_model, lr_model):
    """Generates confusion matrices for the high-precision data tier."""
    
    # Isolate High-Precision data
    X_hp = X_test[hp_mask]
    y_hp = y_test[hp_mask]
    
    # Predictions
    odr_hp_preds = odr_model.predict(X_hp)
    lr_hp_preds = lr_model.predict(X_hp)
    
    # Setup Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ODR Confusion Matrix
    cm_odr = confusion_matrix(y_hp, odr_hp_preds)
    disp_odr = ConfusionMatrixDisplay(confusion_matrix=cm_odr, display_labels=['Star', 'Galaxy'])
    disp_odr.plot(ax=ax1, cmap='Blues', colorbar=False)
    ax1.set_title(f"ODR Confusion Matrix\n(High-Precision Tier)")
    
    # LR Confusion Matrix
    cm_lr = confusion_matrix(y_hp, lr_hp_preds)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Star', 'Galaxy'])
    disp_lr.plot(ax=ax2, cmap='Greens', colorbar=False)
    ax2.set_title(f"Standard LR Confusion Matrix\n(High-Precision Tier)")
    
    plt.tight_layout()
    plt.show()
    
    return cm_odr, cm_lr
# --- Main Execution ---

n_samples = 10000
correct_imbalance = True
ODR_C = 30.

# 1. Load and Expand Data
(X_train, X_test, y_train, y_test, E_train, E_test), poly = prepare_sdss_linear_odr(n_samples=n_samples)

# (X_train, X_test, y_train, y_test, E_train, E_test), poly = prepare_sdss_polynomial_odr()

# 2. Scaling (Essential for Polynomial ODR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale errors by the same factor to maintain physical proportions
E_train_scaled = E_train / scaler.scale_
E_test_scaled = E_test / scaler.scale_

# Avoid tiny error dominating the fit
E_train_scaled = np.maximum(E_train_scaled, 0.1)
E_test_scaled = np.maximum(E_test_scaled, 0.1)

# 3. Governance
suggested_C, mix_idx = get_governed_C(X_train_scaled, y_train)

# 4. Model Training
# Assuming OrthogonalDistanceLogisticRegressionOVR is defined in your environment
# ODR filters the noise from the data

odr = OrthogonalDistanceLogisticRegressionOVR(C=ODR_C, tol=1e-8, maxit=2000)
odr.fit(X_train_scaled, y_train, X_err=E_train_scaled, correct_imbalance=correct_imbalance)
odr_mean_proba, odr_std_err = odr.predict_proba_with_model_uncertainty(X_train_scaled)
                                     
odr_multi = OrthogonalDistanceMultinomialLogisticRegression(C=ODR_C, tol=1e-8, maxit=2000)
odr_multi.fit(X_train_scaled, y_train, X_err=E_train_scaled, correct_imbalance=correct_imbalance)

E_train_scaled[:] = 1.0
odr_NoErr = OrthogonalDistanceLogisticRegressionOVR(C=suggested_C, tol=1e-8, maxit=2000)
odr_NoErr.fit(X_train_scaled, y_train, X_err=E_train_scaled, correct_imbalance=correct_imbalance)

# LR fits the data + noise
lr = LogisticRegression(C=suggested_C, max_iter=2000)
lr.fit(X_train_scaled, y_train)

# 5. Evaluation
odr_preds = odr.predict(X_test_scaled)
odr_multi_preds = odr_multi.predict(X_test_scaled)

lr_preds = lr.predict(X_test_scaled)

# Using the 3-point average probability
odr_sigma = odr.predict_robust(X_test_scaled, E_test_scaled)

E_test_scaled[:] = 1.0
odr_NoErr = odr_NoErr.predict_robust(X_test_scaled, E_test_scaled)

print()
print(f"\n=== SDSS 5-Band Linear/Polynomial Stress Test ===")
print(f"N samples: {n_samples}")
print(f"Mixing Index: {mix_idx:.4f}")
print(f"Adaptive C: {suggested_C:.4f}")
print(f"LR Accuracy:  {accuracy_score(y_test, lr_preds):.4f}")
print(f"ODR OVR No Error Accuracy using adaptive C: {accuracy_score(y_test, odr_NoErr):.4f}")
print(f"OVR with Errors, C: {ODR_C}")
print(f"ODR OVR Accuracy: {accuracy_score(y_test, odr_preds):.4f}")
print(f"Robust ODR OVR Accuracy: {accuracy_score(y_test, odr_sigma):.4f}")
print(f"Average Model-Induced Uncertainty (Sigma-Beta): {np.mean(odr_std_err):.4f}")
print(f"ODR MULTI Accuracy: {accuracy_score(y_test, odr_multi_preds):.4f}")

print(f"Mean ODR Residual: {np.mean(np.abs(odr.epsilon_)):.4f}")
# Execute the report
hp_mask = precision_tiered_report(X_test_scaled, y_test, E_test_scaled, odr, lr)

cm_odr, cm_lr = plot_precision_confusion_matrices(X_test_scaled, y_test, hp_mask, odr, lr)

print(f"ODR confusion matrix on high-quality data:")
print(cm_odr)
print()
print(f"OLSconfusion matrix on high-quality data:")
print(cm_lr)

# Compare this to your previous ODR confusion matrix
print()
print(f"ODR no Error confusion matrix on all data:")
print(confusion_matrix(y_test, odr_NoErr))
print()
print(f"ODR confusion matrix on all data:")
print(confusion_matrix(y_test, odr_preds))
print()
print("Robust ODR Confusion Matrix (Sigma-Check):")
print(confusion_matrix(y_test, odr_sigma))
print()
print("ODR Multi Confusion Matrix (Sigma-Check):")
print(confusion_matrix(y_test, odr_multi_preds))
print()
print(f"OLS confusion matrix on all data:")
print(confusion_matrix(y_test, lr_preds))


# Extract feature names from the PolynomialFeatures object
feature_names = poly.get_feature_names_out(['u-g', 'g-r', 'r-i', 'i-z'])

# Pair names with ODR coefficients (weights)
# Note: odr.coef_ is usually shape (n_classes, n_features)
weights = odr.coef_[0] 
feature_importance = sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)

print("\n=== ODR Polynomial Coefficient Analysis ===")
print(f"{'Feature Term':<20} | {'Weight (Coefficient)':<20}")
print("-" * 45)
for name, weight in feature_importance:
    print(f"{name:<20} | {weight:>20.4f}")
    
h_ovr = odr.calculate_entropy(X_test, E_test_scaled)
h_multi = odr_multi.calculate_entropy(X_test, E_test_scaled)

print("OVR model")
report_entropy_vs_performance(y_test, odr_preds, h_ovr, X_test, E_test, odr, n_draws=100)

print("Multinomial model")
report_entropy_vs_performance(y_test, odr_preds, h_multi, X_test, E_test, odr_multi, n_draws=100)

