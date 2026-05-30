# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:22:00 2026

ODR vs Random Forest: Multiclass Robustness Stress Test

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from ODLinear_fast import OrthogonalDistanceLogisticRegressionOVR

# --- 1. DATA GENERATION ENGINE ---
def generate_stress_data(n_samples=3000, outlier_rate=0.05):
    """Generates 3-class data with asymmetric noise and optional catastrophic outliers."""
    np.random.seed(101)
    y = np.random.choice([0, 1, 2], n_samples)
    X_true = np.zeros((n_samples, 2))
    
    # Class Centroids
    X_true[y == 0] = np.random.multivariate_normal([0, 0], [[0.15, 0], [0, 0.15]], sum(y==0))
    X_true[y == 1] = np.random.multivariate_normal([0.6, 1.0], [[0.2, 0.1], [0.1, 0.2]], sum(y==1))
    X_true[y == 2] = np.random.multivariate_normal([1.2, 0.0], [[0.15, -0.05], [-0.05, 0.15]], sum(y==2))
    
    # Asymmetric Gaussian Noise (The Attenuation Source)
    X_err = np.array([0.35, 0.15]) 
    X_noisy = X_true + np.random.normal(0, X_err, X_true.shape)
    
    # Add Catastrophic Outliers (Teleportation)
    if outlier_rate > 0:
        n_outliers = int(outlier_rate * n_samples)
        out_idx = np.random.choice(n_samples, n_outliers, replace=False)
        # Scatter outliers across the entire observed range
        X_noisy[out_idx] = np.random.uniform(-1.5, 2.5, (n_outliers, 2))
        
    return X_noisy, y, X_err

# --- 2. PLOTTING ENGINE ---
def plot_results(X, y, err_vals, rf_model, odr_model, filename='comparison.png'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]

    titles = ["Random Forest (Noise Blind)", "ODR OVR (Noise Aware)"]
    models = [rf_model, odr_model]
    colors = ['purple', 'teal', 'gold']

    for i, ax in enumerate(axes):
        # Decision Surface
        Z = models[i].predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.25, cmap='viridis')
        
        # Data Points
        for cls in range(3):
            idx = (y == cls)
            ax.scatter(X[idx, 0], X[idx, 1], c=colors[cls], s=15, alpha=0.6, label=f'Class {cls}')
        
        # Overlay Error Bars on a subset
        sub = np.random.choice(len(X), 60, replace=False)
        ax.errorbar(X[sub, 0], X[sub, 1], xerr=err_vals[0], yerr=err_vals[1], 
                    fmt='none', ecolor='black', alpha=0.4, elinewidth=0.6)
        
        ax.set_title(titles[i])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("$x_1$ (Noisy)")
        ax.set_ylabel("$x_2$ (Clean)")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    # Generate data with 5% catastrophic outliers
    X_noisy, y, err_vals = generate_stress_data(n_samples=3000, outlier_rate=0.05)
    
    # Split
    split = 2000
    X_train, X_test = X_noisy[:split], X_noisy[split:]
    y_train, y_test = y[:split], y[split:]
    X_err_train = np.tile(err_vals, (len(X_train), 1))
    X_err_test = np.tile(err_vals, (len(X_test), 1))

    # Initialize Models
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    # Note: Ensure your class in ODLinear_fast handles 'robust' internally or via C
    odr = OrthogonalDistanceLogisticRegressionOVR(C=10, tol=1e-4, 
                                                  probability="softmax",
                                                  robust=True)

    # Training
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    
    print("Training ODR (This may take a moment with OVR)...")
    odr.fit(X_train, y_train, X_err=X_err_train)

    # Evaluation
    rf_preds = rf.predict(X_train)
    odr_preds = odr.predict(X_train)


    print("\n" + "="*30)
    print("Training set")
    print(f"RF Accuracy:  {accuracy_score(y_train, rf_preds):.4f}")
    print(f"ODR Accuracy: {accuracy_score(y_train, odr_preds):.4f}")
    print("="*30)


    # Evaluation
    rf_preds = rf.predict(X_test)
    odr_preds = odr.predict(X_test)
    _, odr_errs = odr.predict_proba_MC_error(X_test, X_err_test)


    print("\n" + "="*30)
    print("Test set")
    print(f"RF Accuracy:  {accuracy_score(y_test, rf_preds):.4f}")
    print(f"ODR Accuracy: {accuracy_score(y_test, odr_preds):.4f}")
    print(f"ODR Error: {np.mean(odr_errs, axis=0)}")
    print("="*30)

    print("\nODR PERFORMANCE PROFILE:")
    print(classification_report(y_test, odr_preds))

    # Plot
    plot_results(X_test, y_test, err_vals, rf, odr, 'robust_benchmark.png')