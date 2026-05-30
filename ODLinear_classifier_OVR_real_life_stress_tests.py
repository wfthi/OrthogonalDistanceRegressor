"""
Advanced Physical Science Stress Tests for OrthogonalDistanceLogisticRegressionOVR

Stress tests for real-world applications

Key Features of Each Test

1. Heteroscedasticity Test
Purpose: Tests handling of varying measurement uncertainties
Scenario: Different noise levels per sample (0.01 to 5.0)
Metric: Whether noisy points drag the boundary vs. high-precision anchors
Test: Error-aware fitting - Uses measurement uncertainties properly
      Adaptive regularization - Adjusts based on data topology

2. Collinearity Test
Purpose: Tests stability with correlated features
Scenario: 99% correlated features (common in atmospheric physics)
Metric: Coefficient stability (avoiding explosion to ±∞)

3. Boundary Overlap Test
Purpose: Tests behavior with inseparable classes
Scenario: Identical class means with different covariance shapes
Metric: Prediction entropy (should be high in overlap regions)
Test: Uncertainty quantification - Provides meaningful confidence estimates

4. Adversarial Outliers Test
Purpose: Tests robustness to malicious data
Scenario: 5% of Class A points labeled as Class B and placed in Class A territory
Metric: Residual analysis (outliers should have large residuals)
Test: Robust outlier handling - Identifies and mitigates misleading data

5. Concept Drift Test
Purpose: Tests adaptability to changing conditions
Scenario: Train on summer data, test on autumn data (slightly shifted)
Metric: Accuracy drop vs. quasi-chi-square stability

6. Error Sensitivity Sweep
Purpose: Benchmark against standard methods
Scenario: Vary noise levels and compare ODR vs. standard LR
Metric: Accuracy vs. noise scale curves
Why These Tests Matter
These tests directly address the computational and physical limits that matter 
in real scientific applications:
Astrophysics: Heteroscedasticity from detector effects, collinearity from 
instrumental calibration
Atmospheric Physics: Non-stationarity from seasonal changes, extreme events in
 noisy environments
Robustness: Real-world data rarely follows textbook assumptions

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ODLinear_fast import OrthogonalDistanceLogisticRegressionOVR
from sklearn.neighbors import NearestNeighbors

def compute_mixing_index(X, y, n_neighbors=10):
    """
    Computes a 'Mixing Index' based on local class purity.
    0.0 = Perfectly separated classes.
    1.0 = Completely mixed (random noise).
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X)
    
    # Find neighbors for each point (excluding the point itself)
    distances, indices = nn.kneighbors(X)
    neighbor_labels = y[indices[:, 1:]]
    
    # Calculate what fraction of neighbors are 'different' from the point's own label
    mismatch_counts = np.sum(neighbor_labels != y[:, np.newaxis], axis=1)
    purity_per_point = mismatch_counts / n_neighbors
    
    mixing_index = np.mean(purity_per_point)
    
    # Heuristic for C: Inverse relationship
    # If mixing is 0.8 (high), C should be low (e.g., 0.1)
    # If mixing is 0.1 (low), C can be high (e.g., 100)
    suggested_C = 1.0 / (mixing_index + 1e-5) * 0.1 
    
    print(f"Mixing Index: {mixing_index:.4f}")
    print(f"Suggested C range: {suggested_C:.2f}")
    
    return mixing_index, suggested_C

def get_governed_C(X, y, C_max=10.0, alpha=7.0, n_neighbors=10):
    """
    Dynamically adjusts C based on class overlap (Mixing Index).
    C_opt = C_max * exp(-alpha * MixingIndex)
    """
    # 1. Compute Mixing Index: Adaptive regularization - Adjusts based on 
    # data topology
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    
    neighbor_labels = y[indices[:, 1:]]
    mismatch_counts = np.sum(neighbor_labels != y[:, np.newaxis], axis=1)
    mixing_index = np.mean(mismatch_counts / n_neighbors)
    
    # 2. Apply Exponential Decay
    # If mixing_index is 0.58 and alpha is 7.0, C_opt will be ~0.17
    C_opt = C_max * np.exp(-alpha * mixing_index)
    
    # 3. Safety Clamps
    C_opt = np.clip(C_opt, 0.01, C_max)
    
    print(f"Topology Analysis:")
    print(f"  > Mixing Index: {mixing_index:.4f}")
    print(f"  > Adaptive C:   {C_opt:.4f}")
    
    return C_opt

def test_heteroscedasticity():
    """
    Heteroscedasticity Stress Test: 
    Tests if the model correctly prioritizes high-precision samples over noisy ones.
    """
    print("\n=== Heteroscedasticity Test ===")
    np.random.seed(42)

    # 1. GROUND TRUTH (The two physical classes)
    # Class 0 at (-2, -2), Class 1 at (2, 2)
    n_clean = 400
    X_0 = np.random.normal(loc=-2.0, scale=1.0, size=(n_clean, 2))
    X_1 = np.random.normal(loc=2.0, scale=1.0, size=(n_clean, 2))
    
    # 2. THE CONTAMINATION (The "Liar" points)
    # We place Class 0 points right on top of the Class 1 region
    # But we give them MASSIVE noise levels (e.g., 8.0)
    n_outliers = 150
    X_outliers = np.random.normal(loc=2.0, scale=1.0, size=(n_outliers, 2))
    y_outliers = np.zeros(n_outliers) # Liars: Class 0 in Class 1 territory
    noise_outliers = np.full(n_outliers, 8.0)
    
    # Training Data
    X_train = np.vstack([X_0, X_1, X_outliers])
    y_train = np.concatenate([np.zeros(n_clean), np.ones(n_clean), y_outliers])
    noise_train = np.concatenate([np.full(n_clean, 0.1), np.full(n_clean, 0.1), noise_outliers])
    
    # Test Data (Pure Ground Truth - No Contamination)
    X_test_0 = np.random.normal(loc=-2.0, scale=1.0, size=(150, 2))
    X_test_1 = np.random.normal(loc=2.0, scale=1.0, size=(150, 2))
    X_test = np.vstack([X_test_0, X_test_1])
    y_test = np.concatenate([np.zeros(150), np.ones(150)])
    n_test = np.full(len(X_test), 0.1) # Test set is high-precision

    # 3. Fit Models
    X_err_train = np.column_stack([noise_train, noise_train])
    suggested_C = get_governed_C(X_train, y_train, C_max=10.0)
    
    # 5. Fit ODR (The 'Physics' Aware Model)
    odr = OrthogonalDistanceLogisticRegressionOVR(C=suggested_C, tol=1e-4)
    odr.fit(X_train, y_train, X_err=X_err_train)
    
    # 6. Fit Standard LR (The 'Weight-Blind' Model)
    lr = LogisticRegression(C=suggested_C, random_state=42)
    lr.fit(X_train, y_train)
    
    # --- Metrics ---
    odr_accuracy = accuracy_score(y_test, odr.predict(X_test))
    lr_accuracy = accuracy_score(y_test, lr.predict(X_test))
    
    print(f"Topology Mixing Index: {compute_mixing_index(X_train, y_train)[0]:.4f}")
    print(f"Adaptive C: {suggested_C:.4f}")
    print(f"ODR Accuracy: {odr_accuracy:.4f}")
    print(f"Standard LR Accuracy: {lr_accuracy:.4f}")

    # --- Verification of Intent ---
    # We check if ODR ignored the high-noise points by looking at the epsilon
    low_noise_mask = n_test < 1.0
    high_noise_mask = n_test > 1.0
    
    # Only compute if the mask actually contains samples
    if np.any(low_noise_mask):
        acc_low = accuracy_score(y_test[low_noise_mask], odr.predict(X_test[low_noise_mask]))
        print(f"ODR Accuracy on High-Precision subset: {acc_low:.4f}")
    
    if np.any(high_noise_mask):
        acc_high = accuracy_score(y_test[high_noise_mask], odr.predict(X_test[high_noise_mask]))
        print(f"ODR Accuracy on Low-Precision subset:  {acc_high:.4f}")
    else:
        print("No Low-Precision points in Test Set (Pure Truth).")
    
    # --- Visualization Section ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create a mesh grid to plot boundaries
    h = .05  # step size in the mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Mesh grid predictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_probs = odr.predict_proba(grid_points)
    Z_plot = Z_probs[:, 0].reshape(xx.shape)
    
    Z_lr = lr.predict(grid_points).reshape(xx.shape)

    # Plot 1: ODR Decision Boundaries    
    cont = ax1.contourf(xx, yy, Z_plot, levels=np.linspace(0, 1, 11), 
                    cmap="RdBu", alpha=0.6)
    fig.colorbar(cont, ax=ax1) # Add a legend for the colors
    
    # Scatter points: size proportional to 'precision' (1/noise)
    sizes = 50 / (n_test + 0.5)  # Use n_test as defined in your split 
    scatter1 = ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=sizes, 
                           cmap=plt.cm.Spectral, edgecolors='k', alpha=0.7)
    ax1.set_title(f"ODR (Weighted by Error)\nAccuracy: {odr_accuracy:.4f}")
    ax1.set_xlabel("Feature 1"); ax1.set_ylabel("Feature 2")

    # Plot 2: Standard LR Decision Boundaries
    ax2.contourf(xx, yy, Z_lr, cmap=plt.cm.Spectral, alpha=0.3)
    # Scatter points: uniform size
    scatter2 = ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, 
                           cmap=plt.cm.Spectral, edgecolors='k', alpha=0.7)
    ax2.set_title(f"Standard Logistic Regression\nAccuracy: {lr_accuracy:.4f}")
    ax2.set_xlabel("Feature 1"); ax2.set_ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig('heteroscedasticity_test_plot.png')
    plt.show()
    
    return odr, odr_accuracy, lr_accuracy

def test_collinearity():
    """Feature Correlation (Collinearity) Test
    
    Test is pass if:
       No explosion (would be >1000+) 
       Physically meaningful coefficients
       Stable solution despite 99% correlation
    """
    print("\n=== Collinearity Test ===")
    
    np.random.seed(123)
    n_samples = 800
    
    f0 = np.random.randn(n_samples)
    f1 = 0.99 * f0 + 0.01 * np.random.randn(n_samples) # 99% correlation
    f2 = np.random.randn(n_samples)
    
    X = np.column_stack([f0, f1, f2])
    y = np.random.choice([0, 1, 2], n_samples)
    X[y == 0, :2] += [1, 1]
    X[y == 1, :2] += [-1, 1] 
    X[y == 2, :2] += [0, -1]
    
    X_err = np.array([0.1, 0.1, 0.1])
    X_noisy = X + np.random.normal(0, X_err, X.shape)
    
    # IMPROVED: Random Splitting
    X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)
    X_err_train = np.tile(X_err, (len(X_train), 1))
    
    suggested_C = get_governed_C(X_train, y_train, C_max=10.0)
    odr = OrthogonalDistanceLogisticRegressionOVR(C=suggested_C, tol=1e-4, robust=True)
    odr.fit(X_train, y_train, X_err=X_err_train)
    
    odr_accuracy = accuracy_score(y_test, odr.predict(X_test))
    max_coeff = np.max(np.abs(odr.coef_))
    
    print(f"Adaptive C: {suggested_C:.4f}")
    print(f"Accuracy: {odr_accuracy:.4f} | Max Coeff: {max_coeff:.4f}")
    return odr, odr_accuracy, max_coeff > 1000

def test_boundary_overlap():
    """Boundary Overlap (Bayes Error) Test
    
    Ideal behavior - High entropy in overlapping regions shows:
     Appropriate skepticism when classes are inseparable  
     No false confidence in ambiguous areas
     Proper uncertainty quantification
    """
    print("\n=== Boundary Overlap Test ===")
    
    np.random.seed(456)
    n_samples = 1200
    
    # Class A and B overlap completely in mean, differ in shape
    X_A = np.random.multivariate_normal([0, 0], [[0.5, 0.1], [0.1, 0.5]], n_samples)
    X_B = np.random.multivariate_normal([0, 0], [[0.2, -0.1], [-0.1, 0.5]], n_samples)
    X_C = np.random.multivariate_normal([2, 2], [[0.3, 0.0], [0.0, 0.3]], n_samples)
    
    X = np.vstack([X_A, X_B, X_C])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples), 2 * np.ones(n_samples)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Higher the C value, the more "confident" is the classifier
    for C in [0.1, 1, 10., 100]:
        odr = OrthogonalDistanceLogisticRegressionOVR(C=C, tol=1e-4)
        odr.fit(X_train, y_train)
    
        probas = odr.predict_proba(X_test)
        probas = np.clip(probas, 1e-10, 1.0) # Prevent log(0)
        entropy = -np.sum(probas * np.log(probas), axis=1)
        mean_entropy = np.mean(entropy)

        print(f"C: {C} ")    
        print(f"Accuracy: {accuracy_score(y_test, odr.predict(X_test)):.4f}")
        print(f"Mean Entropy: {mean_entropy:.4f} (High = Good Uncertainty)")

    return odr, mean_entropy

def test_adversarial_outliers():
    """Adversarial Outlier Injection Test
    A model shows high robustness if:
        It maintains high accuracy despite adversarial contamination
        Low residuals indicate effective outlier identification
        Robust fitting properly handles the "liars"
    """
    print("\n=== Adversarial Outlier Test ===")
    
    np.random.seed(789)
    n_samples = 1000
    y = np.random.choice([0, 1, 2], n_samples)
    X = np.random.randn(n_samples, 2)
    X[y == 0] += [2, 2]
    X[y == 1] += [-2, 2]
    X[y == 2] += [0, -2]
    
    # Inject 5% outliers: Class 0 labeled as Class 1
    n_outliers = 50
    outlier_idx = np.where(y == 0)[0][:n_outliers]
    y[outlier_idx] = 1 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    odr = OrthogonalDistanceLogisticRegressionOVR(C=1e3, robust=True, max_robust_iter=10)
    odr.fit(X_train, y_train)
    
    # Check if robust epsilon (residuals) identified outliers
    # In OVR, epsilon is a list of arrays per class
    avg_eps = np.mean([np.mean(np.abs(e)) for e in odr.epsilon_])
    print(f"Robust Accuracy: {accuracy_score(y_test, odr.predict(X_test)):.4f}")
    print(f"Mean Residual (Epsilon): {avg_eps:.4f}")
    return odr

def test_error_sensitivity_sweep():
    """
    Physical Intervention:
    1. Training data has one-directional bias at low flux.
    2. Test data is the 'Physical Truth' (unbiased).
    3. Accuracy is measured by how well the model recovered the TRUE boundary.
    """
    print("\n=== Error Sensitivity: Bias Recovery Test ===")
    np.random.seed(42)
    n_samples = 2000

    # GROUND TRUTH
    X_true = np.random.uniform(0.1, 10.0, (n_samples, 2))
    y = (X_true[:, 1] > X_true[:, 0]).astype(int)

    noise_factors = np.logspace(0, 2.0, 8)
    odr_accs, lr_accs = [], []

    # Cap the Noise Factor: Logspace from 0 to 1.5 is plenty (1 to ~31)
    noise_factors = np.logspace(0, 1.5, 8) 

    for k in noise_factors:
        # Prevent infinite noise at the origin
        flux_tr = np.sqrt(X_true[:, 0]**2 + X_true[:, 1]**2)
        n_tr = k / (flux_tr + 1.0) # Increased floor to 1.0 to stabilize
    
        X_train_noisy = X_true.copy()
        # 3. Apply the shift, but keep it proportional to the signal
        # Systematic bias: Low flux points drift right
        
        # Bias increases with noise, but we cap it so it doesn't 
        # physically teleport the points to infinity.
        bias_limit = 5.0 
        bias_component = np.clip(n_tr * 1.5, 0, bias_limit)
        X_train_noisy[:, 0] += np.abs(np.random.normal(0, n_tr)) + bias_component
    
        # FIXED TEST SET: High-precision ground truth
        # We want to see if the model recovered the 45-degree law
        X_test = np.random.uniform(1.0, 10.0, (500, 2)) # Test on mid-to-high flux
        y_test = (X_test[:, 1] > X_test[:, 0]).astype(int)

        # Fitting
        # Inform ODR of the specific X-uncertainty
        X_err_tr = np.column_stack([n_tr, np.full_like(n_tr, 0.05)])
        
        # We use a stable C for the sweep to see the 'Stiffness' of ODR
        # Or use the governor but cap it so it doesn't zero out at high noise
        c_val = max(0.1, get_governed_C(X_train_noisy, y))
        
        odr = OrthogonalDistanceLogisticRegressionOVR(C=c_val)
        odr.fit(X_train_noisy, y, X_err=X_err_tr)
        
        lr = LogisticRegression(C=c_val, solver='lbfgs')
        lr.fit(X_train_noisy, y)

        # 4. Record
        odr_accs.append(accuracy_score(y_test, odr.predict(X_test)))
        lr_accs.append(accuracy_score(y_test, lr.predict(X_test)))

    print(f"ODR: {odr_accs}")
    print()
    print(f"OLS: {lr_accs}")
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.semilogx(noise_factors, odr_accs, 'o-', color='blue', label='ODR (Ignores Low-Flux Bias)')
    plt.semilogx(noise_factors, lr_accs, 's--', color='red', label='Standard LR (Fooled by Bias)')
    plt.title('Resistance to Asymmetric Low-Flux Bias')
    plt.xlabel('Noise/Bias Factor (k)'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True, which="both", ls="-")
    plt.show()

if __name__ == "__main__":
    test_heteroscedasticity()
    test_collinearity()
    test_boundary_overlap()
    test_adversarial_outliers()
    test_error_sensitivity_sweep()
    print("\nAll stress tests finished.")

