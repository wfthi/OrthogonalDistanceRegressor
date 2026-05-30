import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Import your classes here
from ODLinear_fast import OrthogonalDistanceMultinomialLogisticRegression

def generate_galaxy_data(n_samples=10000, noise_level=0.5, outlier_ratio=0.05):
    """
    Simulates galaxy features (e.g., u-g, g-r colors) for Spiral (0) vs Elliptical (1).
    Introduces systematic noise and outliers.
    """
    np.random.seed(42)
    
    # Feature 1: g-r color (separation between populations)
    # Feature 2: Concentration index
    X0 = np.random.multivariate_normal([1.2, 2.2], [[0.1, 0.05], [0.05, 0.1]], n_samples // 2)
    X1 = np.random.multivariate_normal([1.8, 3.2], [[0.1, 0.05], [0.05, 0.1]], n_samples // 2)
    
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2))).astype(int)
    
    # Add Standard Measurement Noise (X_err)
    X_err = np.full(X.shape, noise_level)
    X += np.random.normal(0, noise_level, X.shape)
    
    # Introduce SYSTEMATIC OUTLIERS (The "Robust" target)
    # 5% of data is hit with 10x the noise (simulating cosmic rays/blending)
    n_outliers = int(n_samples * outlier_ratio)
    outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
    X[outlier_idx] += np.random.normal(0, noise_level * 10, (n_outliers, 2))
    
    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices], X_err[indices]

def generate_contaminated_galaxy_data(n_samples=10000, contamination_rate=0.10):
    """
    Simulates high-quality calibrated data but with label errors.
    """
    np.random.seed(42)
    # Features (g-r color and concentration)
    X0 = np.random.multivariate_normal([1.2, 2.2], [[0.05, 0.02], [0.02, 0.05]], n_samples // 2)
    X1 = np.random.multivariate_normal([1.8, 3.2], [[0.05, 0.02], [0.02, 0.05]], n_samples // 2)
    
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2))).astype(int)
    
    # DATA QUALITY: Add only standard measurement noise
    X_err = np.full(X.shape, 0.05)
    X += np.random.normal(0, 0.05, X.shape)
    
    # CONTAMINATION: Flip 10% of the labels (The "Invisible" Outliers)
    flip_idx = np.random.choice(n_samples, int(n_samples * contamination_rate), replace=False)
    y[flip_idx] = 1 - y[flip_idx] 
    
    return X, y, X_err

# Run the test

# 1. Generate 10,000 samples
contamination = True
if contamination:
    X, y, X_err = generate_contaminated_galaxy_data(n_samples=10000)
else:
    X, y, X_err = generate_galaxy_data(n_samples=10000)

X_train, X_test, y_train, y_test, X_err_train, X_err_test = train_test_split(X, y, X_err, test_size=0.2, random_state=42)

print(f"--- Dataset: Galaxy Classification (N={len(X_train)}) ---")

# 2. Run Standard Multinomial ODR
print("\n[Running Standard ODR...]")
odlr_std = OrthogonalDistanceMultinomialLogisticRegression(robust=False)
odlr_std.fit(X_train, y_train, X_err=X_err_train)
y_pred_std = odlr_std.predict(X_test)
acc_std = accuracy_score(y_test, y_pred_std)

# 3. Run Robust Multinomial ODR (Should trigger 3-sigma clipping)
print("\n[Running Robust ODR (3-Sigma Clipping)...]")
odlr_rob = OrthogonalDistanceMultinomialLogisticRegression(robust=True, max_robust_iter=5)
odlr_rob.fit(X_train, y_train, X_err=X_err_train)
y_pred_rob = odlr_rob.predict(X_test)
acc_rob = accuracy_score(y_test, y_pred_rob)

print("\n--- FINAL RESULTS ---")
print(f"Standard Accuracy: {acc_std:.4f}")
print(f"Robust Accuracy:   {acc_rob:.4f}")
print(f"Improvement:       {(acc_rob - acc_std)*100:.2f}%")