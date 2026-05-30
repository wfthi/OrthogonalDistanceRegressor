import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal



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

def get_scientific_governance(df, features, error_cols, C_max=30.0):
    """
    Computes a Mixing Index based on physical distribution overlap.
    Relaxes C if errors are high but distributions remain distinct.
    """
    classes = df['class'].unique()
    total_bc = 0
    pairs = 0
    
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            d1 = df[df['class'] == classes[i]]
            d2 = df[df['class'] == classes[j]]
            
            # Means of the features
            mu1 = d1[features].mean().values
            mu2 = d2[features].mean().values
            
            # Covariance informed by measurement errors (E_linear)
            # We treat the errors as the primary source of variance for the boundary
            sig1 = np.diag(d1[error_cols].mean().values**2)
            sig2 = np.diag(d2[error_cols].mean().values**2)
            
            # Bhattacharyya Distance
            sig_avg = (sig1 + sig2) / 2
            # Term 1: Difference in means
            try:
                t1 = 0.125 * np.dot(np.dot((mu1 - mu2), np.linalg.inv(sig_avg)), (mu1 - mu2))
                # Term 2: Ratio of covariances
                t2 = 0.5 * np.log(np.linalg.det(sig_avg) / np.sqrt(np.linalg.det(sig1) * np.linalg.det(sig2)))
                bc = np.exp(-(t1 + t2)) # Bhattacharyya Coefficient
            except np.linalg.LinAlgError:
                bc = 1.0 # If math fails, assume total overlap
                
            total_bc += bc
            pairs += 1
            
    mixing_index = total_bc / pairs
    
    # New Governance: If we have scientific errors, we allow C to stay higher 
    # to avoid "clipping" the physical anchors like SNR.
    alpha_scientific = 2.0 # Less aggressive than the previous 7.0
    C_opt = C_max * np.exp(-alpha_scientific * mixing_index)
    
    return np.clip(C_opt, 5.0, C_max), mixing_index

def get_sophisticated_governed_C(X, y, E, C_max=10.0, alpha=5.0):
    """
    Computes a Mixing Index based on Probabilistic Overlap 
    using Measurement Errors (E).
    """
    classes = np.unique(y)
    total_overlap = 0
    pairs = 0

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            # Extract data for class pairs
            idx_a = (y == classes[i])
            idx_b = (y == classes[j])
            
            mu_a, mu_b = X[idx_a].mean(axis=0), X[idx_b].mean(axis=0)
            # Use E (measurement errors) to define the physical variance
            sigma_a = np.sqrt(np.mean(E[idx_a]**2, axis=0))
            sigma_b = np.sqrt(np.mean(E[idx_b]**2, axis=0))
            
            # Compute Bhattacharyya Coefficient (measure of overlap)
            # BC = 1 (complete overlap), BC = 0 (perfect separation)
            avg_sigma_sq = (sigma_a**2 + sigma_b**2) / 2
            d_b = 0.125 * np.sum((mu_a - mu_b)**2 / avg_sigma_sq) + \
                  0.5 * np.log(np.prod(avg_sigma_sq) / (np.prod(sigma_a) * np.prod(sigma_b)))
            
            overlap = np.exp(-d_b)
            total_overlap += overlap
            pairs += 1

    mixing_index = total_overlap / pairs
    
    # Apply your established governance logic
    C_opt = C_max * np.exp(-alpha * mixing_index)
    return np.clip(C_opt, 0.05, C_max), mixing_index

def generate_scientific_multiclass_data(n_samples_base=1000, imbalance_ratio=0.5):
    """
    Generates a 3-class dataset where boundaries have scientific meaning.
    
    Class 0: "Stable" - High SNR, low overlap with others.
    Class 1: "Transition" - Moderate overlap with Class 0.
    Class 2: "Rare/Noisy" - High overlap with Class 1, lower sample count.
    """
    np.random.seed(42)
    
    # Define class sizes to test imbalance correction
    # Class 0 and 1 are full size, Class 2 is reduced by the imbalance_ratio
    sizes = [n_samples_base, n_samples_base, int(n_samples_base * imbalance_ratio)]
    total_samples = sum(sizes)
    
    data = []
    
    for class_id, n in enumerate(sizes):
        # 1. Generate "True" Physical Values (Ground Truth)
        # We'll use 3 features: 'Signal_A', 'Signal_B', 'Physical_Index'
        if class_id == 0:
            true_a = np.random.normal(loc=10, scale=1.5, size=n)
            true_b = np.random.normal(loc=2, scale=1.0, size=n)
            true_idx = np.random.normal(loc=0.5, scale=0.2, size=n)
        elif class_id == 1:
            # Class 1 overlaps with Class 0 on Signal_A but shifts on Signal_B
            true_a = np.random.normal(loc=8.5, scale=2.0, size=n)
            true_b = np.random.normal(loc=5, scale=1.5, size=n)
            true_idx = np.random.normal(loc=1.2, scale=0.3, size=n)
        else:
            # Class 2 heavily overlaps with Class 1 (The difficult boundary)
            true_a = np.random.normal(loc=7.5, scale=2.5, size=n)
            true_b = np.random.normal(loc=6, scale=2.0, size=n)
            true_idx = np.random.normal(loc=1.5, scale=0.4, size=n)

        # 2. Generate Measurement Errors (E_linear)
        # In real science, errors are often heteroscedastic (depend on the signal)
        err_a = 0.1 + 0.05 * true_a + np.random.uniform(0, 0.2, size=n)
        err_b = 0.2 + 0.03 * true_b + np.random.uniform(0, 0.1, size=n)
        err_idx = np.full(n, 0.05) # Constant instrument error for the index
        
        # 3. Observed Values = True Values + Measurement Noise
        obs_a = true_a + np.random.normal(0, err_a)
        obs_b = true_b + np.random.normal(0, err_b)
        obs_idx = true_idx + np.random.normal(0, err_idx)
        
        # Calculate SNR as it is a critical anchor in your ODR method
        snr_a = obs_a / err_a
        snr_b = obs_b / err_b

        # Collect into a list of dictionaries
        for i in range(n):
            data.append({
                'class': class_id,
                'feat_A': obs_a[i],
                'feat_B': obs_b[i],
                'feat_Index': obs_idx[i],
                'err_A': err_a[i],
                'err_B': err_b[i],
                'err_Index': err_idx[i],
                'SNR_A': snr_a[i],
                'SNR_B': snr_b[i]
            })

    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Dataset Generated: {len(df)} samples")
    print(f"Class Distribution:\n{df['class'].value_counts(normalize=True) * 100}")
    
    return df

# --- Execution ---
if __name__ == "__main__":
    scientific_df = generate_scientific_multiclass_data(n_samples_base=1000, imbalance_ratio=0.3)

    # FIX: Ensure X and E have the same columns (the base physical features)
    features_with_errors = ['feat_A', 'feat_B', 'feat_Index']
    X = scientific_df[features_with_errors].values
    E = scientific_df[['err_A', 'err_B', 'err_Index']].values
    y = scientific_df['class'].values

    print("\nSample Data (First 5 rows):")
    print(scientific_df.head())

    # The shapes will match (3 features and 3 error columns)
    suggested_C_knn, mixing_index_knn = get_governed_C(X, y)
    print(f"\nMixing index (knn): {mixing_index_knn}")
    print(f"Suggested C (knn): {suggested_C_knn}")

    suggested_C_soph, mixing_index_soph = get_sophisticated_governed_C(X, y, E)
    print(f"Mixing index (Bhattacharyya): {mixing_index_soph}")
    print(f"Suggested C (Bhattacharyya): {suggested_C_soph}")

    # For the final governance test
    suggested_C_sci, sci_mixing = get_scientific_governance(scientific_df, features_with_errors, ['err_A', 'err_B', 'err_Index'])
    print(f"Scientific Mixing Index (Overlap): {sci_mixing:.4f}")
    print(f"Adjusted Physical C: {suggested_C_sci:.4f}")

