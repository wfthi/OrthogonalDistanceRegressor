import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def process_gaia_classification_data(file_path, sig_Av_default=0.1):
    """
    Reads Gaia DR3 data and processes it into X, y, and E for ODR.
    Incorporates extinction-inflated errors and SNR anchors.
    """
    df = pd.read_csv(file_path)
    
    # 1. Basic Cleaning: Remove rows missing critical classification data
    df = df.dropna(subset=['Gmag', 'BPmag', 'RPmag', 'pscol', 'SpType-ELS'])
    
    # 2. Encode Labels (Spectral Type)
    le = LabelEncoder()
    y = le.fit_transform(df['SpType-ELS'])
    
    # 3. Estimate Extinction (A_v) along line of sight
    # In a full pipeline, you might use a 3D map like Bayestar.
    # Here we use a distance-scaled proxy if A_v isn't directly in the CSV.
    # Dist is in parsecs (1/parallax).
    dist = df['Dist']
    # Simplified galactic model: extinction increases with distance and lower latitude
    # This provides the sig_Av used for error inflation.
    sig_Av = sig_Av_default + (dist * 0.0001) # Example scaling: 0.1 mag per kpc
    
    # 4. Color Features
    df['BP_RP'] = df['BPmag'] - df['RPmag']
    df['G_RP'] = df['Gmag'] - df['RPmag']
    
    # 5. Error Inflation (The Covariance Method via Box Approximation)
    # k-coefficients for Gaia DR3
    k_G, k_BP, k_RP = 0.84, 1.10, 0.60
    
    # Inflate instrumental errors with extinction uncertainty
    e_G = np.sqrt(df['e_Gmag']**2 + (k_G * sig_Av)**2)
    e_BP = np.sqrt(df['e_BPmag']**2 + (k_BP * sig_Av)**2)
    e_RP = np.sqrt(df['e_RPmag']**2 + (k_RP * sig_Av)**2)
    
    # Quadrature sum for color errors
    e_BP_RP = np.sqrt(e_BP**2 + e_RP**2)
    e_G_RP = np.sqrt(e_G**2 + e_RP**2)
    e_pscol = np.full(len(df), 0.05) # Instrumental constant if not provided
    
    # 6. SNR Anchors (Critical for ODR stability)
    df['SNR_BPRP'] = df['BP_RP'] / e_BP_RP
    df['SNR_GRP'] = df['G_RP'] / e_G_RP
    
    # 7. Final Feature Assembly
    # X: [BP-RP, G-RP, pscol, SNR_BPRP, SNR_GRP]
    features = ['BP_RP', 'G_RP', 'pscol', 'SNR_BPRP', 'SNR_GRP']
    X = df[features].values
    
    # E: Error matrix matching the first 3 physical features (for ODR sx)
    # SNR errors are usually treated as unit-less or derived during the fit
    E = np.vstack([e_BP_RP, e_G_RP, e_pscol]).T
    
    return X, y, E, le, df

# Execution
if __name__ == "__main__":
    X, y, E, encoder, processed_df = process_gaia_classification_data("dataGaia.csv")
    breakpoint()
