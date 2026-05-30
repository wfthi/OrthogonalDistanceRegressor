import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_uq_dataset(n_samples=500):
    np.random.seed(42)
    
    # 1. Independent variable
    x = np.linspace(0, 10, n_samples)
    
    # 2. True underlying function (Ground Truth)
    y_true = np.sin(x) * x
    
    # 3. Modeled Uncertainty (The "Error Bar" values)
    # We'll make uncertainty grow linearly with x to simulate sensor drift/noise
    uncertainty = 0.2 + 0.15 * x 
    
    # 4. Observed Measurements (y_true + random noise scaled by uncertainty)
    noise = np.random.normal(0, uncertainty)
    y_measured = y_true + noise
    
    return pd.DataFrame({
        'feature_x': x,
        'label_measured': y_measured,
        'label_true': y_true,
        'uncertainty': uncertainty
    })

def generate_classification_uq(n_samples=1000):
    np.random.seed(42)
    # Generate two clusters (Class 0 and Class 1)
    x = np.concatenate([np.random.normal(2, 1, n_samples//2), 
                        np.random.normal(5, 1, n_samples//2)])
    classes = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Generate heteroscedastic uncertainty (higher uncertainty for higher X values)
    # This simulates a sensor that becomes less reliable at high ranges
    uncertainty = 0.1 + 0.2 * np.abs(x) 

    # Add noise to create 'measured' features
    x_measured = x + np.random.normal(0, uncertainty)
       
    return pd.DataFrame({'feature': x_measured, 'uncertainty': uncertainty, 'class': classes})

df_class = generate_classification_uq()
print(df_class.head())


# Create the data
df = generate_uq_dataset()

# Quick Visualization
plt.figure(figsize=(10, 5))
plt.errorbar(df['feature_x'], df['label_measured'], yerr=df['uncertainty'], 
             fmt='o', alpha=0.3, label='Measured (with error bars)', color='gray')
plt.plot(df['feature_x'], df['label_true'], color='red', lw=2, label='True Signal')
plt.legend()
plt.title("Synthetic ML Dataset with Measurement Uncertainties")
plt.show()

print(df.head())
