# -*- coding: utf-8 -*-
"""
Kelly 2007 Dataset Test: OLS vs ODR Comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ODLinear_fast import OrthogonalDistanceLinearRegression

# Kelly 2007 dataset
x = np.array([ 0.27160351, -0.05666712,  0.0148909 ,  0.94171762, -0.88913964,
       -1.25785874,  0.39363033,  0.37059422, -0.90615775,  0.90498088,
       -0.429674  , -0.17900806,  0.80771356,  0.38811852,  0.06503306,
        0.18258487,  1.15785868, -0.81248739, -0.71272798, -2.14394123,
       -0.07862118, -3.11106395, -0.56047   , -0.38415866, -1.01591122,
       -0.49949083, -3.38368868, -3.02169256,  0.42445945, -0.05229724,
       -2.53952609,  0.1634791 ,  0.85623914, -1.85489074, -0.43861255,
       -2.60017811, -3.48509755, -0.58991145, -0.48885896,  0.56396916,
        0.47765034,  2.16433331, -0.1611202 , -1.53842563,  0.21220742,
       -0.946258  , -0.87039763, -2.09499142, -3.51970367,  1.33948891,
        0.60438143,  0.34631407, -0.65521324, -0.85458591, -4.04603174,
       -0.8639032 , -0.11514712,  0.11577957,  0.20972625, -1.3154218 ,
        2.15515339,  0.19347424, -0.33177445,  0.3480446 , -0.69119449,
       -1.63395732, -1.00720278,  0.46108507, -4.39920489, -1.16512153,
       -3.28932748, -2.4817363 ,  0.66512237,  0.01089362, -0.52887875,
       -2.04475302, -0.41618787, -0.34146875, -1.50809465, -0.52268816,
       -0.70120567, -0.11154978,  0.38761169, -3.51271498, -0.98794272,
        0.01309817, -0.21184287,  0.37346753,  0.00978954, -0.85901428,
       -2.60222494, -0.22878616, -3.2751648 , -0.49515905,  1.22209584,
       -0.98303944, -0.09605539,  0.12466544, -0.31125185, -0.17043109])

y = np.array([ 1.9403452 ,  2.20121033,  1.5688266 ,  1.27581206,  0.9887312 ,
        1.32955877,  3.98865354,  2.12817836,  2.87784209,  2.78422471,
        0.11834964,  1.33360154,  2.11279652,  3.4183064 ,  1.42902325,
        2.91069526,  2.3794769 ,  0.4653009 ,  2.19031481,  0.45864241,
        2.11506926, -1.50835983,  2.46727348,  0.68083957,  2.82352079,
        3.46825359, -1.68298361, -1.65614146,  2.55442456,  1.43317162,
       -1.12446367,  2.34748129,  1.69837237,  0.99670571,  1.32785069,
        1.0353256 , -1.14612591,  1.26251048,  1.82566504,  0.98617896,
        2.34281863,  5.09947263,  0.49996771,  1.03522287,  2.61974461,
        1.68288214,  0.60087425, -1.22664364, -0.95435192,  3.73897021,
        2.71506283,  1.91248441,  0.93869886,  0.69355672,  0.63351139,
        0.0216733 ,  2.99086795,  2.75086281,  1.11010988,  1.6109603 ,
        2.61530663,  2.81544981,  2.77804655,  1.23821835,  2.14018352,
        0.51018906,  1.5861735 ,  3.66769748, -2.16893689, -0.51321169,
       -1.2152835 , -1.87788335,  2.52897934,  2.26847585,  1.89125027,
        0.57411836,  1.82025556,  2.09387741,  0.30905545,  3.79707928,
        1.82960935,  1.31867364,  0.72150546, -0.40210889,  1.44418028,
        2.30601349,  1.90669214,  2.72687905,  1.86273425, -0.78581536,
       -0.78621764,  2.53490638, -1.73854337,  0.71706329,  3.1135282 ,
        2.02042593,  0.65058547,  2.13559449,  2.13962426,  0.1633826 ])

dx = np.array([0.34792637, 0.21526068, 0.26034054, 0.49426219, 0.48145707,
       0.18477939, 0.17961414, 0.25601451, 0.58454177, 0.47060719,
       0.31775888, 0.28956841, 0.22066886, 0.36056287, 0.18734918,
       0.18863182, 0.14403579, 0.30884015, 0.25329124, 0.34268208,
       0.5793057 , 0.09785248, 0.15576476, 0.19101586, 0.31588901,
       0.61922534, 0.17217101, 0.34677166, 0.23957815, 0.47190709,
       0.26642411, 0.45163955, 0.43941935, 0.26865147, 1.03196718,
       0.99072725, 0.53366165, 0.3634675 , 0.1839337 , 0.34992985,
       0.31724138, 0.38370251, 0.25352348, 0.58368017, 0.33760359,
       0.24125591, 0.27003604, 0.31060333, 0.30810911, 0.2273499 ,
       0.09516074, 0.55540946, 0.09358684, 0.57453882, 2.24043792,
       0.45822339, 0.25049226, 0.10829074, 0.61983833, 0.22819577,
       0.90931357, 0.11063252, 0.15399022, 0.43879749, 0.14044182,
       0.33896633, 0.14926931, 0.31215474, 0.30347853, 0.43248467,
       0.2108651 , 0.28900034, 0.22508312, 1.12717619, 0.14615611,
       0.12696759, 0.28988536, 0.09929357, 0.11477696, 0.20806213,
       0.13936955, 0.16443002, 1.43788544, 0.37838277, 0.19941002,
       0.1221484 , 0.96417888, 1.35163611, 0.32539372, 2.08271853,
       0.33302706, 0.24660901, 0.64228609, 0.32699552, 0.17409932,
       0.45981818, 0.56723148, 0.33719716, 0.69818741, 0.10384423])

dy = np.array([0.16899589, 0.29040872, 0.15017146, 0.18698631, 0.20189375,
       0.2540919 , 0.2631942 , 0.61064304, 0.07707025, 0.11977142,
       0.14109067, 0.18903318, 0.3596953 , 0.30409919, 0.55551801,
       0.27829561, 0.40203549, 0.06556261, 0.10024742, 0.16428946,
       0.33183352, 0.38642706, 0.19753492, 0.09776092, 0.54253786,
       0.1207547 , 0.27919898, 0.2457685 , 0.16749474, 0.32348111,
       0.20451198, 0.34513951, 0.26339027, 0.10777028, 0.16190357,
       0.10802101, 0.28188131, 0.12020127, 0.68747171, 0.1675439 ,
       0.11541532, 0.71322868, 0.27946466, 0.1676212 , 0.09349124,
       0.26341508, 0.12320032, 0.07415928, 0.11973435, 0.19932773,
       0.13586064, 0.26907182, 0.16328208, 0.19275255, 0.1387571 ,
       0.23217088, 0.13389544, 0.2337624 , 0.10084675, 0.2178546 ,
       1.10863677, 0.37659195, 0.17092918, 0.06590881, 0.10327223,
       1.39064382, 0.5880967 , 0.08672661, 0.11581428, 0.04985173,
       0.24063076, 0.32865019, 0.16492754, 0.2440002 , 0.09189336,
       0.25011499, 0.20912284, 0.36703564, 0.10767048, 0.34948506,
       0.13232271, 0.05981153, 0.81260886, 0.07212808, 0.14392913,
       0.41835831, 0.16082594, 0.10964386, 0.31276716, 0.12766477,
       0.06868627, 0.25034552, 0.3705307 , 0.18875167, 0.1666866 ,
       0.08908801, 0.19306925, 0.23480059, 0.3779272 , 0.70807727])

def test_kelly_dataset():
    """Test Kelly 2007 dataset with OLS vs ODR comparison"""
    print("=== Kelly 2007 Dataset: OLS vs ODR Comparison ===")
    
    # Prepare data
    X = x.reshape(-1, 1)
    
    # Create error arrays
    X_err = dx.reshape(-1, 1)
    y_err = dy.reshape(-1, 1)
    
    # Fit OLS (standard linear regression)
    ols = LinearRegression()
    ols.fit(X, y)
    ols_slope = ols.coef_[0]
    ols_intercept = ols.intercept_
    ols_pred = ols.predict(X)
    
    # Fit ODR (orthogonal distance regression)
    odr = OrthogonalDistanceLinearRegression(C=5.0, maxit=100, error_type='std', verbose=False)
    odr.fit(x, y, X_err=dx, y_err=dy)
    odr_slope = odr.coef_[0]
    odr_intercept = odr.intercept_
    odr_pred = odr.predict(x)
    
    # Calculate statistics
    ols_rss = np.sum((y - ols_pred) ** 2)
    odr_rss = np.sum((y - odr_pred) ** 2)
    
    print(f"OLS: Slope = {ols_slope:.4f}, Intercept = {ols_intercept:.4f}")
    print(f"ODR: Slope = {odr_slope:.4f}, Intercept = {odr_intercept:.4f}")
    print(f"OLS RSS: {ols_rss:.4f}")
    print(f"ODR RSS: {odr_rss:.4f}")
    
    # Create detailed comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Basic comparison
    ax1.errorbar(x, y, xerr=dx, yerr=dy, fmt='o', alpha=0.6, 
                label='Data with errors', color='blue', ecolor='gray')
    
    # OLS line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_ols_line = ols_slope * x_line + ols_intercept
    ax1.plot(x_line, y_ols_line, 'r-', linewidth=2, label=f'OLS: y = {ols_slope:.3f}x + {ols_intercept:.3f}')
    
    # ODR line
    y_odr_line = odr_slope * x_line + odr_intercept
    ax1.plot(x_line, y_odr_line, 'g-', linewidth=2, label=f'ODR: y = {odr_slope:.3f}x + {odr_intercept:.3f}')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Kelly 2007 Dataset: OLS vs ODR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals comparison
    ols_residuals = y - ols_pred
    odr_residuals = y - odr_pred
    
    ax2.scatter(ols_pred, ols_residuals, alpha=0.6, color='red', label='OLS Residuals')
    ax2.scatter(odr_pred, odr_residuals, alpha=0.6, color='green', label='ODR Residuals')
    
    # Add horizontal line at y=0
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kelly_dataset_comparison.png')
    plt.show()
    
    # Additional statistics
    print("\n=== Detailed Statistics ===")
    ols_r2 = 1 - ols_rss/np.sum((y - np.mean(y))**2)
    odr_r2 = 1 - odr_rss/np.sum((y - np.mean(y))**2)
    print(f"OLS R²: {ols_r2:.4f}")
    print(f"ODR R²: {odr_r2:.4f}")
    
    # Show some sample predictions
    print("\n=== Sample Comparisons ===")
    sample_indices = [0, 5, 10, 15, 20]
    for i in sample_indices:
        print(f"Point {i}: x={x[i]:.3f}, y={y[i]:.3f}")
        print(f"  OLS pred: {ols_pred[i]:.3f} (residual: {ols_residuals[i]:.3f})")
        print(f"  ODR pred: {odr_pred[i]:.3f} (residual: {odr_residuals[i]:.3f})")
        print()
    
    return ols, odr, ols_slope, odr_slope

def test_error_weighting_impact():
    """Demonstrate the impact of error weighting"""
    print("\n=== Error Weighting Impact Analysis ===")
    
    # Create a simplified example to show the difference
    x_simple = np.array([1, 2, 3, 4, 5])
    y_simple = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship
    
    # Add some noise with different error levels
    x_err = np.array([0.1, 0.5, 0.1, 1.0, 0.2])  # Different measurement uncertainties
    y_err = np.array([0.1, 0.2, 0.1, 0.3, 0.1])
    
    # Fit both models
    ols_simple = LinearRegression()
    ols_simple.fit(x_simple.reshape(-1, 1), y_simple)
    
    odr_simple = OrthogonalDistanceLinearRegression(C=1e5, verbose=False)
    odr_simple.fit(x_simple, y_simple, X_err=x_err, y_err=y_err)
    
    print(f"Simple Example:")
    print(f"OLS: y = {ols_simple.coef_[0]:.3f}x + {ols_simple.intercept_:.3f}")
    print(f"ODR: y = {odr_simple.coef_[0]:.3f}x + {odr_simple.intercept_:.3f}")
    
    # Show how error weighting affects the result
    print(f"\nError weighting effect:")
    print(f"Points with low error (x_err=0.1): More influential in ODR")
    print(f"Points with high error (x_err=1.0): Less influential in ODR")
    
    # Plot the simple example
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_simple, y_simple, xerr=x_err, yerr=y_err, fmt='o', 
                alpha=0.7, label='Data with errors')
    
    x_line = np.linspace(0, 6, 100)
    y_ols = ols_simple.coef_[0] * x_line + ols_simple.intercept_
    y_odr = odr_simple.coef_[0] * x_line + odr_simple.intercept_
    
    plt.plot(x_line, y_ols, 'r--', linewidth=2, label=f'OLS: y = {ols_simple.coef_[0]:.3f}x + {ols_simple.intercept_:.3f}')
    plt.plot(x_line, y_odr, 'g-', linewidth=2, label=f'ODR: y = {odr_simple.coef_[0]:.3f}x + {odr_simple.intercept_:.3f}')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simple Example: Error Weighting Impact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('error_weighting_example.png')
    plt.show()

if __name__ == "__main__":
    # Test the Kelly dataset
    try:
        ols_model, odr_model, ols_slope, odr_slope = test_kelly_dataset()
        print("Kelly dataset test completed successfully!")
    except Exception as e:
        print(f"Error in Kelly dataset test: {e}")
        # Fallback to basic test
        print("Running basic comparison...")
        ols = LinearRegression()
        ols.fit(x.reshape(-1, 1), y)
        print(f"Basic OLS: slope = {ols.coef_[0]:.4f}")

    # Test error weighting impact
    try:
        test_error_weighting_impact()
        print("Error weighting test completed successfully!")
    except Exception as e:
        print(f"Error in error weighting test: {e}")
    
    print("\n=== Summary ===")
    print("The Kelly 2007 dataset demonstrates:")
    print("1. ODR accounts for measurement uncertainties in both x and y")
    print("2. OLS ignores measurement errors entirely")
    print("3. When errors vary significantly, ODR can give very different results")
    print("4. ODR is more robust to measurement noise")

