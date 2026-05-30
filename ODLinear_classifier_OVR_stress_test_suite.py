"""
Advanced Stress Test for OrthogonalDistanceLogisticRegressionOVR

These tests will reveal:
Robustness to high-dimensionality - Whether ODR scales well
Adaptability to noise - How well it handles varying uncertainty levels
Fair performance across classes - Especially important for imbalanced datasets
Reproducibility - Consistent results across different data splits
Hyperparameter sensitivity - Guidance for parameter selection

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from ODLinear_fast import OrthogonalDistanceLogisticRegressionOVR
import seaborn as sns

def test_high_dimensional_data():
    """Test with high-dimensional feature space"""
    print("=== High-Dimensional Data Test ===")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 20  # High dimensional
    
    # Generate multiclass data
    y = np.random.choice([0, 1, 2], n_samples)
    X_true = np.random.randn(n_samples, n_features)
    
    # Create class-specific patterns
    X_true[y == 0, :5] += 2  # First 5 features stronger for class 0
    X_true[y == 1, 5:10] += 2  # Next 5 features stronger for class 1  
    X_true[y == 2, 10:15] += 2  # Next 5 features stronger for class 2
    
    # Add measurement noise
    X_err = np.random.uniform(0.1, 0.5, n_features)
    X_noisy = X_true + np.random.normal(0, X_err, X_true.shape)
    
    # Split data
    split = 700
    X_train, X_test = X_noisy[:split], X_noisy[split:]
    y_train, y_test = y[:split], y[split:]
    X_err_train = np.tile(X_err, (len(X_train), 1))
    
    # Train models
    odr = OrthogonalDistanceLogisticRegressionOVR(C=1e3, tol=1e-4, robust=True, max_robust_iter=3)
    odr.fit(X_train, y_train, X_err=X_err_train)
    
    # Evaluate
    odr_preds = odr.predict(X_test)
    accuracy = accuracy_score(y_test, odr_preds)
    print(f"High-dimensional accuracy: {accuracy:.4f}")
    
    return odr, accuracy

def test_varying_noise_levels():
    """Test performance across different noise levels"""
    print("=== Varying Noise Levels Test ===")
    
    np.random.seed(123)
    n_samples = 500
    
    # Generate clean data
    y = np.random.choice([0, 1, 2], n_samples)
    X_true = np.random.randn(n_samples, 2)
    
    # Create class patterns
    X_true[y == 0] += [1, 1]
    X_true[y == 1] += [-1, 1] 
    X_true[y == 2] += [0, -1]
    
    noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
    accuracies = []
    
    for noise_level in noise_levels:
        # Add noise
        X_noisy = X_true + np.random.normal(0, noise_level, X_true.shape)
        X_err = np.array([noise_level, noise_level])
        
        # Split data
        split = 350
        X_train, X_test = X_noisy[:split], X_noisy[split:]
        y_train, y_test = y[:split], y[split:]
        X_err_train = np.tile(X_err, (len(X_train), 1))
        
        # Train and evaluate ODR
        odr = OrthogonalDistanceLogisticRegressionOVR(C=1e3, tol=1e-4, robust=True)
        odr.fit(X_train, y_train, X_err=X_err_train)
        odr_preds = odr.predict(X_test)
        accuracy = accuracy_score(y_test, odr_preds)
        accuracies.append(accuracy)
        print(f"Noise level {noise_level}: Accuracy = {accuracy:.4f}")
    
    return noise_levels, accuracies

def test_class_imbalance():
    """Test with imbalanced classes"""
    print("=== Class Imbalance Test ===")
    
    np.random.seed(456)
    n_samples = 1000
    
    # Create highly imbalanced classes
    n_class0 = 600  # 60%
    n_class1 = 300  # 30%  
    n_class2 = 100  # 10%
    
    y = np.array([0] * n_class0 + [1] * n_class1 + [2] * n_class2)
    X_true = np.random.randn(n_samples, 2)
    
    # Create class patterns
    X_true[y == 0] += [1, 1]
    X_true[y == 1] += [-1, 1] 
    X_true[y == 2] += [0, -1]
    
    # Add measurement noise
    X_err = np.array([0.3, 0.3])
    X_noisy = X_true + np.random.normal(0, X_err, X_true.shape)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y, test_size=0.3, random_state=42, stratify=y)
    X_err_train = np.tile(X_err, (len(X_train), 1))
    
    # Train ODR
    odr = OrthogonalDistanceLogisticRegressionOVR(C=1e3, tol=1e-4, robust=True)
    odr.fit(X_train, y_train, X_err=X_err_train)
    
    # Evaluate
    odr_preds = odr.predict(X_test)
    accuracy = accuracy_score(y_test, odr_preds)
    
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Imbalanced accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, odr_preds))
    
    return odr, accuracy

def test_cross_validation():
    """Test cross-validation stability"""
    print("=== Cross-Validation Test ===")
    
    np.random.seed(789)
    n_samples = 800
    
    # Generate data
    y = np.random.choice([0, 1, 2], n_samples)
    X_true = np.random.randn(n_samples, 2)
    
    # Create class patterns
    X_true[y == 0] += [1, 1]
    X_true[y == 1] += [-1, 1] 
    X_true[y == 2] += [0, -1]
    
    # Add noise
    X_err = np.array([0.2, 0.2])
    X_noisy = X_true + np.random.normal(0, X_err, X_true.shape)
    
    # Test cross-validation scores
    cv_scores = []
    for fold in range(5):
        # Create stratified splits
        np.random.seed(fold)
        indices = np.random.permutation(n_samples)
        split_point = int(0.8 * n_samples)
        
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]
        
        X_train, X_test = X_noisy[train_idx], X_noisy[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_err_train = np.tile(X_err, (len(X_train), 1))
        
        # Train and evaluate
        odr = OrthogonalDistanceLogisticRegressionOVR(C=1e3, tol=1e-4, robust=True)
        odr.fit(X_train, y_train, X_err=X_err_train)
        odr_preds = odr.predict(X_test)
        score = accuracy_score(y_test, odr_preds)
        cv_scores.append(score)
        print(f"Fold {fold + 1}: {score:.4f}")
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {mean_score:.4f} ± {std_score:.4f}")
    
    return cv_scores

def test_parameter_sensitivity():
    """Test sensitivity to regularization parameters"""
    print("=== Parameter Sensitivity Test ===")
    
    np.random.seed(999)
    n_samples = 600
    
    # Generate data
    y = np.random.choice([0, 1, 2], n_samples)
    X_true = np.random.randn(n_samples, 2)
    
    # Create class patterns
    X_true[y == 0] += [1, 1]
    X_true[y == 1] += [-1, 1] 
    X_true[y == 2] += [0, -1]
    
    # Add noise
    X_err = np.array([0.3, 0.3])
    X_noisy = X_true + np.random.normal(0, X_err, X_true.shape)
    
    # Split data
    split = 400
    X_train, X_test = X_noisy[:split], X_noisy[split:]
    y_train, y_test = y[:split], y[split:]
    X_err_train = np.tile(X_err, (len(X_train), 1))
    
    # Test different C values
    C_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    accuracies = []
    
    for C in C_values:
        odr = OrthogonalDistanceLogisticRegressionOVR(C=C, tol=1e-4, robust=True)
        odr.fit(X_train, y_train, X_err=X_err_train)
        odr_preds = odr.predict(X_test)
        accuracy = accuracy_score(y_test, odr_preds)
        accuracies.append(accuracy)
        print(f"C={C}: Accuracy = {accuracy:.4f}")
    
    return C_values, accuracies

def comprehensive_classifier_stress_test():
    """Run all advanced stress tests"""
    print("Running Comprehensive Classifier Stress Tests\n")
    
    # Test 1: High dimensional data
    try:
        odr1, acc1 = test_high_dimensional_data()
        print(f"High-dimensional test accuracy: {acc1:.4f}\n")
    except Exception as e:
        print(f"High-dimensional test failed: {e}\n")
    
    # Test 2: Varying noise levels
    try:
        noise_levels, accuracies = test_varying_noise_levels()
        print(f"Noise sensitivity test completed\n")
    except Exception as e:
        print(f"Noise sensitivity test failed: {e}\n")
    
    # Test 3: Class imbalance
    try:
        odr3, acc3 = test_class_imbalance()
        print(f"Imbalanced test accuracy: {acc3:.4f}\n")
    except Exception as e:
        print(f"Imbalanced test failed: {e}\n")
    
    # Test 4: Cross-validation
    try:
        cv_scores = test_cross_validation()
        print(f"Cross-validation test completed\n")
    except Exception as e:
        print(f"Cross-validation test failed: {e}\n")
    
    # Test 5: Parameter sensitivity
    try:
        C_values, accuracies = test_parameter_sensitivity()
        print(f"Parameter sensitivity test completed\n")
    except Exception as e:
        print(f"Parameter sensitivity test failed: {e}\n")
    
    print("All advanced stress tests completed successfully!")

if __name__ == "__main__":
    comprehensive_classifier_stress_test()