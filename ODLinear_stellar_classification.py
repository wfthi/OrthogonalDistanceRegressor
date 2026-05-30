import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Import your stable classes
try:
    from ODLinear_fast import (OrthogonalDistanceMultinomialLogisticRegression, 
                              OrthogonalDistanceLogisticRegressionOVR)
except ImportError:
    from ODLinear import (OrthogonalDistanceMultinomialLogisticRegression, 
                          OrthogonalDistanceLogisticRegressionOVR)

# 1. GENERATE DATA (Physical Units)
np.random.seed(42)
n_samples = 30
x0 = np.random.multivariate_normal([1.0, 0.3], [[0.05, 0], [0, 0.01]], n_samples)
x1 = np.random.multivariate_normal([2.5, 0.8], [[0.1, 0], [0, 0.02]], n_samples)
x2 = np.random.multivariate_normal([4.0, 0.5], [[0.2, 0], [0, 0.05]], n_samples)

X = np.vstack([x0, x1, x2])
y = np.concatenate([np.zeros(n_samples), np.ones(n_samples), np.full(n_samples, 2)])

# 2. NOISE & OUTLIERS
X_err_vec = np.array([0.3, 0.15]) 
X_obs = X + np.random.normal(0, X_err_vec, X.shape)
X_obs[35] = [1.2, 0.35] 
X_obs[75] = [2.6, 0.85] 

# 3. PREPARE DATA: Raw vs Scaled
# Raw
X_err_raw = np.tile(X_err_vec, (X_obs.shape[0], 1))

# Scaled
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_obs)
X_err_scaled = np.tile(X_err_vec / scaler.scale_, (X_obs.shape[0], 1))

# 4. FIT ALL CASES
print("--- RUNNING FULL COMPARISON ---")

# CASE A: RAW PHYSICAL UNITS
odr_ovr_raw = OrthogonalDistanceLogisticRegressionOVR(C=15.0).fit(X_obs, y, X_err=X_err_raw)
odr_multi_raw = OrthogonalDistanceMultinomialLogisticRegression(C=15.0).fit(X_obs, y, X_err=X_err_raw)

# CASE B: SCALED UNITS (with warm-start hint to prevent collapse)
std_logit_scaled = LogisticRegression(multi_class='multinomial', C=1.0).fit(X_scaled, y)
hint = np.hstack([std_logit_scaled.coef_, std_logit_scaled.intercept_.reshape(-1, 1)]).flatten()

odr_ovr_scaled = OrthogonalDistanceLogisticRegressionOVR(C=1.0).fit(X_scaled, y, X_err=X_err_scaled)
odr_multi_scaled = OrthogonalDistanceMultinomialLogisticRegression(C=1.0).fit(X_scaled, y, X_err=X_err_scaled, initial_guess=hint)

# 5. REPORTING FUNCTION
def report(model, X_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

print("\n--- RESULTS SUMMARY ---")
report(odr_ovr_raw, X_obs, "OVR (Raw)")
report(odr_multi_raw, X_obs, "Multinomial (Raw)")
report(odr_ovr_scaled, X_scaled, "OVR (Scaled)")
report(odr_multi_scaled, X_scaled, "Multinomial (Scaled)")

# 6. VISUALIZATION (4-Panel Comparison)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

def plot_case(model, X_data, title, ax):
    h = .02
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(X_data[:, 0], X_data[:, 1], c=y, edgecolors='k', cmap='viridis', s=20)
    ax.set_title(title)

plot_case(odr_ovr_raw, X_obs, "OVR - Raw (The Physical Peak)", ax1)
plot_case(odr_multi_raw, X_obs, "Multinomial - Raw (The Symmetry Trap)", ax2)
plot_case(odr_ovr_scaled, X_scaled, "OVR - Scaled (ML Standard)", ax3)
plot_case(odr_multi_scaled, X_scaled, "Multinomial - Scaled (The Middle-Ground)", ax4)

plt.tight_layout()
plt.show()