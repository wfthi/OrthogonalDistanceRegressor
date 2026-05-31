"""
    Machine learning regressor and classifiers (One-versus-the-rest and
    Multinomial) based on the Orthogonal Distance algorithm.

    The methods and attributes are compatible with the scikit-learn Application
    Programming Interface such that the algorithms here can be used as
    scikit-learn regressor and classifiers.

    The Orthogonal Distance algorithm uses the quasi-chi-squared as the
    cost function instead of the least-square method.

    The quasi-chi-squared" is defined to be the
        [total weighted sum of squares]/dof,
    i.e. same as numpy.sum((residual/sigma_odr)**2)/dof
    or
    numpy.sum(((output.xplus-x)/x_sigma)**2+((y_data-output.y)/y_sigma)**2)/dof.
    It converges to conventional chi-square for zero uncertainties in the
    independent variables x.

    The quasi-chi-squared accounts for uncertainties in both independent
    and dependent variables, i.e. both x and y can be associated with errors.
    This is advantagous when both values come from obervational or experimental
    measurements.

    The classes are essentially wrappers of the scipy scipy.odr classes,
    which in turn are wrapper of the Fortran77 rountines in the ODRpack.
    In contrast to many linear and logistic regression implementation,
    whose solver is based on the gradient search method, ODRpack uses an an
    efficient and stable trust region Levenberg-Marquardt procedure.

    The routine uses a specifically-designed regularization that scales
    the coefficients instead of penalizaing the cost function (here the
    quasi-chi-squared). The parameter C controles the level of regularization
    with a low value of C corresponding to a high regularization and
    a high value of C meaning no and small regularization.

    By choice, the default is quasi no regularization (C=1e4), which can lead
    to overfitting. The user is encouraged to play with the value of C to
    find the right balance between overfitting and underfitting.

    The method is much slower than the other algorithms implemented in
    scikit-learn. In addition the OvR methods should have only up to 100
    features and the Multinomial method should be ony be used for
    dimensions < 10.

    Ref. ODRpack: P. T. Boggs and J. E. Rogers, "Orthogonal
        Distance Regression,"
        in "Statistical analysis of measurement error models and
        applications: proceedings of the AMS-IMS-SIAM joint summer research
        conference held June 10-16, 1989," Contemporary Mathematics,
        vol. 112, pg. 186, 1990.

        P Bevington & D. Keith Robinson Data Reduction and Error
        Analysis for the Physical Sciences
        (for a description of the Levenberg-Marquardt procedure)

    For information on scipy.odr, see
    http://docs.scipy.org/doc/scipy/reference/odr.html

    Author: Wing-Fai Thi <wingfai.thi googlemail. com>

    License: GNU v3.0

    Copyright (C) 2024  Wing-Fai Thi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
    Python : > version 3

    History:
       Version 1.0 (12/3/2018)

    Package requirements:
    ---------------------
    numpy
    scipy

    Parameters
    ----------
    see specific class

    Attributes
    ----------
    see specific regressor and classifiers

    Methods
    -------
    fit(X,y)
        Fit the model using X as training data and y as target values

        Parameters X array of shape(n_samples,n_features)

        Returns None

    predict(X)
        Returns the predicted label for X. the fit method has to
        be run before

        Parameters X array of shape(n_samples,n_features)

        Returns ndarray, shape (n_samples)

    predict_proba(X)
        Returns the predicted class probability for each
        instance in the input X

        Parameters X array of shape(n_samples,n_features)

        Returns ndarray, shape (n_samples,number of classes)

    score((self, X, y, sample_weight=None)
    For linear regression, it returns the coefficient of determination
    R^2 of the prediction. The coefficient R^2 is defined as (1 - u/v),
    where u is the residual sum of squares
    ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
    ((y_true - y_true.mean()) ** 2).sum().
    The best possible score is 1.0 and it can be negative (because
    the model can be arbitrarily worse).

    get_params(self, deep=True)
        Get parameters for this estimator

    set_params(self, **params)
        Set the parameters of this estimator.

    Examples
    --------
    See the provided example files

"""
import numpy as np
from scipy import stats
from scipy.odr import Data, RealData, Model, ODR
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
    # 1. Compute Mixing Index
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

def get_governed_C_bhatta(X, y, E, C_max=30.0, alpha=3.0, C_min=5.0):
    """
    Computes a Mixing Index based on Bhattacharyya Distance with 
    measurement errors. Combines scientific rigor with efficient implementation.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Class labels
    E : array-like, shape (n_samples, n_features)
        Measurement errors (standard deviations)
    C_max : float, default=30.0
        Maximum regularization parameter
    alpha : float, default=3.0
        Governance aggressiveness (2=conservative, 5=aggressive)
    C_min : float, default=5.0
        Minimum C to prevent underfitting
    
    Returns
    -------
    C_opt : float
        Optimized regularization parameter
    mixing_index : float
        Class overlap measure (0=separated, 1=overlapping)
    """
    classes = np.unique(y)
    total_overlap = 0
    pairs = 0

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            # Extract data for class pairs
            idx_a = (y == classes[i])
            idx_b = (y == classes[j])
            
            if np.sum(idx_a) < 2 or np.sum(idx_b) < 2:
                continue  # Skip if insufficient samples
            
            mu_a, mu_b = X[idx_a].mean(axis=0), X[idx_b].mean(axis=0)
            
            # Use E (measurement errors) to define physical variance
            sigma_a_sq = np.mean(E[idx_a]**2, axis=0) + 1e-10  # Floor to prevent div/0
            sigma_b_sq = np.mean(E[idx_b]**2, axis=0) + 1e-10
            
            # Bhattacharyya Distance (element-wise for efficiency)
            avg_sigma_sq = (sigma_a_sq + sigma_b_sq) / 2
            
            # Term 1: Difference in means (weighted by variance)
            t1 = 0.125 * np.sum((mu_a - mu_b)**2 / avg_sigma_sq)
            
            # Term 2: Ratio of variances
            t2 = 0.5 * np.sum(np.log(avg_sigma_sq) - 0.5 * (np.log(sigma_a_sq) + np.log(sigma_b_sq)))
            
            # Bhattacharyya Coefficient
            overlap = np.exp(-(t1 + t2))
            total_overlap += overlap
            pairs += 1

    mixing_index = total_overlap / pairs if pairs > 0 else 1.0
    
    # Governance: Conservative enough to prevent underfitting
    C_opt = C_max * np.exp(-alpha * mixing_index)
    
    return np.clip(C_opt, C_min, C_max), mixing_index

# Option 1: Using quantiles for approximate median (faster for large arrays)
def fast_mad_approximate(error, method='quantile'):
    if len(error) > 10000:  # Only use approximation for very large datasets
        # Use quantile approach - much faster than exact median
        median_approx = np.quantile(error, 0.5)
        mad_approx = np.quantile(np.abs(error - median_approx), 0.5)
        return median_approx, mad_approx
    else:
        # Fall back to exact calculation for smaller datasets
        return np.median(error), np.median(np.abs(error - np.median(error)))

# Option 2: Using scipy's more efficient median implementations
def fast_mad_scipy(error):
    if len(error) > 10000:
        # Use scipy's optimized median
        median_approx = stats.mstats.mquantiles(error, prob=0.5)[0]
        mad_approx = stats.mstats.mquantiles(np.abs(error - median_approx), prob=0.5)[0]
        return median_approx, mad_approx
    else:
        return np.median(error), np.median(np.abs(error - np.median(error)))

# Option 3: Statistical sampling approach for extremely large datasets
def fast_mad_sampling(error, sample_size=10000):
    if len(error) > sample_size:
        # Sample the data for faster computation
        indices = np.random.choice(len(error), sample_size, replace=False)
        sampled_error = error[indices]
        median_sampled = np.median(sampled_error)
        mad_sampled = np.median(np.abs(sampled_error - median_sampled))
        return median_sampled, mad_sampled
    else:
        return np.median(error), np.median(np.abs(error - np.median(error)))

class ODRBase(object):
    def check_Xy(self, X, y):
        X = np.array(X)
        n_samples = X.shape[0]
        y = np.array(y)
        if (n_samples != y.size):
            raise ValueError("Dimensions do not match")
        return self

    def check_XY(self, X, Y):
        X = np.array(X)
        n_samples = X.shape[0]
        Y = np.array(Y)
        if (n_samples != Y.shape[0]):
            print("X shape = ", X.shape)
            print("y size = ", Y.shape[0])
            raise ValueError("Dimensions do not match")
        return self

    def check_XX(self, X1, X2):  # check that two arrays have the same shape
        X1 = np.array(X1)
        X2 = np.array(X2)
        dimX1 = X1.ndim
        dimX2 = X2.ndim
        if (dimX1 > 1):
            X1_dim0, X1_dim1 = X1.shape
        else:
            X1_dim0 = dimX1
            X1_dim1 = 0
        if (dimX2 > 1):
            X2_dim0, X2_dim1 = X2.shape
        else:
            X2_dim0 = dimX2
            X2_dim1 = 0
        if (X1_dim0 != X2_dim0):
            print("X1 shape = ", X1_dim0, X1_dim1)
            print("X2 shape = ", X2_dim0, X2_dim1)
            raise ValueError("Dimensions do not match dim = 0")
        if (X1_dim1 != X2_dim1):
            print("X1 shape = ", X1_dim0, X1_dim1)
            print("X2 shape = ", X2_dim0, X2_dim1)
            raise ValueError("Dimensions do not match dim = 1")
        return self

    def modify(self, X):
        X = np.array(X).astype(float)
        if (X.ndim > 1):
            n_samples, n_features = X.shape
        else:
            n_samples, n_features = X.size, 1
        X = X.T 
        return X, n_samples, n_features

    def _regularize_coeffs(self, p, C, l1=0.0, l2=0.0, alpha_en=0.5):
        """
        Scaling regularization with explicit 
        Elastic Net shrinkage added at the end.
        """
        if (p.ndim > 1):
            alpha = 1. / (1. + np.abs(p).sum(axis=1) / C)
            rcoeffs = np.empty((p.shape))
            for i, a in enumerate(alpha):
                rcoeffs[i, :] = p[i, :] * a
                
            # --- Explicit Elastic Net for Multi-class ---
            if (l1 > 0.0 or l2 > 0.0):
                for i in range(rcoeffs.shape[0]):
                    # Apply to weights only, skip intercept (last column)
                    for j in range(rcoeffs.shape[1] - 1):
                        val = rcoeffs[i, j]
                        if (l2 > 0.0):
                            val = val / (1.0 + (1.0 - alpha_en) * l2)
                        if (l1 > 0.0):
                            l1_threshold = alpha_en * l1
                            if (val > l1_threshold):
                                val = val - l1_threshold
                            elif (val < -l1_threshold):
                                val = val + l1_threshold
                            else:
                                val = 0.0
                        rcoeffs[i, j] = val
        else:
            rcoeffs = p / (1. + abs(p).sum() / C)
            
            # --- Explicit Elastic Net for Binary ---
            if (l1 > 0.0 or l2 > 0.0):
                # Apply to weights only, skip intercept (last element)
                for j in range(rcoeffs.size - 1):
                    val = rcoeffs[j]
                    if (l2 > 0.0):
                        val = val / (1.0 + (1.0 - alpha_en) * l2)
                    if (l1 > 0.0):
                        l1_threshold = alpha_en * l1
                        if (val > l1_threshold):
                            val = val - l1_threshold
                        elif (val < -l1_threshold):
                            val = val + l1_threshold
                        else:
                            val = 0.0
                    rcoeffs[j] = val
        return rcoeffs

    def lin_func(self, p, X, C, l1=0.0, l2=0.0, alpha=0.5):
        n_features = p.size - 1
        p = self._regularize_coeffs(p, C, l1, l2, alpha)
        if (n_features > 1):
            y = (X.T).dot(p[0:n_features]) + p[n_features]
        else:
            a, b = p
            y = a*X + b
        return y.reshape(y.size)

    def multi_lin_func(self, p, X, C, l1=0.0, l2=0.0, alpha=0.5):
        n_features = X.shape[0]
        n_classes = int(p.size / (n_features + 1))
        pm = p.reshape(n_classes, n_features + 1)
        pm = self._regularize_coeffs(pm, C, l1, l2, alpha)
        ylin = np.dot(pm[:, 0:n_features], X)
        for i in range(X.shape[1]):
            ylin[:, i] = ylin[:, i] + pm[:, n_features]
        return ylin

    def logit_func(self, p, X, C, func, l1=0.0, l2=0.0, alpha_en=0.5):
        ylin = self.lin_func(p, X, C, l1, l2, alpha_en)
        if (func == 'sigmoid'):
            return self._sigmoid(ylin)
        elif (func == 'tanh'):
            return self._tanh(ylin)

    def multinomial_func(self, p, X, C, l1=0.0, l2=0.0, alpha=0.5):
        Ylin = self.multi_lin_func(p, X, C, l1, l2, alpha)
        proba = []
        for y in Ylin.T:
            proba.append(self.softmax(y))
        return np.array(proba).T

    def _sigmoid(self, x):
        if (np.any(np.isnan(x))):
            raise ValueError("NaN or Inf found while calling sigmoid")
        z = np.empty(x.size)
        z[x > 30.] = 1.
        z[(x <= 30.) & (x > 20.)] = 1. - np.exp(-x[(x <= 30.) & (x > 20.)])
        z[(x <= 20.) & (x >= 0.)] = 1. / (1. + np.exp(-x[(x <= 20.) & (x >= 0.)]))
        z[x < -30.] = 0.
        z[(x >= -30.) & (x < -20.)] = np.exp(x[(x >= -30.) & (x < -20.)])
        w6 = (x < 0.) & (x >= -20.)
        z6 = np.exp(x[w6])
        z[w6] = z6 / (1 + z6)
        return z

    def _tanh(self, x):
        z = np.empty(x.size)
        z[x > 20.] = 1.
        z[x < -20.] = 0.
        w3 = ((x >= -20.) & (x <= 20.))
        z[w3] = 0.5*(np.tanh(x[w3])+1.)
        return z

    def softmax(self, x):
        # softmax function for a vector
        if (x.ndim > 1):  # use x-m to scale the values down to avoid overflow
            raise ValueError("softmax function works on vectors only")
        m = np.max(x)
        u = np.exp(x-m)
        return u / u.sum()  # u is a vector

#########################################################
class OrthogonalDistanceLinearRegression(ODRBase, object):
    """
    Linear Regression using the Orthogonal Distance methiod.
    Errors in the predcitor and in the target values are possible.
    Without errors, the standatd least square method is used and will
    give the same results than LinearRegression from scikit-learn.

    See the scipy.odr manual for more details.
    
    when $\sigma_x \gg \sigma_y$, the solution becomes highly sensitive to 
    individual point placement (high leverage).

    Parameters
    ----------
    C : float, default 1e5
        level of regularization, a higher value means less regularization.
        A high regularization permits to avoid overfitting.

    maxit : int, default: 100
        Maximum number of iterations taken for the solvers to converge.

    error_type : string, default: 'std' (valid types: 'std' and 'weights')
        Only std (standard deviation) type of error for X and y have been
        tested so far.

    tol : float, default: eps**(1/2)~1e-8, where eps is the machine precision
        for double precision
        Tolerance for stopping criteria.

    robust : bolean, default: False
        iterate up to max_robust_iter times the fit by assigning the fit
        residuals to the noise in y.
        The aim is to be less sensitive to outliers, similar to the
        RANSAC, Theil-Sen, ir Huber regressors

    max_robust_iter : integer, default = 3 (positive values)
        The maximum iteration performed in the context of robust fitting.
        This value and the error_crit plays the same role. max_robust_iter
        is more arbitrary while the error_crit criterion is based on the data.

    error_crit : string, default: 'mean', valid inputs 'mean' 'mode',
        'sum_abs', 'lorentz' and 'std'
        The criterion to stop earlier the robust fit iteration based either on
        an increase of the mean of the residual error or on a decrease of the
        standard deviation (std) of the residual error.
        Current tests suggest that mean should be used whent the fit in
        actually linear and std should be used in fits
        with more complex functions. In any case, the user should experiment
        with bothe the maxiumum number of iterations and the error criterion.
        One should try the mode option as well.
        Ref. Numerical Recipes in Fortran Chapter 15.7

    verbose : int, default: False
        level of screen output

    Example
    -------
    >>> # Code source: Jaques Grobler for the sklearn part
    >>> # License: BSD 3 clause
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.metrics import mean_squared_error, r2_score
    >>> # Load the diabetes dataset
    >>> diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    >>> # Use only one feature
    >>> diabetes_X = diabetes_X[:, np.newaxis, 2]
    >>> # Split the data into training/testing sets
    >>> diabetes_X_train = diabetes_X[:-20]
    >>> diabetes_X_test = diabetes_X[-20:]
    >>> # Split the targets into training/testing sets
    >>> diabetes_y_train = diabetes_y[:-20]
    >>> diabetes_y_test = diabetes_y[-20:]
    >>> # Create linear regression object
    >>> regr = linear_model.LinearRegression()
    >>> # Train the model using the training sets
    >>> regr.fit(diabetes_X_train, diabetes_y_train)
    >>> # Make predictions using the testing set
    >>> diabetes_y_pred = regr.predict(diabetes_X_test)
    >>> # The coefficients
    >>> print("Coefficients: \n", regr.coef_)
    >>> # The mean squared error
    >>> print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test,
    ...       diabetes_y_pred))
    >>> # The coefficient of determination: 1 is perfect prediction
    >>> print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test,
    ...       diabetes_y_pred))
    >>> # Now the ODLR
    >>> from ODLinear import OrthogonalDistanceLinearRegression as ODLR
    >>> regr_odlr = ODLR(error_type='std', max_robust_iter=5, C=1e3, verbose=True)
    >>> regr_odlr.fit(diabetes_X_train.flatten(), diabetes_y_train)   
    >>> X_err = 1e-2
    >>> y_err = 10.
    >>> regr_odlr.fit(diabetes_X_train.flatten(), diabetes_y_train,
    ...               X_err=X_err, y_err=y_err)
    >>> diabetes_y_odlr_pred = regr_odlr.predict(diabetes_X_test)
    >>> # Plot outputs
    >>> plt.scatter(diabetes_X_train, diabetes_y_train, color="blue")
    >>> plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    >>> plt.errorbar(diabetes_X_test.flatten(), diabetes_y_test,
    ...              xerr=X_err, yerr=y_err, fmt='none' )
    >>> plt.plot(diabetes_X_test, diabetes_y_pred, color="blue",
    ...          label='Ordinary Linear Regression', linewidth=3)
    >>> plt.plot(diabetes_X_test, diabetes_y_odlr_pred, color="red",
    ...          label='Orthogonal Distance Linear Regression', linewidth=3)
    >>> plt.show()
    """
    def __init__(self, C=1e5, maxit=100, error_type='std', tol=1e-8,
                 robust=False, max_robust_iter=3, error_crit='mean',
                 verbose=False, l1=0.0, l2=0.0, alpha=0.5):
        """
        If error_type is std then the error are Standard deviations of x. 
        sx are standard deviations of x and are converted to weights by
        dividing 1.0 by their squares.
        
        """
        self.C = C
        self.maxit = maxit
        self.error_type = error_type
        self.tol = tol
        self.robust = robust
        self.max_robust_iter = max_robust_iter
        self.error_crit = error_crit
        self.verbose = verbose
        self.fit_model_ = False
        if (error_type not in ['std', 'weights']):
            raise ValueError('Invalid error_type: choose std or weights')
        if (error_crit not in ['mean', 'std', 'mode', 'sum_abs', 'lorentz']):
            raise ValueError('Invalid robust fit stopping error criterion.'
                             'It has to be mean, std, sum_abs,'
                             'lorentz or mode')
        assert C > 0
        assert maxit > 0
        assert tol > 0
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha # Elastic Net mixing parameter

    def set_params(self, **parameters):
        if not parameters:
            # direct return in no parameter
            return self
        valid_params = self.get_params(deep=True)
        for parameter, value in parameters.items():
            if parameter not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (parameter, self.__class__.__name__))
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        self.params = {'C': self.C,
                       'maxit': self.maxit,
                       'error_type': self.error_type,
                       'tol': self.tol,
                       'robust': self.robust,
                       'max_robust_iter': self.max_robust_iter,
                       'error_crit': self.error_crit,
                       'verbose': self.verbose,
                        'l1': self.l1, 
                        'l2': self.l2,
                        'alpha': self.alpha}
        return self.params

    def fit(self, X, y, X_err=None, y_err=None,
            initial_guess=None, **kwargs):

        y = np.array(y)
        if (y.ndim > 1):
            ODRBase().check_XY(X, y)
        else:
            ODRBase().check_Xy(X, y)
        # transpose the value for X
        X, n_samples, n_features = ODRBase().modify(X)
        if (n_samples < n_features):
            raise ValueError('n_samples < n_features')
        self.n_samples = n_samples
        self.n_features = n_features

        # guess the starting coefficients to one
        if (initial_guess is None):
            initial_guess = np.ones(n_features+1)

        if (X_err is not None):
            if isinstance(X_err, (int, float)):
                if X_err < 1e-16:
                    X_err = 1e-16
                if n_features == 1:
                    X_err = np.repeat(X_err, n_samples)
                else:
                    X_err = np.tile(X_err, (n_samples, n_features))
            else:
                w = X_err < 1e-16
                X_err[w] = 1e-16

            if (self.error_type == 'std'):
                X_err = X_err.T              # transpose X_err like X
                # check that X and X_err have the same shape
                ODRBase().check_XX(X, X_err)

        if (y_err is not None):
            if isinstance(y_err, (int, float)):
                y_err = np.repeat(y_err, n_samples)
            if (y.ndim > 1):
                ODRBase().check_XX(y, y_err)
            else:
                ODRBase().check_Xy(X.T, y_err)
            y_err0 = np.copy(y_err)
        else:
            y_err0 = np.zeros(y.shape)

        if ((X_err is not None) or (y_err is not None) or (self.robust)):
            self.fit_type_ = 0  # explicit ODR
        else:
            self.fit_type_ = 2  # no error: use least-square
        if (self.verbose is True):
            print("solver type:", self.fit_type_)
        self.epsilon_ = np.zeros(y.shape)
        error = 0.
        self.fit_model_ = True
        mean_error = 1e9
        mode_error = 1e9
        std_error = 1e9
        error_sum = 1e9
        lorentz_error = 1e9
        for _ in range(self.max_robust_iter):
            # The data, with weightings as actual standard deviations
            if (self.error_type == 'std'):
                """
                The data, with weightings as actual standard deviations
                and/or covariances.
                sx and sy are converted to weights by dividing 1.0 by their
                squares. For example, wd = 1./numpy.power(`sx`, 2).
                """
                data = RealData(X, y, sx=X_err, sy=y_err)
            else:
                """
                The we argument weights the effect a deviation in the response
                variable has on the fit. The `wd` argument weights the effect a
                deviation in the input variable has on the fit. To handle
                multidimensional inputs and responses easily, the structure of
                these arguments has the n'th dimensional axis first.
                """
                data = Data(X, y, wd=X_err, we=y_err)
            # linear function
            model = Model(ODRBase().lin_func, extra_args=[self.C, self.l1, self.l2, self.alpha])
            odr = ODR(data, model, beta0=initial_guess,
                      maxit=self.maxit, sstol=self.tol, **kwargs)
            odr.set_job(fit_type=self.fit_type_)
            out = odr.run()
            if (self.verbose):
                out.pprint()
            error = abs(out.eps)

            # --- Convergence Check ---
            if (self.error_crit == 'mean'):
                if (np.mean(error) > mean_error):
                    break  # exit without updating the model
            if (self.error_crit == 'std'):
                if (np.std(error) < std_error):
                    break
            if (self.error_crit == 'mode'):
                if (stats.mode(error)[0] > mode_error):
                    break
            if (self.error_crit == 'sum_abs'):
                if (error.sum() > error_sum):
                    break
            if (self.error_crit == 'lorentz'):
                if (np.log(1+0.5*error**2).sum() > lorentz_error):
                    break

            w = self._regularize_coeffs(out.beta, self.C)
            self.beta = out.beta
            self.coef_ = w[0:n_features]
            self.intercept_ = w[n_features]
            self.uncertainty_ = out.sd_beta
            # statistics on the fit to X
            self.quasi_chisq_ = out.res_var   # Residual variance.
            self.delta = out.delta.T   # estimated x-component of the residuals
            self.epsilon = out.eps     # estimated y-component of the residuals
            self.Xplus = out.xplus
            self.sum_square = out.sum_square
            self.sum_square_delta = out.sum_square_delta
            self.sum_square_eps = out.sum_square_eps

            if (self.robust is False):
                break  # run only once if self.robust = False
            
            # --- Update Metrics for Robust Loop ---
            mean_error = np.mean(error)  # better for pure linear model
            std_error = np.std(error)    # better for polynomial model
            # mode (most frequent value in a distribution)
            mode_error = stats.mode(error)[0]
            error_sum = error.sum()
            lorentz_error = np.log(1+0.5*error**2).sum()
            
            # --- MAD SUPERIOR METHOD ---
            # Calculate Median Absolute Deviation of residuals
            median_eps = np.median(error)
            mad_eps = np.median(np.abs(error - median_eps))
            # 1.4826 scale makes MAD consistent with the standard deviation of a normal distribution
            scale = 1.4826 * mad_eps
            
            # Feedback: Update y_err using the scaled residual
            # This prevents the uncertainty from blowing up to 10^10 by 
            # normalizing the residual "punishment"
            # add the scaled y-residual to the initial error in y
            y_err = y_err0 + (error / (scale + 1e-16)) 
            initial_guess = out.beta
        self.out = out
        self.final_y_err_ = y_err  # Save the robust weights for diagnostic use
        y_err = y_err0             # Reset local variable
        return self

    def predict(self, X):
        """
        Return the precicted values

        Parameters:
        -----------

        X: array of shape (n_samples, n_features)
            input independent variables

        Return:
        -------
        array of shape (n_samples,)
        The predicted values
        """
        X, _, n_features = ODRBase().modify(X)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        return ODRBase().lin_func(self.beta, X, self.C, l1=self.l1, l2=self.l2, alpha=self.alpha)

    def predict_MC_error(self, X, X_err,
                         Number_of_MC_iterations=1000):
        """
        Return the error on the probability estimates using
        a Monte-Carlo method given the error X_err
        """
        X, _, n_features = ODRBase().modify(X)
        X_err = X_err.T
        ODRBase().check_XX(X, X_err)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")
        pred_MC = []
        for i in range(Number_of_MC_iterations):
            X_MC = np.random.normal(X, X_err)
            pred_MC.append(ODRBase().lin_func(self.beta, X_MC, self.C, l1=self.l1, l2=self.l2, alpha=self.alpha).T)
        pred_MC = np.array(pred_MC)
        std_pred = np.std(pred_MC, axis=0)
        return std_pred

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determinton R2
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        ODRBase().check_Xy(X, y)
        if (sample_weight is not None):
            sample_weight = np.array(sample_weight)
            ODRBase().check_Xy(X, sample_weight)
            sample_weight = sample_weight/np.sum(sample_weight)*len(y)
            u = sum(sample_weight*(self.predict(X) - y)**2)
        else:
            u = sum((self.predict(X) - y)**2)
        return 1-u/sum((y - np.mean(y))**2)

#########################################################
class OrthogonalDistanceLogisticRegression(ODRBase, object):
    def __init__(self, C=1e5, maxit=100, error_type='std', tol=1e-8, 
                 func='sigmoid', verbose=False, l1=0.0, l2=0.0, alpha=0.5,
                 robust=False, max_robust_iter=10, error_crit='lorentz'):
        self.C = C
        self.maxit = maxit
        self.error_type = error_type
        self.tol = tol
        self.verbose = verbose
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha 
        self.fit_model_ = False
        self.func = func
        # Robust parameters
        self.robust = robust
        self.max_robust_iter = max_robust_iter
        self.error_crit = error_crit

    def fit(self, X, y, X_err=None, y_err=None, initial_guess=None, **kwargs):
        ODRBase().check_Xy(X, y)
        X_trans, n_samples, n_features = ODRBase().modify(X)
        self.n_samples, self.n_features = n_samples, n_features
        self.classes_ = np.unique(y).astype(int)
        
        y_copy = np.copy(y)
        wmin = (y == min(y))
        wmax = (y == max(y))
        y_copy[wmin] = 0
        y_copy[wmax] = 1
        
        if (initial_guess is None):
            initial_guess = np.zeros(n_features+1)
        
        # Consistent X_err handling to match your tested original code
        sx = X_err.T if X_err is not None else None
        
        # Baseline Y error initialization
        if y_err is not None:
            y_err_orig = np.array(y_err)
        elif self.robust:
            y_err_orig = np.ones(y_copy.shape)
        else:
            y_err_orig = None

        curr_y_err = np.copy(y_err_orig) if y_err_orig is not None else None

        # Set fit type (0=ODR, 2=OLS)
        self.fit_type_ = 0 if (X_err is not None or self.robust) else 2

        best_crit_val = 1e9
        beta_guess = initial_guess

        for i in range(self.max_robust_iter):
            # Pass errors exactly like the original working version to avoid corruption
            if self.error_type == 'std':
                data = RealData(X_trans, y_copy, sx=sx, sy=curr_y_err)
            else:
                data = Data(X_trans, y_copy, wd=sx, we=curr_y_err)

            model = Model(self.logit_func, extra_args=[self.C, self.func, self.l1, self.l2, self.alpha])
            odr = ODR(data, model, beta0=beta_guess, maxit=self.maxit, sstol=self.tol, **kwargs)
            odr.set_job(fit_type=self.fit_type_)
            out = odr.run()

            error = np.abs(out.eps)
    
            # --- Robust Criterion Assessment ---
            median_eps = np.median(error)
            mad_eps = np.median(np.abs(error - median_eps))
            
            crit_map = {
                'mean': np.mean(error),
                'median': median_eps,
                'mad': mad_eps,
                'std': np.std(error),
                'sum_abs': np.sum(error),
                'lorentz': np.sum(np.log(1 + 0.5 * error**2))
            }
            current_crit_val = crit_map.get(self.error_crit, np.mean(error))

            if i > 0 and current_crit_val > best_crit_val:
                break
            best_crit_val = current_crit_val

            # Save Current Iteration Results
            w = self._regularize_coeffs(out.beta, self.C, self.l1, self.l2, self.alpha)
            self.beta = out.beta
            self.coef_ = w[0:n_features]
            self.intercept_ = w[n_features]
            self.uncertainty_ = out.sd_beta
            self.quasi_chisq_ = out.res_var
            self.out_ = out
            
            if not self.robust:
                break

            # --- DUAL-STRATEGY ROBUST LOGIC ---
            # --- DAMPED RANK-BASED PENALTY (Small Data) ---
            if n_samples <= 100:
                max_err = np.max(error)
                if max_err > 0:
                    # We cap the inflation so it can NEVER more than double 
                    # the original error. This prevents the "Cliff" effect.
                    relative_error = error / max_err
                    damping = 0.5  # Only apply 50% of the robust correction
                    inflation = error * relative_error * damping
                    
                    base = y_err_orig if y_err_orig is not None else np.ones(y_copy.shape)
                    curr_y_err = base + inflation
                else:
                    curr_y_err = y_err_orig
            else:
                # 3-SIGMA CLIPPING (Large Data: N > 100)
                # Standard thresholding for datasets where distributions are meaningful.
                threshold = 3.0 * (1.4826 * mad_eps)
                if threshold > 0:
                    # Clip only those beyond the 3-sigma equivalent
                    inflation = np.where(error > threshold, error, 0)
                    base = y_err_orig if y_err_orig is not None else np.ones(y_copy.shape)
                    curr_y_err = base + inflation
                else:
                    curr_y_err = y_err_orig
                
            beta_guess = out.beta

        self.fit_model_ = True
        return self

    def fit_(self, X, y, X_err=None, initial_guess=None, **kwargs):
        ODRBase().check_Xy(X, y)
        X_trans, n_samples, n_features = ODRBase().modify(X)
        self.n_samples, self.n_features = n_samples, n_features
        self.classes_ = np.unique(y).astype(int)
        
        y_copy = np.copy(y)
        wmin = (y == min(y))
        wmax = (y == max(y))
        y_copy[wmin] = 0
        y_copy[wmax] = 1
        
        if (initial_guess is None):
            initial_guess = np.zeros(n_features+1)
        
        sx = X_err.T if X_err is not None else None
        self.fit_type_ = 0 if X_err is not None else 2

        data = RealData(X_trans, y_copy, sx=sx) if self.error_type == 'std' else Data(X_trans, y_copy, wd=sx)
        
        # Explicitly passing all parameters to the model
        model = Model(self.logit_func, extra_args=[self.C, self.func, self.l1, self.l2, self.alpha])
        
        odr = ODR(data, model, beta0=initial_guess, maxit=self.maxit, sstol=self.tol, **kwargs)
        odr.set_job(fit_type=self.fit_type_)
        out = odr.run()

        # Regularize result using explicit Elastic Net settings
        w = self._regularize_coeffs(out.beta, self.C, self.l1, self.l2, self.alpha)
        self.beta = out.beta
        self.coef_ = w[0:n_features]
        self.intercept_ = w[n_features]
        self.uncertainty_ = out.sd_beta
        self.quasi_chisq_ = out.res_var
        self.fit_model_ = True
        self.final_y_err_ = y_err
        return self

    def predict_proba(self, X):
        X_trans, _, _ = ODRBase().modify(X)
        res = self.logit_func(self.beta, X_trans, self.C, self.func, self.l1, self.l2, self.alpha).T
        return np.column_stack((1. - res, res))

    def predict_proba_MC_error(self, X, X_err, Number_of_MC_iterations=1000):
        X_trans, _, _ = ODRBase().modify(X)
        if isinstance(X_err, (int, float)):
            X_err_trans = np.full(X_trans.shape, X_err)
        else:
            X_err_trans = X_err.T
        proba_MC = []
        for i in range(Number_of_MC_iterations):
            X_MC = np.random.normal(X_trans, X_err_trans)
            proba_MC.append(self.logit_func(self.beta, X_MC, self.C, self.func, self.l1, self.l2, self.alpha).T)
        proba_array = np.array(proba_MC)
        return np.mean(proba_array, axis=0), np.std(proba_array, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return self.classes_[(np.floor(proba + 0.5)).astype(int)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def logloss_score(self, y, p):
        y, p = np.array(y), np.array(p)
        logloss, eps = 0., 1e-15
        for i in range(y.size):
            a = max([p[i], eps])
            b = max([1.-p[i], eps])
            logloss += y[i]*np.log(a) + (1.-y[i])*np.log(b)
        return -logloss/float(y.size)

    def predict_proba_sigma_check(self, X, X_err):
        """
        Evaluates probability at the mean and +/- 1 sigma points.
        Returns the average probability across these 3 points.
        """
        X_trans, _, _ = ODRBase().modify(X)
    
        if isinstance(X_err, (int, float)):
            X_err_trans = np.full(X_trans.shape, X_err)
        else:
            X_err_trans = X_err.T

        # 1. Evaluate at the center (Standard)
        p_center = self.logit_func(self.beta, X_trans, self.C, self.func, self.l1, self.l2, self.alpha).T
    
        # 2. Evaluate at +1 Sigma
        p_plus = self.logit_func(self.beta, X_trans + X_err_trans, self.C, self.func, self.l1, self.l2, self.alpha).T
    
        # 3. Evaluate at -1 Sigma
        p_minus = self.logit_func(self.beta, X_trans - X_err_trans, self.C, self.func, self.l1, self.l2, self.alpha).T
    
        # Average probability: A 3-point approximation of the integral over the error
        p_avg = (p_center + p_plus + p_minus) / 3.0
    
        return np.column_stack((1. - p_avg, p_avg))

    def predict_robust(self, X, X_err, threshold=0.5):
        """
        Uses the sigma-averaged probability for a more 'stable' classification.
        """
        proba = self.predict_proba_sigma_check(X, X_err)[:, 1]
        return self.classes_[(proba >= threshold).astype(int)]

#########################################################
class OrthogonalDistanceLogisticRegressionOVR(ODRBase):
    def __init__(self, C=1e5, maxit=100, error_type='std',
                 probability='softmax', 
                 tol=1e-8, func='sigmoid', verbose=False,
                 l1=0.0, l2=0.0, alpha=0.5,
                 robust=False, max_robust_iter=10, error_crit='mad'):
        self.C = C
        self.maxit = maxit
        self.error_type = error_type
        self.probability = probability
        self.tol = tol
        self.func = func
        self.verbose = verbose
        # Elastic Net parameters to be passed to each binary classifier
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.fit_model_ = False
        # Robust parameters
        self.robust = robust
        self.max_robust_iter = max_robust_iter
        self.error_crit = error_crit
        self.fit_model_ = False

    def fit(self, X, y, X_err=None, y_err=None, initial_guess=None, correct_imbalance=False,
            **kwargs):
        ODRBase().check_Xy(X, y)
        X_trans, n_samples, n_features = ODRBase().modify(X)
        self.classes_ = np.unique(y).astype(int)
        self.n_classes_ = self.classes_.size
        self.n_samples, self.n_features = n_samples, n_features
        
        if (self.verbose):
            print(f"Fitting OVR (n_samples: {n_samples}, n_classes: {self.n_classes_})")

        # Protect original X_err
        sx = X_err.T if (X_err is not None and self.error_type == 'std') else X_err
        self.fit_type_ = 0 if (X_err is not None or self.robust) else 2
        
        if (initial_guess is None):
            initial_guess = np.zeros((self.n_classes_, n_features + 1))

        w, quasi_chisq, uncertainty, delta, eps = [], [], [], [], []

        for i in self.classes_:
            if (self.verbose):
                print(f"Binary Loop - Class: {i}")
            
            y_bin = np.where(y == i, 1, 0)
            
            # --- FIX: Ensure y_err_orig is NEVER None ---
            if y_err is not None:
                y_err_orig = np.array(y_err)
            else:
                # Default to ones even if robust=False, 
                # because we still need a numerical base for weight_pos math
                y_err_orig = np.ones(n_samples)

            if (correct_imbalance):
                # Calculate weights to balance the classes
                count_pos = np.sum(y_bin == 1)
                count_neg = np.sum(y_bin == 0)
            
                # Add a small epsilon to avoid division by zero
                weight_pos = count_neg / (count_pos + 1e-10) 
            
                # Cap the weight to prevent numerical explosion
                weight_pos = np.clip(weight_pos, 1.0, 100.0)

                # Apply balancing: Minority class (1) gets smaller error -> higher weight
                # This will now work because y_err_orig is guaranteed to be an array
                y_err_balanced = np.where(y_bin == 1, y_err_orig / weight_pos, y_err_orig)
            
            else:
                y_err_balanced = np.copy(y_err_orig)
                
            curr_y_err = np.copy(y_err_balanced)
            
            beta_guess = initial_guess[i, :]
            best_crit_val = 1e9
            out_final = None

            # --- Robust Iteration Loop per Class ---
            for r_iter in range(self.max_robust_iter if self.robust else 1):
                if (self.error_type == 'std'):
                    data = RealData(X_trans, y_bin, sx=sx, sy=curr_y_err)
                else:
                    data = Data(X_trans, y_bin, wd=sx, we=curr_y_err)
                
                model = Model(self.logit_func, extra_args=[self.C, self.func, self.l1, self.l2, self.alpha])
                odr = ODR(data, model, beta0=beta_guess, maxit=self.maxit, sstol=self.tol, **kwargs)
                odr.set_job(fit_type=self.fit_type_)
                out = odr.run()

                error = np.abs(out.eps)
   
                # Robust Criterion Selection
                median_eps = np.median(error)
                mad_eps = np.median(np.abs(error - median_eps))
                crit_map = {'mean': np.mean(error), 'median': median_eps, 'mad': mad_eps, 'std': np.std(error)}
                current_crit_val = crit_map.get(self.error_crit, np.mean(error))

                if r_iter > 0 and current_crit_val > best_crit_val:
                    break
              
                best_crit_val = current_crit_val
                out_final = out
                beta_guess = out.beta

                if not self.robust:
                    break

                # --- Dual-Strategy Robust Update ---
                if n_samples <= 100:
                    # Linear Rank-Based (Stable for small N)
                    max_err = np.max(error)
                    inflation = error * (error / max_err) if max_err > 0 else 0
                    curr_y_err = y_err_balanced + inflation
                else:
                    # 3-Sigma Clipping (Statistical for large N)
                    threshold = 3.0 * (1.4826 * mad_eps)
                    inflation = np.where(error > threshold, error, 0) if threshold > 0 else 0
                    curr_y_err = y_err_balanced + inflation

            # Store results for this class
            w.append(out_final.beta)
            quasi_chisq.append(out_final.res_var)
            uncertainty.append(out_final.sd_beta)
            delta.append(out_final.delta.T)
            eps.append(out_final.eps)

        self.beta = w
        w_reg = self._regularize_coeffs(np.array(w), self.C, self.l1, self.l2, self.alpha)
        self.coef_ = w_reg[:, 0:n_features]
        self.intercept_ = w_reg[:, n_features]
        self.quasi_chisq_ = quasi_chisq
        self.uncertainty_ = uncertainty
        self.delta_ = delta
        self.epsilon_ = eps
        self.fit_model_ = True
        self.final_y_err_ = y_err
        return self

    def predict_proba(self, X):
        """
        Predict probabilities by aggregating outputs from all OVR classifiers.
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        X, n_samples, n_features = ODRBase().modify(X)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")
        if (self.probability == "softmax"):
            ylin = []
            for w in self.beta:
                ylin.append(ODRBase().lin_func(w, X, self.C))
            ylin = np.array(ylin)
            probabilities = []
            for i in range(n_samples):
                probabilities.append(ODRBase().softmax(ylin[:, i]))
            probabilities = np.array(probabilities)
        elif (self.probability == "ovr"):
            proba_logit = []
            for w in self.beta:
                proba_logit.append(ODRBase().logit_func(w, X,
                                                        self.C, self.func, self.l1, self.l2, self.alpha))
            proba_logit = np.array(proba_logit)
            normalization = np.sum(proba_logit, axis=0)
            probabilities = (proba_logit / (normalization + 1e-16)).T
        else:
            raise ValueError("Wrong probabilty type. Use softmax or ovr")
        return probabilities

    def predict_proba_MC_error(self, X, X_err, Number_of_MC_iterations=1000):
        proba_MC = []
        for i in range(Number_of_MC_iterations):
            X_MC = np.random.normal(X, X_err)
            proba_MC.append(self.predict_proba(X_MC))
        proba_MC = np.array(proba_MC)
        mean_proba = np.mean(proba_MC, axis=0)
        std_proba = np.std(proba_MC, axis=0)
        
        return mean_proba, std_proba

    def predict(self, X):
        """
        Return the class with the highest probability.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]

    def predict_proba_sigma_check(self, X, X_err):
        """
        Multiclass (OVR) evaluation at mean and +/- 1, 2, and 3 sigma points.
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")

        # 1. Transform X: returns (n_features, n_samples)
        X_mid, n_samples, n_features = ODRBase().modify(X)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")

        # 2. Correct the broadcasting: E_trans must be (n_features, n_samples)
        if isinstance(X_err, (int, float)):
            E_trans = np.full(X_mid.shape, X_err)
        else:
            # Check orientation and force it to (14, 3000)
            if X_err.shape == (n_samples, n_features):
                E_trans = X_err.T
            else:
                E_trans = X_err

        # 3. Define evaluation points - shapes are now consistent (14, 3000)
        points = [X_mid, X_mid + E_trans, X_mid - E_trans, 
                  X_mid + 2. * E_trans, X_mid - 2. * E_trans,
                  X_mid + 3. * E_trans, X_mid - 3. * E_trans]
        
        all_p = []
        for pt in points:
            if self.probability == "softmax":
                ylin = []
                for w in self.beta:
                    ylin.append(ODRBase().lin_func(w, pt, self.C))
                ylin = np.array(ylin)
                
                pt_probs = []
                for i in range(n_samples):
                    pt_probs.append(ODRBase().softmax(ylin[:, i]))
                all_p.append(np.array(pt_probs))

            elif self.probability == "ovr":
                proba_logit = []
                for w in self.beta:
                    proba_logit.append(ODRBase().logit_func(w, pt,
                                        self.C, self.func, self.l1, self.l2, self.alpha))
                proba_logit = np.array(proba_logit)
                
                normalization = np.sum(proba_logit, axis=0)
                # Ensure result is (n_samples, n_classes)
                pt_probs = (proba_logit / (normalization + 1e-16)).T
                all_p.append(pt_probs)

#        Convert to array for statistical calculations: (7, n_samples, n_classes)
        all_p_array = np.array(all_p)
        
        # Calculate mean across the 7-point distribution
        mean_probabilities = np.mean(all_p_array, axis=0)
        
        # Calculate standard deviation to quantify classification sensitivity
        std_probabilities = np.std(all_p_array, axis=0)
        
        return mean_probabilities, std_probabilities

    def predict_robust(self, X, X_err):
        """
        Multiclass robust prediction using the sigma-averaged probability.
        Identifies the class with the highest average probability 
        across the error distribution.
        """
        # 1. Get the (n_samples, n_classes) averaged probability matrix
        # from your new OVR sigma-check routine.
        proba, _ = self.predict_proba_sigma_check(X, X_err)
        
        # 2. Select the index of the highest probability for each sample
        class_indices = np.argmax(proba, axis=1)
        
        # 3. Return the actual class labels (e.g., [3, 6])
        return self.classes_[class_indices]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The score is the linear predictor (X * beta + intercept).
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        
        # Format X to handle the internal intercept logic
        X_trans, n_samples, n_features = ODRBase().modify(X)
        
        if self.n_features != n_features:
            raise ValueError(f"Feature mismatch: model has {self.n_features}, input has {n_features}")

        # Compute X * beta + intercept for each class
        scores = []
        for w in self.beta:
            # We use the raw linear function result before any probability transformation
            scores.append(ODRBase().lin_func(w, X_trans, self.C, self.l1, self.l2, self.alpha))
        
        # Return as (n_samples, n_classes)
        return np.array(scores).T

    def calculate_entropy(self, X, X_err=None):
        """
        Calculates Shannon entropy for predictions.
        If X_err is provided, it uses the robust sigma-averaged probabilities.
        """
        if X_err is not None:
            # Use your unique sigma-check mean probabilities
            probs, _ = self.predict_proba_sigma_check(X, X_err)
        else:
            probs = self.predict_proba(X)

        # H = -sum(p * log(p))
        # Adding a tiny epsilon to avoid log(0)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        return entropy

    def predict_proba_with_model_uncertainty(self, X, n_draws=100):
        """
        Estimates uncertainty by sampling from the distribution of weights (beta).
        """
        if not hasattr(self, 'uncertainty_') or self.uncertainty_ is None:
            raise ValueError("Model covariance (sd_beta) not found. Fit the model first.")

        # Original weights
        beta_mean = self.beta
        # standard deviations of weights from ODR
        beta_std = self.uncertainty_ 

        all_draw_probs = []
        
        for _ in range(n_draws):
            # 1. Build an alternative model by perturbing beta
            # We sample each beta weight from its own normal distribution
            beta_perturbed = np.random.normal(beta_mean, beta_std)
            
            # 2. Run inference with this alternative model
            # Note: We use the internal multinomial_func directly
            probs = ODRBase().multinomial_func(beta_perturbed, X.T, self.C,
                                               l1=self.l1, l2=self.l2, alpha=self.alpha).T
            all_draw_probs.append(probs)

        # 3. Compute the mean and std across the alternative models
        all_draw_probs = np.array(all_draw_probs)
        model_mean_proba = np.mean(all_draw_probs, axis=0)
        model_std_err = np.std(all_draw_probs, axis=0)
        
        return model_mean_proba, model_std_err

#########################################################
class OrthogonalDistanceMultinomialLogisticRegression(ODRBase, object):
    def __init__(self, C=1e5, maxit=100, error_type='std',
                 tol=1e-8, verbose=False, l1=0.0, l2=0.0, alpha=0.5,
                 robust=False, max_robust_iter=10, error_crit='mad'):
        self.C = C
        self.maxit = maxit
        self.error_type = error_type
        self.tol = tol
        self.fit_model_ = False
        self.verbose = verbose
        if (error_type not in ['std', 'weights']):
            raise ValueError('Invalid error_type: choose std or weights,'
                             ' default is std')
        assert C > 0
        assert maxit > 0
        assert tol > 0
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        # Minimal additions for robust mode
        self.robust = robust
        self.max_robust_iter = max_robust_iter
        self.error_crit = error_crit

    def set_params(self, **parameters):
        """
        One can set the parameters after the initialization with this method.
        Standard scikit-learn method
        """
        if not parameters:
            # direct return in no parameter
            return self
        valid_params = self.get_params(deep=True)
        for parameter, value in parameters.items():
            if parameter not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                            'Check the list of available parameters '
                            'with `estimator.get_params().keys()`.' %
                            (parameter, self.__class__.__name__))
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """
            Return the parameters. Standard scikit-learn method
        """
        self.params = {'C': self.C,
                       'maxit': self.maxit,
                       'error_type': self.error_type,
                       'tol': self.tol,
                       'verbose': self.verbose,
                       'l1': self.l1, 'l2': self.l2, 'alpha': self.alpha}
        return self.params

    def _hot_encoding(self, y):
        """
        Create the Y (n_samples ,n_classes) array using the hot-encoding method
        """
        Y = np.zeros((self.n_samples, self.n_classes_))
        for i in range(self.n_samples):
            ind = np.argmax(self.classes_ == y[i].astype(int))
            Y[i, ind] = 1.
        return Y.T

    def fit(self, X, y, X_err=None, y_err=None, initial_guess=None, correct_imbalance=False, **kwargs):
        """
        Fit the model according to the given training data.
        """
        ODRBase().check_XY(X, y)
        X, n_samples, n_features = ODRBase().modify(X)
        self.classes_ = np.unique(y).astype(int)
        self.n_classes_ = self.classes_.size
        n_classes = self.n_classes_
        if (self.verbose):
            print("Number of classes:", n_classes)
        if (n_samples < (n_features + 1) * n_classes):
            print("Training set size:", n_samples)
            print("Number of parameters:", (n_features + 1) * n_classes)
            raise ValueError('n_samples < (n_features + 1) * n_classe')
        if (self.verbose):
            print("Fitting (n_samples, n_features):", n_samples, n_features)
        self.n_samples = n_samples
        self.n_features = n_features
        # get a hot-encoding matrix from the target array
        Y = self._hot_encoding(y)
        if (initial_guess is None):
            initial_guess = np.zeros(self.n_classes_ * (n_features + 1))
        else:
            initial_guess = initial_guess.reshape(self.n_classes_ *
                                                  (n_features + 1))
            if (self.verbose):
                print("initial_guess:", initial_guess)

        # Protect original X_err
        if (X_err is not None):
            self.fit_type_ = 0  # explicit ODR + central finite differences
            if (self.error_type == 'std'):
                X_err_in = X_err.T
                ODRBase().check_XX(X, X_err_in)
            else: # no error: use least-square + central finite differences
                X_err_in = X_err
        else:
            self.fit_type_ = 0 if self.robust else 2
            X_err_in = None

        # Y Error Initialization for robust mode
        y_err_orig = np.array(y_err) if y_err is not None else (np.ones(Y.shape) if self.robust else None)
        curr_y_err = np.copy(y_err_orig) if y_err_orig is not None else None

        # --- CLASS IMBALANCE CORRECTION ---
        if correct_imbalance and y_err_orig is not None:
            # For multinomial, we need to handle class balancing differently
            # Since each sample belongs to exactly one class, we can't directly 
            # apply the OVR approach. Instead, we'll balance the error contributions
            # by adjusting the error weights based on class frequencies
            
            # Count samples per class
            class_counts = np.bincount(y.astype(int))
            total_samples = len(y)
            
            # Calculate inverse frequency weights for each sample
            # This ensures minority classes get more "attention" in fitting
            sample_weights = np.array([1.0 / (class_counts[label] / total_samples + 1e-10) 
                                     for label in y])
            
            # Apply weights to y_err_orig (this is a bit tricky for multinomial)
            # For multinomial, we need to think about this differently
            # We'll scale the error by the inverse of class frequency
            # But since y_err_orig is (n_samples, n_classes), we need to be careful
            
            # Simple approach: scale each row by inverse class frequency
            # This is a reasonable approximation for multinomial
            if y_err_orig is not None:
                # Normalize sample weights to be reasonable
                normalized_weights = sample_weights / np.mean(sample_weights)
                # Apply to y_err_orig - scale each sample's error by its weight
                # For multinomial, we apply the same scaling to all classes for each sample
                curr_y_err = y_err_orig * normalized_weights[:, np.newaxis]

        beta_guess = initial_guess
        best_crit_val = 1e9
        out = None

        # Robust Iteration Loop
        for r_iter in range(self.max_robust_iter if self.robust else 1):
            if (self.error_type == 'std'):
                data = RealData(X, Y, sx=X_err_in, sy=curr_y_err)
            else:
                data = Data(X, Y, wd=X_err_in, we=curr_y_err)
            
            if (self.verbose):
                print("Number of parameters:", beta_guess.size)
                
            model = Model(ODRBase().multinomial_func, extra_args=[self.C, self.l1, self.l2, self.alpha])
            odr = ODR(data, model, beta0=beta_guess,
                      maxit=self.maxit, sstol=self.tol, **kwargs)
            odr.set_job(fit_type=self.fit_type_)
            out = odr.run()

            if not self.robust:
                break

            # Robustness evaluation
            error = np.abs(out.eps)
            median_eps = np.median(error)
            mad_eps = np.median(np.abs(error - median_eps))
            crit_map = {'mean': np.mean(error), 'median': median_eps, 'mad': mad_eps, 'std': np.std(error)}
            current_crit_val = crit_map.get(self.error_crit, np.mean(error))

            if r_iter > 0 and current_crit_val > best_crit_val:
                break
            
            best_crit_val = current_crit_val
            beta_guess = out.beta

            # Dual-Strategy Update
            if n_samples <= 100:
                max_err = np.max(error)
                inflation = error * (error / max_err) if max_err > 0 else 0
                curr_y_err = y_err_orig + inflation
            else:
                threshold = 3.0 * (1.4826 * mad_eps)
                inflation = np.where(error > threshold, error, 0) if threshold > 0 else 0
                curr_y_err = y_err_orig + inflation

        if (self.verbose):
            out.pprint()
            
        self.beta = out.beta
        w = self.beta.reshape(n_classes, n_features+1)
        w = self._regularize_coeffs(w, self.C, self.l1, self.l2, self.alpha)
        self.coef_ = w[:, 0:n_features]
        self.intercept_ = w[:, n_features]
        self.quasi_chisq_ = out.res_var
        self.uncertainty_ = out.sd_beta
        self.delta_ = out.delta.T  # estimated x-component of the residuals
        self.epsilon_ = out.eps    # estimated y-component of the residuals
        self.Xplus_ = out.xplus
        self.sum_square = out.sum_square
        self.sum_square_delta = out.sum_square_delta
        self.sum_square_eps = out.sum_square_eps
        self.fit_model_ = True
        self.final_y_err_ = y_err
        return self

    def predict(self, X):
        """
        Predict the class as the class with whith the highest probability
        """
        class_index = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[class_index]

    def predict_proba(self, X):
        """
        Returns the probability for each classe
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        X, _, n_features = ODRBase().modify(X)
        if (self.n_features != n_features):
            print("X_train has:", self.n_features, " features")
            print("X has:", n_features, " features")
            raise ValueError("Numbers of features do not match")
        proba = ODRBase().multinomial_func(self.beta, X, self.C, l1=self.l1, l2=self.l2, alpha=self.alpha).T
        return proba

    def score(self, X, y, sample_weight=None):
        """
        Returns the accuracy score: sum(predict = true) / len(true)
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        ODRBase().check_Xy(X, y)
        if (sample_weight is not None):
            sample_weight = np.array(sample_weight)
            ODRBase().check_Xy(X, sample_weight)
            sample_weight = sample_weight / np.sum(sample_weight) * len(y)
            return sum(sample_weight * (self.predict(X) == y)) * 1. / len(y)
        else:
            return sum(self.predict(X) == y)*1. / len(y)

    def predict_proba_MC_error(self, X, X_err, Number_of_MC_iterations=1000):
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        ODRBase().check_XX(X, X_err)
        proba_MC = []
        for _ in range(Number_of_MC_iterations):
            X_MC = np.random.normal(X, X_err)
            proba_MC.append(self.predict_proba(X_MC))
        proba_MC = np.array(proba_MC)        
        mean_proba = np.mean(proba_MC, axis=0)
        std_proba = np.std(proba_MC, axis=0)
        
        return mean_proba, std_proba

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The score is the linear predictor (X * beta + intercept).
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        
        # Format X to handle the internal intercept logic
        X_trans, n_samples, n_features = ODRBase().modify(X)
        
        if self.n_features != n_features:
            raise ValueError(f"Feature mismatch: model has {self.n_features}, input has {n_features}")

        # Compute X * beta + intercept for each class
        scores = []
        for w in self.beta:
            # We use the raw linear function result before any probability transformation
            scores.append(ODRBase().lin_func(w, X_trans, self.C, self.l1, self.l2, self.alpha))
        
        # Return as (n_samples, n_classes)
        return np.array(scores).T

    def predict_proba_sigma_check(self, X, X_err):
        """
        Multiclass evaluation at mean and +/- 1, 2, and 3 sigma points.
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")

        # 1. Transform X: returns (n_features, n_samples)
        X_mid, n_samples, n_features = ODRBase().modify(X)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")

        # 2. Correct the broadcasting: E_trans must be (n_features, n_samples)
        if isinstance(X_err, (int, float)):
            E_trans = np.full(X_mid.shape, X_err)
        else:
            # Check orientation and force it to (n_features, n_samples)
            if X_err.shape == (n_samples, n_features):
                E_trans = X_err.T
            else:
                E_trans = X_err

        # 3. Define evaluation points - shapes are now consistent (n_features, n_samples)
        points = [X_mid, X_mid + E_trans, X_mid - E_trans, 
                  X_mid + 2. * E_trans, X_mid - 2. * E_trans,
                  X_mid + 3. * E_trans, X_mid - 3. * E_trans]
        
        all_p = []
        for pt in points:
            # For multinomial, we use the same approach as OVR but adapted
            pt_probs = ODRBase().multinomial_func(self.beta, pt, self.C, l1=self.l1, l2=self.l2, alpha=self.alpha).T
            all_p.append(pt_probs)

        # Convert to array for statistical calculations: (7, n_samples, n_classes)
        all_p_array = np.array(all_p)
        
        # Calculate mean across the 7-point distribution
        mean_probabilities = np.mean(all_p_array, axis=0)
        
        # Calculate standard deviation to quantify classification sensitivity
        std_probabilities = np.std(all_p_array, axis=0)
        
        return mean_probabilities, std_probabilities

    def predict_robust(self, X, X_err):
        """
        Multiclass robust prediction using the sigma-averaged probability.
        Identifies the class with the highest average probability 
        across the error distribution.
        """
        # 1. Get the (n_samples, n_classes) averaged probability matrix
        # from your new sigma-check routine.
        proba, _ = self.predict_proba_sigma_check(X, X_err)
        
        # 2. Select the index of the highest probability for each sample
        class_indices = np.argmax(proba, axis=1)
        
        # 3. Return the actual class labels (e.g., [3, 6])
        return self.classes_[class_indices]

    def calculate_entropy(self, X, X_err=None):
        """
        Calculates Shannon entropy for predictions.
        If X_err is provided, it uses the robust sigma-averaged probabilities.
        """
        if X_err is not None:
            # Use your unique sigma-check mean probabilities
            probs, _ = self.predict_proba_sigma_check(X, X_err)
        else:
            probs = self.predict_proba(X)

        # H = -sum(p * log(p))
        # Adding a tiny epsilon to avoid log(0)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        return entropy

    def predict_proba_with_model_uncertainty(self, X, n_draws=100):
        """
        Estimates uncertainty by sampling from the distribution of weights (beta).
        """
        if not hasattr(self, 'uncertainty_') or self.uncertainty_ is None:
            raise ValueError("Model covariance (sd_beta) not found. Fit the model first.")

        # Original weights
        beta_mean = self.beta
        # standard deviations of weights from ODR
        beta_std = self.uncertainty_ 

        all_draw_probs = []
        
        for _ in range(n_draws):
            # 1. Build an alternative model by perturbing beta
            # We sample each beta weight from its own normal distribution
            beta_perturbed = np.random.normal(beta_mean, beta_std)
            
            # 2. Run inference with this alternative model
            # Note: We use the internal multinomial_func directly
            probs = ODRBase().multinomial_func(beta_perturbed, X.T, self.C, 
                                               l1=self.l1, l2=self.l2, alpha=self.alpha).T
            all_draw_probs.append(probs)
            
        # 3. Compute the mean and std across the alternative models
        all_draw_probs = np.array(all_draw_probs)
        model_mean_proba = np.mean(all_draw_probs, axis=0)
        model_std_err = np.std(all_draw_probs, axis=0)
        
        return model_mean_proba, model_std_err

#########################################################

