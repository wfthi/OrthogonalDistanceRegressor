import numpy as np
from scipy import stats
from scipy.odr import Data, RealData, Model, ODR


class ODRBase(object):
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
    def check_Xy(self, X, y):
        """
        X Upper case means a matrix
        y lower case means a vector
        """
        X = np.array(X)
        n_samples = X.shape[0]
        y = np.array(y)
        if (n_samples != y.size):
            print("X shape = ", X.shape)
            print("y size = ", y.size)
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
        dimX = X.ndim
        if (dimX > 1):
            n_samples, n_features = X.shape
        else:
            n_samples = X.size
            n_features = 1
        X = X.T  # odrpack is based on a Fortran code
        return X, n_samples, n_features

    def _regularize_coeffs(self, p, C):
        """
        Specifically-designed regularization that
        scales the coefficients instead of penalizaing the cost
        """
        if (p.ndim > 1):
            # function of models with high coefficient values
            alpha = 1. / (1. + np.abs(p).sum(axis=1) / C)
            # high value of C means low regularization
            rcoeffs = np.empty((p.shape))
            # a low value of C means a high regularization
            for i, a in enumerate(alpha):
                rcoeffs[i, :] = p[i, :] * a
        else:
            rcoeffs = p / (1. + abs(p).sum() / C)
        return rcoeffs

    def multi_lin_func(self, p, X, C):
        n_features = X.shape[0]
        n_classes = int(p.size / (n_features + 1))
        pm = p.reshape(n_classes, n_features + 1)
        pm = self._regularize_coeffs(pm, C)
        ylin = np.dot(pm[:, 0:n_features], X)
        for i in range(X.shape[1]):
            ylin[:, i] = ylin[:, i] + pm[:, n_features]
        return ylin

    def lin_func(self, p, X, C):
        n_features = p.size - 1
        p = self._regularize_coeffs(p, C)
        if (n_features > 1):
            y = (X.T).dot(p[0:n_features]) + p[n_features]
        else:
            a, b = p
            y = a*X + b
        return y.reshape(y.size)

    def logit_func(self, p, X, C, func):
        """ run the linear function than the sigmoid function """
        ylin = self.lin_func(p, X, C)
        if (func == 'sigmoid'):
            return self._sigmoid(ylin)  # choose the sigmoid function
        elif (func == 'tanh'):
            return self._tanh(ylin)

    def _sigmoid(self, x):
        """ compute the sigmoid value for a vector x """
        if (np.any(np.isnan(x))):
            raise ValueError("NaN or Inf found while calling sigmoid")
        threshold1 = 30.   # numerically stable x is a numpy array
        threshold2 = 20.
        z = np.empty(x.size)
        w1 = (x > threshold1)
        z[w1] = 1.
        w2 = (x > threshold2)
        z[w2] = 1. - np.exp(-x[w2])
        w3 = (x >= 0.)
        z[w3] = 1. / (1. + np.exp(-x[w3]))
        w4 = (x < -threshold1)
        z[w4] = 0.
        w5 = (x < -threshold2)
        z[w5] = np.exp(x[w5])
        w6 = (x < 0.)
        z6 = np.exp(x[w6])
        z[w6] = z6 / (1 + z6)
        return z

    def _tanh(self, x):
        # hyperbolic tangent, an alternative to sigmoid
        threshold = 20.
        z = np.empty(x.size)
        w1 = (x > threshold)
        z[w1] = 1.
        w2 = (x < -threshold)
        z[w2] = 0.
        w3 = ((x >= -threshold) & (x <= threshold))
        z[w3] = 0.5*(np.tanh(x[w3])+1.)
        return z

    def softmax(self, x):
        # softmax function for a vector
        if (x.ndim > 1):  # use x-m to scale the values down to avoid overflow
            raise ValueError("softmax function works on vectors only")
        m = np.max(x)
        u = np.exp(x-m)
        return u / u.sum()  # u is a vector

    def multinomial_func(self, p, X, C):
        Ylin = self.multi_lin_func(p, X, C)
        proba = []
        for y in Ylin.T:
            proba.append(self.softmax(y))
        return np.array(proba).T

def linear_func(p, x):
    m, c = p
    return m*x + c

#########################################################
class OrthogonalDistanceLinearRegression(ODRBase, object):
    """
    Linear Regression using the Orthogonal Distance methiod.
    Errors in the predcitor and in the target values are possible.
    Without errors, the standatd least square method is used and will
    give the same results than LinearRegression from scikit-learn.

    See the scipy.odr manual for more details.

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
                 verbose=False):
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
                       'verbose': self.verbose}
        return self.params

    def fit(self, X, y, X_err=None, y_err=None,
            initial_guess=None, **kwargs):
        """
        Fit the independent variables X with optional error
        X_err to the dependent variable y with optional error y_err.
        Without error the least-square method is use, with error the
        method perfoms a Orthogonal Distance regression.

        Parameters:
        -----------

        X: array of shape (n_samples, n_features)
            input independent variables

        y : array of shape (n_samples,)

        X_err : scalar or array of same shape as X, optional, default=None
            the errors for X
            If X_err is a scalar, then that value is used for all data points
            (and all dimensions of the response variable

        y_err : scalar or array of same shape as y, optional, default=None
            the errors for y
            If y_err is a scalar, then that value is used for all data points
            (and all dimensions of the response variable
       
        Returns
        -------
        self : object
        Returns self.

        Example
        -------
        >>> import random
        >>> from scipy.odr import *
        >>> import numpy as np
        >>> from ODLinear import OrthogonalDistanceLinearRegression as ODLR
        >>> # Initiate some data, giving some randomness using random.random().
        >>> x = np.array([0, 1, 2, 3, 4, 5])
        >>> y = np.array([i**2 + random.random() for i in x])
        >>> # Define a function (quadratic in our case) to fit the data with.
        >>> def linear_func(p, x):
        ...    m, c = p
        ...    return m*x + c
        >>> # Create a model for fitting.
        >>> linear_model = Model(linear_func)
        >>> # Create a RealData object using our initiated data from above.
        >>> data = RealData(x, y, sx = np.repeat(0.1, 6))
        >>> # Set up ODR with the model and data.
        >>> odr = ODR(data, linear_model, beta0=[0., 1.])
        >>> # Run the regression.
        >>> out = odr.run()
        >>> # Use the in-built pprint method to give us results.
        >>> out.pprint()
        >>> regr_odlr = ODLR(error_type='std', max_robust_iter=5, verbose=True)
        >>> regr_odlr.fit(x, y)
        >>> ypred = regr_odlr.predict(x)
        >>> X_err = 0.1
        >>> regr_odlr.fit(x, y, X_err=X_err)
        >>> ypred = regr_odlr.predict(x)
        """
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
        std_error = 0.
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
            model = Model(ODRBase().lin_func, extra_args=[self.C])
            odr = ODR(data, model, beta0=initial_guess,
                      maxit=self.maxit, sstol=self.tol, **kwargs)
            odr.set_job(fit_type=self.fit_type_)
            out = odr.run()
            if (self.verbose):
                out.pprint()
            error = abs(out.eps)
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
            self.coef = w[0:n_features]
            self.intercept = w[n_features]
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
            mean_error = np.mean(error)  # better for pure linear model
            std_error = np.std(error)    # better for polynomial model
            # mode (most frequent value in a distribution)
            mode_error = stats.mode(error)[0]
            error_sum = error.sum()
            lorentz_error = np.log(1+0.5*error**2).sum()
            # add the y-residual to the initial error in y
            y_err = y_err0 + error
            initial_guess = out.beta
        self.out = out
        y_err = y_err0
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
        return ODRBase().lin_func(self.beta, X, self.C)

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
            pred_MC.append(ODRBase().lin_func(self.beta, X_MC, self.C).T)
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


############################################################
class OrthogonalDistanceLogisticRegression(ODRBase, object):
    """
    Binary logistic Regression based on the Orthongonal Distance
    Linear Regression

    Similar concept as a one-layed neural-network

    Logistic

    Parameters
    ----------
    C : float, default: 1e5, ie no regularization
    Inverse of regularization strength; must be a positive float.
    Like in support vector machines, smaller values specify stronger
    regularization.

    max_iter : int, default: 100
    Maximum number of iterations taken for the solvers to converge.

    error_type : string, default: 'std' (valid types: 'std' and 'weights')

    probability : string, default: 'softmax' (valid methods:
                  "softmax" and "ovr")
    The method to compute the multiclass probabilities. "ovr" uses a
    simple normlization

    tol : float, default: eps**(1/2)~1e-8
    Tolerance for stopping criteria.

    verbose : int, default: 0
    level of screen output

    """
    def __init__(self, C=1e5, maxit=100, error_type='std',
                 tol=1e-8, func='sigmoid', verbose=False):
        self.C = C
        self.maxit = maxit
        self.error_type = error_type
        self.tol = tol
        self.verbose = verbose
        self.func = 'sigmoid'
        self.fit_model_ = False
        if (error_type not in ['std', 'weights']):
            raise ValueError('Invalid error_type: choose std or weights,'
                             ' default is std')
        if (func not in ['sigmoid', 'tanh']):
            raise ValueError('Invalid logit func '
                             'use sigmoid or tanh, default is sigmoid')
        assert C > 0
        assert maxit > 0
        assert tol > 0

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
                       'func': self.func,
                       'verbose': self.verbose}
        return self.params

    def fit(self, X, y, X_err=None,
            initial_guess=None, **kwargs):
        """
        Fit the input data X to the target y with optinal error in X X_err
        using the ODR algorithm.
        """
        ODRBase().check_Xy(X, y)
        X, n_samples, n_features = ODRBase().modify(X)
        if (n_samples < n_features):
            raise ValueError('n_samples < n_features')
        self.n_samples = n_samples
        self.n_features = n_features
        nuq = np.unique(y).size
        if (nuq > 2):
            if (np.all((np.min(y) >= 0.) & (np.max(y) <= 1.))):
                self.classes_ = np.array([0, 1])
                self.n_classes_ = 2
            else:
                raise ValueError(
                """
                y has more than two classes.
                Use OrthogonalDistanceLogisticRegressionOVR or
                OrthogonalDistanceLogisticRegression
                or OrthogonalDistanceMultinomialLogisticRegression instead.
                """)
        else:
            self.classes_ = np.unique(y).astype(int)
            self.n_classes_ = 2
        wmin = (y == min(y))
        wmax = (y == max(y))
        y[wmin] = 0
        y[wmax] = 1
        if (initial_guess is None):
            initial_guess = np.zeros(n_features+1)
        if (X_err is not None):
            self.fit_type_ = 0  # explicit ODR + central finite differences
            if (self.error_type == 'std'):
                X_err = X_err.T
                ODRBase().check_XX(X, X_err)
        else:   # no error: use least-square + central finite differences
            self.fit_type_ = 2
        # The data, with weightings as actual standard deviations
        if (self.error_type == 'std'):
            data = RealData(X, y, sx=X_err)
        else:
            data = Data(X, y, wd=X_err)
        model = Model(ODRBase().logit_func, extra_args=[self.C, self.func])
        odr = ODR(data, model, beta0=initial_guess,
                  maxit=self.maxit, sstol=self.tol, **kwargs)
        # 0: explicit ODR, 2 least-square
        odr.set_job(fit_type=self.fit_type_)
        out = odr.run()
        if (self.verbose):
            out.pprint()
        w = self._regularize_coeffs(out.beta, self.C)
        self.beta = out.beta
        self.coef = w[0: n_features]
        self.intercept_ = w[n_features]
        self.quasi_chisq_ = out.res_var
        self.uncertainty_ = out.sd_beta
        self.delta_ = out.delta.T  # estimated x-component of the residuals
        self.epsilon_ = out.eps    # estimated y-component of the residuals
        self.Xplus_ = out.xplus
        self.sum_square = out.sum_square
        self.sum_square_delta = out.sum_square_delta
        self.sum_square_eps = out.sum_square_eps
        self.fit_model_ = True
        return self

    def predict(self, X):
        """
        Returns the prediction for the preditor array X. The output value is
        either 0 or 1
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        proba = self.predict_proba(X)[:, 1]
        return self.classes_[(np.floor(proba + 0.5)).astype(int)]

    def predict_proba(self, X):
        """
        Returns the logistic probability
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        X, n_samples, n_features = ODRBase().modify(X)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")
        res = ODRBase().logit_func(self.beta, X, self.C, self.func).T
        return np.dstack((1.-res, res)).reshape(n_samples, 2)

    def predict_proba_error(self, X, X_err):
        """
        Returns the uncertainty on the probabilities using a simplified method.
        It requires the error matrix X_err
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        X, _, n_features = ODRBase().modify(X)
        X_err = X_err.T
        ODRBase().check_XX(X, X_err)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")
        res1 = ODRBase().logit_func(self.beta, X+X_err, self.C, self.func).T
        res2 = ODRBase().logit_func(self.beta, X-X_err, self.C, self.func).T
        proba_error = abs(res1-res2)*0.5
        return proba_error

    def predict_proba_MC_error(self, X, X_err,
                               Number_of_MC_iterations=1000):
        """
        Returns the uncertainty on the probabilities using
        a Monte-Carlo method.
        It requires the error matrix X_err
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        X, _, n_features = ODRBase().modify(X)
        X_err = X_err.T
        ODRBase().check_XX(X, X_err)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")
        proba_MC = []
        for i in range(Number_of_MC_iterations):
            X_MC = np.random.normal(X, X_err)
            proba_MC.append(ODRBase().logit_func(self.beta,
                                                 X_MC, self.C, self.func).T)
        proba_MC = np.array(proba_MC)
        std_proba = np.std(proba_MC, axis=0)
        return std_proba

    def decision_function(self, X):
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        return self.predict_proba(X)[:, 1]

    def score(self, X, y, sample_weight=None):
        """
        returns the accuracy: tp+tn/(tp+tn+fp+fn)
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        ODRBase().check_Xy(X, y)
        if (sample_weight is not None):
            sample_weight = np.array(sample_weight)
            ODRBase().check_Xy(X, sample_weight)
            sample_weight = sample_weight/np.sum(sample_weight)*len(y)
            return sum(sample_weight*(self.predict(X) == y))*1. / len(y)
        else:
            return sum(self.predict(X) == y)*1. / len(y)

    def logloss_score(self, y, p):
        """
        returns the binary logloss score (lower is better)
        y is the truth, p is the prediction
        """
        y = np.array(y)
        p = np.array(p)
        if (y.ndim > 1):
            raise ValueError("The truth array has more than one dimension")
        if (p.ndim > 1):
            raise ValueError("The prediction array has more than one dimension")
        if (y.size != p.size):
            raise ValueError("The truth and predictions have different size!")
        logloss = 0.
        eps = 1e-30
        for i in range(y.size):
            a = max([p[i], eps])
            b = max([1.-p[i], eps])
            logloss = logloss + y[i]*np.log(a)+(1.-y[i]) * np.log(b)
        logloss = -logloss/float(y.size)
        return logloss


##############################################################
class OrthogonalDistanceLogisticRegressionOVR(ODRBase, object):
    """
    Multiclass classification using the Orthogonal Distance Logistic Regression
    One Versus the Rest method. First the routine runs n_classes binary
    classification fit where classifier fit one class against all the others
    gouped into one single class.
    The probabiities are stored and normalized directly or throught a softmax
    function to give the final probability for each class.

    Parameters
    ----------
    C : float, default: 1e5, ie no regularization
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    max_iter : int, default: 100
        Maximum number of iterations taken for the solvers to converge.

    error_type : string, default: 'std' (valid types: 'std' and 'weights')

    probability : string, default: 'softmax' (valid methods:
        "softmax" and "ovr")
        The method to compute the multiclass probabilities. "ovr" uses a
        simple normlization

    tol : float, default: eps**(1/2)~1e-8
        Tolerance for stopping criteria.

    verbose : int, default: 0
        level of screen output

    Attributes
    ----------
    n_classes_ : integer
        number of classes

    classes_: array of shape(n_classes_)
        labels of the classes

    coef_ : array of shape(n_classes_,n_features)

    intercept_ : array of shape(n_classes_)

    Methods
    -------
    fit(X,y)
        Fit the model using X as training data and y as target values

        Parameters X array of shape(n_samples,n_features)

        Returns self

    predict(X)
        Returns the predicted label for X. the fit method has to been
        run before

        Parameters  X array of shape(n_samples,n_features)

        Returns ndarray, shape (n_samples

    predict_proba(X)
        Returns the predicted class probability for each instance
        in the input X

        Parameters X array of shape(n_samples,n_features)

        Returns ndarray, shape (n_samples,number of classes)

    predict_proba_MR_error(X)
        Returns the predicted class probability error (1 stdanard deviation)
        for each instance in the input X

        Parameters
            X array of shape(n_samples,n_features): the independent variables
            (predictors)
            X_err array of same shape than X: the error of the independant
            variables

        Returns ndarray, shape (n_samples,number of classes)

    score((self, X, y, sample_weight=None)
        Returns the mean accuracy on the given test data and labels.

    get_params((self, deep=True)
        Get parameters for this estimator

    set_params(self, **params)
        Set the parameters of this estimator.
    """
    def __init__(self, C=1., maxit=100, error_type='std',
                 probability='softmax',
                 tol=1e-8, func='sigmoid', verbose=False):
        self.C = C
        self.maxit = maxit
        self.error_type = error_type
        self.probability = probability
        self.tol = tol
        self.func = 'sigmoid'
        self.fit_model_ = False
        self.verbose = verbose
        if (error_type not in ['std', 'weights']):
            raise ValueError('Invalid error_type: choose std or weights,'
                             ' default is std')
        if (func not in ['sigmoid', 'tanh']):
            raise ValueError('Invalid logit func '
                             'use sigmoid or tanh, default is sigmoid')
        assert C > 0
        assert maxit > 0
        assert tol > 0

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
                       'probability': self.probability,
                       'func': self.func,
                       'tol': self.tol,
                       'verbose': self.verbose}
        return self.params

    def fit(self, X, y, X_err=None, initial_guess=None, **kwargs):
        """
        Fit the model according to the given training data.

        The One-versus-the-rest (OVR) method is used.
        First the probability for each class against
        all the other grouped into one unique class is estimated.
        For each instance the probabilites
        are normalized using a simple normalization by setting
        probability="ovr" or using a softmax
        function by setting probability='softmax').
        The attribute C controles the amount of regularization

        The routine uses the Orthogonal Distance Regression algorithm.
        This method accounts for errors in the independent variables
        (predictors) and the dependent variables (targets).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        X_err : array of the same shape than X (n_samples, n_features), optional
            Array of error for each predictor. Without error, a standard least
            square method is used.

        initial_guess : array of shape (n_features+1), optional
            An array of the inital guess for the coefficients including
            the intercept.

        **kwargs : extra optinal arguments
            They are passed to the scipy.ord solver

        Returns
        -------
            self : object
            Returns self.
        """
        ODRBase().check_Xy(X, y)
        X, n_samples, n_features = ODRBase().modify(X)
        y.astype(int)  # ensure that the class labels are integers
        self.classes_ = np.unique(y).astype(int)
        self.n_classes_ = self.classes_.size
        n_classes = self.n_classes_
        if (self.verbose):
            print("Number of classes:", n_classes)
        if (n_samples < (n_features + 1) * n_classes):
            print("Training set size:", n_samples)
            print("Number of parameters:", (n_features + 1) * n_classes)
            raise ValueError('n_samples < (n_features + 1) * n_classes')
        if (self.verbose):
            print("Fitting (n_samples, n_features):", n_samples, n_features)
        self.n_samples = n_samples
        self.n_features = n_features
        if (n_samples < n_features):
            raise ValueError('n_samples < n_features')
        if (X_err is not None):
            if (self.error_type == 'std'):
                X_err = X_err.T
                ODRBase().check_XX(X, X_err)
            # explicit ODR + central finite differences then 10
            self.fit_type_ = 0
        else:  # no error: use least-square (+ central finite differences : 12)
            self.fit_type_ = 2
        if (self.verbose):
            print("solver type:", self.fit_type_)
        w = []
        quasi_chisq = []
        uncertainty = []
        delta = []
        eps = []
        # The fit will not converge if the guess is poor.
        if (initial_guess is None):
            initial_guess = np.zeros((self.n_classes_, n_features+1))
        if (self.verbose):
            print("Number of classes:", self.n_classes_)
        if (self.verbose):
            print("Starting binary classification loop")
        for i in self.classes_:
            y_copy = np.where(y == i, 1, 0)
            # The data, with weightings as actual standard deviations
            if (self.error_type == 'std'):
                data = RealData(X, y_copy, sx=X_err)
            else:
                data = Data(X, y_copy, wd=X_err)
            model = Model(ODRBase().logit_func,
                          extra_args=[self.C, self.func])
            odr = ODR(data, model, beta0=initial_guess[i, :],
                      maxit=self.maxit, sstol=self.tol, **kwargs)
            odr.set_job(fit_type=self.fit_type_)
            out = odr.run()
            if (self.verbose):
                out.pprint()
            w.append(out.beta)
            quasi_chisq.append(out.res_var)
            uncertainty.append(out.sd_beta)
            delta.append(out.delta.T)
            eps.append(out.eps)
            if (self.verbose):
                print("class:", i, "quasi-rms:", out.res_var)
        self.beta = w  # the coefficients of the fits
        w = np.array(w)
        w = self._regularize_coeffs(w, self.C)
        self.coef_ = w[:, 0:n_features]
        self.intercept_ = w[:, n_features]
        self.quasi_chisq_ = quasi_chisq
        self.uncertainty_ = uncertainty
        self.delta_ = delta  # estimated x-component of the residuals
        self.epsilon_ = eps  # estimated y-component of the residuals
        self.Xplus_ = out.xplus
        self.fit_model_ = True
        return self

    def predict(self, X):
        """
        Predict the class of each instance in X

        Parameters:
        -----------
        X : predictor array of shape (n_samples,n_features)
        """
        if not (self.fit_model_):
            raise ValueError("You need to fit the model first")
        X, _, n_features = ODRBase().modify(X)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")
        predictions = []
        for w in self.beta:
            predictions.append(ODRBase().logit_func(w, X, self.C, self.func))
        return self.classes_[np.argmax(predictions, axis=0)]

    def predict_proba_MC_error(self, X,
                               X_err, Number_of_MC_iterations=1000):
        """
        Find the standard deviation of the probability in a Monte-Carlo fashion

        Parameters:
        -----------
        X : predictor array of shape (n_samples,n_features)

        X_err: predictor error array of the same shape than X

        Number_of_MC_iterations : array of number of MC predictions,
        default 1000
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        X, n_samples, n_features = ODRBase().modify(X)
        X_err = X_err.T
        ODRBase().check_XX(X, X_err)
        if (self.n_features != n_features):
            raise ValueError("Numbers of features do not match")
        proba_MC = []
        for i in range(Number_of_MC_iterations):
            X_MC = np.random.normal(X, X_err)
            if (self.probability == "softmax"):
                ylin = []
                for w in self.beta:
                    ylin.append(ODRBase().lin_func(w, X_MC, self.C))
                ylin = np.array(ylin)
                for j in range(n_samples):
                    proba_MC.append(ODRBase().softmax(ylin[:, j]))
            elif (self.probability == "ovr"):
                proba_logit = []
                for w in self.beta:
                    proba_logit.append(ODRBase().logit_func(w, X,
                                                            self.C, self.func))
                proba_logit = np.array(proba_logit)
                proba_tot = np.sum(proba_logit, axis=0).T
                normalization = np.vstack((proba_tot, proba_tot, proba_tot))
                proba_MC.append((proba_logit / normalization).T)
            else:
                raise ValueError("Wrong probabilty type. Use softmax or ovr")
        proba_MC = np.array(proba_MC).reshape(Number_of_MC_iterations,
                                              self.n_classes_, n_samples)
        return np.std(proba_MC, axis=0).T

    def predict_proba(self, X):
        """
        Probability estimates.

        The softmax function is used to find the predicted probability
        of each class by default
        if the class is instantiated with probability = "ovr", the probability
        of each class is calculated using the logistic function and normalized
        across all the classes

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
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
                                                        self.C, self.func))
            proba_logit = np.array(proba_logit)
            proba_tot = np.sum(proba_logit, axis=0).T
            normalization = np.vstack((proba_tot, proba_tot, proba_tot))
            probabilities = (proba_logit / normalization).T
        else:
            raise ValueError("Wrong probabilty type. Use softmax or ovr")
        return probabilities

    def score(self, X, y, sample_weight=None):
        """
        Returns the accuracy (one float) of the predictions:
            sum(predict = true) / len(true)
        """
        if not self.fit_model_:
            raise ValueError("You need to fit the model first")
        ODRBase().check_Xy(X, y)
        if (sample_weight is not None):
            sample_weight = np.array(sample_weight)
            ODRBase().check_Xy(X, sample_weight)
            sample_weight = sample_weight/np.sum(sample_weight)*len(y)
            return sum(sample_weight*(self.predict(X) == y))*1. / len(y)
        else:
            return sum(self.predict(X) == y)*1. / len(y)


##############################################################
class OrthogonalDistanceMultinomialLogisticRegression(ODRBase, object):
    """
    Multiclass classification using the Orthogonal Distance Logistic Regression
    multinomial method, ie the different classes are converted into a matrix by
    hot encoding and the data are fit to determine directly their probability
    like for a binary classification. The function to obtain the probabities is
    the softmax function instead of the sigmoid function.
    This method is slow than the One-Versus-the-Rest (OVR) method and can give
    different predictions than the OVR method.

    Parameters
    ----------
    C : float, default: 1e5, ie no regularization
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    max_iter : int, default: 100
        Maximum number of iterations taken for the solvers to converge.

    error_type : string, default: 'std' (valid types: 'std' and 'weights')

    probability : string, default: 'softmax' (valid methods: "softmax"
        and "ovr")
        The method to compute the multiclass probabilities. "ovr" uses a simple
        normlization

    tol : float, default: eps**(1/2)~1e-8
        Tolerance for stopping criteria.

    verbose : int, default: 0
        level of screen output

    Attributes
    ----------
    n_classes_ : integer
        number of classes

    classes_: array of shape(n_classes_)
        labels of the classes

    coef_ : array of shape(n_classes_,n_features)

    intercept_ : array of shape(n_classes_)

    Methods
    -------
    fit(X,y)
        Fit the model using X as training data and y as target values

        Parameters X array of shape(n_samples,n_features)

        Returns self

    predict(X)
        Returns the predicted label for X. the fit method has to been run
        before

        Parameters  X array of shape(n_samples,n_features)

        Returns ndarray, shape (n_samples

    predict_proba(X)
        Returns the predicted class probability for each instance in the
        input X

        Parameters X array of shape(n_samples,n_features)
 
        Returns ndarray, shape (n_samples,number of classes)

    predict_proba_MR_error(X)
        Returns the predicted class probability error (1 stdanard deviation)
        for each instance in the input X

        Parameters
            X array of shape(n_samples,n_features): the independent variables
            (predictors)
            X_err array of same shape than X: the error of the independant
            variables

        Returns ndarray, shape (n_samples,number of classes)

    score((self, X, y, sample_weight=None)
        Returns the mean accuracy on the given test data and labels.

    get_params((self, deep=True)
        Get parameters for this estimator

    set_params(self, **params)
        Set the parameters of this estimator.
    """

    def __init__(self, C=1e5, maxit=100, error_type='std',
                 tol=1e-8, verbose=False):
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
                       'verbose': self.verbose}
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

    def fit(self, X, y, X_err=None, initial_guess=None, **kwargs):
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
        if (X_err is not None):
            self.fit_type_ = 0  # explicit ODR + central finite differences
            if (self.error_type == 'std'):
                X_err = X_err.T
                ODRBase().check_XX(X, X_err)
        else:   # no error: use least-square + central finite differences
            self.fit_type_ = 2
        if (self.verbose):
            print("solver type:", self.fit_type_)
        # The data, with weightings as actual standard deviations
        if (self.error_type == 'std'):
            data = RealData(X, Y, sx=X_err)
        else:
            data = Data(X, Y, wd=X_err)
        if (self.verbose):
            print("Number of parameters:", initial_guess.size)
        model = Model(ODRBase().multinomial_func, extra_args=[self.C])
        odr = ODR(data, model, beta0=initial_guess,
                  maxit=self.maxit, sstol=self.tol, **kwargs)
        odr.set_job(fit_type=self.fit_type_)  # 0: explicit ODR, 2 least-square
        out = odr.run()
        if (self.verbose):
            out.pprint()
        self.beta = out.beta
        w = self.beta.reshape(n_classes, n_features+1)
        w = self._regularize_coeffs(w, self.C)
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
        proba = ODRBase().multinomial_func(self.beta, X, self.C).T
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
        std_proba = np.std(proba_MC, axis=0)
        return std_proba
