"""
    Test the Logistic classification on the O-ring incident vs temperatures to
    explain the Challenger space shuttle accident in avril 1984.
    
    Use the Orthogonal Distance Logistic algorithm to account for potential
    uncertainties in the temperature measurements.

    The standard logistic regression does not account for the uncertainties in the
    independent variable.

    Temperatures at the day of the Challenger accident are in Farenheit and converted
    first to Centigrade Celsius

    Author Wing-Fai Thi

    License: GNU v3.0
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from ODLinear import OrthogonalDistanceLogisticRegression

xFahrenheit  = np.array([66, 70, 69, 
           68, 67, 72, 
           73, 70, 57,
           63, 70, 78,
           67, 53, 67,
           75,70, 81, 
           76, 79, 75,
           76,58])
xtemp = (xFahrenheit - 32.)*(5./9.) # conversion to Celsius
Tacc = (31. - 32.)*(5./9.)

print("T(C) the day of the accident:", Tacc)
ydamage = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
           0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]

# Arbitrary uncertainty in the temperature prediction at the launch
temp_uncertainty = 2.  # degree Celsius as an arbitrary value

Xerr = np.full(len(xtemp),2.)

x_fit = np.linspace(min(xtemp)-20, max(xtemp)+10, 100)


# Orthogonal Distance Logistic Regression - the regularization is suppressed
odlr = OrthogonalDistanceLogisticRegression(verbose=0,
                                            error_type='std', 
                                            C=1e4, func='sigmoid')

odlr.fit(xtemp , ydamage, X_err=Xerr)
print('ODR logisitic model fit?', odlr.fit_model_)
y_err_fit = odlr.predict_proba(x_fit)[:,1]
print('ODr logistic fit score:', odlr.score(xtemp, ydamage))

# sklearn Logistic Regression - use a high value of C to limit regularization
xtemp = xtemp.reshape(-1, 1)
sklearn_logi = LogisticRegression(C=1e4)
sklearn_logi.fit(xtemp, ydamage)
x_fit = x_fit.reshape(-1, 1)
y_fit = sklearn_logi.predict_proba(x_fit)[:, 1]

# plot the logitic predictions
plt.figure(figsize=(7, 5))
plt.scatter(xtemp, ydamage, label='Data from previous launches')
plt.plot(x_fit, y_fit, color='red', label='OLS Logistic')
plt.plot(x_fit, y_err_fit, color='black', 
         label='OD Logistic with 2 degrees of uncertainty')
plt.scatter([Tacc], [1.], s=50,marker="*", color='red', label='Temperature at launch')
plt.xlabel("Temperature in Centigrade")
plt.ylabel("Damage incident?")
plt.xlim(-10, 35)
plt.legend(loc='lower left')
plt.title("Defects of the Space Shuttle O-Ring")
plt.savefig("ODLinear_Challenger.png")
plt.show()
