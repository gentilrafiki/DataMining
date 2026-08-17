"""
ISLP Lab 3.6 — Linear Regression
Source: An Introduction to Statistical Learning with Applications in Python
         (James, Witten, Hastie, Tibshirani, Taylor)

Run this file section by section in PyCharm (Shift+Enter on each
"# %%" cell if you enable Scientific Mode, or just run top to bottom).

Datasets used: Boston, Carseats (both ship with the ISLP package)
"""

# %% 3.6.1 Importing packages
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                          summarize,
                          poly)

# Inspecting objects and namespaces (optional, just to explore)
print(dir())

A = np.array([3, 5, 11])
print(dir(A))
print(A.sum())  # 19


# %% 3.6.2 Simple Linear Regression
Boston = load_data("Boston")
print(Boston.columns)

# Build model matrix by hand: intercept + lstat
X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]),
                   'lstat': Boston['lstat']})
print(X[:4])

y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()

# summarize() gives coef, std err, t, and P>|t| (the p-value!)
print(summarize(results))

# --- Using ModelSpec() (the general, recommended approach) ---
design = MS(['lstat'])
design = design.fit(Boston)
X = design.transform(Boston)
print(X[:4])

# Equivalent, combined in one step
design = MS(['lstat'])
X = design.fit_transform(Boston)
print(X[:4])

# Full statsmodels summary
print(results.summary())

# Coefficients
print(results.params)

# Predictions with confidence and prediction intervals
new_df = pd.DataFrame({'lstat': [5, 10, 15]})
newX = design.transform(new_df)
print(newX)

new_predictions = results.get_prediction(newX)
print(new_predictions.predicted_mean)

# 95% confidence interval for the mean response
print(new_predictions.conf_int(alpha=0.05))

# 95% prediction interval for a new observation
print(new_predictions.conf_int(obs=True, alpha=0.05))

# --- Plot medv vs lstat with the regression line ---
def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)

ax = Boston.plot.scatter('lstat', 'medv')
abline(ax,
       results.params[0],
       results.params[1],
       'r--',
       linewidth=3)

# --- Diagnostic plots ---
ax = subplots(figsize=(8, 8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')

# Leverage statistics
infl = results.get_influence()
ax = subplots(figsize=(8, 8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
print(np.argmax(infl.hat_matrix_diag))  # observation with highest leverage


# %% 3.6.3 Multiple Linear Regression
X = MS(['lstat', 'age']).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
print(summarize(results1))

# Regress on ALL predictors using shorthand
terms = Boston.columns.drop('medv')
print(terms)

X = MS(terms).fit_transform(Boston)
model = sm.OLS(y, X)
results = model.fit()
print(summarize(results))

# All predictors EXCEPT age
minus_age = Boston.columns.drop(['medv', 'age'])
Xma = MS(minus_age).fit_transform(Boston)
model1 = sm.OLS(y, Xma)
print(summarize(model1.fit()))


# %% 3.6.4 Multivariate Goodness of Fit
print(results.rsquared)          # R^2
print(np.sqrt(results.scale))    # RSE

# Variance Inflation Factors (VIF) — check for multicollinearity
vals = [VIF(X, i) for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif': vals}, index=X.columns[1:])
print(vif)

# Equivalent, written as an explicit loop instead of list comprehension
vals = []
for i in range(1, X.values.shape[1]):
    vals.append(VIF(X.values, i))


# %% 3.6.5 Interaction Terms
X = MS(['lstat',
        'age',
        ('lstat', 'age')]).fit_transform(Boston)
model2 = sm.OLS(y, X)
print(summarize(model2.fit()))


# %% 3.6.6 Non-linear Transformations of the Predictors
X = MS([poly('lstat', degree=2), 'age']).fit_transform(Boston)
model3 = sm.OLS(y, X)
results3 = model3.fit()
print(summarize(results3))

# Compare linear (results1) vs quadratic (results3) fit
print(anova_lm(results1, results3))

ax = subplots(figsize=(8, 8))[1]
ax.scatter(results3.fittedvalues, results3.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')


# %% 3.6.7 Qualitative Predictors
Carseats = load_data('Carseats')
print(Carseats.columns)

allvars = list(Carseats.columns.drop('Sales'))
y = Carseats['Sales']
final = allvars + [('Income', 'Advertising'),
                    ('Price', 'Age')]
X = MS(final).fit_transform(Carseats)
model = sm.OLS(y, X)
print(summarize(model.fit()))