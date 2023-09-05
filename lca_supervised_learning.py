import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn import neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from scipy.stats import iqr

from numpy.random import MT19937
from numpy.random import RandomState
from numpy.random import SeedSequence

SEED = 42
RANDOM_STATE = RandomState(MT19937(SeedSequence(SEED)))

def score_classification(X, y):
    """Binary classifier for y>0 (chaos detector)."""
    finite_idx = np.isfinite(X[:,0])
    X, y = X[finite_idx], y[finite_idx]
    clf = RFC(random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y>0, scoring="f1")
    return np.mean(scores)

def score_regression(X, y):
    """Regressor for y (chaos quantifier) using Random Forest."""
    finite_idx = np.isfinite(X[:,0])
    X, y = X[finite_idx], y[finite_idx]
    clf = RFR(random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, scoring="neg_mean_squared_error")
    return np.mean(scores)/iqr(y)

def score_regression_pos(X, y):
    """Regressor for positive y (chaos quantifier) using Random Forest."""
    finite_idx = np.isfinite(X[:,0])
    X, y = X[finite_idx], y[finite_idx]
    clf = RFR(random_state=RANDOM_STATE)
    pos_mask = y > 0
    scores = cross_val_score(clf, X[pos_mask,:], y[pos_mask], scoring="neg_mean_squared_error")
    return np.mean(scores)/iqr(y[pos_mask])

def score_regression_KNN(X, y):
    """Regressor for y (chaos quantifier) using KNN."""
    finite_idx = np.isfinite(X[:,0])
    X, y = X[finite_idx], y[finite_idx]
    clf = neighbors.KNeighborsRegressor(n_neighbors=10)
    # clf = Pipeline([('poly', PolynomialFeatures(degree=10)), ('linear', SVR())])
    # clf = Pipeline([('scale', StandardScaler()), ('neigbours', neighbors.RadiusNeighborsRegressor(radius=2.0))])
    scores = cross_val_score(clf, X, y, scoring="neg_mean_squared_error")
    return np.mean(scores)/iqr(y)

def score_regression_pos_KNN(X, y):
    """Regressor for positive y (chaos quantifier) using KNN."""
    finite_idx = np.isfinite(X[:,0])
    X, y = X[finite_idx], y[finite_idx]
    clf = neighbors.KNeighborsRegressor(n_neighbors=10)
    # clf = Pipeline([('poly', PolynomialFeatures(degree=10)), ('linear', SVR())])
    # clf = Pipeline([('scale', StandardScaler()), ('neigbours', neighbors.RadiusNeighborsRegressor(radius=2.0))])
    pos_mask = y > 0
    scores = cross_val_score(clf, X[pos_mask,:], y[pos_mask], scoring="neg_mean_squared_error")
    return np.mean(scores)/iqr(y[pos_mask])

