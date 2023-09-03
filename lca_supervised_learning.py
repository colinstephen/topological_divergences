import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score

from scipy.stats import iqr

from numpy.random import MT19937
from numpy.random import RandomState
from numpy.random import SeedSequence

SEED = 42
RANDOM_STATE = RandomState(MT19937(SeedSequence(SEED)))

def score_classification(X, y):
    """Binary classifier for y>0 (chaos detector)."""
    clf = RFC(random_state=RANDOM_STATE, n_estimators=50)
    scores = cross_val_score(clf, X, y>0, cv=10, scoring="f1")
    return np.mean(scores)

def score_regression(X, y):
    """Regressor for y (chaos quantifier)."""
    clf = RFR(random_state=RANDOM_STATE, n_estimators=50)
    scores = cross_val_score(clf, X, y, cv=10, scoring="neg_mean_absolute_error")
    return np.mean(scores)/iqr(y)

def score_regression_pos(X, y):
    """Regressor for positive y (chaos quantifier)."""
    clf = RFR(random_state=RANDOM_STATE, n_estimators=50)
    pos_mask = y > 0
    scores = cross_val_score(clf, X[pos_mask,:], y[pos_mask], cv=10, scoring="neg_mean_absolute_error")
    return np.mean(scores)/iqr(y[pos_mask])
