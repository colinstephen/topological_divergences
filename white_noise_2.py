# %%
BASELINES=False  # compute the classic and TDA baselines
DISCRETE=True  # discrete or piecewise linear merge tree
OFFSET_MIN, OFFSET_MAX, OFFSET_STEP = 1, 252, 25  # merge tree offset range
SEED = 54321  # consistent random number generation
SAMPLES = 5000  # number of trajectories for training
TEST_SAMPLES = 1001  # number of trajectories for testing
LENGTH = 2000  # number of points per trajectory
SYSTEM = "tinkerbell"

# %%
# Provide clients to an ipyparallel cluster for faster parallel processing
from ipyparallel import Client
clients = Client(profile="testprofile")
dv = clients.direct_view()
lbv = clients.load_balanced_view()


# %% [markdown]
# # Can Topological Divergences Help Predict the Largest Lyapunov Exponent?

# %% [markdown]
# ## Overview
# 
# This notebook generates dynamic system trajectory data then analyses multiple features for supervised learning of the largest Lyapunov exponent (classification and regression). Classical numeric methods, TDA-based methods, Horizontal Visibility methods, and our newly introduced topological divergences are compared.
# 
# - classic neighbour-tracing estimators from Rosenstein, Eckmann, and Kantz
# - ordinal partition network embedded persistence measures from Myers
# - $k$-nearest neighbour graph embedded persistence measures from Myers
# - Betti vector norms on embedded trajectories from Güzel
# - topological divergences (the main contribution)
# 
# Topological divergences are scalar or vector valued measures of the difference between the sublevel and superlevel filtrations over a scalar function.

# %%
# collect imports for cells below

import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy import stats
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from trajectories import generate_trajectories




# %%
# Preprocessing function to remove scale bias in supervised learning

def scale(ts):
    """Make range of ts fall between 0 and 1"""
    scaler = MinMaxScaler()
    return scaler.fit_transform(ts.reshape(-1, 1)).flatten()


# %% [markdown]
# ## Chaotic system data

# %%
# Generate the actual system data to analyse

import pickle
import os

filename_train_data = os.path.join("outputs/white_noise", "__".join(map(str, [SEED, LENGTH, SAMPLES])) + "__train_data.pkl")
if not os.path.exists(filename_train_data):
    with open(filename_train_data, "wb") as file:
        data_ = generate_trajectories(RANDOM_SEED=SEED, TS_LENGTH=LENGTH, CONTROL_PARAM_SAMPLES=SAMPLES)
        pickle.dump(data_, file)

with open(filename_train_data, "rb") as file:
    system_training_data = pickle.load(file)


# %%
# Remove relative scale (amplitude) as a feature that could be used in supevised learning

for system in system_training_data:
    trajectories = system_training_data[system]["trajectories"]
    trajectories = list(map(scale, trajectories))
    system_training_data[system]["trajectories"] = trajectories


# %% [markdown]
# ## Supervised learning

# %%
# Define machine learning models to train on the Lyapunov estimates

def score_features_train(feature_names, features, y_true, cv=5, n_repeats=5, ML_SEED=123):
    """Score various supervised ML models on supplied features give a ground truth.
    
    For classification, assumes ground truth y_true>0 is the positive class.
    """

    # assume vectorial features; if scalar, add an extra dimension
    features = np.array(features)
    if features.ndim == 2:
        features = features[..., np.newaxis]
    n_samples, n_features, feature_vector_length = features.shape

    CLASSIFIER_CV = RepeatedStratifiedKFold(n_splits=cv, random_state=ML_SEED, n_repeats=n_repeats)
    REGRESSOR_CV = RepeatedKFold(n_splits=cv, random_state=ML_SEED*2, n_repeats=n_repeats)

    y = y_true
    pos_mask = y>0
    y_classes = y>0

    classification_scorer = "f1"
    regression_scorer = "neg_mean_squared_error"


    for i in range(n_features):
        feature_name = feature_names[i]
        X = features[:, i, :].reshape(n_samples, -1)

        SVC_pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(random_state=ML_SEED))])
        SVC_clf = GridSearchCV(SVC_pipe, {'svc__C':[0.01, 0.1, 1, 10, 100]}, scoring=classification_scorer, n_jobs=-2, refit=True, cv=CLASSIFIER_CV)
        SVC_clf.fit(X, y_classes)
        SVC_scores = cross_val_score(SVC_clf.best_estimator_, X, y_classes, scoring=classification_scorer, cv=CLASSIFIER_CV, n_jobs=-2)

        KNC_pipe = Pipeline([('scaler', StandardScaler()), ('knc', KNeighborsClassifier())])
        KNC_clf = GridSearchCV(KNC_pipe, {'knc__n_neighbors':[5, 10, 15, 20, 25, 30]}, scoring=classification_scorer, n_jobs=-2, refit=True, cv=CLASSIFIER_CV)
        KNC_clf.fit(X, y_classes)
        KNC_scores = cross_val_score(KNC_clf.best_estimator_, X, y_classes, scoring=classification_scorer, cv=CLASSIFIER_CV, n_jobs=-2)

        MLPC_pipe = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(learning_rate='adaptive', random_state=ML_SEED, max_iter=800))])
        MLPC_clf = GridSearchCV(MLPC_pipe, {'mlp__alpha':[0.00001, 0.0001, 0.001, 0.01], 'mlp__hidden_layer_sizes':[(10,), (20,), (10,5,), (20,5)]}, scoring=classification_scorer, n_jobs=-2, refit=True, cv=CLASSIFIER_CV)
        MLPC_clf.fit(X, y_classes)
        MLPC_scores = cross_val_score(MLPC_clf.best_estimator_, X, y_classes, scoring=classification_scorer, cv=CLASSIFIER_CV, n_jobs=-2)

        KNR_all_pipe = Pipeline([('scaler', StandardScaler()), ('knr', KNeighborsRegressor(weights='distance'))])
        KNR_all_clf = GridSearchCV(KNR_all_pipe, {'knr__n_neighbors': [5, 10, 15, 20, 25, 30]}, n_jobs=-2, scoring=regression_scorer, cv=REGRESSOR_CV, refit=True)
        KNR_all_clf.fit(X, y)
        KNR_all_scores = cross_val_score(KNR_all_clf.best_estimator_, X, y, scoring=regression_scorer, cv=REGRESSOR_CV, n_jobs=-2)

        SVR_all_pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
        SVR_all_clf = GridSearchCV(SVR_all_pipe, {'svr__C':[0.01, 0.1, 1, 10, 100]}, scoring=regression_scorer, n_jobs=-2, refit=True, cv=REGRESSOR_CV)
        SVR_all_clf.fit(X, y)
        SVR_all_scores = cross_val_score(SVR_all_clf.best_estimator_, X, y, scoring=regression_scorer, cv=REGRESSOR_CV, n_jobs=-2)

        MLPR_all_pipe = Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(learning_rate='adaptive', random_state=ML_SEED, max_iter=800))])
        MLPR_all_clf = GridSearchCV(MLPR_all_pipe, {'mlp__alpha':[0.00001, 0.0001, 0.001, 0.01], 'mlp__hidden_layer_sizes':[(10,), (20,), (10,5,), (20,5)]}, scoring=regression_scorer, n_jobs=-2, refit=True, cv=REGRESSOR_CV)
        MLPR_all_clf.fit(X, y)
        MLPR_all_scores = cross_val_score(MLPR_all_clf.best_estimator_, X, y, scoring=regression_scorer, cv=REGRESSOR_CV, n_jobs=-2)

        yield {
            feature_name: {
                "scores": {
                    "SVC": SVC_scores,
                    "SVR": SVR_all_scores,
                    "MLPC": MLPC_scores,
                    "MLPR": MLPR_all_scores,
                    "KNC": KNC_scores,
                    "KNR": KNR_all_scores,
                },
                "models": {
                    "SVC": SVC_clf,
                    "SVR": SVR_all_clf,
                    "MLPC": MLPC_clf,
                    "MLPR": MLPR_all_clf,
                    "KNC": KNC_clf,
                    "KNR": KNR_all_clf,
                }
            }
        }


# %%
# Apply trained machine models to features from new unseen data

def score_features_test(feature_names, features, y_true, trained_models):
    """Predict using features as input to trained models and score against ground truth.
    
    For classification, assumes ground truth y_true>0 is the positive class.
    """

    # assume vectorial features; if scalar, add an extra dimension
    features = np.array(features)
    if features.ndim == 2:
        features = features[..., np.newaxis]
    n_samples, n_features, feature_vector_length = features.shape
    
    is_classifier = lambda clf: hasattr(clf, "classes_")

    pos_mask = y_true>0

    for i in range(n_features):
        feature_name = feature_names[i]
        X = features[:, i, :].reshape(n_samples, -1)

        yield {
            feature_name: {
                "predictions": {
                    model_name: trained_model.predict(X) for model_name, trained_model in trained_models[feature_name].items()
                },
                "r2_scores": {
                    model_name: trained_model.score(X, (pos_mask if is_classifier(trained_model) else y_true))
                    for model_name, trained_model in trained_models[feature_name].items()
                },
            }
        }

        

# %% [markdown]
# #### Test Data

# %%
TEST_SEED = SEED * 2
TEST_LENGTH = LENGTH


# %%
# Generate the test system data to analyse

import pickle
import os

filename_test_data = os.path.join("outputs/white_noise", "__".join(map(str, [SEED, LENGTH, SAMPLES, TEST_SEED, TEST_LENGTH, TEST_SAMPLES])) + "__test_data.pkl")
if not os.path.exists(filename_test_data):
    with open(filename_test_data, "wb") as file:
        data_ = generate_trajectories(RANDOM_SEED=TEST_SEED, TS_LENGTH=TEST_LENGTH, CONTROL_PARAM_SAMPLES=TEST_SAMPLES)
        pickle.dump(data_, file)

with open(filename_test_data, "rb") as file:
    system_test_data = pickle.load(file)


# %%
# Remove relative scale (amplitude) as a feature that could be used in supevised learning

for system in system_test_data:
    trajectories = system_test_data[system]["trajectories"]
    trajectories = list(map(scale, trajectories))
    system_test_data[system]["trajectories"] = trajectories


# %%
# define utility functions

def get_column_mins(arr):
    """Find minimums of each (feature) column over its finite values."""
    arr_isinf = np.isinf(arr)
    return np.min(ma.masked_array(arr, mask=arr_isinf, fill_value=np.inf), axis=0)

def make_inf_column_finite(arr, col_mins=None):
    """Convert -inf and +inf to min finite value in each column."""

    if col_mins is None:
        # e.g. array contains features from training data
        col_mins = get_column_mins(arr)

    for row_idx in range(arr.shape[0]):
        for col_idx in range(arr.shape[1]):
            if np.isinf(arr[row_idx, col_idx]):
                arr[row_idx, col_idx] = col_mins[col_idx]

    arr[np.isnan(arr)] = -1e-12
    return arr

def get_scores_from_predictions(y_pred, y_true=None):
    """Compute f1 and negative mean squared error scores for predictions."""
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import f1_score
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    chaos = y_true > 0
    mse_all = mean_squared_error(y_true, y_pred)
    f1_all = f1_score(chaos, y_pred>0)
    spearmanr_all = stats.spearmanr(y_pred, y_true)[0]
    pearsonr_all = stats.pearsonr(y_pred, y_true)[0]

    results = {
        "Raw F1": f1_all,
        "Raw MSE": -mse_all,
        "Raw Spearman": spearmanr_all,
        "Raw Pearson": pearsonr_all,
    }

    return results


# %% [markdown]
# ## $\lambda_{\max}$ Estimator Pipeline

# %% [markdown]
# A generic pipeline to compute a set of features and apply them to predicting $\lambda_{\max}$.

# %%
# set up caching
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

# import the feature function and the list of names of features
from hvg_estimates import get_hvg_estimates, hvg_names
from tree_offset_divergence import get_offset_divergences_vec, div_names
from crocker_estimates import get_crocker_estimates, crocker_names
from point_summary_estimates import get_point_summary_estimates, point_summary_names
from classic_estimators import get_classic_estimates, classic_names

# %%
def feature_scoring(feature_func, feature_names, trajectories_train, trajectories_test, y_train, y_test, lbv, cache_key_info=None):

    # compute features for training and test data sets
    batch_size = 1000
    # if "0D Betti Norm" in feature_names:
    #     # vietoris-rips based estimates use a huge amount of memory
    #     batch_size = 50

    train_data_features = []
    for batch_start_idx in range(0, len(trajectories_train), batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, len(trajectories_train))
        train_data_features += list(lbv.map_sync(feature_func, trajectories_train[batch_start_idx:batch_end_idx]))
    train_data_features = np.array(train_data_features)

    test_data_features = []
    for batch_start_idx in range(0, len(trajectories_test), batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, len(trajectories_test))
        test_data_features += list(lbv.map_sync(feature_func, trajectories_test[batch_start_idx:batch_end_idx]))
    test_data_features = np.array(test_data_features)
    
    if "Rosenstein" in feature_names:
        # we may have non-finite values which will mess up the supervised models
        # they correspond to non-chaotic trajectories
        # so replace them with column minima (calculated from training data)
        col_mins = get_column_mins(train_data_features)
        train_data_features = make_inf_column_finite(train_data_features, col_mins=col_mins)
        test_data_features = make_inf_column_finite(test_data_features, col_mins=col_mins)

    # train the models and gather the results
    training_results = {}
    for result in score_features_train(feature_names, train_data_features, y_train):
        training_results |= result

    # extract scores and trained models
    training_scores = {k:v["scores"] for k,v in training_results.items()}
    trained_models = {k:v["models"] for k,v in training_results.items()}

    # average the scores for each feature and model over all cross validation runs
    training_scores_df = pd.DataFrame(training_scores)
    training_scores_df = training_scores_df.applymap(np.mean).T

    # apply the trained models to new data and gather the results
    test_results = {}
    for result in score_features_test(feature_names, test_data_features, y_test, trained_models):
            test_results |= result

    # extract scores, predictions, and correlations on the test data
    test_scores = {k:v["r2_scores"] for k,v in test_results.items()}
    test_predictions = {k:v["predictions"] for k,v in test_results.items()}
    test_correlations = {
        k: {
            "SVR Spearman": stats.spearmanr(v["SVR"], y_test)[0],
            "SVR Pearson": stats.pearsonr(v["SVR"], y_test)[0],
            "MLPR Spearman": stats.spearmanr(v["MLPR"], y_test)[0],
            "MLPR Pearson": stats.pearsonr(v["MLPR"], y_test)[0],
            "KNR Spearman": stats.spearmanr(v["KNR"], y_test)[0],
            "KNR Pearson": stats.pearsonr(v["KNR"], y_test)[0],
        }
        for k,v in test_predictions.items()
    }

    # get scores for each feature and model as a dataframe
    test_scores_df = pd.DataFrame(test_scores).T
    test_correlations_df = pd.DataFrame(test_correlations).T

    # also get correlations and scoring metrics for the raw feature values (when they are scalars)
    if test_data_features.ndim == 2:
        # features are single scalar values
        raw_scores = map(partial(get_scores_from_predictions, y_true=y_test), test_data_features.T)
        raw_scores_df = pd.DataFrame(raw_scores)
        raw_scores_df.index = feature_names
    else:
        raw_scores_df = pd.DataFrame()


    if test_data_features.ndim == 2:
        r2_features = test_data_features[...,np.newaxis]
    else:
        r2_features = test_data_features.copy()

    # fit an ordinary least squares model and correlate the predictions with y_true
    r2_scores = []
    n_samples, n_features, feature_vector_length = r2_features.shape
    for i in range(n_features):
        feature_name = feature_names[i]
        X = r2_features[:, i, :].reshape(n_samples, -1)
        clf = LinearRegression()
        clf.fit(X, y_test)
        y_pred = clf.predict(X)
        r2 = r2_score(y_test, y_pred)
        r2_adjusted = 1 - (1-r2)*(n_samples-1)/(n_samples-feature_vector_length-1)
        score = {
            "Raw R2": r2,
            "Raw R2 Adjusted": r2_adjusted,
        }
        r2_scores.append(score)
    r2_scores_df = pd.DataFrame(r2_scores)
    r2_scores_df.index = feature_names

    return training_scores_df, test_scores_df, test_correlations_df, raw_scores_df, r2_scores_df, test_data_features, test_predictions

feature_scoring = memory.cache(feature_scoring, ignore=["feature_func", "lbv"])



# %%

if BASELINES:
    classic_results = feature_scoring(
        get_classic_estimates,
        classic_names,
        system_training_data[SYSTEM]["trajectories"],
        system_test_data[SYSTEM]["trajectories"],
        system_training_data[SYSTEM]["lces"],
        system_test_data[SYSTEM]["lces"],
        lbv,
    )




# %%
if BASELINES:
    hvg_results = feature_scoring(
        get_hvg_estimates,
        hvg_names,
        system_training_data[SYSTEM]["trajectories"],
        system_test_data[SYSTEM]["trajectories"],
        system_training_data[SYSTEM]["lces"],
        system_test_data[SYSTEM]["lces"],
        lbv,
    )


# %%
if BASELINES:
    point_summary_results = feature_scoring(
        get_point_summary_estimates,
        point_summary_names,
        system_training_data[SYSTEM]["trajectories"],
        system_test_data[SYSTEM]["trajectories"],
        system_training_data[SYSTEM]["lces"],
        system_test_data[SYSTEM]["lces"],
        lbv,
    )

# %%
if BASELINES:
    crocker_results = feature_scoring(
        get_crocker_estimates,
        crocker_names,
        system_training_data[SYSTEM]["trajectories"],
        system_test_data[SYSTEM]["trajectories"],
        system_training_data[SYSTEM]["lces"],
        system_test_data[SYSTEM]["lces"],
        lbv,
    )

# %%
offsets=range(OFFSET_MIN, OFFSET_MAX, OFFSET_STEP)
get_offset_divergences_vec_func = partial(get_offset_divergences_vec, offsets=offsets, discrete=DISCRETE)

# %%

if not BASELINES:
    divergence_results = feature_scoring(
        get_offset_divergences_vec_func,
        div_names,
        system_training_data[SYSTEM]["trajectories"],
        system_test_data[SYSTEM]["trajectories"],
        system_training_data[SYSTEM]["lces"],
        system_test_data[SYSTEM]["lces"],
        lbv,
        cache_key_info={"offsets":offsets, "discrete":DISCRETE}
    )


# %%
if BASELINES:
    results_dfs = [classic_results, crocker_results, hvg_results, point_summary_results]
    results_dfs_names = ["Classic Neighbour Tracing", "Betti Vector Norms", "HVG Degree Distributions", "kNN and OPN Point Summaries"]
else:
    results_dfs = [divergence_results]
    if DISCRETE:
        results_dfs_names = ["Topological Divergences"]
    else:
        results_dfs_names = ["Topological Divergences (PL)"]

all_results = []
for dfs, name in zip(results_dfs, results_dfs_names):
    column_groups = ["Train Scores", "Test Scores", "Test Correlations", "1D Feature Scores", "R2 Feature Scores"]
    new_rows = pd.concat([*dfs[:5]], axis=1, keys=column_groups)
    all_results.append(pd.concat([new_rows], keys=[name]))
all_results_df = pd.concat(all_results)

# %%
if BASELINES:
    test_data_features = {
        "classic": classic_results[5],
        "crocker": crocker_results[5],
        "hvg": hvg_results[5],
        "point_summary": point_summary_results[5],
    }

    test_data_predictions = {
        "classic": classic_results[6],
        "crocker": crocker_results[6],
        "hvg": hvg_results[6],
        "point_summary": point_summary_results[6],
    }
else:
    test_data_features = {
        "divergence": divergence_results[5],
    }
    test_data_predictions = {
        "divergence": divergence_results[6],
    }

# %%
# save the results data for this system and set of trajectory parameters

import pickle

filename = "__".join(
    map(
        str,
        [
            SEED,
            SAMPLES,
            LENGTH,
            TEST_SEED,
            TEST_LENGTH,
            TEST_SAMPLES,
            SYSTEM,
            DISCRETE,
            OFFSET_MIN,
            OFFSET_MAX,
            OFFSET_STEP,
            BASELINES,
        ],
    )
)
with open(
    f"./outputs/data/white_noise/{filename}.pkl",
    "wb",
) as file:
    pickle.dump(
        {
            "results": all_results_df,
            "test_features": test_data_features,
            "test_predictions": test_data_predictions,
        },
        file,
    )


# %%
# all_results_df

# %%



