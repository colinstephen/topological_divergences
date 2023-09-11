# %% [markdown]
# # Topological Divergences of Time Series
# 
# Data analysis for time series generated by chaotic maps, over a range of control parameter values, using various topological divergences. Classical Lyapunov estimators and recent TDA/HVG measures are also computed as baselines. 

# %% [markdown]
# ## Set up parallel processing
# 
# Ensure cluster is running before executing the code below.
# 
# Start a cluster with 32 cores with `ipcluster start -n 32`.
# 
# Ensure cluster is stopped after code is complete.
# 
# Stop the cluster with `ipcluster stop` in a separate terminal.

# %%
import ipyparallel as ipp
clients = ipp.Client()
dv = clients.direct_view()
lbv = clients.load_balanced_view()

# %%
len(dv)

# %%
# clients.shutdown()

# %% [markdown]
# ## Import modules, classes, and functions

# %%
import sys
import pickle
import numpy as np
import networkx as nx

from dataclasses import dataclass
from dataclasses import asdict
from functools import partial
from datetime import datetime
from scipy import stats

from numpy.random import MT19937
from numpy.random import RandomState
from numpy.random import SeedSequence

from nolds import lyap_r
from nolds import lyap_e

from teaspoon.SP import network_tools
from teaspoon.SP.network import knn_graph
from teaspoon.SP.network_tools import remove_zeros
from teaspoon.SP.tsa_tools import takens
from teaspoon.TDA.PHN import PH_network
from teaspoon.TDA.PHN import point_summaries
from gtda.time_series import takens_embedding_optimal_parameters

from LogisticMapLCE import logistic_lce
from HenonMapLCE import henon_lce
from IkedaMapLCE import ikeda_lce
from TinkerbellMapLCE import tinkerbell_lce

from TimeSeriesHVG import TimeSeriesHVG as TSHVG
from TimeSeriesMergeTreeSimple import TimeSeriesMergeTree as TSMT
from TimeSeriesPersistence import TimeSeriesPersistence as TSPH

# %% [markdown]
# ## Configure the experiment data

# %%
# USE THIS WHEN RUNNING AN INTERACTIVE NOTEBOOK
SEED = 42
LENGTH = 200
SAMPLES = 50

# %%
# WARNING: ONLY RUN THIS VIA EXPORTED PYTHON SCRIPT
n_args = len(sys.argv)
SEED = int(sys.argv[1]) if n_args > 1 else 42
LENGTH = int(sys.argv[2]) if n_args > 2 else 200
SAMPLES = int(sys.argv[3]) if n_args > 3 else 50

print(f"Experiment config -- SEED:{SEED}, LENGTH:{LENGTH}, SAMPLES:{SAMPLES}")

# %%

class EXPERIMENT_CONFIG:
    SEED = SEED
    RANDOM_STATE = RandomState(MT19937(SeedSequence(SEED)))
    TIME_SERIES_LENGTH = LENGTH
    NUM_CONTROL_PARAM_SAMPLES = SAMPLES

# %% [markdown]
# ### Allow converting configurations to dictionaries for saving

# %%
def configdict(cls):
    """Given a configuration class, convert it and its properties to a dictionary."""
    attributes = {attr: getattr(cls, attr) for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")}
    return {cls.__name__: attributes}

# %% [markdown]
# ### Set up wrapper for easy saving of results

# %%
def save_result(filename, data, extra_metadata=None, RESULTS_DIR="outputs/data"):
    """Save arbitrary data/results for future reference."""

    # use the experimental config to identify the file
    filename_parts = [
        f"SEED_{EXPERIMENT_CONFIG.SEED}",
        f"LENGTH_{EXPERIMENT_CONFIG.TIME_SERIES_LENGTH}",
        f"SAMPLES_{EXPERIMENT_CONFIG.NUM_CONTROL_PARAM_SAMPLES}",
        filename,
        f"{datetime.utcnow()}",
    ]
    full_filename = "__".join(filename_parts)
    path = f"{RESULTS_DIR}/{full_filename}.pkl"

    if extra_metadata is None:
        extra_metadata = {}
    metadata = configdict(EXPERIMENT_CONFIG) | extra_metadata

    data_to_save = dict(
        metadata=metadata,
        data=data,
    )
    with open(path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Saved results: {path}")


# %% [markdown]
# ### Normalisation
# 
# We $z$-normalise all sequences. This helps ensure divergence/distance hyper-parameters (such as the merge tree interleaving distance mesh size) are working at the same scale across the data sets considered.

# %%
def z_normalise(ts: np.array) -> np.array:
    mean = np.mean(ts)
    std = np.std(ts)
    return (ts - mean) / std

# %% [markdown]
# ### Logistic

# %%
class LOGISTIC_CONFIG:
    R_MIN = 3.5
    R_MAX = 4.0
    TRAJECTORY_DIM = 0

logistic_control_params = [
    dict(r=r)
    for r in np.sort(
        EXPERIMENT_CONFIG.RANDOM_STATE.uniform(LOGISTIC_CONFIG.R_MIN, LOGISTIC_CONFIG.R_MAX, EXPERIMENT_CONFIG.NUM_CONTROL_PARAM_SAMPLES)
    )
]
logistic_dataset = [
    logistic_lce(mapParams=params, nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH, includeTrajectory=True)
    for params in logistic_control_params
]
logistic_trajectories = [
    z_normalise(data["trajectory"][:, LOGISTIC_CONFIG.TRAJECTORY_DIM])
    for data in logistic_dataset
]
logistic_lces = np.array([data["lce"][0] for data in logistic_dataset])


# %% [markdown]
# ### Hénon

# %%
class HENON_CONFIG:
    A_MIN = 0.8
    A_MAX = 1.4
    B = 0.3
    TRAJECTORY_DIM = 0

henon_control_params = [
    dict(a=a, b=HENON_CONFIG.B)
    for a in np.sort(
        EXPERIMENT_CONFIG.RANDOM_STATE.uniform(HENON_CONFIG.A_MIN, HENON_CONFIG.A_MAX, EXPERIMENT_CONFIG.NUM_CONTROL_PARAM_SAMPLES)
    )
]
henon_dataset = [
    henon_lce(mapParams=params, nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH, includeTrajectory=True)
    for params in henon_control_params
]
henon_trajectories = [
    z_normalise(data["trajectory"][:, HENON_CONFIG.TRAJECTORY_DIM]) for data in henon_dataset
]
henon_lces = np.array([data["lce"][0] for data in henon_dataset])


# %% [markdown]
# ### Ikeda

# %%
class IKEDA_CONFIG:
    A_MIN = 0.5
    A_MAX = 1.0
    TRAJECTORY_DIM = 0

ikeda_control_params = [
    dict(a=a)
    for a in np.sort(
        EXPERIMENT_CONFIG.RANDOM_STATE.uniform(IKEDA_CONFIG.A_MIN, IKEDA_CONFIG.A_MAX, EXPERIMENT_CONFIG.NUM_CONTROL_PARAM_SAMPLES)
    )
]
ikeda_dataset = [
    ikeda_lce(mapParams=params, nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH, includeTrajectory=True)
    for params in ikeda_control_params
]
ikeda_trajectories = [
    z_normalise(data["trajectory"][:, IKEDA_CONFIG.TRAJECTORY_DIM]) for data in ikeda_dataset
]
ikeda_lces = np.array([data["lce"][0] for data in ikeda_dataset])


# %% [markdown]
# ### Tinkerbell

# %%
class TINKERBELL_CONFIG:
    A_MIN = 0.7
    A_MAX = 0.9
    TRAJECTORY_DIM = 0

tinkerbell_control_params = [
    dict(a=a)
    for a in np.sort(
        EXPERIMENT_CONFIG.RANDOM_STATE.uniform(
            TINKERBELL_CONFIG.A_MIN, TINKERBELL_CONFIG.A_MAX, EXPERIMENT_CONFIG.NUM_CONTROL_PARAM_SAMPLES
        )
    )
]
tinkerbell_dataset = [
    tinkerbell_lce(
        mapParams=params, nIterates=EXPERIMENT_CONFIG.TIME_SERIES_LENGTH, includeTrajectory=True
    )
    for params in tinkerbell_control_params
]
tinkerbell_trajectories = [
    z_normalise(data["trajectory"][:, TINKERBELL_CONFIG.TRAJECTORY_DIM])
    for data in tinkerbell_dataset
]
tinkerbell_lces = np.array([data["lce"][0] for data in tinkerbell_dataset])


# %% [markdown]
# ### Simplify access to LCEs and configs

# %%
system_lces = {
    "logistic": logistic_lces,
    "henon": henon_lces,
    "ikeda": ikeda_lces,
    "tinkerbell": tinkerbell_lces
}

# %%
system_configs = {
    "logistic": LOGISTIC_CONFIG,
    "henon": HENON_CONFIG,
    "ikeda": IKEDA_CONFIG,
    "tinkerbell": TINKERBELL_CONFIG,
}

# %% [markdown]
# ## Build time series representations

# %%
def build_representation(dataset, rep_class, rep_class_kwargs):
    trajectories = [data["trajectory"][:,0] for data in dataset]
    return [rep_class(ts, **rep_class_kwargs) for ts in trajectories]

# %%
HVG_CONFIG = dict(
    DEGREE_DISTRIBUTION_MAX_DEGREE=100,
    DEGREE_DISTRIBUTION_DIVERGENCE_P_VALUE=1.0,
    directed=None,
    weighted=None,
    penetrable_limit=0,
)

# %%
MT_CONFIG = dict(MESHES=[0.5, 0.4, 0.3, 0.2, 0.1], THRESHES=[None])

DMT_CONFIG = MT_CONFIG | dict(discrete=True)

# %%
PH_CONFIG = dict(
    ENTROPY_SUMMARY_RESOLUTION=100,
    BETTI_CURVE_RESOLUTION=100,
    BETTI_CURVE_NORM_P_VALUE=1.0,
    SILHOUETTE_RESOLUTION=100,
    SILHOUETTE_WEIGHT=1,
    LIFESPAN_CURVE_RESOLUTION=100,
    IMAGE_BANDWIDTH=0.2,
    IMAGE_RESOLUTION=20,
    ENTROPY_SUMMARY_DIVERGENCE_P_VALUE=2.0,
    PERSISTENCE_STATISTICS_DIVERGENCE_P_VALUE=2.0,
    WASSERSTEIN_DIVERGENCE_P_VALUE=1.0,
    BETTI_CURVE_DIVERGENCE_P_VALUE=1.0,
    PERSISTENCE_SILHOUETTE_DIVERGENCE_P_VALUE=2.0,
    PERSISTENCE_LIFESPAN_DIVERGENCE_P_VALUE=2.0,
)

# %% [markdown]
# ### Logistic

# %%
logistic_tshvgs = build_representation(logistic_dataset, TSHVG, HVG_CONFIG)
logistic_tsmts = build_representation(logistic_dataset, TSMT, MT_CONFIG)
logistic_tsdmts = build_representation(logistic_dataset, TSMT, DMT_CONFIG)
logistic_tsphs = build_representation(logistic_dataset, TSPH, PH_CONFIG)

# %% [markdown]
# ### Hénon

# %%
henon_tshvgs = build_representation(henon_dataset, TSHVG, HVG_CONFIG)
henon_tsmts = build_representation(henon_dataset, TSMT, MT_CONFIG)
henon_tsdmts = build_representation(henon_dataset, TSMT, DMT_CONFIG)
henon_tsphs = build_representation(henon_dataset, TSPH, PH_CONFIG)

# %% [markdown]
# ### Ikeda

# %%
ikeda_tshvgs = build_representation(ikeda_dataset, TSHVG, HVG_CONFIG)
ikeda_tsmts = build_representation(ikeda_dataset, TSMT, MT_CONFIG)
ikeda_tsdmts = build_representation(ikeda_dataset, TSMT, DMT_CONFIG)
ikeda_tsphs = build_representation(ikeda_dataset, TSPH, PH_CONFIG)

# %% [markdown]
# ### Tinkerbell

# %%
tinkerbell_tshvgs = build_representation(tinkerbell_dataset, TSHVG, HVG_CONFIG)
tinkerbell_tsmts = build_representation(tinkerbell_dataset, TSMT, MT_CONFIG)
tinkerbell_tsdmts = build_representation(tinkerbell_dataset, TSMT, DMT_CONFIG)
tinkerbell_tsphs = build_representation(tinkerbell_dataset, TSPH, PH_CONFIG)

# %% [markdown]
# ### Simplify access to system representations and their configs

# %%
system_reps = {
    "logistic": {
        "hvg": logistic_tshvgs,
        "mt": logistic_tsmts,
        "dmt": logistic_tsdmts,
        "ph": logistic_tsphs,
    },
    "henon": {
        "hvg": henon_tshvgs,
        "mt": henon_tsmts,
        "dmt": henon_tsdmts,
        "ph": henon_tsphs,
    },
    "ikeda": {
        "hvg": ikeda_tshvgs,
        "mt": ikeda_tsmts,
        "dmt": ikeda_tsdmts,
        "ph": ikeda_tsphs,
    },
    "tinkerbell": {
        "hvg": tinkerbell_tshvgs,
        "mt": tinkerbell_tsmts,
        "dmt": tinkerbell_tsdmts,
        "ph": tinkerbell_tsphs,
    },
}


# %%
system_rep_configs = {
    "hvg": HVG_CONFIG,
    "mt": MT_CONFIG,
    "dmt": DMT_CONFIG,
    "ph": PH_CONFIG,
}

# %% [markdown]
# ## Compute topological divergences and correlations with LCEs

# %% [markdown]
# ### Lyapunov exponents (ground truth)
# 
# Calculated using numerical integration and the Benettin algorithm above.

# %% [markdown]
# ### Helper functions

# %%
def dict_of_arrays(list_of_dicts):
    """Convert list of dictionaries with equal keys to a dictionary of numpy arrays.
    
    Example
        Input
            [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        Output
            {'a': np.array([1, 3]), 'b': np.array([2, 4])}
    """
    return {key: np.array([d[key] for d in list_of_dicts]) for key in list_of_dicts[0]}


# %%

def topological_divergences(ts_representations):
    divergences = lbv.map_sync(lambda rep: rep.divergences, ts_representations)
    return dict_of_arrays(divergences)

# %%
def lyapunov_correlations(system_name, divergence_type, divergence_name, lces, divergences):
    # find out how well a given estimator correlates with the lyapunov exponent
    
    # eliminate spurious values
    divergence_nan_mask = ~np.isnan(divergences)
    lces = lces[divergence_nan_mask]

    divergences = divergences[divergence_nan_mask]
    positive_mask = np.array(lces) > 0

    result = dict(
        system_name = system_name,
        divergence_type = divergence_type,
        divergence_name = divergence_name,
        pearsonr_full = stats.pearsonr(lces, divergences)[0],
        spearmanr_full = stats.spearmanr(lces, divergences)[0],
        kendalltau_full = stats.kendalltau(lces, divergences)[0],
        pearsonr_pos = stats.pearsonr(lces[positive_mask], divergences[positive_mask])[0],
        spearmanr_pos = stats.spearmanr(lces[positive_mask], divergences[positive_mask])[0],
        kendalltau_pos = stats.kendalltau(lces[positive_mask], divergences[positive_mask])[0],
    )

    return result


# %%
def compute_and_save_divs_and_corrs(sys_name: str, div_type: str):
    reps = system_reps[sys_name][div_type]
    divs = topological_divergences(reps)
    sys_config = system_configs[sys_name]
    rep_config = system_rep_configs[div_type]
    lces = system_lces[sys_name]

    meta = dict()
    meta |= configdict(sys_config)
    meta |= {f"{div_type.upper()}_CONFIG": rep_config}

    save_result(f"{sys_name}_{div_type}_divergences", divs, extra_metadata=meta)

    correlations = dict()
    for div_name, div in divs.items():
        correlations[sys_name, div_type, div_name] = lyapunov_correlations(
            sys_name, div_type, div_name, lces, div
        )

    save_result(
        f"{sys_name}_{div_type}_correlations", correlations, extra_metadata=meta
    )


# %% [markdown]
# ### HVG divergences
# 
# Wasserstein and $L_p$ divergences of the time series HVG degree distributions.

# %%
# compute_and_save_divs_and_corrs("logistic", "hvg")
# compute_and_save_divs_and_corrs("ikeda", "hvg")
# compute_and_save_divs_and_corrs("tinkerbell", "hvg")
# compute_and_save_divs_and_corrs("henon", "hvg")

# %% [markdown]
# ### Persistent homology divergences
# 
# Various divergences based on the superlevel and sublevel persistence diagrams.

# %%
# compute_and_save_divs_and_corrs("logistic", "ph")
# compute_and_save_divs_and_corrs("ikeda", "ph")
# compute_and_save_divs_and_corrs("tinkerbell", "ph")
# compute_and_save_divs_and_corrs("henon", "ph")

# %% [markdown]
# ### Merge tree divergences
# 
# Interleaving divergence and leaf-to-offset-leaf path length distribution divergence.

# %%
compute_and_save_divs_and_corrs("logistic", "mt")
compute_and_save_divs_and_corrs("ikeda", "mt")
compute_and_save_divs_and_corrs("tinkerbell", "mt")
compute_and_save_divs_and_corrs("henon", "mt")

# %% [markdown]
# #### DISCRETE merge tree versions

# %%
# compute_and_save_divs_and_corrs("logistic", "dmt")
# compute_and_save_divs_and_corrs("ikeda", "dmt")
# compute_and_save_divs_and_corrs("tinkerbell", "dmt")
# compute_and_save_divs_and_corrs("henon", "dmt")
# test = []
# for i, tsdmt in enumerate(ikeda_tsdmts):
#     div = tsdmt.cophenetic_matrix_entropy_divergence
#     test.append(div)
# print(test)

# %% [markdown]
# ### Correlations between divergences and Lyapunov exponents

# %% [markdown]
# ### Plots and visualisations

# %% [markdown]
# def plot_lce_and_topo_divergence(lces_to_plot, mt_div_to_plot):
#     from matplotlib import pyplot as plt
#     no_mask = lces_to_plot > -np.inf
#     positive_mask = lces_to_plot > 0
#     
#     pearsonr_full = stats.pearsonr(lces_to_plot, mt_div_to_plot)[0]
#     spearmanr_full = stats.spearmanr(lces_to_plot, mt_div_to_plot)[0]
#     kendalltau_full = stats.kendalltau(lces_to_plot, mt_div_to_plot)[0]
#     pearsonr_pos = stats.pearsonr(lces_to_plot[positive_mask], mt_div_to_plot[positive_mask])[0]
#     spearmanr_pos = stats.spearmanr(lces_to_plot[positive_mask], mt_div_to_plot[positive_mask])[0]
#     kendalltau_pos = stats.kendalltau(lces_to_plot[positive_mask], mt_div_to_plot[positive_mask])[0]
# 
#     mask = no_mask
#     plt.plot(lces_to_plot, lw=0.7)
#     plt.show()
#     plt.plot(mt_div_to_plot[mask], lw=0.7)
#     plt.show()
#     plt.scatter(lces_to_plot[mask], mt_div_to_plot[mask], s=1)
#     plt.title(f"P=({pearsonr_full}, {pearsonr_pos}),\n S=({spearmanr_full}, {spearmanr_pos}),\n T=({kendalltau_full}, {kendalltau_pos})")
#     plt.show()
# 
# plot_lce_and_topo_divergence(logistic_lces, logistic_hvg_divergences["degree_wasserstein"])
# plot_lce_and_topo_divergence(henon_lces, henon_hvg_divergences["degree_wasserstein"])
# plot_lce_and_topo_divergence(ikeda_lces, ikeda_hvg_divergences["degree_wasserstein"])
# plot_lce_and_topo_divergence(tinkerbell_lces, tinkerbell_hvg_divergences["degree_wasserstein"])
# 
# plot_lce_and_topo_divergence(logistic_lces, logistic_mt_divergences["interleaving"])
# plot_lce_and_topo_divergence(henon_lces, henon_mt_divergences["interleaving"])
# plot_lce_and_topo_divergence(ikeda_lces, ikeda_mt_divergences["interleaving"])
# plot_lce_and_topo_divergence(tinkerbell_lces, tinkerbell_mt_divergences["interleaving"])
# 
# plot_lce_and_topo_divergence(logistic_lces, logistic_mt_divergences["leaf_to_leaf_path_length"][:,0])
# plot_lce_and_topo_divergence(henon_lces, henon_mt_divergences["leaf_to_leaf_path_length"][:,0])
# plot_lce_and_topo_divergence(ikeda_lces, ikeda_mt_divergences["leaf_to_leaf_path_length"][:,0])
# plot_lce_and_topo_divergence(tinkerbell_lces, tinkerbell_mt_divergences["leaf_to_leaf_path_length"][:,0])

# %% [markdown]
# ## Baselines
# 
# Other measures that might approximate or estimate the largest Lyapunov exponent of the trajectory data.

# %% [markdown]
# ### Classical measures
# 
# The Rosenstein and Eckmann estimates from Python `nolds`.

# %% [markdown]
# logistic_rosenstein_estimates = np.array(dv.map_sync(lyap_r, logistic_trajectories))
# henon_rosenstein_estimates = np.array(dv.map_sync(lyap_r, henon_trajectories))
# ikeda_rosenstein_estimates = np.array(dv.map_sync(lyap_r, ikeda_trajectories))
# tinkerbell_rosenstein_estimates = np.array(dv.map_sync(lyap_r, tinkerbell_trajectories))
# 

# %% [markdown]
# logistic_rosenstein_estimates

# %% [markdown]
# logistic_eckmann_estimates = np.array([x[0] for x in dv.map_sync(lyap_e, logistic_trajectories)])
# henon_eckmann_estimates = np.array([x[0] for x in dv.map_sync(lyap_e, henon_trajectories)])
# ikeda_eckmann_estimates = np.array([x[0] for x in dv.map_sync(lyap_e, ikeda_trajectories)])
# tinkerbell_eckmann_estimates = np.array([x[0] for x in dv.map_sync(lyap_e, tinkerbell_trajectories)])
# 

# %% [markdown]
# logistic_eckmann_estimates

# %% [markdown]
# ### HVG-based measures
# 
# The $L_1$ distance between degree distributions of top and bottom HVGs. See Hasson _et. al._ (2018).
# 
# This is already computed above as `logistic_hvg_divergences["degree_lp"]`.

# %% [markdown]
# ### TDA-based measures
# 
# 1. The point summary statistics of persistent homology of kNN graphs of Takens embeddings of time series. See Myers _et. al._ (2019).
# 2. The norm of Betti curves of the Vietorix Rips persistence on the full n-dimensional state space trajectory. See Güzel _et. al._ (2022). 

# %% [markdown]
# ### Helper functions

# %% [markdown]
# def DistanceMatrixFixed(A):
#     """Get the all-pairs unweighted shortest path lengths in the graph A.
# 
#     Fixes an issue in the `teaspoon` library such that distance matrix computation
#     fails with disconnected graphs A.
#     """
# 
#     A = network_tools.remove_zeros(A)
#     np.fill_diagonal(A, 0)
#     A = A + A.T
# 
#     A_sp = np.copy(A)
#     N = len(A_sp)
#     D = np.zeros((N,N))
# 
#     A_sp[A_sp > 0] = 1
#     G = nx.from_numpy_matrix(A_sp)
#     lengths = dict(nx.all_pairs_shortest_path_length(G))    
#     for i in range(N-1):
#         for j in range(i+1, N):
#             D[i][j] = lengths.get(i, {}).get(j, np.inf)
#     D = D + D.T
#     return D
# 

# %% [markdown]
# ### $k$-NN persistence point summaries

# %% [markdown]
# EMBED_MAX_DIM = 5
# EMBED_MAX_DELAY = 40
# 
# logistic_optimal_embedding_params = [
#     takens_embedding_optimal_parameters(
#         ts, max_dimension=EMBED_MAX_DIM, max_time_delay=EMBED_MAX_DELAY
#     )
#     for ts in logistic_trajectories
# ]
# henon_optimal_embedding_params = [
#     takens_embedding_optimal_parameters(
#         ts, max_dimension=EMBED_MAX_DIM, max_time_delay=EMBED_MAX_DELAY
#     )
#     for ts in henon_trajectories
# ]
# ikeda_optimal_embedding_params = [
#     takens_embedding_optimal_parameters(
#         ts, max_dimension=EMBED_MAX_DIM, max_time_delay=EMBED_MAX_DELAY
#     )
#     for ts in ikeda_trajectories
# ]
# tinkerbell_optimal_embedding_params = [
#     takens_embedding_optimal_parameters(
#         ts, max_dimension=EMBED_MAX_DIM, max_time_delay=EMBED_MAX_DELAY
#     )
#     for ts in tinkerbell_trajectories
# ]
# 

# %% [markdown]
# logistic_embeddings = [
#     takens(ts, n=dim, tau=tau)
#     for ts, (tau, dim) in zip(logistic_trajectories, logistic_optimal_embedding_params)
# ]
# henon_embeddings = [
#     takens(ts, n=dim, tau=tau)
#     for ts, (tau, dim) in zip(henon_trajectories, henon_optimal_embedding_params)
# ]
# ikeda_embeddings = [
#     takens(ts, n=dim, tau=tau)
#     for ts, (tau, dim) in zip(ikeda_trajectories, ikeda_optimal_embedding_params)
# ]
# tinkerbell_embeddings = [
#     takens(ts, n=dim, tau=tau)
#     for ts, (tau, dim) in zip(
#         tinkerbell_trajectories, tinkerbell_optimal_embedding_params
#     )
# ]
# 

# %% [markdown]
# K_NEIGHBOURS = 4
# 
# knn_graph_parallel = partial(knn_graph, k=K_NEIGHBOURS)
# 
# logistic_knn_graphs = dv.map_sync(knn_graph_parallel, logistic_trajectories)
# henon_knn_graphs = dv.map_sync(knn_graph_parallel, henon_trajectories)
# ikeda_knn_graphs = dv.map_sync(knn_graph_parallel, ikeda_trajectories)
# tinkerbell_knn_graphs = dv.map_sync(knn_graph_parallel, tinkerbell_trajectories)

# %% [markdown]
# logistic_knn_graphs = list(map(remove_zeros, logistic_knn_graphs))
# henon_knn_graphs = list(map(remove_zeros, henon_knn_graphs))
# ikeda_knn_graphs = list(map(remove_zeros, ikeda_knn_graphs))
# tinkerbell_knn_graphs = list(map(remove_zeros, tinkerbell_knn_graphs))

# %% [markdown]
# def distance_matrix_fixed(A):
#     """Get the all-pairs unweighted shortest path lengths in the graph A.
# 
#     Fixes an issue in the `teaspoon` library such that distance matrix computation
#     fails with disconnected graphs A.
#     """
# 
#     # include imports so function can run in ipyparallel workers
#     import numpy as np
#     import networkx as nx
#     from teaspoon.SP.network_tools import remove_zeros
# 
#     A = remove_zeros(A)
#     np.fill_diagonal(A, 0)
#     A = A + A.T
# 
#     A_sp = np.copy(A)
#     N = len(A_sp)
#     D = np.zeros((N,N))
# 
#     A_sp[A_sp > 0] = 1
#     G = nx.from_numpy_matrix(A_sp)
#     lengths = dict(nx.all_pairs_shortest_path_length(G))    
#     for i in range(N-1):
#         for j in range(i+1, N):
#             D[i][j] = lengths.get(i, {}).get(j, np.inf)
#     D = D + D.T
#     return D
# 

# %% [markdown]
# logistic_distance_matrices = dv.map_sync(distance_matrix_fixed, logistic_knn_graphs)
# henon_distance_matrices = dv.map_sync(distance_matrix_fixed, henon_knn_graphs)
# ikeda_distance_matrices = dv.map_sync(distance_matrix_fixed, ikeda_knn_graphs)
# tinkerbell_distance_matrices = dv.map_sync(distance_matrix_fixed, tinkerbell_knn_graphs)

# %% [markdown]
# logistic_knn_diagrams = dv.map_sync(PH_network, logistic_distance_matrices)
# henon_knn_diagrams = dv.map_sync(PH_network, henon_distance_matrices)
# ikeda_knn_diagrams = dv.map_sync(PH_network, ikeda_distance_matrices)
# tinkerbell_knn_diagrams = dv.map_sync(PH_network, tinkerbell_distance_matrices)

# %% [markdown]
# logistic_knn_stats = dv.map_sync(point_summaries, logistic_knn_diagrams, logistic_knn_graphs)
# henon_knn_stats = dv.map_sync(point_summaries, henon_knn_diagrams, henon_knn_graphs)
# ikeda_knn_stats = dv.map_sync(point_summaries, ikeda_knn_diagrams, ikeda_knn_graphs)
# tinkerbell_knn_stats = dv.map_sync(point_summaries, tinkerbell_knn_diagrams, tinkerbell_knn_graphs)

# %% [markdown]
# logistic_knn_stats

# %% [markdown]
# 


