"""
High level helper function to build PD, merge tree, and HVG based divergence functions. 
"""


from tsph import hvg_degree_distribution, hvg_from_time_series, lp_distance, merge_tree_from_time_series_dmt


import numpy as np
from dtaidistance import dtw as dtw_distance
from gudhi import bottleneck_distance
from gudhi.wasserstein import wasserstein_distance
from scipy.spatial.distance import cosine as cosine_distance, euclidean as euclidean_distance
from scipy.stats import wasserstein_distance as wasserstein_distance_for_1d_distributions


def tda_divergence(
    time_series,
    sublevel_rep_func: str = None,
    sublevel_rep_params: dict = None,
    sublevel_rep_postprocess: list = None,
    sublevel_rep_vectorisation: str = None,
    sublevel_rep_vectorisation_params: dict = None,
    superlevel_rep_func: str = None,
    superlevel_rep_params: dict = None,
    superlevel_rep_postprocess: list = None,
    superlevel_rep_vectorisation: str = None,
    superlevel_rep_vectorisation_params: dict = None,
    distance_func: str = None,
    distance_params: dict = None,
):
    """
    Helper function. Use distance func to compare sub vs super levelset filtrations of a time series.

    Parameters
    ----------
    time_series : array
        The time series to analyse
    sublevel_rep : str
        Name of the sublevel set topology representation of the time series.
    sublevel_rep_params : dict
        Parameters for the sublevel set representation function.
    sublevel_rep_postprocess : list
        List of post-processing steps to apply to sublevel set representation before vectorisation.
    sublevel_rep_vectorisation : str
        Name of the vectorisation to apply to the representation.
    sublevel_rep_vectorisation_params : dict
        Parameters for the sublevel representation vectorisation.
    superlevel_rep : str
        Name of the superlevel set topology representation of the time series.
    superlevel_rep_params : dict
        Parameters for the superlevel set representation function.
    superlevel_rep_postprocess : list
        List of post-processing steps to apply to superlevel set representation before vectorisation.
    superlevel_rep_vectorisation : str
        Name of the vectorisation to apply to the representation.
    superlevel_rep_vectorisation_params : dict
        Parameters for the superlevel representation vectorisation.
    distance_func : str
        Name of the distance function to apply between the representations.
    distance_params : dict
        Parameters for the distance function.

    Notes
    -----
    1. Generally the sublevel and superlevel representations should be the same.
    2. Generally the sublevel and superlevel vectorisations should be the same.
    3. The distance function should be compatible with the chosen vectorisation.
    """

    time_series = np.array(time_series)

    representations = {
        "none": lambda x: x,
        "persistence_diagram": persistence_diagram_from_time_series,
        "merge_tree": merge_tree_from_time_series_dmt,
        "horizontal_visibility_graph": hvg_from_time_series,
    }

    if sublevel_rep_func is None:
        sublevel_rep_func = "none"

    if superlevel_rep_func is None:
        superlevel_rep_func = "none"

    assert (
        sublevel_rep_func in representations.keys()
    ), "Invalid time series sublevel representation."
    assert (
        superlevel_rep_func in representations.keys()
    ), "Invalid time series superlevel representation."

    if sublevel_rep_params is None:
        sublevel_rep_params = dict()

    if superlevel_rep_params is None:
        superlevel_rep_params = dict()

    postprocessing_chain = {
        "none": [],
        "flip_persistence": [flip_super_and_sub_level_persistence_points],
    }

    if sublevel_rep_postprocess is None:
        sublevel_rep_postprocess = "none"

    if superlevel_rep_postprocess is None:
        superlevel_rep_postprocess = "none"

    assert (
        sublevel_rep_postprocess in postprocessing_chain.keys()
    ), "Invalid sublevel postprocessing chain"
    assert (
        superlevel_rep_postprocess in postprocessing_chain.keys()
    ), "Invalid superlevel postprocessing chain"

    vectorisations = {
        "none": lambda x: x,
        "pd_statistics": persistence_statistics_vector,
        "pd_entropy": entropy_summary_function,
        "pd_betti": betti_curve_function,
        "pd_silhouette": persistence_silhouette_function,
        "pd_lifespan": persistence_lifespan_curve_function,
        "pd_image": persistence_image,
        "hvg_degree_distribution": hvg_degree_distribution,
    }

    if sublevel_rep_vectorisation is None:
        sublevel_rep_vectorisation = "none"

    if superlevel_rep_vectorisation is None:
        superlevel_rep_vectorisation = "none"

    assert (
        sublevel_rep_vectorisation in vectorisations.keys()
    ), "Invalid sublevel vectorisation"
    assert (
        superlevel_rep_vectorisation in vectorisations.keys()
    ), "Invalid superlevel vectorisation"

    if sublevel_rep_vectorisation_params is None:
        sublevel_rep_vectorisation_params = dict()

    if superlevel_rep_vectorisation_params is None:
        superlevel_rep_vectorisation_params = dict()

    distances = {
        "l2_dist": euclidean_distance,
        "lp_dist": lp_distance,
        "dtw_dist": dtw_distance,
        "cosine_dist": cosine_distance,
        "bottleneck_dist": bottleneck_distance,
        "wasserstein_dist": wasserstein_distance,
        "wasserstein_distance_for_1d_distributions": wasserstein_distance_for_1d_distributions,
    }

    assert distance_func in distances.keys(), "Invalid distance function"

    if distance_params is None:
        distance_params = dict()

    sub_rep = representations[sublevel_rep_func](time_series, **sublevel_rep_params)
    super_rep = representations[superlevel_rep_func](
        time_series, **superlevel_rep_params
    )

    for sublevel_postprocess_func in postprocessing_chain[sublevel_rep_postprocess]:
        sub_rep = sublevel_postprocess_func(sub_rep)

    for superlevel_postprocess_func in postprocessing_chain[superlevel_rep_postprocess]:
        super_rep = superlevel_postprocess_func(super_rep)

    sub_vec = vectorisations[sublevel_rep_vectorisation](
        sub_rep, **sublevel_rep_vectorisation_params
    )
    super_vec = vectorisations[superlevel_rep_vectorisation](
        super_rep, **superlevel_rep_vectorisation_params
    )

    dist = distances[distance_func](sub_vec, super_vec, **distance_params)

    return dist