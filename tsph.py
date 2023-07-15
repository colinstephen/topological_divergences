import ts2vg
import higra as hg
import numpy as np
import gudhi as gd
import networkx as nx
# import gudhi.wasserstein
import matplotlib.pyplot as plt
import vectorization as vec
from scipy.cluster import hierarchy
from decorated_merge_trees.DMT_tools import MergeTree

# Avoid namespace issues in Jupyter notebooks when using `from tsph import *`
__all__ = [
    "flip_super_and_sub_level_persistence_points",
    "logistic_map",
    "lyapunov_approximation_for_logistic_map",
    "merge_tree_from_time_series_dmt",
    "merge_tree_from_time_series_higra",
    "persistence_diagram_from_time_series",
    "plot_extended_persistence_diagrams",
    "plot_merge_tree_as_dendrogram",
    "plot_merge_tree_as_dendrogram_scipy",
    "plot_merge_tree_as_graph",
    "plot_persistence_diagram",
    "plot_time_series",
    "white_noise",
]

############################
## Time series generators ##
############################


def logistic_map(r, x0, n_iterations, skip_iterations=10000):
    """
    Generates a logistic map time series.

    Parameters
    ----------
    r : float
        Parameter r representing the growth rate where 0.0 < r <= 4.0
    x0 : float
        Initial value where 0 < x0 < 1
    n_iterations : int
        Number of iterations in the returned sequence.
    skip_iterations : int, optional, default=10000
        Number of iterations to ignore for burn in.

    Returns
    -------
    np.array
        A NumPy array containing the generated time series.
    """
    if not (0 < r <= 4.0):
        raise ValueError("Parameter r out of range")

    if not (0 < x0 < 1):
        raise ValueError("Initial value x0 out of range")

    # initialise value of map
    x = x0

    def apply_map(x):
        return r * x * (1 - x)

    # ignore burn in iterations
    for i in range(skip_iterations):
        x = apply_map(x)

    # initialise an array to return
    time_series = np.zeros(n_iterations)

    # generate the values of the map
    time_series[0] = x
    for i in range(1, n_iterations):
        time_series[i] = apply_map(time_series[i - 1])

    return time_series


def white_noise(length, mean=0, std_dev=1):
    """
    Generates a sequence of white noise samples of specified length.

    Parameters
    ----------
    length : int
        The length of the white noise sequence.
    mean : float, optional
        The mean of the white noise distribution (default: 0).
    std_dev : float, optional
        The standard deviation of the white noise distribution (default: 1).

    Returns
    -------
    numpy.ndarray
        A numpy array of white noise samples.
    """

    return np.random.normal(mean, std_dev, length)


##################################
## Lyapunov exponent estimation ##
##################################


def lyapunov_approximation_for_logistic_map(r_values, x0=0.5, n_iterations=10000, skip_iterations=1000):
    """
    Approximate the largest Lyapunov exponent of the Logistic map for each r value provided.

    Parameters
    ----------
    r_values : array
        Array of values for the control parameter `r` in the map `f(x) = r * x * (1 - x)`
    x0 : float
        Initial value for the trajectory
    n_iterations : int
        Number of iterations of the map with which to compute the approximation
    skip_iterations : int
        Number of initial iterations of the map to ignore before beginning `n_iterations`

    Returns
    -------
    array
        The approximate largest Lyapunov exponent values for each of the input `r_values`
    """

    # Ensure we can apply array-wise operations
    r_values = np.array(r_values)
    
    # Initialize all trajectories with the starting value
    x = x0 * np.ones(r_values.shape)

    # Discard the transient on all trajectories
    for _ in range(skip_iterations):
        x = r_values * x * (1 - x)

    # Then iterate n times and compute the sum for the Lyapunov exponent
    lyapunov_exp = np.zeros(r_values.shape)
    for _ in range(n_iterations):
        # update all trajectories
        x = r_values * x * (1 - x)
        # update the exponent approximation for each trajectory
        lyapunov_exp += np.log(abs(r_values - 2 * r_values * x))

    # Average over the number of iterations to get the final Lyapunov exponent
    lyapunov_exp /= n_iterations

    return lyapunov_exp


#####################################################
## Persistent homology and merge tree computations ##
#####################################################


def _sublevel_set_filtration(time_series):
    """
    Helper function. Creates the sublevel set filtration for the given time series.

    Parameters
    ----------
    time_series : array_like
        A list or numpy array of the input time series.

    Returns
    -------
    list
        The filtration: a list of (simplex, value) pairs representing the simplices in the sublevel set filtration.
    """

    filtration = []
    n = len(time_series)

    for i in range(n):
        filtration.append(([i], time_series[i]))

        if i < n - 1:
            edge_value = max(time_series[i], time_series[i + 1])
            filtration.append(([i, i + 1], edge_value))

    return filtration


def _superlevel_set_filtration(time_series):
    """
    Helper function. Creates the superlevel set filtration for the given time series.

    Parameters
    ----------
    time_series : array_like
        A list or numpy array of the input time series.

    Returns
    -------
    list
        The filtration: a list of (simplex, value) pairs representing the superlevel set filtration.

    Notes
    -----
    The values of the simplices are the negations of the original time series values, to ensure they are increasing over the filtration.
    """

    # invert the sequence (make peaks into pits and vice versa)
    time_series = -1 * np.array(time_series)

    # apply the sublevel algorithm to the inverted sequence
    return _sublevel_set_filtration(time_series)


def _persistent_homology_simplex_tree(filtration):
    """
    Helper function. Construct a Gudhi simplex tree representing the given filtration.

    Parameters
    ----------
    filtration : list
        List of (simplex, value) tuples in the filtration

    Returns
    -------
    gudhi.SimplexTree
        A Gudhi SimplexTree object
    """

    # Create a simplex tree object
    st = gd.SimplexTree()

    # Insert the filtration into the simplex tree
    for simplex, value in filtration:
        st.insert(simplex, value)

    return st


def _persistence_diagram_from_simplex_tree(
    simplex_tree, dimension=0, superlevel_filtration=False
):
    """
    Helper function. Construct the persistent homology diagram induced by the given simplex tree.

    Parameters
    ----------
    simplex_tree : a Gudhi SimplexTree object
    dimension : integer, optional
        The dimension of the homology features to be returned.
    superlevel_filtration : Boolean, optional
        Does the simplex tree arise from a superlevel set filtration?
        Default is `False` implying we have a sublevel set simplex tree.

    Returns
    -------
    list
        The persistence diagram: a list of the persistent homology (birth, death) pairs in the given dimension.

    Notes
    -----
    1. When superlevel set adjustment is applied by setting `superlevel_filtration=True`, the intervals returned are (-birth, -death) rather than (birth, death). This is to account for decreasing function values used when building a superlevel set filtration, and the assumption that a simplex tree corresponding to a superlevel set filtration on `f(x)` has been built using the sublevel set filtration of the negated function `-f(x)`. Note that in this situation critical values will satisfy `death<birth`.
    2. The infinite point (corresponding to the global minimum) is removed from the resulting set of points, to ensure downstream functions such as entropy always have finite data to work with.
    """

    if not (dimension >= 0):
        raise ValueError("Requested homology dimension out of range")

    simplex_tree.compute_persistence()
    persistence_diagram = simplex_tree.persistence_intervals_in_dimension(dimension)

    # remove nonfinite points
    persistence_diagram = [
        (b, d) for (b, d) in persistence_diagram if np.isfinite(b) and np.isfinite(d)
    ]

    # reflip the axes if the filtration was built with superlevels
    if superlevel_filtration:
        persistence_diagram = [(-b, -d) for (b, d) in persistence_diagram]

    return persistence_diagram


def flip_super_and_sub_level_persistence_points(persistence_diagram):
    """
    Reflect the points in a persistence diagram across the birth=death diagonal.

    Parameters
    ----------
    persistence_diagram : list
        A list of the persistent homology (birth, death) intervals in the given dimension.

    Returns
    -------
    list
        A list of the persistent homology (death, birth) intervals in the given dimension.

    Notes
    -----
    Use this function to make superlevel and sublevel set persistence diagrams comparable using standard persistence diagram metrics.
    """

    return [(d, b) for (b, d) in persistence_diagram]


def persistence_diagram_from_time_series(time_series, superlevel_filtration=False):
    """
    Compute the persistence diagram of the piecewise linear interpolation of a discrete time series.

    Parameters
    ----------
    time_series : array_like
        The discrete time series over which to compute persistent homology.
    superlevel_filtration : boolean, optional
        Is this diagram for a superlevel set filtration? Default: sublevel set filtration.
    """

    filter_function = (
        _superlevel_set_filtration
        if superlevel_filtration
        else _sublevel_set_filtration
    )
    filtration = filter_function(time_series)
    simplex_tree = _persistent_homology_simplex_tree(filtration)
    persistence_diagram = _persistence_diagram_from_simplex_tree(
        simplex_tree, superlevel_filtration=superlevel_filtration
    )

    return persistence_diagram


############################################
## Vectorisations of persistence diagrams ##
############################################


def persistence_statistics_vector(persistence_diagram):
    return vec.GetPersStats(persistence_diagram)


def entropy_summary_function(persistence_diagram, resolution=100):
    return vec.GetEntropySummary(persistence_diagram, res=resolution)


def betti_curve_function(persistence_diagram, resolution=100):
    return vec.GetBettiCurveFeature(persistence_diagram, res=resolution)


def persistence_silhouette_function(persistence_diagram, resolution=100, weight_factor=1):
    return vec.GetPersSilhouetteFeature(persistence_diagram, res=resolution, w=weight_factor)


def persistence_lifespan_curve_function(persistence_diagram, resolution=100):
    return vec.GetPersLifespanFeature(persistence_diagram, res=resolution)


def persistence_image(persistence_diagram, bandwidth=0.2, resolution=20):
    return vec.GetPersImageFeature(persistence_diagram, bw=bandwidth, r=resolution)


#################################
## Graph-based representations ##
#################################


def _chain_graph(number_of_vertices):
    """
    Helper function. Build a simple chain graph of the given length.

    Parameters
    ----------
    number_of_vertices : int
        How long the resulting chain should be.

    Returns
    -------
    higra.UndirectedGraph
        A Higra graph object representing the chain.
    """

    graph = hg.UndirectedGraph(number_of_vertices)
    vertices = np.arange(number_of_vertices)
    graph.add_edges(vertices[:-1], vertices[1:])

    return graph


def _chain_graph_from_time_series(time_series):
    """
    Helper function. Generate a chain graph representation of a time series.

    Parameters
    ----------
    time_series : array_like
        Sequence of values in the time series.

    Returns
    -------
    tuple : (chain_graph, vertex_weights)
        A Higra graph object and an array of its vertex weights.

    Summary
    -------
    Given a discrete time series `T` of length `|T|=N`, generate a chain graph whose nodes `{0, 1, ..., N-1}` are labelled with the time series values `T[0], T[1], ..., T[N-1]`.
    """

    return _chain_graph(len(time_series)), np.array(time_series)


################################
## Tree-based representations ##
################################


def _higra_component_tree_2_merge_tree(tree, altitudes):
    """
    Helper function. Convert a Higra component tree to a Persistent Homology merge tree.

    Parameters
    ----------
    tree : higra.Tree
        The Higra component tree to be converted
    altitudes : array
        The altitudes of nodes in the tree

    Returns
    -------
    tuple : (higra.Tree, array)
        The persistent homology merge tree and its node altitudes
    """

    # prune then smooth the incoming component tree (NB: order of these operations matters)
    tree, altitudes = _remove_redundant_leaves(tree, altitudes)
    tree, altitudes = _remove_degree_two_vertices(tree, altitudes)

    return tree, altitudes


def _remove_degree_two_vertices(tree, altitudes):
    """
    Helper function. Simplify a tree by removing any internal degree-two vertices.

    Parameters
    ----------
    tree : higra.Tree
        The raw tree to be simplified
    altitudes : array
        The altitudes of nodes in the tree

    Returns
    -------
    tuple : (higra.Tree, array)
        The simplified tree and remaining altitudes
    """

    degrees = np.array([tree.degree(v) for v in tree.vertices()])
    filter = np.array(degrees == 2)
    smoothed_tree, node_map = hg.simplify_tree(tree, filter)
    smoothed_tree_altitudes = altitudes[node_map]

    return smoothed_tree, smoothed_tree_altitudes


def _remove_redundant_leaves(tree, altitudes):
    """
    Helper function. Simplify a tree by removing any leaves at the same altitude as their parent.

    Parameters
    ----------
    tree : higra.Tree
        The raw tree to be simplified
    altitudes : array
        The altitudes of the nodes in the tree

    Returns
    -------
    tuple : (higra.Tree, altitudes)
        The simplified tree and remaining altitudes
    """

    # compute outgoing edge weights in the tree based on altitudes
    weights = altitudes - altitudes[tree.parents()]
    
    # create a filter based on whether an outgoing weight is zero
    filter = np.array(weights == 0)

    # use the filter in the Higra simplify method
    pruned_tree, node_map = hg.simplify_tree(tree, filter, process_leaves=True)

    # use the resulting node map to the original tree to get the altitudes
    pruned_tree_altitudes = altitudes[node_map]

    return pruned_tree, pruned_tree_altitudes


def _higra_merge_tree_2_dmt_merge_tree(hg_merge_tree, altitudes):
    """
    Helper function. Convert a Higra merge tree to a `DMT_tools` merge tree.
    
    Parameters
    ----------
    hg_merge_tree : higra.Tree
        The tree to be converted for use by the decorated merge tree code.
    altitudes : array
        Heights of the nodes in the input tree.

    Returns
    -------
    DMT_tools.MergeTree
        A `networkx`-based version of the merge tree suitable for use in `DMT_tools` functions.
    """

    hg_vertices = list(hg_merge_tree.vertices())
    hg_parents = list(hg_merge_tree.parents())
    hg_edges = list(zip(hg_vertices, hg_parents))

    nx_vertices = hg_vertices
    nx_edges = hg_edges[:-1]  # omit the self-loop edge Higra uses
    nx_heights = {v: h for v, h in zip(nx_vertices, altitudes)}

    nx_merge_tree = nx.Graph()
    nx_merge_tree.add_nodes_from(nx_vertices)
    nx_merge_tree.add_edges_from(nx_edges)

    merge_tree = MergeTree(tree=nx_merge_tree, height=nx_heights)

    return merge_tree


def merge_tree_from_time_series_higra(time_series, superlevel_filtration=False, make_increasing=False):
    """
    Given a discrete time series compute the merge tree of its piecewise linear interpolation.

    Parameters
    ----------
    time_series : array_like
        List or numpy array of time series values.
    superlevel_filtration : boolean, optional
        Generate the superlevel set filtration merge tree? Default is the sublevel set filtration merge tree.
    make_increasing : boolean, optional
        Only applied if superlevel_filtration is True.
        Whether to align root altitude with the sublevel filtration and make paths from leaves to root increasing.
    
    Returns
    -------
    tuple : (higra.CptHierarchy, array)
        Tuple containing the Higra tree structure, of type higra.CptHierarchy, and an array of its node altitudes.
    """

    # flip the time series values if we're doing a superlevel filtration
    time_series = -1 * np.array(time_series) if superlevel_filtration else time_series

    # apply Higra's component tree algorithm over a time series chain graph
    chain_graph, vertex_weights = _chain_graph_from_time_series(time_series)
    tree, altitudes = hg.component_tree_min_tree(chain_graph, vertex_weights)

    # reflip the node altitudes if we flipped the time series originally
    altitudes = -1 * np.array(altitudes) if superlevel_filtration else altitudes

    # simplify the component tree to retain only persistence merge tree information
    merge_tree, merge_tree_altitudes = _higra_component_tree_2_merge_tree(tree, altitudes)

    # make superlevel trees comparable with sublevel trees if required
    if superlevel_filtration and make_increasing:
        max_altitude = np.max(merge_tree_altitudes)
        min_altitude = np.min(merge_tree_altitudes)
        merge_tree_altitudes = -1 * merge_tree_altitudes + min_altitude + max_altitude

    return merge_tree, merge_tree_altitudes


def merge_tree_from_time_series_dmt(time_series, superlevel_filtration=False, make_increasing=False):
    """
    Given a discrete time series compute the merge tree of its piecewise linear interpolation.

    Parameters
    ----------
    time_series : array_like
        List or numpy array of time series values.
    superlevel_filtration : boolean, optional
        Generate the superlevel set filtration merge tree? Default is the sublevel set filtration merge tree.
    make_increasing : boolean, optional
        Only applied if superlevel_filtration is True.
        Whether to align root altitude with the sublevel filtration and make paths from leaves to root increasing.
    
    Returns
    -------
    DMT_tools.MergeTree
        A `networkx`-based version of the merge tree suitable for use in `DMT_tools` functions.
    """

    # Build a Higra-based merge tree    
    higra_merge_tree = merge_tree_from_time_series_higra(time_series, superlevel_filtration=superlevel_filtration, make_increasing=make_increasing)

    # Convert it to the desired DMT_tools format
    merge_tree = _higra_merge_tree_2_dmt_merge_tree(*higra_merge_tree)
    
    return merge_tree


##############
## Plotting ##
##############


def plot_merge_tree_as_graph(tree, altitudes, with_labels=False, node_size=5):
    """
    Use NetworkX to plot the merge tree and label vertices with their altitudes.

    Paramaters
    ----------
    tree : higra.Tree
        The merge tree to plot
    altitudes : array
        The altitudes of the vertices in the tree

    Returns
    -------
    None
    """

    nx_tree = nx.Graph()

    for vertex in tree.vertices():
        nx_tree.add_node(vertex)
    for edge in tree.edges():
        nx_tree.add_edge(edge[0], edge[1])

    labels = {vertex: altitude for vertex, altitude in zip(tree.vertices(), altitudes)}
    nx.draw_kamada_kawai(nx_tree, with_labels=with_labels, labels=labels, node_color='darkblue', node_size=node_size)
    plt.show()


def plot_merge_tree_as_dendrogram(tree, altitudes, superlevel_filtration=False):
    """
    Use Higra to plot the dendrogram of the given merge tree.

    Paramaters
    ----------
    tree : higra.Tree
        The merge tree to plot
    altitudes : array
        The altitudes of the vertices in the tree
    superlevel_filtration : Boolean
        Is the merge tree from a superlevel set filtration?

    Returns
    -------
    None
    """

    # Higra expects clusters to merge at increasing altitudes
    if superlevel_filtration:
        altitudes = -1 * altitudes + np.max(altitudes)

    hg.plot_partition_tree(hg.Tree(tree.parents()), altitudes=altitudes)


def plot_merge_tree_as_dendrogram_scipy(tree, altitudes, superlevel_filtration=False, dendrogram_params=None):
    """
    Plot a dendrogram of a merge tree direcly with Scipy, bypassing Higra, to allow for custom output.

    Parameters
    ----------
    tree : higra.Tree
        The merge tree to plot
    altitudes : array
        The altitudes of the vertices in the tree
    superlevel_filtration : Boolean
        Is the merge tree from a superlevel set filtration?
    dendrogram_params : dict
        Kwargs to pass to the Scipy `hierarchy.dendrogram()` method

    Returns
    -------
    None
    """
    Z = hg.binary_hierarchy_to_scipy_linkage_matrix(hg.Tree(tree.parents()), altitudes)
    if dendrogram_params is None:
        dendrogram_params = {}
    hierarchy.dendrogram(Z, **dendrogram_params)
    plt.show()


def plot_extended_persistence_diagrams(
    persistence_diagrams, title="Extended Persistence Diagrams"
):
    dgms = persistence_diagrams
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(
        [dgms[0][i][1][0] for i in range(len(dgms[0]))],
        [dgms[0][i][1][1] for i in range(len(dgms[0]))],
    )
    axs[0].plot([-2, 2], [-2, 2])
    axs[0].set_title("Ordinary PD")
    axs[1].scatter(
        [dgms[1][i][1][0] for i in range(len(dgms[1]))],
        [dgms[1][i][1][1] for i in range(len(dgms[1]))],
    )
    axs[1].plot([-2, 2], [-2, 2])
    axs[1].set_title("Relative PD")

    fig.suptitle(title)
    plt.show()


def plot_time_series(
    time_series,
    title="Time Series",
    xlabel="Time",
    ylabel="Value",
    show=True,
    plot_kwargs=None,
):
    """
    Plots a univariate time series.

    Parameters
    ----------
    time_series : array_like
        A list or numpy array of the input time series
    title : str
        A string representing the title of the plot (default: 'Time Series')
    xlabel : str
        A string representing the label of the x-axis (default: 'Time')
    ylabel : str
        A string representing the label of the y-axis (default: 'Value')
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    plt.plot(time_series, **plot_kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()


def plot_persistence_diagram(
    persistence_diagram,
    title="Persistence Diagram",
    xlabel="Birth",
    ylabel="Death",
):
    """
    Plots a persistence diagram using Gudhi's diagram formatting.

    Parameters
    ----------
    persistence_diagram : array_like
        A list of persistent homology intervals e.g. from the `compute_persistent_homology` function
    title : str
        A string representing the title of the plot (default: 'Persistence Diagram')
    xlabel : str
        A string representing the label of the x-axis (default: 'Birth')
    ylabel : str
        A string representing the label of the y-axis (default: 'Death')
    """
    gd.plot_persistence_diagram(persistence_diagram)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


###########################################
## Horizontal visibility representations ##
###########################################


def hvg_from_time_series(time_series, directed=None, weighted=None, penetrable_limit=0, bottom_hvg=False):
    """
    Construct HVG. Passes most keyword args to the `ts2vg.HorizontalVG` constructor.

    Parameters
    ----------
    time_series : array
        The time series to compute the HVG for
    directed : str
        Type of directed graph to produce. See ts2vg docs.
    weighted : str
        Type of edge weights to include. See ts2vg docs.
    penetrable_limit : int
        Number of intermediate bars edges can penetrate. See ts2vg docs.
    bottom_hvg : boolean
        Construct the "bottom" HVG? Defaults to `False` and builds the standard "top" HVG.
    
    Returns
    -------
    ts2vg.HorizontalVG
        The "top" or "bottom" horizontal visibility graph generated by ts2vg.
    """
    ts2vg_kwargs = dict(directed=directed, weighted=weighted, penetrable_limit=penetrable_limit)
    time_series = -1 * np.array(time_series) if bottom_hvg else np.array(time_series)
    hvg = ts2vg.HorizontalVG(**ts2vg_kwargs)
    hvg.build(time_series)


############################
## Vectorisations of HVGs ##
############################


def hvg_degree_distribution(hvg : ts2vg.HorizontalVG, max_degree : int = 100):
    """
    Empirical degree distribution of a horizontal visibility graph.

    Parameters
    ----------
    hvg : ts2hvg.HorizontalVG
        The HVG for which to compute the distribution.
    max_degree : int
        Length of the vector of probabilities to return.

    Returns
    -------
    np.array
        Empirical probabilities of degrees 1, 2, ..., max_degree in the HVG.
    """

    ks, ps = hvg.degree_distribution
    probabilities = np.zeros(max_degree)
    for k, p in zip(ks, ps):
        probabilities[k-1] = p

    return probabilities


def hvg_statistics_vector(hvg):
    pass


###########################################################################
## Divergences of superlevel and sublevel set filtration representations ##
###########################################################################


def compute_extended_persistence_divergence(ordinary_pd, relative_pd):
    ordinary_pd = [(b, d) for (_, (b, d)) in ordinary_pd]
    relative_pd = [(d, b) for (_, (b, d)) in relative_pd]
    ordinary_pd = np.array(ordinary_pd)
    relative_pd = np.array(relative_pd)
    # return gudhi.wasserstein.wasserstein_distance(ordinary_pd, relative_pd)
    return gd.bottleneck_distance(ordinary_pd, relative_pd)




def autocorrelation(sequence):
    if isinstance(sequence, list):
        sequence = np.array(sequence)

    if not isinstance(sequence, np.ndarray) or sequence.dtype != float:
        raise ValueError("Input should be a numpy array or a list of floats.")

    n = len(sequence)
    mean = np.mean(sequence)
    autocorr = np.zeros(n)

    for lag in range(n):
        numerator = 0.0
        denominator = 0.0
        for i in range(n - lag):
            numerator += (sequence[i] - mean) * (sequence[i + lag] - mean)
            denominator += (sequence[i] - mean) * (sequence[i] - mean)
        autocorr[lag] = numerator / denominator

    return autocorr


def main():
    """
    Run some example computations and plots of time series persistent homology.
    """

    def run_example(time_series):
        diagram = compute_persistent_homology(time_series)
        extended_diagrams = compute_extended_persistent_homology(time_series)
        plot_time_series(time_series)
        plot_persistence_diagram(diagram)
        plot_extended_persistence_diagrams(extended_diagrams)
        print("divergence:")
        print(compute_extended_persistence_divergence(*extended_diagrams))

    # Simple sequence
    time_series = np.array([1.0, 0.5, 0.8, 0.2, 0.9, 0.6])
    run_example(time_series)

    # Logistic map:
    r, x0, n_iterations = 4.0, 0.90, 100
    time_series = logistic_map(r, x0, n_iterations)
    run_example(time_series)

    # White noise:
    time_series = white_noise(100, seed=42)
    run_example(time_series)

    # Extended persistence divergences wrt r and x0 of logistic map
    r_count = 10000
    r_min, r_max = 3.5, 4.0
    x0_min, x0_max = 0.01, 0.99
    n_iterations = 10000
    # rr = np.linspace(r_min, r_max, r_count + 1)[:-1]  # exclude the max value
    rr = np.random.uniform(r_min, r_max, r_count)
    # xx0 = np.linspace(x0_min, x0_max, x0_count)
    xx0 = [0.5]
    x0_count = len(xx0)
    divs = np.zeros((r_count, x0_count))

    for i, r in enumerate(rr):
        for j, x0 in enumerate(xx0):
            time_series = logistic_map(r, x0, n_iterations)
            extended_diagrams = compute_extended_persistent_homology(time_series)
            divs[i, j] = compute_extended_persistence_divergence(*extended_diagrams)
    plt.imshow(
        divs, cmap="hot", interpolation="nearest", extent=[x0_min, x0_max, r_min, r_max]
    )
    plt.title("Extended Persistence Divergences")
    plt.ylabel("$r$")
    plt.xlabel("$x_0$")
    plt.colorbar()
    plt.show()

    mean_div = np.mean(divs, axis=1)
    # median_div = np.median(divs, axis=1)
    min_div = np.min(divs, axis=1)
    max_div = np.max(divs, axis=1)
    # std_div = np.std(divs, axis=1)

    plt.scatter(rr, mean_div, label="Mean", s=1**2)
    # plt.plot(rr, median_div, label="Median")
    # plt.plot(rr, max_div, label="Max")
    # plt.plot(rr, min_div, label="Min")
    # plt.plot(rr, std_div, label="Std")

    plt.xlabel("$r$")
    plt.ylabel("Persistence Divergence")
    plt.legend(loc="upper left")
    plt.show()

    # auto_div = np.apply_along_axis(autocorrelation, arr=divs, axis=1)
    # for autocor in auto_div[::10]:
    #     plt.plot(autocor)
    #     plt.title("Autocorrelation")
    #     plt.show()


if __name__ == "__main__":
    main()
