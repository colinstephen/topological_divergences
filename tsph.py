import ts2vg
import higra as hg
import numpy as np
import gudhi as gd
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.cluster import hierarchy
from tda_divergence import tda_divergence
from decorated_merge_trees.DMT_tools import MergeTree


############################
## Time series generators ##
############################


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


def merge_tree_from_time_series_higra(
    time_series, superlevel_filtration=False, make_increasing=False
):
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
    merge_tree, merge_tree_altitudes = _higra_component_tree_2_merge_tree(
        tree, altitudes
    )

    # make superlevel trees comparable with sublevel trees if required
    if superlevel_filtration and make_increasing:
        max_altitude = np.max(merge_tree_altitudes)
        min_altitude = np.min(merge_tree_altitudes)
        merge_tree_altitudes = -1 * merge_tree_altitudes + min_altitude + max_altitude

    return merge_tree, merge_tree_altitudes


def merge_tree_from_time_series_dmt(
    time_series, superlevel_filtration=False, make_increasing=False
):
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
    higra_merge_tree = merge_tree_from_time_series_higra(
        time_series,
        superlevel_filtration=superlevel_filtration,
        make_increasing=make_increasing,
    )

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
    nx.draw_kamada_kawai(
        nx_tree,
        with_labels=with_labels,
        labels=labels,
        node_color="darkblue",
        node_size=node_size,
    )
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


def plot_merge_tree_as_dendrogram_scipy(
    tree, altitudes, superlevel_filtration=False, dendrogram_params=None
):
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


def hvg_from_time_series(
    time_series, directed=None, weighted=None, penetrable_limit=0, bottom_hvg=False
):
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
    ts2vg_kwargs = dict(
        directed=directed, weighted=weighted, penetrable_limit=penetrable_limit
    )
    time_series = -1 * np.array(time_series) if bottom_hvg else np.array(time_series)
    hvg = ts2vg.HorizontalVG(**ts2vg_kwargs)
    hvg.build(time_series)

    return hvg


############################
## Vectorisations of HVGs ##
############################


def hvg_degree_distribution(hvg: ts2vg.HorizontalVG, max_degree: int = 100):
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
        probabilities[k - 1] = p

    return probabilities


def hvg_statistics_vector(hvg):
    pass


###########################################################################
## Divergences of superlevel and sublevel set filtration representations ##
###########################################################################


def lp_distance(a, b, p=2):
    a = np.array(a)
    b = np.array(b)
    return norm(a - b, ord=p)


def hvg_peak_pit_divergence(
    time_series,
    distance_func="wasserstein_distance_for_1d_distributions",
    distance_params=None,
):
    """
    Distance between HVG peak and pit degree distributions.
    """
    return tda_divergence(
        time_series,
        sublevel_rep_func="horizontal_visibility_graph",
        superlevel_rep_func="horizontal_visibility_graph",
        superlevel_rep_params=dict(bottom_hvg=True),
        sublevel_rep_vectorisation="hvg_degree_distribution",
        superlevel_rep_vectorisation="hvg_degree_distribution",
        distance_func=distance_func,
        distance_params=distance_params,
    )


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
