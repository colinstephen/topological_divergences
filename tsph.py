import higra as hg
import numpy as np
import gudhi as gd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy


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
