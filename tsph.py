import numpy as np
import gudhi as gd
import higra as hg
import gudhi.wasserstein
import matplotlib.pyplot as plt
import vectorization as vec

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


#####################################################
## Persistent homology and merge tree computations ##
#####################################################


def sublevel_set_filtration(time_series):
    """
    Creates the sub level set filtration for the given time series based on the specified filtration type.

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


def superlevel_set_filtration(time_series):
    """
    Creates the super level set filtration for the given time series based on the specified filtration type.

    Parameters
    ----------
    time_series : array_like
        A list or numpy array of the input time series.

    Returns
    -------
    list
        The filtration: a list of (simplex, value) pairs representing the superlevel set filtration.
    """

    # invert the sequence (make peaks into pits and vice versa)
    time_series = -1 * np.array(time_series)

    # apply the sublevel algorithm to the inverted sequence
    return sublevel_set_filtration(time_series)


def persistent_homology_simplex_tree(filtration):
    """
    Construct a Gudhi simplex tree representing the given filtration.

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


def persistence_diagram_from_simplex_tree(
    simplex_tree, dimension=0, superlevel_filtration=False
):
    """
    Construct the persistent homology diagram induced by the given simplex tree.

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
    When superlevel set adjustment is applied by setting `superlevel_filtration=True`, the intervals returned are (-birth, -death) rather than (birth, death). This is to account for decreasing function values used when building a superlevel set filtration, and the assumption that a simplex tree corresponding to a superlevel set filtration on $f(x)$ has been built using the sublevel set filtration of the negated function $-f(x)$. Note that in this situation critical values will satisfy `death<birth`.
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


############################################
## Vectorisations of persistence diagrams ##
############################################


def persistence_statistics_vector(persistence_diagram):
    pass


def entropy_summary_function(persistence_diagram):
    pass


def betti_curve_function(persistence_diagram):
    pass


def persistence_silhouette_function(persistence_diagram):
    pass


def persistence_lifespan_curve_function(persistence_diagram):
    pass


def persistence_image(persistence_diagram):
    pass


#################################
## Graph-based representations ##
#################################


def time_series_chain_graph(time_series, edge_function=None):
    """
    Generate a chain graph representation of a time series.

    Parameters
    ----------
    time_series : array_like
        Sequence of values in the time series.
    edge_function : [int, int] -> int
        Optional. A function mapping edge vertex values to an edge weight.
        Defaults to minimum.

    Returns
    -------
    higra.UndirectedGraph
        A Higra graph object.

    Summary
    -------
    Given a discrete time series T of length |T|=N, generate a chain graph whose nodes {0, 1, ..., N-1} are labelled with the time series values T[0], T[1], ..., T[N-1], and whose edges {(0,1), (1,2), ..., (N-2,N-1)} are labelled with the minimum value of the incident nodes.
    """


################################
## Tree-based representations ##
################################


def merge_tree_from_time_series(time_series, superlevel_filtration=False):
    """
    Given a discrete time series compute the merge tree of its piecewise linear interpolation.

    Parameters
    ----------
    time_series : array_like
        List or numpy array of time series values.
    superlevel_filtration : boolean, optional
        Generate the superlevel set filtration merge tree? Default is the sublevel set filtration merge tree.
    """

    pass


###########################################
## Horizontal visibility representations ##
###########################################


def hvg_from_time_series(time_series, weighted_output=False):
    pass


############################
## Vectorisations of HVGs ##
############################


def hvg_degree_distribution(hvg, max_degree=100):
    pass


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
