import numpy as np
import gudhi as gd
import gudhi.wasserstein
import matplotlib.pyplot as plt


def logistic_map(r, x0, n_iterations, skip_iterations=1000):
    """
    Generates a logistic map time series.

    Parameters
    ----------
    r : float
        Parameter r representing the growth rate where 0 < r < 4
    x0 : float
        Initial value where 0 < x0 < 1
    n_iterations : int
        Number of iterations to generate the time series for.
    skip_iterations : int
        Number of iterations to ignore while the system settles.

    Returns
    -------
    np.array
        A NumPy array containing the generated time series.
    """
    if not (0 < x0 < 1):
        raise ValueError("x0 must be strictly between 0 and 1")

    x = x0
    for i in range(skip_iterations):
        x = r * x * (1 - x)

    time_series = np.zeros(n_iterations)
    time_series[0] = x

    for i in range(1, n_iterations):
        x = time_series[i - 1]
        time_series[i] = r * x * (1 - x)

    return time_series


def white_noise(length, mean=0, std_dev=1, seed=None):
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
    seed : int, optional
        The random seed for reproducibility (default: None).

    Returns
    -------
    white_noise : numpy.ndarray
        A numpy array of white noise samples.
    """
    if seed is not None:
        np.random.seed(seed)

    white_noise = np.random.normal(mean, std_dev, length)
    return white_noise


def create_sublevel_set_filtration(time_series):
    """
    Creates the level set filtration for the given time series based on the specified filtration type.

    Parameters
    ----------
    time_series : array_like
        A list or numpy array of the input time series.

    Returns
    -------
    filtration : list
        A list of tuples (simplex, value) representing the level set filtration.
    """

    filtration = []
    n = len(time_series)

    for i in range(n):
        filtration.append(([i], time_series[i]))

        if i < n - 1:
            edge_value = max(time_series[i], time_series[i + 1])
            filtration.append(([i, i + 1], edge_value))

    return filtration


def compute_persistent_homology_simplex_tree(time_series):
    """
    Computes the persistent homology of the given time series and returns the simplex tree.

    Parameters
    ----------
    time_series : array_like
        A list or numpy array of the input time series

    Returns
    -------
    gudhi.SimplexTree
        A Gudhi SimplexTree object
    """

    filtration = create_sublevel_set_filtration(time_series)

    # Create a simplex tree object
    st = gd.SimplexTree()

    # Insert the filtration into the simplex tree
    for simplex, value in filtration:
        st.insert(simplex, value)

    # Compute persistent homology
    st.persistence()

    return st


def compute_extended_persistent_homology_simplex_tree(time_series):
    """
    Computes the extended persistent simplex tree of a time series.

    Parameters
    ----------
    time_series : array_like
        A list or numpy array of the input time series.

    Returns
    -------
    gudhi.SimplexTree
        A Gudhi SimplexTree object
    """

    st = compute_persistent_homology_simplex_tree(time_series)
    st.extend_filtration()
    return st


def compute_extended_persistent_homology(time_series):
    """
    Computes the extended persistent homology diagrams of the given time series.

    Parameters
    ----------
    time_series : array_like
        A list or numpy array of the input time series.

    Returns
    -------
    persistence_diagrams : list
        A list of the ordinary, extended, and relative persistence diagams.
    """

    simplex_tree = compute_extended_persistent_homology_simplex_tree(time_series)
    persistence_diagrams = simplex_tree.extended_persistence(min_persistence=1e-5)

    return persistence_diagrams[:2]


def compute_persistent_homology(time_series):
    """
    Computes the persistent homology diagram of the given time series.

    Parameters
    ----------
    time_series : array_like
        A list or numpy array of the input time series.

    Returns
    -------
    persistence_diagram : list
        A list of persistent homology intervals.
    """

    simplex_tree = compute_persistent_homology_simplex_tree(time_series)
    persistence_diagram = simplex_tree.persistence_intervals_in_dimension(0)

    return persistence_diagram


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
