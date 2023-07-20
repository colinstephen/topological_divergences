import numpy as np


class LogisticMapDataset:
    """
    Class to manage generation of logistic map trajectories and their Lyapunov exponents.
    """

    def __init__(
        self,
        MIN_R=3.3,
        MAX_R=4.0,
        NUM_R=1000,
        RANDOM_R=True,
        X0=0.2,
        TIME_SERIES_LENGTH=500,
    ):
        """
        Constructor for LogisticMapDataset class.

        Parameters
        ----------
        MIN_R : float, optional
            Minimum value of the parameter `r` for the logistic map (default: 3.3).
        MAX_R : float, optional
            Maximum value of the parameter `r` for the logistic map (default: 4.0).
        NUM_R : int, optional
            Number of `r` values to generate (default: 1000).
        RANDOM_R : bool, optional
            Whether to randomly sample `r` values or use evenly spaced values (default: True).
        X0 : float, optional
            Initial value for the logistic map trajectory (default: 0.2).
        TIME_SERIES_LENGTH : int, optional
            Length of the generated time series for each trajectory (default: 500).
        """
        self.MIN_R = MIN_R
        self.MAX_R = MAX_R
        self.NUM_R = NUM_R
        self.RANDOM_R = RANDOM_R
        self.X0 = X0
        self.TIME_SERIES_LENGTH = TIME_SERIES_LENGTH
        self._generate_r_values()
        self._generate_trajectories()
        self._generate_lyapunov_exponents()

    def _generate_r_values(self):
        """
        Generate `r` values based on the specified configuration.
        """
        if self.RANDOM_R:
            # Randomly sample `r` values
            self._r_values = np.random.uniform(
                self.MIN_R, self.MAX_R, self.NUM_R
            )  # Excludes MAX_R
            self._r_values = np.sort(self._r_values)
        else:
            # Generate evenly spaced `r` values
            self._r_values = np.linspace(
                self.MIN_R, self.MAX_R, self.NUM_R
            )  # Includes MAX_R

    def _generate_trajectories(self):
        """
        Generate logistic map trajectories based on the generated `r` values.
        """
        self._trajectories = [
            LogisticMapDataset.logistic_map(r, self.X0, self.TIME_SERIES_LENGTH)
            for r in self.r_values
        ]

    def _generate_lyapunov_exponents(self):
        """
        Compute the Lyapunov exponents for the logistic map trajectories.
        """
        self._lyapunov_exponents = (
            LogisticMapDataset.lyapunov_approximation_for_logistic_map(
                self.r_values, self.X0
            )
        )

    @property
    def r_values(self):
        """
        Get the generated `r` values.

        Returns
        -------
        np.array
            Array of `r` values.
        """
        if not len(self._r_values):
            self._generate_r_values()
        return self._r_values

    @property
    def trajectories(self):
        """
        Get the generated logistic map trajectories.

        Returns
        -------
        list
            List of logistic map trajectories.
        """
        if not len(self._trajectories):
            self._generate_trajectories()
        return self._trajectories

    @property
    def lyapunov_exponents(self):
        """
        Get the computed Lyapunov exponents.

        Returns
        -------
        np.array
            Array of Lyapunov exponents.
        """
        if not len(self._lyapunov_exponents):
            self._generate_lyapunov_exponents()
        return self._lyapunov_exponents

    @staticmethod
    def logistic_map(r, x0, n_iterations, skip_iterations=10000):
        """
        Generate a logistic map time series.

        Parameters
        ----------
        r : float
            Parameter `r` representing the growth rate where 0.0 < r <= 4.0.
        x0 : float
            Initial value where 0 < x0 < 1.
        n_iterations : int
            Number of iterations in the returned sequence.
        skip_iterations : int, optional
            Number of iterations to ignore for burn-in (default: 10000).

        Returns
        -------
        np.array
            A NumPy array containing the generated time series.
        """
        if not (0 < r <= 4.0):
            raise ValueError("Parameter r out of range")

        if not (0 < x0 < 1):
            raise ValueError("Initial value x0 out of range")

        # Initialize the value of the map
        x = x0

        def apply_map(x):
            return r * x * (1 - x)

        # Ignore burn-in iterations
        for _ in range(skip_iterations):
            x = apply_map(x)

        # Initialize an array to return
        time_series = np.zeros(n_iterations)

        # Generate the values of the map
        time_series[0] = x
        for i in range(1, n_iterations):
            time_series[i] = apply_map(time_series[i - 1])

        return time_series

    @staticmethod
    def lyapunov_approximation_for_logistic_map(
        r_values, x0=0.2, n_iterations=10000, skip_iterations=1000
    ):
        """
        Approximate the largest Lyapunov exponent of the logistic map for each `r` value provided.

        Parameters
        ----------
        r_values : np.array
            Array of values for the control parameter `r` in the map `f(x) = r * x * (1 - x)`.
        x0 : float, optional
            Initial value for the trajectory (default: 0.2).
        n_iterations : int, optional
            Number of iterations of the map with which to compute the approximation (default: 10000).
        skip_iterations : int, optional
            Number of initial iterations of the map to ignore before beginning `n_iterations` (default: 1000).

        Returns
        -------
        np.array
            The approximate largest Lyapunov exponent values for each of the input `r_values`.
        """
        # Ensure we can apply array-wise operations
        r_values = np.array(r_values)

        # Initialize all trajectories with the starting value
        x = x0 * np.ones(r_values.shape)

        # Discard the transient on all trajectories
        for _ in range(skip_iterations):
            x = r_values * x * (1 - x)

        # Then iterate `n_iterations` times and compute the sum for the Lyapunov exponent
        lyapunov_exp = np.zeros(r_values.shape)
        for _ in range(n_iterations):
            # Update all trajectories
            x = r_values * x * (1 - x)
            # Update the exponent approximation for each trajectory
            lyapunov_exp += np.log(abs(r_values - 2 * r_values * x))

        # Average over the number of iterations to get the final Lyapunov exponent
        lyapunov_exp /= n_iterations

        return lyapunov_exp
