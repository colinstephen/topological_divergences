from typing import Any
import numpy as np
import ts2vg
from numpy.linalg import norm
from scipy.stats import wasserstein_distance


class TimeSeriesHVG:
    """
    Generate and manage horizontal visibility representations of time series.

    Properties include vector representations and divergences of them.
    """

    def __init__(
        self,
        time_series: np.array,
        DEGREE_DISTRIBUTION_MAX_DEGREE=100,
        DEGREE_DISTRIBUTION_DIVERGENCE_P_VALUE=1.0,
        directed=None,
        weighted=None,
        penetrable_limit=0,
    ) -> None:
        self._time_series = time_series
        self._top_hvg = None
        self._bottom_hvg = None
        self.DEGREE_DISTRIBUTION_MAX_DEGREE = DEGREE_DISTRIBUTION_MAX_DEGREE
        self.DEGREE_DISTRIBUTION_DIVERGENCE_P_VALUE = (
            DEGREE_DISTRIBUTION_DIVERGENCE_P_VALUE
        )
        self.directed = directed
        self.weighted = weighted
        self.penetrable_limit = penetrable_limit

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in ["directed", "weighted", "penetrable_limit"]:
            # Reset HVG whenever we change the graph type
            self._top_hvg = None
            self._bottom_hvg = None
        self.__dict__[__name] = __value

    @property
    def top_hvg(self):
        if self._top_hvg is None:
            self._top_hvg = TimeSeriesHVG.hvg_from_time_series(
                self._time_series,
                directed=self.directed,
                weighted=self.weighted,
                penetrable_limit=self.penetrable_limit,
            )
        return self._top_hvg

    @property
    def bottom_hvg(self):
        if self._bottom_hvg is None:
            self._bottom_hvg = TimeSeriesHVG.hvg_from_time_series(
                self._time_series,
                directed=self.directed,
                weighted=self.weighted,
                penetrable_limit=self.penetrable_limit,
                bottom_hvg=True,
            )
        return self._bottom_hvg

    @staticmethod
    def hvg_from_time_series(
        time_series, directed=None, weighted=None, penetrable_limit=0, bottom_hvg=False
    ):
        """
        Construct HVG. Most keyword args go to the `ts2vg.HorizontalVG` constructor.

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
        time_series = (
            -1 * np.array(time_series) if bottom_hvg else np.array(time_series)
        )
        hvg = ts2vg.HorizontalVG(**ts2vg_kwargs)
        hvg.build(time_series)

        return hvg

    ############################
    ## Vectorisations of HVGs ##
    ############################

    @property
    def vectorisations(self):
        return dict(
            degree_top=self.degree_distribution_top,
            degree_bottom=self.degree_distribution_bottom,
            degree_all=self.degree_distribution_all,
        )

    @property
    def degree_distribution_top(self):
        return self.hvg_degree_distribution()

    @property
    def degree_distribution_bottom(self):
        return self.hvg_degree_distribution(bottom_hvg=True)

    @property
    def degree_distribution_all(self):
        return np.concatenate(
            (self.degree_distribution_top, self.degree_distribution_bottom)
        )

    def hvg_degree_distribution(self, bottom_hvg=False):
        """
        Empirical degree distribution of a horizontal visibility graph.

        Parameters
        ----------
        bottom_hvg : boolean
            Return the top (default) or bottom HVG degree distribution?

        Returns
        -------
        np.array
            Empirical probabilities of degrees 1, 2, ..., max_degree in the HVG.
        """

        hvg = self.bottom_hvg if bottom_hvg else self.top_hvg

        ks, ps = hvg.degree_distribution
        probabilities = np.zeros(self.DEGREE_DISTRIBUTION_MAX_DEGREE)
        for k, p in zip(ks, ps):
            probabilities[k - 1] = p

        return probabilities

    #######################################################################
    ## Divergences of super and sub level set filtration representations ##
    #######################################################################

    @property
    def divergences(self):
        return dict(
            degree_wasserstein=self.degree_wasserstein_divergence,
            degree_lp=self.degree_lp_divergence,
        )

    @property
    def degree_wasserstein_divergence(self):
        return wasserstein_distance(
            self.degree_distribution_top, self.degree_distribution_bottom
        )

    @property
    def degree_lp_divergence(self):
        return norm(
            self.degree_distribution_top - self.degree_distribution_bottom,
            ord=self.DEGREE_DISTRIBUTION_DIVERGENCE_P_VALUE,
        )