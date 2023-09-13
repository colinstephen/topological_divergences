import numpy as np
from functools import wraps
from nolds import lyap_r
from nolds import lyap_e
from gtda.time_series import takens_embedding_optimal_parameters
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
tseriesChaos = importr("tseriesChaos")


def kantz_estimator(
    ts,
    theiler_window=5,
    k_neighbours=2,
    max_num_points=400,
    num_iterations=20,
    neighbour_radius=1.1,
):
    """Use R's `tseriesChaos` function `lyap_k` to estimate max of Lyapunov spectrum."""

    delay, dim = takens_embedding_optimal_parameters(ts, 50, 8)
    num_points = min(max_num_points, len(ts) - (((dim - 1) * delay) + 1))

    lyapunov_spectrum = tseriesChaos.lyap_k(
        ts,
        dim,
        delay,
        t=theiler_window,
        k=k_neighbours,
        ref=num_points,
        s=num_iterations,
        eps=neighbour_radius,
    )
    return max(lyapunov_spectrum)


def safe_lyap(func):
    """Add exception handling to Lyapunov estimation functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as err:
            message = getattr(err, "message", repr(err))
            print("WARNING: Lyapunov function failed for some reason")
            print("MESSAGE:", message)
            return np.nan
        return result

    return wrapper


def get_classic_estimates(time_series):
    rosenstein_estimate = safe_lyap(lyap_r)(time_series)
    eckmann_estimate = max(safe_lyap(lyap_e)(time_series))
    kantz_estimate = safe_lyap(kantz_estimator)(robjects.FloatVector(time_series))
    return np.array([rosenstein_estimate, eckmann_estimate, kantz_estimate])

classic_names = ["Rosenstein", "Eckmann", "Kantz"]
