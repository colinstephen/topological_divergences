import numpy as np
from ts2vg import HorizontalVG as HVG
from scipy.stats import wasserstein_distance

def get_hvg_estimates(time_series):

    time_series = np.array(time_series)

    # build the horizontal visibility graphs
    hvg_top = HVG(directed="left_to_right").build(time_series, only_degrees=True)
    hvg_bot = HVG(directed="left_to_right").build(-1 * time_series, only_degrees=True)

    # extract the node degrees
    ks_top, ps_top = hvg_top.degree_distribution
    ks_bot, ps_bot = hvg_bot.degree_distribution

    # make it easy to look up probability for given k
    prob_top = {k: p for k, p in zip(ks_top, ps_top)}
    prob_bot = {k: p for k, p in zip(ks_bot, ps_bot)}

    # get all possible k values over both graphs
    all_ks = list(set(list(ks_top) + list(ks_bot)))

    # compute the l1, l2, and linf distance between distributions
    ell1_delta = 0
    ell2_delta = 0
    ellinf_delta = -np.inf
    for k in all_ks:
        k_diff = abs(prob_top.get(k, 0) - prob_bot.get(k, 0))
        ell1_delta += k_diff
        ell2_delta += k_diff * k_diff
        ellinf_delta = max(k_diff, ellinf_delta)
    ell2_delta = np.sqrt(ell2_delta)

    # compute the wasserstein disstance between distributions
    wass_delta = wasserstein_distance(ks_top, ks_bot, ps_top, ps_bot)

    return np.array([ell1_delta, ell2_delta, ellinf_delta, wass_delta])

hvg_names = ["HVG L1", "HVG L2", "HVG Linf", "HVG Wasserstein"]
