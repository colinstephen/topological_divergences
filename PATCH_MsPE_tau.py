# Patch for teaspoon.parameter_selection.MsPE.MsPE_tau (line 59)

"""
MsPE for time delay (tau) and dimension (n).
=================================================

This function implements Multi-scale Permutation Entropy (MsPE) for the selection of n and tau for permutation entropy. 
Additionally, it only requires a single time series, is robust to additive noise, and has a fast computation time.
"""


def MsPE_tau(time_series, delay_end=200, plotting=False):
    """This function takes a time series and uses Multi-scale Permutation Entropy (MsPE) to find the optimum
    delay based on the first maxima in the MsPE plot

    Args:
       ts (array):  Time series (1d).
       delay_end (int): maximum delay in search. default is 200.

    Kwargs:
       plotting (bool): Plotting for user interpretation. defaut is False.

    Returns:
       (int): tau, The embedding delay for permutation formation.

    """
    trip = 0.9
    from pyentrp import entropy as ent
    import math
    import numpy as np

    MSPE = []
    delays = []
    ds, de = 0, delay_end
    m = 3
    start = False
    end = False
    delay = ds
    NPE_previous = 0
    while end == False:
        delay = delay+1
        ME = np.log2(math.factorial(m))
        PE = ent.permutation_entropy(time_series, m, delay)
        NPE = PE/ME
        if NPE < trip:
            start = True
        if NPE > trip and start == True and end == False:
            if NPE < NPE_previous:
                delay_peak = delay-1
                end = True
            NPE_previous = NPE
        MSPE = np.append(MSPE, NPE)
        delays.append(delay)

        if delay > de:
            delay = 1
            trip = trip-0.05
            if trip < 0.5:
                delay_peak = 1  # FIXED: previously "delay = 1"
                end = True

    if plotting == True:
        import matplotlib.pyplot as plt
        plt.figure(2)
        TextSize = 17
        plt.figure(figsize=(8, 3))
        plt.plot(delays, MSPE, marker='.')
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.ylabel(r'$h(3)$', size=TextSize)
        plt.xlabel(r'$\tau$', size=TextSize)
        plt.show()
    return delay_peak
