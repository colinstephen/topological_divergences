import numpy as np
from ripser import ripser
from teaspoon.TDA.Persistence import BettiCurve, maxPers
from teaspoon.SP.tsa_tools import takens

def get_crocker_estimates(
    time_series,
    num_stops=100,
):

    # phase space embedding
    embedding = takens(time_series)

    # compute embedded persistence diagrams
    embedded_pds = ripser(embedding)
    embedded_pd_0D = embedded_pds["dgms"][0]
    embedded_pd_1D = embedded_pds["dgms"][1]

    # get the max persistence of each
    max_0D_pers = maxPers(embedded_pd_0D)
    max_1D_pers = maxPers(embedded_pd_1D)

    # compute the Betti vectors for embedding Vietoris Rips persistence diagrams
    betti_vec_0D = BettiCurve(embedded_pd_0D, maxEps=max_0D_pers, numStops=num_stops)
    betti_vec_1D = BettiCurve(embedded_pd_1D, maxEps=max_1D_pers, numStops=num_stops)

    # compute magnitudes
    betti_vec_norm_0D = np.linalg.norm(betti_vec_0D, ord=1)
    betti_vec_norm_1D = np.linalg.norm(betti_vec_1D, ord=1)

    return np.array([betti_vec_norm_0D, betti_vec_norm_1D])

crocker_names = ["0-Betti vector norm", "1-Betti vector norm"]
