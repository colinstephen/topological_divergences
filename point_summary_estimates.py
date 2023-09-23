# patch bug in Teaspoon's `teaspoon.parameter_selection.MsPE`
# can cause a reference error by not defining the delay peak
from teaspoon.parameter_selection import MsPE
from PATCH_MsPE_tau import MsPE_tau
MsPE.MsPE_tau = MsPE_tau

# patch bug in Teaspoon's `teaspoon.TDA.PHN.DistanceMatrix`
# can crash when the input graph is not connected
from teaspoon.TDA import PHN
from PATCH_DistanceMatrix import DistanceMatrix
PHN.DistanceMatrix = DistanceMatrix

# patch bug in Teaspoon's `teaspoon.SP.network`
# can crash due to tsa_tools not being defined
from teaspoon.SP import tsa_tools
from teaspoon.SP import network
network.tsa_tools = tsa_tools

# regular imports
import numpy as np
from teaspoon.SP.network import knn_graph
from teaspoon.SP.network import ordinal_partition_graph
from teaspoon.SP.network_tools import remove_zeros
from teaspoon.TDA.PHN import PH_network
from teaspoon.TDA.PHN import point_summaries


def get_point_summary_estimates(time_series):

    graph_point_summaries = []
    
    # compute the k-NN graph
    graph_knn = knn_graph(time_series)

    # compute the persistence diagram
    graph_knn_distance = DistanceMatrix(graph_knn)
    graph_knn_pd = PH_network(graph_knn_distance)
    graph_knn_pd = graph_knn_pd if len(graph_knn_pd) > 0 else np.array([[0]])

    # compute the point summary statistics
    graph_point_summaries.extend(point_summaries(graph_knn_pd, graph_knn))

    # compute the ordinal partition graph
    try:
        graph_opn = ordinal_partition_graph(time_series)
    except ValueError as err:
        message = getattr(err, "message", repr(err))
        if "negative dimensions are not allowed" in message:
            print("WARNING: ordinal_partition_graph() tried to use a negative dimension")
            graph_opn = np.array([[0]])
        else:
            raise err

    # compute the persistence diagram
    graph_opn_distance = DistanceMatrix(graph_opn)
    graph_opn_pd = PH_network(graph_opn_distance)
    graph_opn_pd = graph_opn_pd if len(graph_opn_pd) > 0 else np.array([[0]])

    # compute the point summary statistics
    graph_point_summaries.extend(point_summaries(graph_opn_pd, graph_opn))

    return np.nan_to_num(np.array(graph_point_summaries))

point_summary_names = ["kNN R(D)", "kNN E'(D)", "kNN M(D)", "OPN R(D)", "OPN E'(D)", "OPN M(D)"]
