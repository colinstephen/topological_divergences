# patch for teaspoon.TDA.PHN.DistanceMatrix

def DistanceMatrix(A):
    """Get the all-pairs unweighted shortest path lengths in the graph A.

    Does its own imports so it can run on ipyparallel engines.

    Fixes an issue in the `teaspoon` library such that distance matrix computation
    fails with disconnected graphs A.
    """
    import numpy as np
    import networkx as nx
    from teaspoon.SP import network_tools
    
    A = network_tools.remove_zeros(A)
    np.fill_diagonal(A, 0)
    A = A + A.T

    A_sp = np.copy(A)
    N = len(A_sp)
    D = np.zeros((N,N))

    A_sp[A_sp > 0] = 1
    G = nx.from_numpy_matrix(A_sp)
    lengths = dict(nx.all_pairs_shortest_path_length(G))    
    for i in range(N-1):
        for j in range(i+1, N):
            D[i][j] = lengths.get(i, {}).get(j, np.inf)
    D = D + D.T
    return D

