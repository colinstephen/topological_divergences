import networkx as nx
import numpy as np
from TimeSeriesMergeTreeSimple import as_directed_tree, get_root, get_leaves

def path_length_vector(T, offset=1) -> np.array:
    """List of path lengths in T from leaf i to leaf i+offset."""
    
    root = get_root(T)
    leaves = get_leaves(T)
    n = len(leaves)
    leaf_pairs = [(leaves[i], leaves[i+offset]) for i in range(n-offset)]

    T_directed = as_directed_tree(T, root_to_leaf=True)
    lcas = list(nx.tree_all_pairs_lowest_common_ancestor(T_directed, root=root, pairs=leaf_pairs))
    
    path_lengths = []
    
    T_directed = as_directed_tree(T, root_to_leaf=False)
    for ((u, v), lca) in lcas:
        path_node = u
        path_length = 0
        while path_node != lca:
            path_node = list(T_directed.successors(path_node))[0]
            path_length = path_length + 1
        path_node = v
        while path_node != lca:
            path_node = list(T_directed.successors(path_node))[0]
            path_length = path_length + 1
        path_lengths.append(path_length)
    return np.array(path_lengths)


"""Distance between distributions of path lengths between leaf i and i+offset"""

from scipy.stats import wasserstein_distance
from TimeSeriesMergeTreeSimple import make_increasing

def get_offset_divergence(offset, tsmt=None):
    plv1 = path_length_vector(tsmt.merge_tree, offset=offset)
    plv2 = path_length_vector(make_increasing(tsmt.superlevel_merge_tree), offset=offset)
    w_dist = wasserstein_distance(plv1, plv2)
    l1_dist = np.linalg.norm(plv1/np.sum(plv1)-plv2/np.sum(plv2), ord=1)
    l2_dist = np.linalg.norm(plv1/np.sum(plv1)-plv2/np.sum(plv2), ord=2)
    linf_dist = np.linalg.norm(plv1/np.sum(plv1)-plv2/np.sum(plv2), ord=np.inf)
    return w_dist, l1_dist, l2_dist, linf_dist
