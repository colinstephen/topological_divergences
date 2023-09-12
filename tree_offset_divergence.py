import networkx as nx
import numpy as np
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler
from TimeSeriesMergeTreeSimple import as_directed_tree
from TimeSeriesMergeTreeSimple import get_root
from TimeSeriesMergeTreeSimple import tree_leaves
from TimeSeriesMergeTreeSimple import make_increasing
from scipy.stats import wasserstein_distance


def rescale_vector(vec, min_val=0, max_val=1):
    scaler = MinMaxScaler((min_val, max_val))
    return scaler.fit_transform(vec.reshape(-1, 1)).flatten()



def tree_lcas(T, leaf_pairs):
    """List of lowest common ancestors in tree T of given pairs of leaves."""
    root = get_root(T)
    T_directed = as_directed_tree(T, root_to_leaf=True)
    return list(
        nx.tree_all_pairs_lowest_common_ancestor(
            T_directed, root=root, pairs=leaf_pairs
        )
    )


def tree_offset_leaf_pairs(T, offset=1):
    """List of pairs of leaves in tree T separated by the given offset."""
    leaves = tree_leaves(T)
    n = len(leaves)
    return [(leaves[i], leaves[i + offset]) for i in range(n - offset)]


def tree_path_length(T, u, v):
    """Count of edges between u and v in tree T.

    Assumes u and v are on the same branch.
    Assumes v is closer to root.
    """

    if u == v:
        return 0

    # make the tree easier to traverse
    T_directed = as_directed_tree(T, root_to_leaf=False)

    # iterate towards root from u along successor nodes
    current_node = u
    path_length = 0
    while current_node != v:
        current_node = list(T_directed.successors(current_node))[0]
        path_length += 1

    return path_length


def leaf_pair_path_length_vector(T: nx.Graph, offset=1, normalise=True) -> np.array:
    """Array of path lengths in T from leaf i to leaf i+offset."""

    # get the leaf pairs from the offset
    leaf_pairs = tree_offset_leaf_pairs(T, offset=offset)

    # find the lowest common ancestors
    lcas = tree_lcas(T, tuple(leaf_pairs))

    # use the lcas to find the leaf-to-leaf path lengths
    path_lengths = []
    for (u, v), lca in lcas:
        length = 0
        length += tree_path_length(T, u, lca)
        length += tree_path_length(T, v, lca)
        path_lengths.append(length)

    # Normalise if required
    path_lengths = np.array(path_lengths)
    if normalise:
        path_lengths = path_lengths / np.sum(path_lengths)

    return path_lengths


def leaf_pair_path_cophenetic_vector(T: nx.Graph, offset=1, normalise=True) -> np.array:
    """List of path lengths in T from lca of leaves (i, i+offset) to root."""

    # get the leaf pairs from the offset
    leaf_pairs = tree_offset_leaf_pairs(T, offset=offset)

    # find the lowest common ancestors and root
    lcas = tree_lcas(T, tuple(leaf_pairs))
    root = get_root(T)

    # use the lcas to find the cophenetic distances
    path_lengths = []
    for (u, v), lca in lcas:
        length = tree_path_length(T, lca, root)
        path_lengths.append(length)

    # Normalise if required
    path_lengths = np.array(path_lengths)
    if normalise and (np.sum(path_lengths) > 0):
        path_lengths = path_lengths / np.sum(path_lengths)

    return path_lengths


def distribution_vec(samples, dim=100, min_val=0, max_val=1, rescale=True):
    """Vector representing distribution of sample values."""
    if len(samples) == 0:
        return np.zeros(shape=dim)
    if rescale:
        samples = rescale_vector(samples, min_val=min_val, max_val=max_val)
    vec, _ = np.histogram(samples, bins=dim, range=(min_val, max_val), density=True)
    return vec


"""Distance between distributions of path lengths between leaf i and i+offset"""


def get_offset_divergences(offset, tsmt=None):
    # normalised path length vectors in the tree for the given offset
    plv1 = leaf_pair_path_length_vector(tsmt.merge_tree, offset=offset)
    plv2 = leaf_pair_path_length_vector(
        make_increasing(tsmt.superlevel_merge_tree), offset=offset
    )

    # lp distances between path length vectors
    plv_l1 = np.linalg.norm(plv1 - plv2, ord=1)
    plv_l2 = np.linalg.norm(plv1 - plv2, ord=2)
    plv_linf = 0 if len(plv1) == 0 else np.linalg.norm(plv1 - plv2, ord=np.inf)

    # distributions of values in the path length vectors
    plv1_hist = distribution_vec(plv1)
    plv2_hist = distribution_vec(plv2)

    # wasserstein and lp distances between path length distributions
    plv_hist_w = 0 if (sum(plv1)*sum(plv2) == 0) else wasserstein_distance(plv1, plv2)
    plv_hist_l1 = np.linalg.norm(plv1_hist - plv2_hist, ord=1)
    plv_hist_l2 = np.linalg.norm(plv1_hist - plv2_hist, ord=2)
    plv_hist_linf = 0 if len(plv1_hist) == 0 else np.linalg.norm(plv1_hist - plv2_hist, ord=np.inf)

    # normalised cophenetic vectors in the tree for the given offset
    cov1 = leaf_pair_path_cophenetic_vector(tsmt.merge_tree, offset=offset)
    cov2 = leaf_pair_path_cophenetic_vector(
        make_increasing(tsmt.superlevel_merge_tree), offset=offset
    )

    # lp distances between path length vectors
    cov_l1 = np.linalg.norm(cov1 - cov2, ord=1)
    cov_l2 = np.linalg.norm(cov1 - cov2, ord=2)
    cov_linf = 0 if len(cov1) == 0 else np.linalg.norm(cov1 - cov2, ord=np.inf)

    # distributions of values in the path length vectors
    cov1_hist = distribution_vec(cov1)
    cov2_hist = distribution_vec(cov2)

    # wasserstein and lp distances between path length distributions
    cov_hist_w = 0 if (sum(cov1)*sum(cov2) == 0) else wasserstein_distance(cov1, cov2)
    cov_hist_l1 = np.linalg.norm(cov1_hist - cov2_hist, ord=1)
    cov_hist_l2 = np.linalg.norm(cov1_hist - cov2_hist, ord=2)
    cov_hist_linf = 0 if len(cov1_hist) == 0 else np.linalg.norm(cov1_hist - cov2_hist, ord=np.inf)

    div_values = [
        plv_l1,
        plv_l2,
        plv_linf,
        plv_hist_w,
        plv_hist_l1,
        plv_hist_l2,
        plv_hist_linf,
        cov_l1,
        cov_l2,
        cov_linf,
        cov_hist_w,
        cov_hist_l1,
        cov_hist_l2,
        cov_hist_linf,
    ]

    return div_values

div_names = [
    "path_length_l1",
    "path_length_l2",
    "path_length_linf",
    "path_length_hist_w",
    "path_length_hist_l1",
    "path_length_hist_l2",
    "path_length_hist_linf",
    "cophenetic_l1",
    "cophenetic_l2",
    "cophenetic_linf",
    "cophenetic_hist_w",
    "cophenetic_hist_l1",
    "cophenetic_hist_l2",
    "cophenetic_hist_linf",
]
