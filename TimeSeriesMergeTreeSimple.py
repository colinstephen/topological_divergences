import numpy as np
import scipy as sp
import math
import networkx as nx
from collections import Counter
from functools import lru_cache
from scipy.stats import wasserstein_distance
# from scipy.stats import entropy
from numpy.random import MT19937
from numpy.random import RandomState
from numpy.random import SeedSequence
from decorated_merge_trees.DMT_tools import MergeTree as DMTMergeTree
from decorated_merge_trees.DMT_tools import merge_tree_interleaving_distance

SEED = 123
randomState = RandomState(MT19937(SeedSequence(SEED)))


def perturb_array(arr, epsilon=1e-10):
    """Perturb array values slightly to make them distinct."""

    arr = np.array(arr)

    # get number of finite elements
    finite_array = arr[np.isfinite(arr)]
    num_finite = len(finite_array)

    # initialise a perturbed array and its finite elements
    perturbed = arr
    finite_perturbed = finite_array

    while len(np.unique(finite_perturbed)) != num_finite:

        noise = np.random.uniform(-epsilon, epsilon, len(arr))
        perturbed = np.array(arr) + noise
        finite_perturbed = perturbed[np.isfinite(perturbed)]

    return perturbed


def merge_tree(arr):
    """Sublevel set merge tree of given time series array.

    This computes the merge tree of the piecewise linear interpolation of
    the input array.

    Returns the tree as a `nx.Graph` object with node attributes `height` and
    `idx`. Height gives the filtration value at the node and `idx` gives the
    corresponding index of the node in `array`.
    """

    # Sort the array but keep original indices
    sorted_indices = sorted(range(len(arr)), key=lambda k: arr[k])
    sorted_values = [arr[i] for i in sorted_indices]

    # Initialize structures
    G = nx.Graph()
    components = {}  # current components, map from index to root
    next_node = 0  # next available node id for the tree

    # Process values in sorted order
    for idx, val in zip(sorted_indices, sorted_values):
        left = components.get(idx - 1)
        right = components.get(idx + 1)

        if left is None and right is None:
            # New component
            components[idx] = next_node
            G.add_node(next_node, height=val, idx=idx)
            next_node += 1

        elif left is not None and right is None:
            # Extend the left component
            components[idx] = left

        elif left is None and right is not None:
            # Extend the right component
            components[idx] = right

        else:
            # Merge left and right
            if left != right:
                G.add_node(next_node, height=val, idx=idx)
                G.add_edge(next_node, left)
                G.add_edge(next_node, right)
                components[idx] = next_node
                # Merge everything to new root
                for k, v in components.items():
                    if v == left or v == right:
                        components[k] = next_node
                next_node += 1
            else:
                # Same component, just extend
                components[idx] = left

    return G


def superlevel_merge_tree(arr):
    """Superlevel set merge tree of given time series array.

    This computes the superlevel merge tree of the piecewise linear
    interpolation of the input array.

    Implementation negates the array, computes the sublevel merge tree, then
    negates the resulting node heights.
    """

    # Convert the problem to sublevel sets by negating the array
    negated_array = [-val for val in arr]

    # Use the sublevel merge tree function
    G = merge_tree(negated_array)

    # Correct the heights (negate them back)
    for _, data in G.nodes(data=True):
        data["height"] = -data["height"]

    return G


def merge_tree_discrete(arr):
    """Sublevel set discrete merge tree of given time series array.

    This computes the merge tree over a chain graph whose edges are weighted
    with the values in `array` and whose nodes are weighted with `-np.inf`.

    Implementation simply interleaves `-np.inf` into `array` before computing
    the regular merge tree. This is equivalent.
    """
    new_array = [-np.inf]
    for x in arr:
        new_array.append(x)
        new_array.append(-np.inf)
    return merge_tree(new_array)


def superlevel_merge_tree_discrete(arr):
    """Superlevel set discrete merge tree of given time series array.

    This computes the superlevel merge tree over a chain graph whose edges are
    weighted with the values in `array` and whose nodes are weighted with
    `-np.inf`.

    Implementation negates the array, computes the sublevel discrete merge tree,
    then negates the resulting node heights.
    """
    negated_array = [-val for val in arr]
    G = merge_tree_discrete(negated_array)
    for _, data in G.nodes(data=True):
        data["height"] = -data["height"]
    return G


@lru_cache
def make_increasing(T):
    """Alter tree node heights so leaves are minimal and root is maximal."""

    tree = T.copy()
    min_height = minimum_finite_height(T)
    max_height = maximum_finite_height(T)

    # Update the heights of nodes in the transformed tree
    for _, data in tree.nodes(data=True):
        data["height"] = min_height + max_height - data["height"]

    return tree


@lru_cache
def as_directed_tree(T: nx.Graph, root_to_leaf=False) -> nx.DiGraph:
    """Copy of T that is a DiGraph
    
    Default direction of edges is from leaves to root.
    Note: assumes height attribute increases towards root.
    """
    tree = nx.DiGraph()
    for n, data in T.nodes(data=True):
        tree.add_node(n, **data)
    for u, v in T.edges():
        hu, hv = T.nodes[u]["height"], T.nodes[v]["height"]
        if root_to_leaf:
            if hu < hv:
                tree.add_edge(v, u)
            else:
                tree.add_edge(u, v)
        else:
            if hu < hv:
                tree.add_edge(u, v)
            else:
                tree.add_edge(v, u)
    return tree


def dmt_merge_tree(T):
    """Initialise a DMT-compatible tree from a standard NetworkX merge tree."""
    tree = T.__class__()
    tree.add_nodes_from(T)
    tree.add_edges_from(T.edges)
    height = {n: h for n, h in T.nodes(data="height")}
    return DMTMergeTree(tree=tree, height=height)


def tree_leaves(T, order_by_attr="idx"):
    """Get a list of degree-1 nodes."""
    leaves = [node for node, degree in T.degree() if degree == 1]
    if order_by_attr is not None:
        leaves = sorted(leaves, key=lambda n: T.nodes[n][order_by_attr])
    return leaves


def get_root(T: nx.Graph):
    """Find the degree-2 node in the tree."""
    for node, degree in T.degree():
        if degree == 2:
            return node
    # No degree 2 node, for some reason. Return the max height node.
    max_node = None
    max_node_height = -np.inf
    for node, data in T.nodes(data=True):
        if data["height"] > max_node_height:
            max_node = node
            max_node_height = data["height"]
    return max_node


def get_heights(T: nx.Graph):
    """Heights of nodes in the tree."""
    return np.array([data["height"] for _, data in T.nodes(data=True)])


def get_heights_finite(T: nx.Graph):
    """Finite heights of nodes in the tree."""
    heights = get_heights(T)
    return heights[np.isfinite(heights)]


def minimum_finite_height(T: nx.Graph):
    """Minimum (finite) height in the tree."""
    return min(get_heights_finite(T))


def maximum_finite_height(T: nx.Graph):
    """Maximum (finite) height in the tree."""
    return max(get_heights_finite(T))



@lru_cache
def sum_of_edge_lengths(T: nx.Graph):
    """Sum of edge lengths inferred from node heights."""
    length = 0
    for u, v in T.edges:
        h1, h2 = T.nodes[u]["height"], T.nodes[v]["height"]
        if not np.all(np.isfinite([h1, h2])):
            continue
        length += abs(h1 - h2)
    return length


def get_idx_pairs(n: int) -> list[tuple]:
    """List of distinct pairs of values (modulo order) between 0 and n-1."""
    idx_pairs = []
    for i in range(n-1):
        for j in range(i+1, n):
            idx_pairs.append((i,j))
    return idx_pairs


def get_leaf_pairs(T: nx.Graph) -> list:
    """List of pairs of leaves in T."""
    leaves = tree_leaves(T)
    n = len(leaves)
    idx_pairs = get_idx_pairs(n)
    leaf_pairs = [(leaves[i], leaves[j]) for i,j in idx_pairs]
    return leaf_pairs


@lru_cache
def get_lcas(T: nx.Graph) -> list[tuple]:
    """List of pairs of leaves and their lca nodes in T."""
    T_directed = as_directed_tree(T, root_to_leaf=True)
    root = get_root(T)
    leaf_pairs = get_leaf_pairs(T)
    lcas = nx.tree_all_pairs_lowest_common_ancestor(T_directed, root=root, pairs=leaf_pairs)
    return lcas


@lru_cache
def cophenetic_vector(T: nx.Graph) -> np.array:
    """List of node heights in T of leaf-pair lcas."""
    lcas = get_lcas(T)
    heights = []
    for ((u, v), lca) in lcas:
        heights.append(T.nodes[lca]["height"])
    return np.array(heights)


@lru_cache
def path_cophenetic_vector(T: nx.Graph) -> np.array:
    """List of path lengths in T from leaf-pair lcas to root."""
    lcas = get_lcas(T)
    root = get_root(T)
    path_lengths = []
    T_directed = as_directed_tree(T, root_to_leaf=False)
    for ((u, v), lca) in lcas:
        path_node = lca
        path_length = 0
        while path_node != root:
            path_node = list(T_directed.successors(path_node))[0]
            path_length = path_length + 1
        path_lengths.append(path_length)
    return np.array(path_lengths)

@lru_cache
def in_path_vector(T: nx.Graph) -> np.array:
    """List of path lengths in T from left of leaf-pair to lca."""
    lcas = get_lcas(T)
    path_lengths = []
    T_directed = as_directed_tree(T, root_to_leaf=False)
    for ((u, v), lca) in lcas:
        path_node = u
        path_length = 0
        while path_node != lca:
            path_node = list(T_directed.successors(path_node))[0]
            path_length = path_length + 1
        path_lengths.append(path_length)
    return np.array(path_lengths)


@lru_cache
def out_path_vector(T: nx.Graph) -> np.array:
    """List of path lengths in T from right of leaf-pair to lca."""
    lcas = get_lcas(T)
    path_lengths = []
    T_directed = as_directed_tree(T, root_to_leaf=False)
    for ((u, v), lca) in lcas:
        path_node = v
        path_length = 0
        while path_node != lca:
            path_node = list(T_directed.successors(path_node))[0]
            path_length = path_length + 1
        path_lengths.append(path_length)
    return np.array(path_lengths)


def monotonize(ts):
    # forget intermediate non-critical points and equalize count of minima/maxima
    new_ts = [ts[0]]
    N = len(ts)
    for idx in range(1,N-1):
        x, y, z = ts[idx-1:idx+2]
        if (((x<y) and (z<y)) or ((x>y) and (z>y))):
            # add the local max/min
            new_ts.append(y)
    if (len(new_ts) % 2) == 1:
        new_ts.append(ts[-1])

    is_monotonic = lambda x: (np.all(x[::2]<x[1::2]) or np.all(x[::2]>x[1::2]))
    assert is_monotonic(new_ts), "new time series has non-critical values, somehow"
    return new_ts



class TimeSeriesMergeTree:
    """Define merge tree vectorisations and topological divergences."""

    def __init__(
        self,
        time_series,
        discrete=False,
        MESHES=[0.5, 0.4],
        THRESHES=[None, 0.1],
        DISTRIBUTION_VECTOR_LENGTH=100,
    ) -> None:
        self.time_series = perturb_array(time_series)
        self.discrete = discrete
        if not self.discrete:
            self.time_series = monotonize(time_series)
        self.MESHES = MESHES
        self.THRESHES = THRESHES
        self.DISTRIBUTION_VECTOR_LENGTH = DISTRIBUTION_VECTOR_LENGTH
        self._merge_tree = None
        self._superlevel_merge_tree = None

    @property
    def merge_tree(self):
        if self._merge_tree is None:
            if self.discrete:
                self._merge_tree = merge_tree_discrete(self.time_series)
            else:
                self._merge_tree = merge_tree(self.time_series)
        return self._merge_tree

    @property
    def superlevel_merge_tree(self):
        if self._superlevel_merge_tree is None:
            if self.discrete:
                self._superlevel_merge_tree = superlevel_merge_tree_discrete(
                    self.time_series
                )
            else:
                self._superlevel_merge_tree = superlevel_merge_tree(self.time_series)
        return self._superlevel_merge_tree


    @property
    def vectorisations(self):
        # vector representations of the merge tree
        pass



    @property
    def divergences(self):
        # dictionary of divergences associated to the merge tree representation
        divs = dict()

        # if self.discrete:
        if True:
            # divergences that require equal leaf counts in the superlevel and sublevel trees
            divs = divs | dict(
                cophenetic=self.cophenetic_vector_divergence,
                cophenetic_reverse=self.cophenetic_reverse_vector_divergence,
                cophenetic_linf=self.cophenetic_vector_divergence_linf,
                cophenetic_reverse_linf=self.cophenetic_reverse_vector_divergence_linf,
                # cophenetic_length_linf=self.length_normalised_cophenetic_vector_divergence_linf,
                # cophenetic_reverse_length_linf=self.length_normalised_cophenetic_reverse_vector_divergence_linf,
                # cophenetic_normed_linf=self.cophenetic_normalised_vector_divergence,
                # cophenetic_reverse_normed_linf=self.cophenetic_reverse_normalised_vector_divergence,
                # path_cophenetic=self.path_cophenetic_vector_divergence,  ## NO GOOD
                # path_copehentic_reverse=self.path_cophenetic_reverse_vector_divergence,  ## NO GOOD
                # path_div_in_top_in_bot=self.path_divergence_in_top_in_bot,  ## NO GOOD
                # path_div_in_top_out_bot=self.path_divergence_in_top_out_bot,  ## NO GOOD
                # path_div_in_top_out_bot=self.path_divergence_in_top_out_bot,  ## NO GOOD
                # path_div_in_top_out_bot_reverse=self.path_divergence_in_top_out_bot_reverse,  ## NO GOOD
                # cophenetic_length=self.length_normalised_cophenetic_vector_divergence,
                # cophenetic_reverse_length=self.length_normalised_cophenetic_reverse_vector_divergence,
                # cophenetic_edge=self.edge_normalised_cophenetic_vector_divergence,
                # cophenetic_reverse_edge=self.edge_normalised_cophenetic_reverse_vector_divergence,
            )
        else:
            divs = divs | dict(
                # cophenetic=self.cophenetic_vector_divergence,
                # cophenetic_reverse=self.cophenetic_reverse_vector_divergence,
                # cophenetic_linf=self.cophenetic_vector_divergence_linf,
                # cophenetic_reverse_linf=self.cophenetic_reverse_vector_divergence_linf,
                # cophenetic_length=self.length_normalised_cophenetic_vector_divergence,
                # cophenetic_reverse_length=self.length_normalised_cophenetic_reverse_vector_divergence,
                # cophenetic_edge=self.edge_normalised_cophenetic_vector_divergence,
                # cophenetic_reverse_edge=self.edge_normalised_cophenetic_reverse_vector_divergence,
            )

        # else:
        #     # interleavings for PL functions (interleavings for discrete MTs are too slow)
        #     meshes = self.MESHES
        #     threshes = self.THRESHES

        #     for mesh in meshes:
        #         for thresh in threshes:

        #             if thresh is not None and mesh <= thresh:
        #                 continue

        #             divs = divs | {
        #                 f"interleaving_{mesh}_{thresh}": self.interleaving_divergence(mesh, thresh),
        #                 f"interleaving_length_{mesh}_{thresh}": self.length_normalised_interleaving_divergence(mesh, thresh),
        #                 f"interleaving_edge_{mesh}_{thresh}": self.edge_normalised_interleaving_divergence(mesh, thresh),
        #             }

        return divs

    @lru_cache()
    def interleaving_divergence(self, mesh, thresh):
        # merge tree interleaving between super and sub level trees
        T1 = self.merge_tree
        T2 = make_increasing(self.superlevel_merge_tree)
        for node, data in T1.nodes(data=True):
            if not np.isfinite(data["height"]):
                neighbour = list(nx.neighbors(T1, node))[0]
                data["height"] = T1.nodes[neighbour]["height"] - 1.5 * mesh
        for node, data in T2.nodes(data=True):
            if not np.isfinite(data["height"]):
                neighbour = list(nx.neighbors(T2, node))[0]
                data["height"] = T2.nodes[neighbour]["height"] - 1.5 * mesh
        MT1 = dmt_merge_tree(T1)
        MT2 = dmt_merge_tree(T2)
        if thresh is not None:
            try:
                MT1_thresh = MT1.copy()
                MT2_thresh = MT2.copy()
                MT1_thresh.threshold(thresh)
                MT2_thresh.threshold(thresh)
                MT1 = MT1_thresh
                MT2 = MT2_thresh
            except Exception as e:
                print("WARNING: threshold operation raised an exception:", e)
        try:
            distance = merge_tree_interleaving_distance(MT1, MT2, mesh, verbose=False)
        except Exception as e:
            print("WARNING: interleaving distance raised an exception:", e)
            distance = -1
        return distance    

    def length_normalised_interleaving_divergence(self, mesh, thresh):
        # normalise interleaving by total height of tree
        l1 = sum_of_edge_lengths(self.merge_tree)
        l2 = sum_of_edge_lengths(self.superlevel_merge_tree)
        return self.interleaving_divergence(mesh, thresh) / (l1 + l2)

    def edge_normalised_interleaving_divergence(self, mesh, thresh):
        # normalise interleaving by total number of edges
        n1 = self.merge_tree.number_of_edges()
        n2 = self.superlevel_merge_tree.number_of_edges()
        return self.interleaving_divergence(mesh, thresh) / (n1 + n2)
    
    @property
    def path_divergence_in_top_in_bot(self):
        v1 = in_path_vector(self.merge_tree)
        v2 = in_path_vector(make_increasing(self.superlevel_merge_tree))
        v1 = v1 / self.merge_tree.number_of_edges()
        v2 = v2 / self.superlevel_merge_tree.number_of_edges()
        return wasserstein_distance(v1, v2)
    
    @property
    def path_divergence_in_top_in_bot_reverse(self):
        v1 = in_path_vector(self.merge_tree)
        v2 = in_path_vector(make_increasing(self.superlevel_merge_tree))[::-1]
        v1 = v1 / self.merge_tree.number_of_edges()
        v2 = v2 / self.superlevel_merge_tree.number_of_edges()
        return wasserstein_distance(v1, v2)
    
    @property
    def path_divergence_in_top_out_bot(self):
        v1 = in_path_vector(self.merge_tree)
        v2 = out_path_vector(make_increasing(self.superlevel_merge_tree))
        v1 = v1 / self.merge_tree.number_of_edges()
        v2 = v2 / self.superlevel_merge_tree.number_of_edges()
        return wasserstein_distance(v1, v2)
    
    @property
    def path_divergence_in_top_out_bot_reverse(self):
        v1 = in_path_vector(self.merge_tree)
        v2 = out_path_vector(make_increasing(self.superlevel_merge_tree))[::-1]
        v1 = v1 / self.merge_tree.number_of_edges()
        v2 = v2 / self.superlevel_merge_tree.number_of_edges()
        return wasserstein_distance(v1, v2)
    
    @property
    def path_cophenetic_vector_divergence(self):
        v1 = path_cophenetic_vector(self.merge_tree)
        v2 = path_cophenetic_vector(make_increasing(self.superlevel_merge_tree))
        v1 = v1 / np.sum(v1)
        v2 = v2 / np.sum(v2)
        return np.linalg.norm(v1 - v2, ord=np.inf)

    @property
    def path_cophenetic_reverse_vector_divergence(self):
        v1 = path_cophenetic_vector(self.merge_tree)
        v2 = path_cophenetic_vector(make_increasing(self.superlevel_merge_tree))
        v1 = v1 / np.sum(v1)
        v2 = v2 / np.sum(v2)
        return np.linalg.norm(v1 - v2, ord=np.inf)

    @property
    @lru_cache
    def cophenetic_vector_divergence(self):
        v1 = cophenetic_vector(self.merge_tree)
        v2 = cophenetic_vector(make_increasing(self.superlevel_merge_tree))
        if len(v1) != len(v2):
            print("WARNING: cophenetic vectors are different lengths. How?!")
            return np.nan
        return np.linalg.norm(v1 - v2)
    
    @property
    @lru_cache
    def cophenetic_normalised_vector_divergence(self):
        v1 = cophenetic_vector(self.merge_tree)
        v1 = v1 / np.sum(v1)
        v2 = cophenetic_vector(make_increasing(self.superlevel_merge_tree))
        v2 = v2 / np.sum(v2)
        if len(v1) != len(v2):
            print("WARNING: cophenetic vectors are different lengths. How?!")
            return np.nan
        return np.linalg.norm(v1 - v2, ord=np.inf)
    
    @property
    @lru_cache
    def cophenetic_reverse_normalised_vector_divergence(self):
        v1 = cophenetic_vector(self.merge_tree)
        v1 = v1 / np.sum(v1)
        v2 = cophenetic_vector(make_increasing(self.superlevel_merge_tree))[::-1]
        v2 = v2 / np.sum(v2)
        if len(v1) != len(v2):
            print("WARNING: cophenetic vectors are different lengths. How?!")
            return np.nan
        return np.linalg.norm(v1 - v2, ord=np.inf)
    
    @property
    @lru_cache
    def cophenetic_reverse_vector_divergence(self):
        v1 = cophenetic_vector(self.merge_tree)
        v2 = cophenetic_vector(make_increasing(self.superlevel_merge_tree))[::-1]
        if len(v1) != len(v2):
            print("WARNING: cophenetic vectors are different lengths. How?!")
            return np.nan
        return np.linalg.norm(v1 - v2)

    @property
    @lru_cache
    def cophenetic_vector_divergence_linf(self):
        v1 = cophenetic_vector(self.merge_tree)
        v2 = cophenetic_vector(make_increasing(self.superlevel_merge_tree))
        if len(v1) != len(v2):
            print("WARNING: cophenetic vectors are different lengths. How?!")
            return np.nan
        return np.linalg.norm(v1 - v2, ord=np.inf)
    
    @property
    @lru_cache
    def cophenetic_reverse_vector_divergence_linf(self):
        v1 = cophenetic_vector(self.merge_tree)
        v2 = cophenetic_vector(make_increasing(self.superlevel_merge_tree))[::-1]
        if len(v1) != len(v2):
            print("WARNING: cophenetic vectors are different lengths. How?!")
            return np.nan
        return np.linalg.norm(v1 - v2, ord=np.inf)

    @property
    def length_normalised_cophenetic_vector_divergence(self):
        # normalise divergence by total height of tree
        l1 = sum_of_edge_lengths(self.merge_tree)
        l2 = sum_of_edge_lengths(self.superlevel_merge_tree)
        return self.cophenetic_vector_divergence / (l1 + l2)

    @property
    def length_normalised_cophenetic_vector_divergence_linf(self):
        # normalise divergence by total height of tree
        l1 = sum_of_edge_lengths(self.merge_tree)
        l2 = sum_of_edge_lengths(self.superlevel_merge_tree)
        return self.cophenetic_vector_divergence_linf / (l1 + l2)

    @property
    def edge_normalised_cophenetic_vector_divergence(self):
        # normalise divergence by total number of edges
        n1 = self.merge_tree.number_of_edges()
        n2 = self.superlevel_merge_tree.number_of_edges()
        return self.cophenetic_vector_divergence / (n1 + n2)

    @property
    def length_normalised_cophenetic_reverse_vector_divergence(self):
        # normalise divergence by total height of tree
        l1 = sum_of_edge_lengths(self.merge_tree)
        l2 = sum_of_edge_lengths(self.superlevel_merge_tree)
        return self.cophenetic_reverse_vector_divergence / (l1 + l2)

    @property
    def length_normalised_cophenetic_reverse_vector_divergence_linf(self):
        # normalise divergence by total height of tree
        l1 = sum_of_edge_lengths(self.merge_tree)
        l2 = sum_of_edge_lengths(self.superlevel_merge_tree)
        return self.cophenetic_reverse_vector_divergence_linf / (l1 + l2)

    @property
    def edge_normalised_cophenetic_reverse_vector_divergence(self):
        # normalise divergence by total number of edges
        n1 = self.merge_tree.number_of_edges()
        n2 = self.superlevel_merge_tree.number_of_edges()
        return self.cophenetic_reverse_vector_divergence / (n1 + n2)
