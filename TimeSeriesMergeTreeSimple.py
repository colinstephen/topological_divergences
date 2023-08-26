import numpy as np
import math
import networkx as nx
from collections import Counter
from functools import lru_cache
from scipy.stats import wasserstein_distance
from numpy.random import MT19937
from numpy.random import RandomState
from numpy.random import SeedSequence
from decorated_merge_trees.DMT_tools import MergeTree as DMTMergeTree
from decorated_merge_trees.DMT_tools import merge_tree_interleaving_distance

SEED = 123
randomState = RandomState(MT19937(SeedSequence(SEED)))


def perturb_array(array, epsilon=1e-10):
    """Perturb array values slightly to make them distinct."""
    noise = randomState.uniform(-epsilon, epsilon, len(array))
    return np.array(array) + noise


def merge_tree(array):
    """Return sublevel set merge tree of given time series array.

    This computes the merge tree of the piecewise linear interpolation of
    the input array.

    Returns the tree as a `nx.Graph` object with node attributes `height` and
    `idx`. Height gives the filtration value at the node and `idx` gives the
    corresponding index of the node in `array`.
    """
    array = perturb_array(array)

    # Sort the array but keep original indices
    sorted_indices = sorted(range(len(array)), key=lambda k: array[k])
    sorted_values = [array[i] for i in sorted_indices]

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


def superlevel_merge_tree(array):
    """Return superlevel set merge tree of given time series array.

    This computes the superlevel merge tree of the piecewise linear
    interpolation of the input array.

    Implementation negates the array, computes the sublevel merge tree, then
    negates the resulting node heights.
    """

    # Convert the problem to sublevel sets by negating the array
    negated_array = [-val for val in array]

    # Use the sublevel merge tree function
    G = merge_tree(negated_array)

    # Correct the heights (negate them back)
    for _, data in G.nodes(data=True):
        data["height"] = -data["height"]

    return G


def merge_tree_discrete(array):
    """Return sublevel set discrete merge tree of given time series array.

    This computes the merge tree over a chain graph whose edges are weighted
    with the values in `array` and whose nodes are weighted with `-np.inf`.

    Implementation simply interleaves `-np.inf` into `array` before computing
    the regular merge tree. This is equivalent.
    """
    new_array = [-np.inf]
    for x in array:
        new_array.append(x)
        new_array.append(-np.inf)
    return merge_tree(new_array)


def superlevel_merge_tree_discrete(array):
    """Return superlevel set discrete merge tree of given time series array.

    This computes the superlevel merge tree over a chain graph whose edges are
    weighted with the values in `array` and whose nodes are weighted with
    `-np.inf`.

    Implementation negates the array, computes the sublevel discrete merge tree,
    then negates the resulting node heights.
    """
    negated_array = [-val for val in array]
    G = merge_tree_discrete(negated_array)
    for _, data in G.nodes(data=True):
        data["height"] = -data["height"]
    return G


def as_directed_tree(T: nx.Graph) -> nx.DiGraph:
    """Return copy of T that is a DiGraph with edges from leaves to root.

    Note: assumes height attribute increases towards root.
    """
    tree = nx.DiGraph()
    for n, data in T.nodes(data=True):
        tree.add_node(n, **data)
    for u, v in T.edges():
        hu, hv = T.nodes[u]["height"], T.nodes[v]["height"]
        if hu < hv:
            tree.add_edge(u, v)
        else:
            tree.add_edge(v, u)
    return tree


def lca(T: nx.DiGraph, u, v, root):
    """Least common ancestor of u,v in T.

    Tree is assumed directed from leaves to `root`.
    """
    assert nx.is_tree(T)
    p1 = nx.shortest_path(T, u, root)
    p2 = nx.shortest_path(T, v, root)
    for n in p1:
        if n in p2:
            return n


def leaf_to_leaf_path_length(T: nx.DiGraph, u, v, root):
    """Count of edges between u and v in T."""
    assert nx.is_tree(T)
    return len(leaf_to_leaf_path(T, u, v, root)) - 1


def leaf_to_leaf_path(T: nx.DiGraph, u, v, root):
    """Edges between leaves u and v in T."""
    assert nx.is_tree(T)
    p1 = nx.shortest_path(T, u, root)  # fast for a digraph
    p2 = nx.shortest_path(T, v, root)  # fast for a digraph
    path = []
    for n in p1:
        path.append(n)
        if n in p2:
            break
    for n in p2[::-1]:
        if n in p1:
            continue
        path.append(n)
    return path


def make_increasing(T):
    """Alter tree node heights so leaves are minimal and root is maximal."""

    tree = T.copy()
    min_height = minimum_height(T)
    max_height = maximum_height(T)

    # Update the heights of nodes in the transformed tree
    for _, data in tree.nodes(data=True):
        data["height"] = min_height + max_height - data["height"]

    return tree


def dmt_merge_tree(T):
    """Initialise a DMT-compatible tree from a standard NetworkX merge tree."""
    tree = T.__class__()
    tree.add_nodes_from(T)
    tree.add_edges_from(T.edges)
    height = {n: h for n, h in T.nodes(data="height")}
    return DMTMergeTree(tree=tree, height=height)


def get_leaves(T, order_by_attr="idx"):
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


def get_heights(T: nx.Graph):
    """List all heights of nodes in the tree."""
    return np.array([data["height"] for _, data in T.nodes(data=True)])


def get_heights_finite(T: nx.Graph):
    """List all finite heights of nodes in the tree."""
    heights = get_heights(T)
    return heights[np.isfinite(heights)]


def minimum_height(T: nx.Graph):
    """Find minimum (finite) height in the tree."""
    return min(get_heights_finite(T))


def maximum_height(T: nx.Graph):
    """Find maximum (finite) height in the tree."""
    return max(get_heights_finite(T))


def leaf_to_leaf_path_lengths(T: nx.Graph, offset=1):
    """List path lengths from leaf `n` to leaf `n + offset` over all `n`.

    Tree `T` is assumed ordered by node attribute `idx`.

    Note: if `T` is a sublevel discrete merge tree and `offset=1` then the
    returned sequence is the ordered degree sequence of nodes in the horizon
    visibility graph for the time series that generated `T`.
    """
    ordered_leaf_nodes = get_leaves(T, order_by_attr="idx")
    num_leaves = len(ordered_leaf_nodes)

    if num_leaves - offset < 1:
        return [0]

    # use a directed tree to speed up path length computations
    T_directed = as_directed_tree(T)
    root = get_root(T)

    lengths = []
    for i in range(num_leaves - offset):
        node1 = ordered_leaf_nodes[i]
        node2 = ordered_leaf_nodes[i + offset]
        length = leaf_to_leaf_path_length(T_directed, node1, node2, root)
        lengths.append(length)

    return lengths


def distribution_from_samples(sample_array: list[int], DISTRIBUTION_VECTOR_LENGTH=100):
    """Empirical distribution of integers in `sample_array`.

    Returns an array, `distribution`, such that `distribution[n]` contains relative
    frequency of occurrence of value `n`.
    """
    item_counts = Counter(sample_array)
    n_samples = len(sample_array)
    empirical_distribution = {int(k): v / n_samples for k, v in item_counts.items()}
    distribution = np.zeros(DISTRIBUTION_VECTOR_LENGTH)
    for k in sorted(empirical_distribution.keys()):
        if k >= DISTRIBUTION_VECTOR_LENGTH:
            break
        distribution[k] = empirical_distribution[k]
    return distribution


def sum_of_edge_lengths(T: nx.Graph):
    """Infer edge lengths from node heights and add them together."""
    length = 0
    for u, v in T.edges:
        h1, h2 = T.nodes[u]["height"], T.nodes[v]["height"]
        if not np.all(np.isfinite([h1, h2])):
            continue
        length += abs(h1 - h2)
    return length


def distance_matrix(T: nx.Graph) -> np.array:
    """Path lengths between all pairs of leaves."""
    leaves = get_leaves(T)
    n = len(leaves)
    D = np.zeros((n, n))
    for offset in range(1, n):
        D[range(n - offset), range(offset, n)] = leaf_to_leaf_path_lengths(
            T, offset=offset
        )
    D = D + D.T
    return D


def cophenetic_matrix(T: nx.Graph) -> np.array:
    """Cophenetic distance between all pairs of leaves."""
    leaves = get_leaves(T)
    n = len(leaves)

    # use a directed tree to speed up path length computations
    T_directed = as_directed_tree(T)
    root = get_root(T)

    D = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            path = leaf_to_leaf_path(T_directed, leaves[i], leaves[j], root)
            heights = [T.nodes[n]["height"] for n in path]
            D[i, j] = max(heights)
    D = D + D.T
    return D


class TimeSeriesMergeTree:
    """Access merge tree based topological divergences."""

    def __init__(
        self,
        time_series,
        discrete=False,
        INTERLEAVING_DIVERGENCE_MESH=0.5,
        DISTRIBUTION_VECTOR_LENGTH=100,
    ) -> None:
        self.time_series = time_series
        self.discrete = discrete
        self.INTERLEAVING_DIVERGENCE_MESH = INTERLEAVING_DIVERGENCE_MESH
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
    def divergences(self):
        # dictionary of divergences associated to the merge tree representation
        divs = dict(
            interleaving=self.interleaving_divergence,
            length_normalised_interleaving=self.length_normalised_interleaving_divergence,
            edge_normalised_interleaving=self.edge_normalised_interleaving_divergence,
        )
        for offset in range(1, 51):
            divs = divs | {
                f"offset_path_length_{offset}": self.offset_path_length_distribution_divergences(
                    offset
                )
            }
        if self.discrete:
            # only defined for discrete time series merge trees
            divs = divs | dict(
                distance_matrix=self.distance_matrix_divergence,
                edge_normalised_distance_matrix=self.edge_normalised_distance_matrix_divergence,
                cophenetic_matrix=self.cophenetic_matrix_divergence,
                length_normalised_cophenetic_matrix=self.length_normalised_cophenetic_matrix_divergence,
            )
        return divs

    @property
    @lru_cache()
    def interleaving_divergence(self):
        # merge tree interleaving between super and sub level trees
        mesh = self.INTERLEAVING_DIVERGENCE_MESH
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
        return merge_tree_interleaving_distance(
            dmt_merge_tree(T1), dmt_merge_tree(T2), mesh, verbose=False
        )

    @property
    def length_normalised_interleaving_divergence(self):
        # normalise interleaving by total height of tree
        l1 = sum_of_edge_lengths(self.merge_tree)
        l2 = sum_of_edge_lengths(self.superlevel_merge_tree)
        return self.interleaving_divergence / (l1 + l2)

    @property
    def edge_normalised_interleaving_divergence(self):
        # normalise interleaving by total number of edges
        n1 = self.merge_tree.number_of_edges()
        n2 = self.superlevel_merge_tree.number_of_edges()
        return self.interleaving_divergence / (n1 + n2)

    def offset_path_length_distribution_divergences(self, offset):
        # wasserstein distance between pairwise leaf-to-leaf path length distributions
        T1 = self.merge_tree
        T2 = make_increasing(self.superlevel_merge_tree)
        l1 = leaf_to_leaf_path_lengths(T1, offset=offset)
        l2 = leaf_to_leaf_path_lengths(T2, offset=offset)
        d1 = distribution_from_samples(
            l1, DISTRIBUTION_VECTOR_LENGTH=self.DISTRIBUTION_VECTOR_LENGTH
        )
        d2 = distribution_from_samples(
            l2, DISTRIBUTION_VECTOR_LENGTH=self.DISTRIBUTION_VECTOR_LENGTH
        )
        return wasserstein_distance(d1, d2)

    @property
    @lru_cache()
    def distance_matrix_divergence(self):
        # frobenius norm of difference between leaf-to-leaf path lengths
        D1 = distance_matrix(self.merge_tree)
        D2 = distance_matrix(make_increasing(self.superlevel_merge_tree))
        return np.linalg.norm(D1 - D2)

    @property
    def edge_normalised_distance_matrix_divergence(self):
        # normalise frobenius norm by total number of edges
        n1 = self.merge_tree.number_of_edges()
        n2 = self.superlevel_merge_tree.number_of_edges()
        return self.distance_matrix_divergence / (n1 + n2)

    @property
    @lru_cache
    def cophenetic_matrix_divergence(self):
        # frobenius norm of difference between leaf-to-leaf cophenetic distances
        D1 = cophenetic_matrix(self.merge_tree)
        D2 = cophenetic_matrix(make_increasing(self.superlevel_merge_tree))
        return np.linalg.norm(D1 - D2)

    @property
    def length_normalised_cophenetic_matrix_divergence(self):
        # normalise interleaving by total height of tree
        l1 = sum_of_edge_lengths(self.merge_tree)
        l2 = sum_of_edge_lengths(self.superlevel_merge_tree)
        return self.cophenetic_matrix_divergence / (l1 + l2)


if __name__ == "__main__":
    # Test the function
    import numpy as np
    from matplotlib import pyplot as plt

    array = np.random.uniform(size=100)

    plt.plot(array)
    plt.show()

    G = merge_tree(array)
    print(G.nodes(data=True))
    print(G.edges())
    print(nx.is_tree(G))
    print(leaf_to_leaf_path_lengths(as_directed_tree(G), offset=1))
    print(leaf_to_leaf_path_lengths(as_directed_tree(G), offset=2))
    nx.draw(
        G,
        with_labels=True,
        labels={n: (data["height"], data["idx"]) for n, data in G.nodes(data=True)},
    )
    plt.show()

    G2 = superlevel_merge_tree(array)
    print(G2.nodes(data=True))
    print(G2.edges())
    print(nx.is_tree(G2))
    nx.draw(
        G2,
        with_labels=True,
        labels={n: (data["height"], data["idx"]) for n, data in G2.nodes(data=True)},
    )
    plt.show()

    G3 = make_increasing(G2)
    print(G3.nodes(data=True))
    print(G3.edges())
    print(nx.is_tree(G3))
    nx.draw(
        G3,
        with_labels=True,
        labels={n: (data["height"], data["idx"]) for n, data in G3.nodes(data=True)},
    )
    plt.show()

    tsmt = TimeSeriesMergeTree(time_series=array, INTERLEAVING_DIVERGENCE_MESH=0.1)
    divergences = tsmt.divergences
    print(divergences)

    G = merge_tree_discrete(array)
    print(G.nodes(data=True))
    print(G.edges())
    print(nx.is_tree(G))
    print(leaf_to_leaf_path_lengths(as_directed_tree(G), offset=1))
    print(leaf_to_leaf_path_lengths(as_directed_tree(G), offset=2))
    nx.draw(
        G,
        with_labels=True,
        labels={n: (data["height"], data["idx"]) for n, data in G.nodes(data=True)},
    )
    plt.show()

    G2 = superlevel_merge_tree_discrete(array)
    print(G2.nodes(data=True))
    print(G2.edges())
    print(nx.is_tree(G2))
    nx.draw(
        G2,
        with_labels=True,
        labels={n: (data["height"], data["idx"]) for n, data in G2.nodes(data=True)},
    )
    plt.show()

    G3 = make_increasing(G2)
    print(G3.nodes(data=True))
    print(G3.edges())
    print(nx.is_tree(G3))
    nx.draw(
        G3,
        with_labels=True,
        labels={n: (data["height"], data["idx"]) for n, data in G3.nodes(data=True)},
    )
    plt.show()

    tsmt = TimeSeriesMergeTree(
        time_series=array, discrete=True, INTERLEAVING_DIVERGENCE_MESH=0.1
    )
    divergences = tsmt.divergences
    print(divergences)
