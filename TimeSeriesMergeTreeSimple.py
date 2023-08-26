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
    """Build the sublevel set merge tree of the given time series array."""
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


def merge_tree_discrete(array):
    # interleave -np.inf into the array before computing the tree
    new_array = [-np.inf]
    for x in array:
        new_array.append(x)
        new_array.append(-np.inf)
    return merge_tree(new_array)


def superlevel_merge_tree_discrete(array):
    negated_array = [-val for val in array]
    G = merge_tree_discrete(negated_array)
    for _, data in G.nodes(data=True):
        data["height"] = -data["height"]
    return G

def superlevel_merge_tree(array):
    # Convert the problem to sublevel sets by negating the array
    negated_array = [-val for val in array]

    # Use the sublevel merge tree function
    G = merge_tree(negated_array)

    # Correct the heights (negate them back)
    for _, data in G.nodes(data=True):
        data["height"] = -data["height"]

    return G


def make_increasing(T):
    # Alter node heights so leaves are minimal and root is maximal

    tree = T.copy()
    min_height = min(height for _, height in tree.nodes(data="height") if np.isfinite(height))
    max_height = max(height for _, height in tree.nodes(data="height") if np.isfinite(height))

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
    leaves = [node for node, degree in T.degree() if degree == 1]
    if order_by_attr is not None:
        leaves = sorted(leaves, key=lambda n: T.nodes[n][order_by_attr])
    return leaves

def get_root(T: nx.Graph):
    for node, degree in T.degree():
        if degree == 2:
            return node

def get_heights(T: nx.Graph):
    return np.array([data["height"] for _, data in T.nodes(data=True)])

def get_heights_finite(T: nx.Graph):
    heights = get_heights(T)
    return heights[np.isfinite(heights)]

def minimum_height(T: nx.Graph):
    return min(get_heights_finite(T))

def maximum_height(T: nx.Graph):
    return max(get_heights_finite(T))

@lru_cache
def leaf_to_leaf_path_lengths(T, offset=1):
    ordered_leaf_nodes = get_leaves(T)
    num_leaves = len(ordered_leaf_nodes)

    if num_leaves - offset < 1:
        return [0]

    lengths = []
    for i in range(num_leaves - offset):
        node1 = ordered_leaf_nodes[i]
        node2 = ordered_leaf_nodes[i + offset]
        length = nx.shortest_path_length(T, source=node1, target=node2)
        lengths.append(length)

    return lengths


def distribution_from_samples(sample_array, DISTRIBUTION_VECTOR_LENGTH=100):
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
    # Infer edge lengths from node heights and add them together
    length = 0
    for u, v in T.edges:
        h1, h2 = T.nodes[u]["height"], T.nodes[v]["height"]
        if not np.all(np.isfinite([h1, h2])):
            continue
        length += abs(h1 - h2)
    return length


def distance_matrix(T: nx.Graph) -> np.array:
    # path lengths between all pairs of leaves 
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
    # cophenetic distance between all pairs of leaves
    leaves = get_leaves(T)
    n = len(leaves)

    D = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1, n):
            path = nx.shortest_path(T, leaves[i], leaves[j])
            heights = [T.nodes[n]["height"] for n in path]
            D[i,j] = max(heights)
    D = D + D.T
    return D

class TimeSeriesMergeTree:
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

    @property
    def merge_tree(self):
        if self.discrete:
            return merge_tree_discrete(self.time_series)
        else:
            return merge_tree(self.time_series)

    @property
    def superlevel_merge_tree(self):
        if self.discrete:
            return superlevel_merge_tree_discrete(self.time_series)
        else:
            return superlevel_merge_tree(self.time_series)

    @property
    def divergences(self):
        divs = dict(
            interleaving=self.interleaving_divergence,
            length_normalised_interleaving=self.length_normalised_interleaving_divergence,
            edge_normalised_interleaving=self.edge_normalised_interleaving_divergence,
            offset_path_lengths=self.offset_path_length_distribution_divergences,
        )
        if self.discrete:
            divs = divs | dict(
                distance_matrix=self.distance_matrix_divergence,
                cophenetic_matrix=self.cophenetic_matrix_divergence,
            )
        return divs

    @property
    @lru_cache()
    def interleaving_divergence(self):
        mesh = self.INTERLEAVING_DIVERGENCE_MESH
        T1 = self.merge_tree
        T2 = make_increasing(self.superlevel_merge_tree)
        for node, data in T1.nodes(data=True):
            if not np.isfinite(data["height"]):
                neighbour = list(nx.neighbors(T1, node))[0]
                data["height"] = T1.nodes[neighbour]["height"] - 2*mesh
        for node, data in T2.nodes(data=True):
            if not np.isfinite(data["height"]):
                neighbour = list(nx.neighbors(T2, node))[0]
                data["height"] = T2.nodes[neighbour]["height"] - 2*mesh
        return merge_tree_interleaving_distance(
            dmt_merge_tree(T1), dmt_merge_tree(T2), mesh, verbose=False
        )

    @property
    def length_normalised_interleaving_divergence(self):
        l1 = sum_of_edge_lengths(self.merge_tree)
        l2 = sum_of_edge_lengths(self.superlevel_merge_tree)
        return self.interleaving_divergence / (l1 + l2)

    @property
    def edge_normalised_interleaving_divergence(self):
        n1 = self.merge_tree.number_of_edges()
        n2 = self.superlevel_merge_tree.number_of_edges()
        return self.interleaving_divergence / (n1 + n2)

    @property
    def offset_path_length_distribution_divergences(self):
        num_samples = len(self.time_series)
        max_scale = int(math.log2(num_samples))

        if self.discrete:
            # there are more leaves in the discrete case
            max_scale += 1

        distances = []
        for scale in range(max_scale):
            offset = 2**scale
            T1 = self.merge_tree
            T2 = self.superlevel_merge_tree
            l1 = leaf_to_leaf_path_lengths(T1, offset=offset)
            l2 = leaf_to_leaf_path_lengths(T2, offset=offset)
            d1 = distribution_from_samples(
                l1, DISTRIBUTION_VECTOR_LENGTH=self.DISTRIBUTION_VECTOR_LENGTH
            )
            d2 = distribution_from_samples(
                l2, DISTRIBUTION_VECTOR_LENGTH=self.DISTRIBUTION_VECTOR_LENGTH
            )
            distances.append(wasserstein_distance(d1, d2))

        return distances

    @property
    def distance_matrix_divergence(self):
        D1 = distance_matrix(self.merge_tree)
        D2 = distance_matrix(self.superlevel_merge_tree)
        return np.linalg.norm(D1-D2)
    
    @property
    def cophenetic_matrix_divergence(self):
        D1 = cophenetic_matrix(self.merge_tree)
        D2 = cophenetic_matrix(make_increasing(self.superlevel_merge_tree))
        return np.linalg.norm(D1-D2)


if __name__ == "__main__":
    # Test the function
    import numpy as np
    from matplotlib import pyplot as plt

    array = np.random.choice(100, 20, True)

    plt.plot(array)
    plt.show()

    G = merge_tree(array)
    print(G.nodes(data=True))
    print(G.edges())
    print(nx.is_tree(G))
    print(leaf_to_leaf_path_lengths(G, offset=1))
    print(leaf_to_leaf_path_lengths(G, offset=2))
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

    tsmt = TimeSeriesMergeTree(time_series=array)
    divergences = tsmt.divergences
    print(divergences)

    G = merge_tree_discrete(array)
    print(G.nodes(data=True))
    print(G.edges())
    print(nx.is_tree(G))
    print(leaf_to_leaf_path_lengths(G, offset=1))
    print(leaf_to_leaf_path_lengths(G, offset=2))
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

    tsmt = TimeSeriesMergeTree(time_series=array, discrete=True)
    divergences = tsmt.divergences
    print(divergences)
