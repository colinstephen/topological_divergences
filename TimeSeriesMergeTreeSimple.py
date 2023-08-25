import numpy as np
import math
import networkx as nx
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
    min_height = min(height for _, height in tree.nodes(data="height"))
    max_height = max(height for _, height in tree.nodes(data="height"))

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


def leaf_to_leaf_path_lengths(T, offset=1):
    leaf_nodes = [node for node, degree in T.degree() if degree == 1]

    num_leaves = len(leaf_nodes)
    if num_leaves - offset < 1:
        return [0]

    ordered_leaf_nodes = sorted(leaf_nodes, key=lambda node: T.nodes[node]["idx"])

    lengths = []
    for i in range(num_leaves - offset):
        node1 = ordered_leaf_nodes[i]
        node2 = ordered_leaf_nodes[i + offset]
        length = len(nx.shortest_path(T, source=node1, target=node2)) - 1
        lengths.append(length)

    return lengths


class TimeSeriesMergeTree:
    def __init__(
        self,
        time_series,
        INTERLEAVING_DIVERGENCE_MESH=0.5,
    ) -> None:
        self.time_series = time_series
        self.INTERLEAVING_DIVERGENCE_MESH = INTERLEAVING_DIVERGENCE_MESH

    @property
    @lru_cache()
    def merge_tree(self):
        return merge_tree(self.time_series)

    @property
    @lru_cache()
    def superlevel_merge_tree(self):
        return superlevel_merge_tree(self.time_series)

    @property
    def divergences(self):
        return dict(
            interleaving=self.interleaving_divergence,
            normalised_interleaving=self.normalised_interleaving_divergence,
            leaf_to_leaf_path_length=self.path_length_divergence,
        )

    @property
    @lru_cache()
    def interleaving_divergence(self):
        mesh = self.INTERLEAVING_DIVERGENCE_MESH
        T1 = self.merge_tree
        T2 = make_increasing(self.superlevel_merge_tree)
        return merge_tree_interleaving_distance(
            dmt_merge_tree(T1), dmt_merge_tree(T2), mesh, verbose=False
        )

    @property
    def normalised_interleaving_divergence(self):
        n1 = sum(1 for _, d in self.merge_tree.degree() if d == 1)
        n2 = sum(1 for _, d in self.superlevel_merge_tree.degree() if d == 1)
        return self.interleaving_divergence / (n1 + n2)

    @property
    def path_length_divergence(self):
        num_samples = len(self.time_series)
        max_scale = int(math.log2(num_samples))

        distances = []
        for scale in range(max_scale):
            offset = 2 ** int(scale)
            T1 = self.merge_tree
            T2 = self.superlevel_merge_tree
            l1 = leaf_to_leaf_path_lengths(T1, offset=offset)
            l2 = leaf_to_leaf_path_lengths(T2, offset=offset)
            distances.append(wasserstein_distance(l1, l2))

        return distances


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
