import higra as hg
import networkx as nx
import numpy as np
from decorated_merge_trees.DMT_tools import MergeTree
from decorated_merge_trees.DMT_tools import decorate_merge_tree_networks
from decorated_merge_trees.DMT_tools import merge_tree_interleaving_distance
from decorated_merge_trees.DMT_tools import DMT_interleaving_distance
from TimeSeriesPersistence import TimeSeriesPersistence as TSP


class TimeSeriesMergeTree:
    def __init__(
        self,
        time_series: np.array,
        INTERLEAVING_DIVERGENCE_MESH=0.5,
        DMT_ALPHA=0.5,
    ) -> None:
        self._time_series = time_series
        self._sublevel_tree_higra = None
        self._superlevel_tree_higra = None
        self._sublevel_tree_dmt = None
        self._superlevel_tree_dmt = None
        self._persistence = None
        self._interleaving_divergence = None
        self._dmt_interleaving_divergence = None
        self.INTERLEAVING_DIVERGENCE_MESH = INTERLEAVING_DIVERGENCE_MESH
        self.DMT_ALPHA = DMT_ALPHA

    @property
    def persistence(self):
        # Decorating merge trees requires persistence information
        if self._persistence is None:
            self._persistence = TSP(self._time_series)
        # Can now access `self.sublevel_diagram` etc. 
        return self._persistence

    @property
    def sublevel_tree_higra(self):
        if self._sublevel_tree_higra is None:
            self._sublevel_tree_higra = (
                TimeSeriesMergeTree.merge_tree_from_time_series_higra(
                    self._time_series,
                    superlevel_filtration=False,
                    make_increasing=False,
                )
            )
        return self._sublevel_tree_higra

    @property
    def superlevel_tree_higra(self):
        if self._superlevel_tree_higra is None:
            self._superlevel_tree_higra = (
                TimeSeriesMergeTree.merge_tree_from_time_series_higra(
                    self._time_series,
                    superlevel_filtration=True,
                    make_increasing=True,
                )
            )
        return self._superlevel_tree_higra

    @property
    def sublevel_tree_dmt(self) -> MergeTree:
        if self._sublevel_tree_dmt is None:
            self._sublevel_tree_dmt = (
                TimeSeriesMergeTree._higra_merge_tree_2_dmt_merge_tree(
                    *self.sublevel_tree_higra
                )
            )
        return self._sublevel_tree_dmt

    @property
    def superlevel_tree_dmt(self) -> MergeTree:
        if self._superlevel_tree_dmt is None:
            self._superlevel_tree_dmt = (
                TimeSeriesMergeTree._higra_merge_tree_2_dmt_merge_tree(
                    *self.superlevel_tree_higra
                )
            )
        return self._superlevel_tree_dmt

    @staticmethod
    def _chain_graph(number_of_vertices):
        """
        Helper function. Build a simple chain graph of the given length.

        Parameters
        ----------
        number_of_vertices : int
            How long the resulting chain should be.

        Returns
        -------
        higra.UndirectedGraph
            A Higra graph object representing the chain.
        """

        graph = hg.UndirectedGraph(number_of_vertices)
        vertices = np.arange(number_of_vertices)
        graph.add_edges(vertices[:-1], vertices[1:])

        return graph

    @staticmethod
    def _chain_graph_from_time_series(time_series):
        """
        Helper function. Generate a chain graph representation of a time series.

        Parameters
        ----------
        time_series : array_like
            Sequence of values in the time series.

        Returns
        -------
        tuple : (chain_graph, vertex_weights)
            A Higra graph object and an array of its vertex weights.

        Summary
        -------
        Given a discrete time series `T` of length `|T|=N`, generate a chain graph whose nodes `{0, 1, ..., N-1}` are labelled with the time series values `T[0], T[1], ..., T[N-1]`.
        """

        return TimeSeriesMergeTree._chain_graph(len(time_series)), np.array(time_series)

    @staticmethod
    def _higra_component_tree_2_merge_tree(tree, altitudes):
        """
        Helper function. Convert a Higra component tree to a Persistent Homology merge tree.

        Parameters
        ----------
        tree : higra.Tree
            The Higra component tree to be converted
        altitudes : array
            The altitudes of nodes in the tree

        Returns
        -------
        tuple : (higra.Tree, array)
            The persistent homology merge tree and its node altitudes
        """

        # prune then smooth the incoming component tree (NB: order of these operations matters)
        tree, altitudes = TimeSeriesMergeTree._remove_redundant_leaves(tree, altitudes)
        tree, altitudes = TimeSeriesMergeTree._remove_degree_two_vertices(
            tree, altitudes
        )

        return tree, altitudes

    @staticmethod
    def _remove_degree_two_vertices(tree, altitudes):
        """
        Helper function. Simplify a tree by removing any internal degree-two vertices.

        Parameters
        ----------
        tree : higra.Tree
            The raw tree to be simplified
        altitudes : array
            The altitudes of nodes in the tree

        Returns
        -------
        tuple : (higra.Tree, array)
            The simplified tree and remaining altitudes
        """

        degrees = np.array([tree.degree(v) for v in tree.vertices()])
        filter = np.array(degrees == 2)
        smoothed_tree, node_map = hg.simplify_tree(tree, filter)
        smoothed_tree_altitudes = altitudes[node_map]

        return smoothed_tree, smoothed_tree_altitudes

    @staticmethod
    def _remove_redundant_leaves(tree, altitudes):
        """
        Helper function. Simplify a tree by removing any leaves at the same altitude as their parent.

        Parameters
        ----------
        tree : higra.Tree
            The raw tree to be simplified
        altitudes : array
            The altitudes of the nodes in the tree

        Returns
        -------
        tuple : (higra.Tree, altitudes)
            The simplified tree and remaining altitudes
        """

        # compute outgoing edge weights in the tree based on altitudes
        weights = altitudes - altitudes[tree.parents()]

        # create a filter based on whether an outgoing weight is zero
        filter = np.array(weights == 0)

        # use the filter in the Higra simplify method
        pruned_tree, node_map = hg.simplify_tree(tree, filter, process_leaves=True)

        # use the resulting node map to the original tree to get the altitudes
        pruned_tree_altitudes = altitudes[node_map]

        return pruned_tree, pruned_tree_altitudes

    @staticmethod
    def _higra_merge_tree_2_dmt_merge_tree(hg_merge_tree, altitudes):
        """
        Helper function. Convert a Higra merge tree to a `DMT_tools` merge tree.

        Parameters
        ----------
        hg_merge_tree : higra.Tree
            The tree to be converted for use by the decorated merge tree code.
        altitudes : array
            Heights of the nodes in the input tree.

        Returns
        -------
        DMT_tools.MergeTree
            A `networkx`-based version of the merge tree suitable for use in `DMT_tools` functions.
        """

        hg_vertices = list(hg_merge_tree.vertices())
        hg_parents = list(hg_merge_tree.parents())
        hg_edges = list(zip(hg_vertices, hg_parents))

        nx_vertices = hg_vertices
        nx_edges = hg_edges[:-1]  # omit the self-loop edge Higra uses
        nx_heights = {v: h for v, h in zip(nx_vertices, altitudes)}

        nx_merge_tree = nx.Graph()
        nx_merge_tree.add_nodes_from(nx_vertices)
        nx_merge_tree.add_edges_from(nx_edges)

        merge_tree = MergeTree(tree=nx_merge_tree, height=nx_heights)

        return merge_tree

    @staticmethod
    def merge_tree_from_time_series_higra(
        time_series, superlevel_filtration=False, make_increasing=False
    ):
        """
        Given a discrete time series compute the merge tree of its piecewise linear interpolation.

        Parameters
        ----------
        time_series : array_like
            List or numpy array of time series values.
        superlevel_filtration : boolean, optional
            Generate the superlevel set filtration merge tree? Default is the sublevel set filtration merge tree.
        make_increasing : boolean, optional
            Only applied if superlevel_filtration is True.
            Whether to align root altitude with the sublevel filtration and make paths from leaves to root increasing.

        Returns
        -------
        tuple : (higra.CptHierarchy, array)
            Tuple containing the Higra tree structure, of type higra.CptHierarchy, and an array of its node altitudes.
        """

        # flip the time series values if we're doing a superlevel filtration
        time_series = (
            -1 * np.array(time_series) if superlevel_filtration else time_series
        )

        # apply Higra's component tree algorithm over a time series chain graph
        chain_graph, vertex_weights = TimeSeriesMergeTree._chain_graph_from_time_series(
            time_series
        )
        tree, altitudes = hg.component_tree_min_tree(chain_graph, vertex_weights)

        # reflip the node altitudes if we flipped the time series originally
        altitudes = -1 * np.array(altitudes) if superlevel_filtration else altitudes

        # simplify the component tree to retain only persistence merge tree information
        (
            merge_tree,
            merge_tree_altitudes,
        ) = TimeSeriesMergeTree._higra_component_tree_2_merge_tree(tree, altitudes)

        # make superlevel trees comparable with sublevel trees if required
        if superlevel_filtration and make_increasing:
            max_altitude = np.max(merge_tree_altitudes)
            min_altitude = np.min(merge_tree_altitudes)
            merge_tree_altitudes = (
                -1 * merge_tree_altitudes + min_altitude + max_altitude
            )

        return merge_tree, merge_tree_altitudes

    @staticmethod
    def merge_tree_from_time_series_dmt(
        time_series, superlevel_filtration=False, make_increasing=False
    ):
        """
        Given a discrete time series compute the merge tree of its piecewise linear interpolation.

        Parameters
        ----------
        time_series : array_like
            List or numpy array of time series values.
        superlevel_filtration : boolean, optional
            Generate the superlevel set filtration merge tree? Default is the sublevel set filtration merge tree.
        make_increasing : boolean, optional
            Only applied if superlevel_filtration is True.
            Whether to align root altitude with the sublevel filtration and make paths from leaves to root increasing.

        Returns
        -------
        DMT_tools.MergeTree
            A `networkx`-based version of the merge tree suitable for use in `DMT_tools` functions.
        """

        # Build a Higra-based merge tree
        higra_merge_tree = TimeSeriesMergeTree.merge_tree_from_time_series_higra(
            time_series,
            superlevel_filtration=superlevel_filtration,
            make_increasing=make_increasing,
        )

        # Convert it to the desired DMT_tools format
        merge_tree = TimeSeriesMergeTree._higra_merge_tree_2_dmt_merge_tree(
            *higra_merge_tree
        )

        return merge_tree

    ####################################################################
    ## Divergences of super and sub level persistence representations ##
    ####################################################################

    @property
    def divergences(self):
        return dict(
            interleaving=self.interleaving_divergence,
            dmt_interleaving=self.dmt_interleaving_divergence
        )

    @property
    def interleaving_divergence(self):
        if self._interleaving_divergence is None:
            mesh = self.INTERLEAVING_DIVERGENCE_MESH
            MT1 = self.sublevel_tree_dmt
            MT2 = self.superlevel_tree_dmt
            self._interleaving_divergence = merge_tree_interleaving_distance(
                MT1, MT2, mesh, verbose=False
            )
        return self._interleaving_divergence

    @property
    def dmt_interleaving_divergence(self):

        if self._dmt_interleaving_divergence is None:

            MT1 = self.sublevel_tree_dmt
            tree1, height1 = MT1.tree, MT1.height
            pd1 = self.persistence.sublevel_diagram
            leaf_barcode1 = decorate_merge_tree_networks(tree1, height1, pd1)
            MT1.fit_barcode(degree=0, leaf_barcode=leaf_barcode1)

            MT2 = self.superlevel_tree_dmt
            tree2, height2 = MT2.tree, MT2.height
            pd2 = self.persistence.superlevel_diagram_flipped
            leaf_barcode2 = decorate_merge_tree_networks(tree2, height2, pd2)
            MT2.fit_barcode(degree=0, leaf_barcode=leaf_barcode2)

            self._dmt_interleaving_divergence = DMT_interleaving_distance(
                MT1, MT2, self.INTERLEAVING_DIVERGENCE_MESH, alpha=self.DMT_ALPHA, verbose=False
            )

        return self._dmt_interleaving_divergence
