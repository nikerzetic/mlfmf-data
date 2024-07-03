from typing import *

import dgl
import networkx as nx

import apaa.data.structures.agda as agda
import apaa.helpers.original as helpers
import apaa.helpers.types as mytypes
import apaa.helpers.utils as utils
from apaa.learning.recommendation.base import BaseRecommender


class GNN(BaseRecommender):

    def __init__(self, node_attributes_file: str):
        # super().__init__("default model", k)
        self.node_attributes = node_attributes_file
        # HACK: I've been following the practices of the original code, but I don't like
        # how instance attributes are assigned in fit method; so I'm choosing to declare
        # them in init. I cannot assign them herre, because graph is only passed to fit
        self.network: dgl.DGLGraph
        self.definitions: dict
        self.embeddings: dict
        self.embeddings_size: int
        self.node_labels_counters: dict
        self.edge_labels_counters: dict

    def fit(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[mytypes.NODE, agda.Definition],
        **kwargs: Any
    ) -> None:
        # The class works on dql.DGLGraph, so we don't keep the original graph as an attribute
        self.definitions = definitions
        self._encode_node_labels(graph)
        self._encode_edge_labels(graph)
        self._add_embeddings_to_nodes(graph, kwargs["embeddings_file_path"])

        self.network = dgl.from_networkx(
            self.graph,
            node_attrs=["encoded label", "embedding"],
            edge_attrs=["encoded label"],
        )

        source_nodes, destination_nodes = self.network.edges()

    def predict_one(self, example: agda.Definition) -> List[Tuple[float, mytypes.NODE]]:
        pass

    def _encode_node_labels(self, graph: nx.DiGraph):
        """
        Adds `"encoded label"` attribute to nodes in `self.network`. It encodes labels
        using One Hot Encoding.
        """
        self.node_labels_counter = {}
        for data in graph.nodes.data():
            label = data[1]["label"]
            if label in self.node_labels_counter:
                self.node_labels_counter[label] += 1
            else:
                self.node_labels_counter[label] = 1

        node_labels_list = list(self.node_labels_counter.keys())

        for node in graph.nodes:
            label = graph.nodes[node]["label"]
            graph.nodes[node]["encoded label"] = [
                float(item == label) for item in node_labels_list
            ]

    def _encode_edge_labels(self, graph: nx.DiGraph):
        """
        Adds `"encoded label"` attribute to edges in `self.network`. It encodes labels
        using One Hot Encoding.
        """
        self.edge_labels_counter = {}
        for data in graph.edges:
            label = data[2]
            if label in self.edge_labels_counter:
                self.edge_labels_counter[label] += 1
            else:
                self.edge_labels_counter[label] = 1

        edge_labels_list = list(self.edge_labels_counter.keys())

        for edge in graph.edges:
            label = edge[2]
            graph.edges[edge]["encoded label"] = [
                float(item == label) for item in edge_labels_list
            ]

    def _add_embeddings_to_nodes(self, graph: nx.DiGraph, embeddings_file_path: str):
        # We assume embeddings are lists of in
        self.embeddings, self.embedding_size = utils.read_embeddings(
            embeddings_file_path
        )
        for node in graph.nodes:
            # HACK: nodes without embedding get embedding 0
            graph.nodes[node]["embedding"] = self.embeddings.get(
                node, [0 for _ in range(self.embedding_size)]
            )
