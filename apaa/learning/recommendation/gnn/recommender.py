from typing import *

import dgl
import torch
import itertools
import networkx as nx
import numpy as np
import scipy.sparse as sp

import apaa.data.structures.agda as agda
import apaa.learning.recommendation.gnn.helpers as helpers
import apaa.helpers.types as mytypes
import apaa.helpers.utils as utils

from apaa.learning.recommendation.base import BaseRecommender
from sklearn.metrics import roc_auc_score


class GNN(BaseRecommender):

    def __init__(self, k: Literal["all"] | int, node_attributes_file: str):
        # super().__init__("default model", k)
        self.node_attributes = node_attributes_file
        # HACK: I've been following the practices of the original code, but I don't like
        # how instance attributes are assigned in fit method; so I'm choosing to declare
        # them in init. I cannot assign them herre, because graph is only passed to fit
        self.name2id = {}
        self.id2name = {}
        # You can replace DotPredictor with MLPPredictor.
        # self.predictor = MLPPredictor(16)
        self.predictor = helpers.DotPredictor()
        self.network: dgl.DGLGraph
        self.definitions: dict
        self.embeddings: dict
        self.embeddings_size: int
        # Counter dicts intialized through fit method, so they reset if passed a new graph
        self.node_labels_counters: dict
        self.edge_labels_counters: dict
        self.model: helpers.GraphSAGE
        self.curent_predictions: torch.Tensor

    def fit(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[mytypes.NODE, agda.Definition],
        **kwargs: Any
    ) -> None:
        # The class works on dql.DGLGraph, so we don't keep the original graph as an attribute
        self.definitions = definitions

        # TODO: do these even contriute anything?
        # TODO: ensure we don't override the original graph
        self._encode_node_labels(graph)
        self._encode_edge_labels(graph)
        self._add_embeddings_to_nodes(graph, self.node_attributes)

        train_pos_network, train_neg_network = self._prepare_train_network(
            graph, definitions
        )

        # TODO: that shape[1] is sus
        self.model = helpers.GraphSAGE(self.network.ndata["encoded label"].shape[1], 16)

        # TODO: make this a parameter
        optimizer = torch.optim.Adam(
            itertools.chain(self.model.parameters(), self.predictor.parameters()),
            lr=0.01,
        )

        self._train(optimizer, train_pos_network, train_neg_network)

    def predict_one(self, node: agda.Definition) -> List[Tuple[float, mytypes.NODE]]:
        pass

    def predict_one_edge(
        self,
        example: agda.Definition,
        other: agda.Definition,
        nearest_neighbours: List[Tuple[float, mytypes.NODE]] | None = None,
    ) -> float:
        example_id = self.name2id[example]
        other_id = self.name2id[other]

        return np.dot(
            self.curent_predictions[example_id], self.curent_predictions[other_id]
        )

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

    def _prepare_train_network(
        self,
        graph: nx.DiGraph,
        rerserved_for_test: Tuple[
            dict[mytypes.Node, agda.Definition],
            list[Tuple[mytypes.Node, mytypes.Node, mytypes.Edge]],
            list[Tuple[mytypes.Node, mytypes.Node, mytypes.Edge]],
        ],
    ):
        # TODO: make this in line with already split graph
        # TODO, HACK: instead of definitions, we take all the edges reserved for testing
        # and exclude them

        self.network = dgl.from_networkx(
            graph,
            node_attrs=["encoded label", "embedding"],
            edge_attrs=["encoded label"],
        )

        # from_networkx relabels nodes in sorted()-ordering, starting at 0
        for name, tensor_id in zip(sorted(graph.nodes), self.network.nodes()):
            id = tensor_id.item()
            self.name2id[name] = id
            self.id2name[id] = name

        test_pos_edges = []
        definitions, positive_edges, negative_eges = rerserved_for_test
        for u, v, _ in positive_edges:
            test_pos_edges.append(
                (
                    self.name2id[definitions[u].name],
                    self.name2id[definitions[v].name],
                )
            )
            
        test_neg_u = []
        test_neg_v = []
        for u, v, _ in negative_eges:
            test_neg_u.append(self.name2id[definitions[u].name])
            test_neg_v.append(self.name2id[definitions[v].name])

        source_nodes, destination_nodes = self.network.edges()

        edge_ids = np.arange(self.network.number_of_edges())
        edge_ids = np.random.permutation(edge_ids)

        test_pos_mask = torch.Tensor(
            [
                (source_nodes[id], destination_nodes[id]) in test_pos_edges
                for id in edge_ids
            ]
        )
        # test_size
        # test_pos_u, test_pos_v = (
        #     source_nodes[edge_ids[:test_size]],
        #     destination_nodes[edge_ids[:test_size]],
        # )
        train_pos_u, train_pos_v = (
            source_nodes[edge_ids[~test_pos_mask]],
            destination_nodes[edge_ids[~test_pos_mask]],
        )

        # Find all negative edges and split them for training and testing
        adj_pos = sp.coo_matrix(
            (
                np.ones(len(source_nodes)),
                (source_nodes.numpy(), destination_nodes.numpy()),
            ),
            shape=(self.network.number_of_nodes(), self.network.number_of_nodes()),
        )
        adj_test = sp.coo_matrix(
            (
                np.ones(len(source_nodes)),
                (test_neg_u, test_neg_v),
            ),
            shape=(self.network.number_of_nodes(), self.network.number_of_nodes()),
        )
        adj_neg = (
            1
            - adj_pos.todense()
            - adj_test.todense()
            - np.eye(self.network.number_of_nodes())
        )
        neg_source_nodes, neg_destination_nodes = np.where(adj_neg != 0)

        # neg_eids = np.random.choice(
        #     len(neg_source_nodes), self.network.number_of_edges()
        # )
        # test_neg_u, test_neg_v = (
        #     neg_source_nodes[neg_eids[:test_size]],
        #     neg_destination_nodes[neg_eids[:test_size]],
        # )
        train_neg_u, train_neg_v = (
            neg_source_nodes,
            neg_destination_nodes,
        )

        train_pos_network = dgl.graph(
            (train_pos_u, train_pos_v), num_nodes=self.network.number_of_nodes()
        )
        train_neg_network = dgl.graph(
            (train_neg_u, train_neg_v), num_nodes=self.network.number_of_nodes()
        )

        return train_pos_network, train_neg_network

        test_pos_network = dgl.graph(
            (test_pos_u, test_pos_v), num_nodes=self.network.number_of_nodes()
        )
        test_neg_network = dgl.graph(
            (test_neg_u, test_neg_v), num_nodes=self.network.number_of_nodes()
        )

    def _train(
        self,
        optimizer: torch.optim.Optimizer,
        train_pos_network: dgl.DGLGraph,
        train_neg_network: dgl.DGLGraph,
    ):
        for e in range(5000):  # TODO: make this a parameter
            # forward
            self.curent_predictions = self.model(self.network, self.ndata["embedding"])
            pos_score = self.predictor(train_pos_network, self.curent_predictions)
            neg_score = self.predictor(train_neg_network, self.curent_predictions)
            loss = helpers.compute_loss(pos_score, neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if e % 500 == 0:
            #     print("In epoch {}, loss: {}".format(e, loss))

    def _test(
        self,
        test_pos_network: dgl.DGLGraph,
        test_neg_network: dgl.DGLGraph,
    ):
        with torch.no_grad():
            pos_score = self.predictor(test_pos_network, self.curent_predictions)
            neg_score = self.predictor(test_neg_network, self.curent_predictions)
            # print("AUC", helpers.compute_auc(pos_score, neg_score))  # TODO: not print
