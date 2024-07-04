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

    def __init__(self, node_attributes_file: str):
        # super().__init__("default model", k)
        self.node_attributes = node_attributes_file
        # HACK: I've been following the practices of the original code, but I don't like
        # how instance attributes are assigned in fit method; so I'm choosing to declare
        # them in init. I cannot assign them herre, because graph is only passed to fit
        # You can replace DotPredictor with MLPPredictor.
        #pred = MLPPredictor(16)
        self.predictor = helpers.DotPredictor()
        self.network: dgl.DGLGraph
        self.definitions: dict
        self.embeddings: dict
        self.embeddings_size: int
        self.node_labels_counters: dict
        self.edge_labels_counters: dict
        self.model: helpers.GraphSAGE

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

        train_network = self._prepare_train_network()

        # TODO: that shape[1] is sus
        self.model = helpers.GraphSAGE(train_network.ndata['encoded label'].shape[1], 16)

        # TODO: make this a parameter
        optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), predictor.parameters()), lr=0.01)



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

    def _prepare_train_network(self):
        # TODO: make this in line with already split graph
        source_nodes, destination_nodes = self.network.edges()

        eids = np.arange(self.network.number_of_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * 0.1)
        train_size = self.network.number_of_edges() - test_size
        test_pos_u, test_pos_v = (
            source_nodes[eids[:test_size]],
            destination_nodes[eids[:test_size]],
        )
        train_pos_u, train_pos_v = (
            source_nodes[eids[test_size:]],
            destination_nodes[eids[test_size:]],
        )

        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix(
            (
                np.ones(len(source_nodes)),
                (source_nodes.numpy(), destination_nodes.numpy()),
            ),
            shape=(self.network.number_of_nodes(), self.network.number_of_nodes()),
        )
        adj_neg = 1 - adj.todense() - np.eye(self.network.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), self.network.number_of_edges())
        test_neg_u, test_neg_v = (
            neg_u[neg_eids[:test_size]],
            neg_v[neg_eids[:test_size]],
        )
        train_neg_u, train_neg_v = (
            neg_u[neg_eids[test_size:]],
            neg_v[neg_eids[test_size:]],
        )

        train_network = dgl.remove_edges(self.network, eids[:test_size])
        # TODO: train_network could just be network

        train_pos_network = dgl.graph((train_pos_u, train_pos_v), num_nodes=self.network.number_of_nodes())
        train_neg_network = dgl.graph((train_neg_u, train_neg_v), num_nodes=self.network.number_of_nodes())

        test_pos_network = dgl.graph((test_pos_u, test_pos_v), num_nodes=self.network.number_of_nodes())
        test_neg_network = dgl.graph((test_neg_u, test_neg_v), num_nodes=self.network.number_of_nodes())

    def _train(self, optimizer: torch.optim.Optimizer):
        all_logits = []
        for e in range(5000): #TODO: make this a parameter
            # forward
            h = self.model(train_g, train_g.ndata["encoded label"])
            pos_score = self.predictor(train_pos_g, h)
            neg_score = self.predictor(train_neg_g, h)
            loss = helpers.compute_loss(pos_score, neg_score)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if e % 500 == 0:
                print('In epoch {}, loss: {}'.format(e, loss))

        with torch.no_grad():
            pos_score = self.predictor(test_pos_g, h)
            neg_score = self.predictor(test_neg_g, h)
            print('AUC', helpers.compute_auc(pos_score, neg_score)) #TODO: not print
