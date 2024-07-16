from typing import *

import dgl
import io
import pstats
import torch
import itertools
import tqdm
import cProfile
import logging
import networkx as nx
import numpy as np
import scipy.sparse as sp

import apaa.data.structures.agda as agda
import apaa.learning.recommendation.gnn.helpers as helpers
import apaa.helpers.types as mytypes
import apaa.helpers.utils as utils

from apaa.learning.recommendation.base import BaseRecommender
from sklearn.metrics import roc_auc_score
from apaa.helpers.utils import myprofile


class GNN(BaseRecommender):

    def __init__(
        self,
        k: Literal["all"] | int,
        node_attributes_file: str,
        predict_file: str,
        label2raw_dict_file: str,
        number_of_epochs: int,
        logger: logging.Logger,
    ):
        # super().__init__("default model", k)
        self.node_attributes_file = node_attributes_file
        self.predictions_file = predict_file
        self.label2raw_file = label2raw_dict_file
        self.number_of_epochs = number_of_epochs
        self.logger = logger
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
        **kwargs: Any,
    ) -> None:
        # The class works on dql.DGLGraph, so we don't keep the original graph as an attribute
        self.definitions = definitions

        # TODO: do these even contriute anything?
        # TODO: ensure we don't override the original graph
        # self._encode_node_labels(graph)
        # self._encode_edge_labels(graph)
        self.logger.info("Adding embeddings to nodes...")
        self._add_embeddings_to_nodes(graph)
        self.logger.info("Done")

        self.logger.info("Preparing train network...")
        train_pos_network, train_neg_network = self._prepare_train_network(
            graph, definitions
        )
        

        self.logger.info("Done")

        # TODO: that shape[1] is sus
        # self.model = helpers.GraphSAGE(self.network.ndata["embedding"].shape[1], 16)

        self.logger.info("Training the model...")
        self.model, self.curent_predictions = self.train(
            self.network.ndata["embedding"].shape[1], 
            16,
            # self.predictor,
            # self.model,
            self.network,
            train_pos_network,
            train_neg_network,
            self.number_of_epochs,
        )

        self.network = dgl.from_networkx(
            graph,
            node_attrs=["embedding"],
            # node_attrs=["encoded label", "embedding"],
            # edge_attrs=["encoded label"],
        )
        

    def predict_one(self, node: agda.Definition) -> List[Tuple[float, mytypes.NODE]]:
        node_id = self.name2id[node.name]
        #TODO: figure out why evaluation.update sometimes cites node itself as an actual neighbour
        node_successor_network = dgl.graph(
            ([float(node_id) for _ in range(self.network.number_of_nodes())], self.network.nodes()), num_nodes=self.network.number_of_nodes()
        )
        # node_successor_network = dgl.graph(
        #     ([float(node_id) for _ in range(self.network.number_of_nodes()-1)], self.network.nodes()[self.network.nodes() != int(node_id)]), num_nodes=self.network.number_of_nodes()
        # )
        node_successor_network.edata["score"] = self.predictor(node_successor_network, self.curent_predictions)
    
        neighbours = [(v.item(), score.item()) for v, score in zip(node_successor_network.edges()[1], node_successor_network.edata["score"])]
        # neighbours: list = []
        # for v, score in zip(node_successor_network.edges()[1], node_successor_network.edata["score"]):
        #     if score > 0.5:
        #         neighbours.append((v.item(), score.item()))
        # Sort by score (descending)
        neighbours.sort(key=lambda x: x[1], reverse=True)
        return [(score, self.id2name[id]) for id, score in neighbours]


    def predict_one_edge(
        self,
        example: agda.Definition,
        other: agda.Definition,
        nearest_neighbours: List[Tuple[float, mytypes.NODE]] | None = None,
    ) -> float:
        example_id = self.name2id[example.name]
        other_id = self.name2id[other.name]

        return np.dot(
            self.curent_predictions[example_id].detach().numpy(), self.curent_predictions[other_id].detach().numpy()
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

    def _add_embeddings_to_nodes(self, graph: nx.DiGraph):
        # We assume embeddings are lists of in
        self.embeddings, self.embedding_size = utils.read_embeddings(
            self.node_attributes_file, self.predictions_file, self.label2raw_file
        )
        for node in graph.nodes:
            # HACK: nodes without embedding get embedding 0
            graph.nodes[node]["embedding"] = self.embeddings.get(
                node, [float(0) for _ in range(self.embedding_size)]
            )

    def _prepare_train_network(
        self,
        graph: nx.DiGraph,
        rerserved_for_test: Tuple[
            dict[mytypes.NodeType, agda.Definition],
            list[Tuple[mytypes.NodeType, mytypes.NodeType, mytypes.EdgeType]],
            list[Tuple[mytypes.NodeType, mytypes.NodeType, mytypes.EdgeType]],
        ],
    ):
        # TODO: make this in line with already split graph
        # TODO, HACK: instead of definitions, we take all the edges reserved for testing
        # and exclude them

        self.network = dgl.from_networkx(
            graph,
            node_attrs=["embedding"]
            # node_attrs=["embedding"],
            # edge_attrs=["encoded label"],
        )

        self.logger.info("...building dictionaries")
        # from_networkx relabels nodes in sorted()-ordering, starting at 0
        for name, tensor_id in zip(sorted(graph.nodes), self.network.nodes()):
            id = tensor_id.item()
            self.name2id[name] = id
            self.id2name[id] = name

        self.logger.info("...translating test pos edges")
        test_pos_edges = []
        definitions, positive_edges, negative_eges = rerserved_for_test
        for u, v, _ in positive_edges:
            test_pos_edges.append(
                [
                    self.name2id[definitions[u].name],
                    self.name2id[definitions[v].name],
                ]
            )
        test_pos_edges = np.array(test_pos_edges)

        self.logger.info("...translating test neg edges")
        test_neg_u = []
        test_neg_v = []
        for u, v, _ in negative_eges:
            test_neg_u.append(self.name2id[definitions[u].name])
            test_neg_v.append(self.name2id[definitions[v].name])

        source_nodes, destination_nodes = self.network.edges()

        edge_ids = np.arange(self.network.number_of_edges())
        edge2id = {}
        for u, v, id in zip(source_nodes, destination_nodes, edge_ids):
            edge2id[u.item()] = {v.item(): id}
        edge_ids = np.random.permutation(edge_ids)

        self.logger.info(f"...building test pos mask")
        test_pos_mask = torch.zeros(len(edge_ids), dtype=bool)
        for e in test_pos_edges:
            # BUG, TODO: there should be no test edges in the training graph
            edge_id = edge2id.get(e[0], {}).get(e[1], False)
            if edge_id:
                test_pos_mask[edge_id] = 1
            # for id in edge_ids:
            #     if not source_nodes[id] == e[0]:
            #         continue
            #     if destination_nodes[id] == e[1]:
            #         test_pos_mask[id] = 1
            #         # HACK: we assume each edge appears only once
            #         break

        counter = [0, 0, 0]
        for e in test_pos_edges:
            eid = edge2id.get(e[0], {}).get(e[1], "good")
            if eid == "good":
                counter[0] += 1
            else:
                counter[1] += 1
                u = self.id2name[e[0]]
                v = self.id2name[e[1]]
                if v in graph.succ[u]:
                    counter[2] += 1
        self.logger.warning(
            f"There are {counter[1]} test edges in train network, and {counter[2]} of them are in the train graph"
        )

        train_pos_u, train_pos_v = (
            source_nodes[edge_ids[~test_pos_mask]],
            destination_nodes[edge_ids[~test_pos_mask]],
        )

        self.logger.info("...finding negative edges")
        # Find all negative edges and split them for training and testing
        adj_pos = sp.coo_matrix(
            (
                np.ones(len(source_nodes)),
                (source_nodes.numpy(), destination_nodes.numpy()),
            ),
            shape=(self.network.number_of_nodes(), self.network.number_of_nodes()),
        )
        self.logger.info("...building test neg edges mask")
        adj_test = sp.coo_matrix(
            (
                np.ones(len(test_neg_u)),
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

        # We randomly choose as many neg test nodes as positive
        neg_eids = np.random.choice(len(neg_source_nodes), len(train_pos_u))

        train_neg_u, train_neg_v = (
            neg_source_nodes[neg_eids],
            neg_destination_nodes[neg_eids],
        )

        self.logger.info("...buildling train pos and neg networks")
        train_pos_network = dgl.graph(
            (train_pos_u, train_pos_v), num_nodes=self.network.number_of_nodes()
        )
        train_neg_network = dgl.graph(
            (train_neg_u, train_neg_v), num_nodes=self.network.number_of_nodes()
        )

        self.network = dgl.remove_edges(self.network, edge_ids[~test_pos_mask])

        return train_pos_network, train_neg_network

        test_pos_network = dgl.graph(
            (test_pos_u, test_pos_v), num_nodes=self.network.number_of_nodes()
        )
        test_neg_network = dgl.graph(
            (test_neg_u, test_neg_v), num_nodes=self.network.number_of_nodes()
        )

    @staticmethod
    def train(
        # predictor: DotPredictor,
        in_feats: int, 
        h_feats: int,
        network: dgl.DGLGraph,
        train_pos_network: dgl.DGLGraph,
        train_neg_network: dgl.DGLGraph,
        number_of_epochs: int,
    ):
        predictor_local = helpers.DotPredictor()
        model_local = helpers.GraphSAGE(in_feats, h_feats)
        network_local = network
        train_pos_network_local = train_pos_network
        train_neg_network_local = train_neg_network
        compute_loss_local = helpers.compute_loss

        # TODO: make this a parameter
        optimizer = torch.optim.Adam(
            itertools.chain(model_local.parameters(), predictor_local.parameters()),
            lr=0.01,
        )

        for e in tqdm.tqdm(range(number_of_epochs)):  # TODO: make this a parameter
            # forward
            prediction = model_local(
                network_local, network_local.ndata["embedding"]
            )
            pos_score = predictor_local(train_pos_network_local, prediction)
            neg_score = predictor_local(train_neg_network_local, prediction)
            loss = compute_loss_local(pos_score, neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if e % 1000 == 0:
            #     logger.info(f"In epoch {e}, loss: {loss}")
        
        return model_local, prediction

    def _test(
        self,
        test_pos_network: dgl.DGLGraph,
        test_neg_network: dgl.DGLGraph,
    ):
        with torch.no_grad():
            pos_score = self.predictor(test_pos_network, self.curent_predictions)
            neg_score = self.predictor(test_neg_network, self.curent_predictions)
            # print("AUC", helpers.compute_auc(pos_score, neg_score))  # TODO: not print
