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

from apaa.learning.recommendation.embedding.node.base import NodeEmbeddingBase
from sklearn.metrics import roc_auc_score
from apaa.helpers.utils import myprofile


class Code2Seq(NodeEmbeddingBase):

    def __init__(
        self,
        node_attributes_file: str,
        predict_file: str,
        label2raw_dict_file: str,
        logger: logging.Logger,
    ):
        # super().__init__("default model", k)
        self.node_attributes_file = node_attributes_file
        self.predictions_file = predict_file
        self.label2raw_file = label2raw_dict_file
        self.logger = logger
        # HACK: I've been following the practices of the original code, but I don't like
        # how instance attributes are assigned in fit method; so I'm choosing to declare
        # them in init. I cannot assign them herre, because graph is only passed to fit
        self.name2id = {}
        self.id2name = {}
        # You can replace DotPredictor with MLPPredictor.
        # self.predictor = MLPPredictor(16)
        self.predictor = helpers.DotPredictor()
        self.sorted_nodes: list[mytypes.NODE]
        self.node_embeddings: mytypes.ARRAY_2D
        self.network: dgl.DGLGraph
        self.definitions: dict
        self.embeddings: dict
        self.embeddings_size: int
        # Counter dicts intialized through fit method, so they reset if passed a new graph
        self.node_labels_counters: dict
        self.edge_labels_counters: dict
        self.model: helpers.GraphSAGE
        self.curent_predictions: torch.Tensor

    def embed(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[mytypes.NODE, agda.Definition],
        **kwargs: Any,
    ) -> Tuple[List[mytypes.NODE], mytypes.ARRAY_2D]:
        # The class works on dql.DGLGraph, so we don't keep the original graph as an attribute
        self.definitions = definitions

        # TODO: do these even contriute anything?
        # TODO: ensure we don't override the original graph
        self.logger.info("Adding embeddings to nodes...")
        self._add_embeddings_to_nodes(graph)
        self.logger.info("Done")

        self.sorted_nodes = graph.nodes

        return graph.nodes, np.array([graph.nodes[node]["embedding"] for node in graph.nodes])

    def _add_embeddings_to_nodes(self, graph: nx.DiGraph):
        # We assume embeddings are lists of float
        self.embeddings, self.embedding_size = utils.read_embeddings(
            self.node_attributes_file, self.predictions_file, self.label2raw_file
        )
        for node in graph.nodes:
            # HACK: nodes without embedding get embedding 0
            graph.nodes[node]["embedding"] = self.embeddings.get(
                node, [float(0) for _ in range(self.embedding_size)]
            )
