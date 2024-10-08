from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import distance as ssd
from sklearn.neighbors import VALID_METRICS

import apaa.data.structures.agda as agda
import apaa.helpers.types as mytypes
from apaa.learning.recommendation.embedding.node.base import NodeEmbeddingBase
from apaa.learning.recommendation.base import KNNRecommender

Node = mytypes.NODE
array1d = mytypes.ARRAY_1D
array2d = mytypes.ARRAY_2D


class KNNNodeEmbeddingRecommender(KNNRecommender):
    def __init__(
        self,
        vectorizer: NodeEmbeddingBase,
        k: Literal["all"] | int = 5,
        metric: str = "cityblock",
        **metric_kwargs: Any,
    ):
        super().__init__(k=k)
        self.embeddings: Optional[array2d] = None
        self.vectorizer = vectorizer
        self.metric = metric
        self.metric_kwargs = metric_kwargs

    def fit(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[mytypes.NODE, agda.Definition],
        **embed_kwargs: Any,
    ):
        self.embed_documents(graph, definitions, **embed_kwargs)
        try:
            self.distance_matrix = ssd.squareform(
                ssd.pdist(self.embeddings, metric=self.metric, **self.metric_kwargs)
            )
        except:
            print("Probably, the matrix does not fit into memory.")

    def embed_documents(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[mytypes.NODE, agda.Definition],
        **embed_kwargs: Any,
    ):
        self.initialize_examples_and_distance_matrix(list(graph.nodes))
        self.graph = graph
        self.definitions = definitions
        self.vectorizer.fit(graph, definitions, **embed_kwargs)
        if self.examples != self.vectorizer.nodes:
            assert self.examples is not None
            assert len(self.examples) == len(self.vectorizer.nodes), (
                len(self.examples),
                len(self.vectorizer.nodes),
            )
            for i, (example, node) in enumerate(
                zip(self.examples, self.vectorizer.nodes)
            ):
                if example != node:
                    raise ValueError(
                        f"First difference at i = {i}: {example} != {node}"
                    )
        self.embeddings = self.vectorizer.node_embeddings
        if not isinstance(self.embeddings, np.ndarray):
            self.embeddings = self.embeddings.todense()  # type: ignore

    def predict_one(self, example: agda.Definition) -> List[Tuple[float, Node]]:
        assert self.embeddings is not None
        node = example.name
        if node not in self.example_to_i:
            raise ValueError(f"Unknown example {node}")
        i = self.example_to_i[node]
        embedding = self.embeddings[i].reshape((1, -1))
        distances = ssd.cdist(
            embedding, self.embeddings, metric=self.metric, **self.metric_kwargs
        )[0]
        return self.postprocess_predictions(
            self.distances_to_tuples(distances), True, False
        )

    def compute_distances(self, example: agda.Definition) -> tuple[array1d, array1d]:
        assert self.embeddings is not None
        node = example.name
        if node not in self.example_to_i:
            raise ValueError(f"Unknown example {node}")
        i = self.example_to_i[node]
        embedding: array1d = self.embeddings[i].reshape((1, -1))
        distances = ssd.cdist(
            embedding, self.embeddings, metric=self.metric, **self.metric_kwargs
        )[0]
        return embedding, distances
