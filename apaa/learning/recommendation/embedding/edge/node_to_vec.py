from typing import Any, Dict, List, Literal, Tuple

import networkx as nx
from sklearn.ensemble import RandomForestClassifier

import apaa.data.structures.agda as agda
import apaa.helpers.types as mytypes
from apaa.learning.node_embedding.graph import NodeToVecEmbedding
from apaa.learning.recommendation.embedding.edge.base import (
    BaseEdgeEmbeddingRecommender, EdgeEmbeddingScheme)

Node = mytypes.NODE
array2d = mytypes.ARRAY_2D


class Node2VecEdgeEmbeddingRecommender(BaseEdgeEmbeddingRecommender):
    def __init__(
        self,
        k: Literal["all"] | int = 5,
        classifier: Literal["knn", "rf"] = "rf",
        classifier_kwargs: dict[str, Any] | None = None,
        edge_embedding_scheme: EdgeEmbeddingScheme = EdgeEmbeddingScheme.MEAN,
        edge_file: str | None = None,
        **node_to_vec_kwargs: Any
    ):
        super().__init__(
            name="node2vec edge embedding",
            k=k,
            predictive_model=BaseEdgeEmbeddingRecommender.create_classifier(
                classifier, classifier_kwargs
            ),
            edge_embedding_scheme=edge_embedding_scheme,
            edge_file=edge_file,
        )
        self.embedder = NodeToVecEmbedding(**node_to_vec_kwargs)

    def embed_nodes(
        self, graph: nx.MultiDiGraph, definitions: Dict[Node, agda.Definition]
    ) -> Tuple[List[Node], array2d]:
        self.embedder.fit(graph, definitions)
        assert self.embedder.node_embeddings is not None
        return self.embedder.nodes, self.embedder.node_embeddings
