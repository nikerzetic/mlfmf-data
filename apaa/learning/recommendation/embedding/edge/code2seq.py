from typing import Any, Dict, List, Literal, Tuple

import networkx as nx
from sklearn.ensemble import RandomForestClassifier

import apaa.data.structures.agda as agda
import apaa.helpers.types as mytypes
import apaa.learning.recommendation.embedding.node.code2seq as embedding
from apaa.learning.recommendation.embedding.edge.base import (
    BaseEdgeEmbeddingRecommender, EdgeEmbeddingScheme)


class Code2Seq(BaseEdgeEmbeddingRecommender):
    def __init__(
        self,
        k: Literal["all"] | int = 5,
        classifier: Literal["knn", "rf"] = "rf",
        classifier_kwargs: dict[str, Any] | None = None,
        edge_embedding_scheme: EdgeEmbeddingScheme = EdgeEmbeddingScheme.MEAN,
        edge_file: str | None = None,
        **code2seq_kwargs: Any
    ):
        super().__init__(
            name="code2seq edge embedding",
            k=k,
            predictive_model=BaseEdgeEmbeddingRecommender.create_classifier(
                classifier, classifier_kwargs
            ),
            edge_embedding_scheme=edge_embedding_scheme,
            edge_file=edge_file,
        )
        self.embedder = embedding.Code2Seq(**code2seq_kwargs)

    def embed_nodes(
        self, graph: nx.MultiDiGraph, definitions: Dict[mytypes.NODE, agda.Definition]
    ) -> Tuple[List[mytypes.NODE], mytypes.ARRAY_2D]:
        self.embedder.fit(graph, definitions)
        assert self.node_embeddings is not None
        return self.embedder.sorted_nodes, self.embedder.node_embeddings
