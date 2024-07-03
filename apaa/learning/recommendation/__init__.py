from .base import BaseRecommender
from .dummy import DummyRecommender
from .embedding.edge.base import (BaseEdgeEmbeddingRecommender,
                                  EdgeEmbeddingScheme)
from .embedding.edge.mixed import (TFIDFAndNode2VecEmbeddingRecommender,
                                   Word2VecAndNode2VecEmbeddingRecommender)
from .embedding.edge.node_to_vec import Node2VecEdgeEmbeddingRecommender
from .embedding.word_simple import BagOfWordsRecommender, TFIDFRecommender
from .embedding.word_vectors import (EmbeddingAnalogiesRecommender,
                                     WordEmbeddingRecommender)
from .two_hops import EdgeWeightScheme, NodeWeightScheme, TwoHops
