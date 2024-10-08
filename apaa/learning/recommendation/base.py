import pickle
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
import numpy as np

import apaa.data.structures.agda as agda
import apaa.helpers.types as mytypes


class BaseRecommender:
    def __init__(self, name: str, k: Literal["all"] | int) -> None:
        self.name = name
        self.definitions: Optional[Dict[mytypes.NodeType, agda.Definition]] = None
        self.graph: Optional[nx.MultiDiGraph] = None
        self._k = self._check_k(k)

    @staticmethod
    def _check_k(k: Literal["all"] | int) -> int:
        if isinstance(k, str):
            if k == "all":
                return -1
            raise ValueError("If str, PredictiveModel.k should equal 'all'.")
        if k > 0:
            return k
        raise ValueError("If int, PredictiveModel.k should be positive")

    @property
    def k(self) -> int:
        return self._k

    def n_predictions(self, n_all: int):
        if self.k < 0:
            return n_all
        else:
            return self.k

    def fit(
        self,
        graph: nx.MultiDiGraph,
        definitions: Dict[mytypes.NodeType, agda.Definition],
        **kwargs: Any
    ) -> None:
        raise NotImplementedError()

    def predict(
        self, example_s: agda.Definition | List[agda.Definition]
    ) -> List[Tuple[float, mytypes.NodeType]] | List[List[Tuple[float, mytypes.NodeType]]]:
        if isinstance(example_s, agda.Definition):
            unpack = True
            e_list = [example_s]
        else:
            unpack = False
            e_list = example_s
        neighbours: List[List[Tuple[float, mytypes.NodeType]]] = []
        for element in e_list:
            neighbours.append(self.predict_one(element))
        if unpack:
            return neighbours[0]
        else:
            return neighbours

    def predict_one(self, example: agda.Definition) -> List[Tuple[float, mytypes.NodeType]]:
        """
        Returns a sorted list of pairs (distance, other id).
        """
        raise NotImplementedError()

    def predict_one_edge(
        self,
        example: agda.Definition,
        other: agda.Definition,
        nearest_neighbours: Optional[List[Tuple[float, mytypes.NodeType]]] = None,
    ) -> float:
        """
        Returns either a score ("probability") or
        a binary decision (1/0) of the presence of the link.
        By default, it calls predict_one(example),
        and computes the final prediction from rank of the other in the
        list from predict_one.

        To speed things up, the nearest neighbours dictionary can be used:
        predict the corresponding similarity if the edge among the neighbours
        and 0 otherwise. Those nearest neighbors have been computed
        before by the same model in recommender system fashion.
        """
        if nearest_neighbours is not None:
            max_similarity = 0.0
            other_similarity = 0.0
            other_name = other.name
            for sim, name in nearest_neighbours:
                if name == other_name:
                    other_similarity = sim
                if sim > max_similarity:
                    max_similarity = sim
            if max_similarity > 0.0:
                return other_similarity / max_similarity
            else:
                return 0.0
        # a slower approach
        scores = self.predict_one(example)
        rank = len(scores)
        for i, candidate in enumerate(scores):
            if candidate == other.name:
                rank = i
                break
        # the higher the results, the higher the confidence
        return 1.0 - rank / len(scores)

    def postprocess_predictions(
        self,
        predictions: list[tuple[float, mytypes.NodeType]],
        needs_normalisation: bool,
        is_similarity: bool,
    ) -> list[tuple[float, mytypes.NodeType]]:
        predictions = [
            pair
            for pair in predictions
            if agda.Definition.is_normal_definition(pair[1])
        ]
        if needs_normalisation or not is_similarity:
            ys = np.array([y for y, _ in predictions])
            if needs_normalisation:
                mini = np.min(ys)
                maxi = np.max(ys)
                if mini < maxi:
                    ys = (ys - mini) / (maxi - mini)
            if not is_similarity:
                ys = 1.0 - ys
            predictions = [(ys[i], predictions[i][1]) for i in range(len(predictions))]
        predictions.sort(reverse=True)
        if self.k < 0:
            return predictions
        else:
            return predictions[: self.k]

    def dump(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def unpickle(file: str) -> "BaseRecommender":
        with open(file, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load(model_dump_file: str):
        return BaseRecommender.unpickle(model_dump_file)


class KNNRecommender(BaseRecommender):
    def __init__(self, k: Literal["all"] | int = 5):
        super().__init__("nearest neighbours", k)
        # a list of ids
        self.examples: Optional[List[mytypes.NodeType]] = None
        self.example_to_i: Dict[mytypes.NodeType, int] = {}
        self.distance_matrix: Optional[
            np.ndarray[Tuple[int, int], np.dtype[np.float_]]
        ] = None

    def initialize_examples_and_distance_matrix(self, examples: list[mytypes.NodeType]):
        self.examples = sorted(examples)
        try:
            self.distance_matrix = np.zeros((len(self.examples), len(self.examples)))
        except:
            print(
                "Error: could not allocate memory for the distance matrix. Skipping this."
            )
        self.example_to_i = {e: i for i, e in enumerate(self.examples)}

    def find_neighbours_for_existing(
        self, i: Union[int, List[int]]
    ) -> Union[
        List[Tuple[float, mytypes.NodeType]], List[List[Tuple[float, mytypes.NodeType]]]
    ]:
        if self.distance_matrix is None:
            raise ValueError("Use .fit(examples) first to initialize the matrix.")
        if isinstance(i, int):
            unpack = True
            i_list = [i]
        else:
            unpack = False
            i_list = i
        neighbours = []
        for element in i_list:
            neighbours.append(
                self.distances_to_tuples(
                    self.distance_matrix[element], for_existing=True
                )
            )
        if unpack:
            return neighbours[0]
        else:
            return neighbours

    def distances_to_tuples(
        self, distances: mytypes.ARRAY_1D, for_existing: bool = False
    ) -> List[Tuple[float, mytypes.NODE]]:
        assert self.examples is not None
        nearest = np.argsort(distances)
        n_predictions = self.n_predictions(len(nearest))
        # does not hurt if upper = len + 1
        upper = n_predictions + int(for_existing)
        nearest = nearest[:upper]
        return [(distances[j], self.examples[j]) for j in nearest]

    def predict_one(self, example: agda.Definition) -> List[Tuple[float, mytypes.NodeType]]:
        raise NotImplementedError()

    @staticmethod
    def unpickle(file: str) -> "KNNRecommender":
        with open(file, "rb") as f:
            return pickle.load(f)
