import logging
import os
import random
import re
from enum import Enum
from typing import Any, Iterator, List, Tuple, Type

import networkx as nx
import numpy as np

import apaa.helpers.types as mytypes


def create_logger(name: str, file: str | None = None):
    if file is None:
        ch = logging.StreamHandler()
    else:
        ch = logging.FileHandler(file, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(filename)s:%(funcName)s:%(lineno)d]:  %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# -------------------------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GIT_ROOT = os.path.join(_THIS_DIR, "..", "..")


class AgdaSyntax:
    KEYWORDS = [
        "=",
        "|",
        "->",
        "→",
        ":",
        "?",
        "\\",
        "λ",
        "∀",
        "..",
        "...",
        "abstract",
        "codata",
        "coinductive",
        "constructor",
        "data",
        "eta-equality",
        "field",
        "forall",
        "hiding",
        "import",
        "in",
        "inductive",
        "infix",
        "infixl",
        "infixr",
        "instance",
        "let",
        "macro",
        "module",
        "mutual",
        "no-eta-equality",
        "open",
        "overlap",
        "pattern",
        "postulate",
        "primitive",
        "private",
        "public",
        "quote",
        "quoteContext",
        "quoteGoal",
        "quoteTerm",
        "record",
        "renaming",
        "rewrite",
        "Set",
        "syntax",
        "tactic",
        "unquote",
        "unquoteDecl",
        "unquoteDef",
        "using",
        "where",
        "with",
    ]
    BLOCK_STARTERS = [
        "record",
        "module",
        "open",
        "private",
        "data",
        "macro",
        "instance",
    ]
    USED_BLOCK_STARTERS = [
        "record",
        "module",
        "macro",
        "instance",
    ]  # need inner modules
    BLOCK_CONTINUATION = ["...", "where"]


class Locations:
    NON_FILE = "non-library file"

    SEXP_DIR_STDLIB = os.path.join(_GIT_ROOT, "../agda-stdlib/src/sexp")
    SEXP_DIR_UNIMATH = os.path.join(_GIT_ROOT, "../agda-unimath/src/sexp")
    SEXP_DIR_AGDA_TEST = os.path.join(_GIT_ROOT, "test_data/agda/test_lib/sexp")
    SEXP_DIR_TYPE_TOPOLOGY = os.path.join(_GIT_ROOT, "../TypeTopology/source/sexp")

    SEXP_DIR_LEAN_TEST = os.path.join(_GIT_ROOT, "test_data/lean/test_lib/sexp")
    SEXP_DIR_MATHLIB = r"D:\mathlib4\sexp"

    NAME_STDLIB = "stdlib"
    NAME_UNIMATH = "unimath"
    NAME_TYPE_TOPOLOGY = "TypeTopology"
    NAME_AGDA_TEST = "test_agda"

    NAME_MATHLIB = "mathlib"
    NAME_LEAN_TEST = "test_lean"

    DUMPS_DIR = os.path.join(_GIT_ROOT, "dumps")
    EXPERIMENTS_DIR = os.path.join(DUMPS_DIR, "experiments")
    EMBEDDINGS_DIR = os.path.join(DUMPS_DIR, "embeddings")
    os.makedirs(DUMPS_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    @staticmethod
    def flat_dump(library: str):
        # naive parsing
        return os.path.join(Locations.DUMPS_DIR, f"{library}_flat.pickle")

    @staticmethod
    def dag_dir(library: str):
        return os.path.join(r"D:\sexp_dumps", f"{library}_dags")

    @staticmethod
    def forest_dump(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_forest.pickle")

    @staticmethod
    def nn_jaccard(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_nn_jaccard.pickle")

    @staticmethod
    def nn_tfidf(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_nn_tfidf.pickle")

    @staticmethod
    def results_jaccard(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_results_jaccard.txt")

    @staticmethod
    def results_tfidf(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_results_tfidf.txt")

    @staticmethod
    def tree_analysis(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_tree_analysis.pickle")

    @staticmethod
    def definitions_pickled(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_definitions.pickle")

    @staticmethod
    def definitions_text_dir(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_definitions")

    @staticmethod
    def knowledge_graph(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_kg.pickle")

    @staticmethod
    def knowledge_graph_pure(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_kg_pure.pickle")

    @staticmethod
    def facts(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_facts.pickle")

    @staticmethod
    def dataset(library: str, p_test: float, p_body_to_keep: float):
        return os.path.join(
            Locations.DUMPS_DIR, f"{library}_dataset_{p_test}_{p_body_to_keep}.pickle"
        )

    @staticmethod
    def library_name_to_internal(library: str, n_folds: int, i_fold: int):
        return f"{library}_internal_{n_folds}_{i_fold}"

    @staticmethod
    def internal_library_name_to_library_name(internal_library: str):
        return internal_library[: internal_library.rfind("_internal_")]

    @staticmethod
    def model_file(library: str, model: Type[Any], file_appendix: str = ""):
        return os.path.join(
            Locations.EXPERIMENTS_DIR,
            f"{library}_model_{model.__name__}{file_appendix}.pickle",
        )

    @staticmethod
    def predictions_file(library: str, model: Type[Any], file_appendix: str = ""):
        return os.path.join(
            Locations.EXPERIMENTS_DIR,
            f"{library}_predictions_{model.__name__}{file_appendix}.txt",
        )

    @staticmethod
    def experiment_meta_file(library: str, model: Type[Any], file_appendix: str = ""):
        return os.path.join(
            Locations.EXPERIMENTS_DIR,
            f"{library}_meta_{model.__name__}{file_appendix}.txt",
        )

    @staticmethod
    def temp_learning_file(library: str, model: Type[Any], file_appendix: str = ""):
        return os.path.join(
            Locations.EXPERIMENTS_DIR,
            f"{library}_{model.__name__}{file_appendix}.temp",
        )

    @staticmethod
    def graph_text_dump(library: str):
        return os.path.join(Locations.DUMPS_DIR, f"{library}_kg.txt")

    @staticmethod
    def vocabulary_file(library: str, extended: bool):
        if extended:
            appendix = "_all"
        else:
            appendix = "_names"
        return os.path.join(
            Locations.EMBEDDINGS_DIR, f"{library}_vocabulary{appendix}.csv"
        )


class Other:
    @staticmethod
    def get_all_dirs(root_dir: str):
        all_dirs: List[str] = [root_dir]
        for candidate in os.listdir(root_dir):
            full_path = os.path.join(root_dir, candidate)
            if os.path.isdir(full_path):
                all_dirs.extend(Other.get_all_dirs(full_path))
        return sorted(all_dirs)

    @staticmethod
    def get_all_files(root_dir: str, file_extension: str = ".agda") -> List[str]:
        all_files: List[str] = []
        for subdir in Other.get_all_dirs(root_dir):
            for candidate in os.listdir(subdir):
                full_path = os.path.join(subdir, candidate)
                if os.path.isfile(full_path) and full_path.endswith(file_extension):
                    all_files.append(full_path)
        return sorted(all_files)

    @staticmethod
    def class_checker(value: Any, expected: Type[Any]) -> Any:
        if isinstance(value, expected):
            return value
        else:
            raise ValueError(
                f"Value {value} should be of type {expected}, but is {type(value)}"
            )

    @staticmethod
    def sample_negative_edges(
        graph: nx.MultiDiGraph,
        definitions_ids: List[mytypes.NODE],
        def_id: mytypes.NODE,
        n_to_sample: int,
    ) -> List[Tuple[mytypes.NODE, mytypes.NODE, mytypes.Edge]]:
        negative: List[mytypes.NODE] = []
        n = len(definitions_ids)
        e_type: mytypes.Edge = mytypes.Edge.REFERENCE_IN_BODY
        for _ in range(n_to_sample):
            start = random.randint(0, n - 1)
            found = False
            # after at most n iterations we should find a nonexisting edge
            for i in range(n):
                candidate = definitions_ids[(start + i) % n]
                if (
                    not graph.has_edge(def_id, candidate, e_type)
                    and candidate not in negative
                ):
                    negative.append(candidate)
                    found = True
                    break
            if not found:
                break
        return [(def_id, c, e_type) for c in negative]


class Embeddings:
    @staticmethod
    def load_embedding(
        file: str, normalize: bool = False
    ) -> Tuple[List[str], mytypes.ARRAY_2D]:
        """
        Loads the embedding as (words, matrix) pair.

        :param file: A path to the file whose contents are

        N D
        word1 x11 x12 ... x1D
        word2 x21 x22 ... x2D
        ...
        wordN xN1 xN2 ... xND

        where all the values are space-separated.

        :param normalize normalize (in Euclidean)

        :return: (list of words, embedding matrix)
        """

        words: List[str] = []
        with open(file, encoding="utf-8") as f:
            _, n_dim = map(int, f.readline().split())
            for line in f:
                words.append(line[: line.find(" ")])
                if not words[-1]:
                    raise ValueError("Not liking space as a word")
        matrix = np.loadtxt(
            file,
            delimiter=" ",
            skiprows=1,
            usecols=list(range(1, n_dim + 1)),
            encoding="utf-8",
            comments=None,
        )
        assert (len(words), n_dim) == matrix.shape, ((len(words), n_dim), matrix.shape)
        return words, matrix

    @staticmethod
    def load_words(file: str, separator: str = " ") -> List[str]:
        """
        Either loads words from embedding file (that contains dimensions in the first row)
        or library word file (that contains WORD,COUNT in the first row).
        :param file: path to the file
        :param separator:
        :return: a list of words
        """
        with open(file, encoding="utf-8") as f:
            f.readline()  # header(-like) row
            words: List[str] = []
            for row in f:
                words.append(row[: row.find(separator)])
        for i, word in enumerate(words):
            if not word:
                raise ValueError(
                    f"Empty word detected in '{file}' at position {i}/{len(words)}."
                )
        return words


class TextManipulation:
    @staticmethod
    def extract_words(text: str) -> List[str]:
        """
        Extracts words from text. Since the names are meaningful (most of the times),
        we also split them to parts.

        :param text: space-separated call
        :return:
        """
        words = re.sub(" +", " ", text).split(" ")
        words = [
            part for word in words for part in TextManipulation.name_to_parts(word)
        ]
        return words

    @staticmethod
    def normalize_tree_text(text: str):
        text = re.sub("[()]", " ", text)
        text = re.sub(" +", " ", text)
        return text.strip()

    @staticmethod
    def name_to_parts(name: str) -> List[str]:
        """
        Splits the name to its atomic parts, as defined by camel case, dashes or underscores.
        Non-ascii characters and numbers are also extracted separately

        :param name: a string, for example ``this₂₂Is_aVery-longName_innit12∘21``

        :return: a list of parts, for example
            ``['this', '₂₂', 'is', 'a', 'very', 'long', 'name', 'innit', '12', '∘', '21']``

        """

        def is_split(i: int):
            """
            Should be split the word just before the i-th character?
            :param i:
            :return:
            """
            if i == 0:
                return False
            this = name[i]
            previous = name[i - 1]
            if this.isnumeric() and previous.isnumeric():
                # two numbers
                return False
            elif (
                this.isalpha()
                and previous.isalpha()
                and not (this.upper() == this and previous.lower() == previous)
            ):
                # two non-camel-case letters
                return False
            else:
                return True

        special = "_-."
        extended: List[str] = []
        for position, character in enumerate(name):
            if is_split(position):
                extended.append("_")
            if character not in special:
                extended.append(character)
        return re.sub("_+", "_", "".join(extended)).lower().split("_")


class NetworkxWrappers:
    """
    More or less experimental thingy that would
    avoid type-checking issues.
    """

    @staticmethod
    def graph_nodes(graph: nx.Graph) -> Iterator[mytypes.NODE]:
        yield from graph.nodes

    @staticmethod
    def graph_edges(
        graph: nx.Graph,
    ) -> Iterator[Tuple[mytypes.NODE, mytypes.NODE, mytypes.Edge]]:
        yield from graph.edges
