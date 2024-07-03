from enum import Enum
from typing import Tuple

import networkx as nx
import numpy as np

NODE = str  # Tuple[str, str, str]

ARRAY_1D = np.ndarray[int, np.dtype[np.float64]]
ARRAY_2D = np.ndarray[Tuple[int, int], np.dtype[np.float64]]

INT_ARRAY_1D = np.ndarray[int, np.dtype[np.int_]]


class Edge(Enum):
    DEFINES = "DEFINES"
    CONTAINS = "CONTAINS"

    REFERENCE_IN_TYPE = "REFERENCE_TYPE"
    REFERENCE_IN_BODY = "REFERENCE_BODY"

    REFERENCE_IN_TYPE_TO_WITH = "REFERENCE_TYPE_TO_WITH"
    REFERENCE_IN_BODY_TO_WITH = "REFERENCE_BODY_TO_WITH"

    REFERENCE_IN_TYPE_TO_REWRITE = "REFERENCE_TYPE_TO_REWRITE"
    REFERENCE_IN_BODY_TO_REWRITE = "REFERENCE_BODY_TO_REWRITE"

    def __str__(self):
        return self.value

    def with_to_normal(self):
        if self == Edge.REFERENCE_IN_TYPE_TO_WITH:
            return Edge.REFERENCE_IN_TYPE
        elif self == Edge.REFERENCE_IN_BODY_TO_WITH:
            return Edge.REFERENCE_IN_BODY
        else:
            raise ValueError(f"Cannot normalize non-with edge type {self}")

    def normal_to_with(self):
        if self == Edge.REFERENCE_IN_TYPE:
            return Edge.REFERENCE_IN_TYPE_TO_WITH
        elif self == Edge.REFERENCE_IN_BODY:
            return Edge.REFERENCE_IN_BODY_TO_WITH
        raise ValueError(f"{self} --> with!?")

    def rewrite_to_normal(self):
        if self == Edge.REFERENCE_IN_BODY_TO_REWRITE:
            return Edge.REFERENCE_IN_BODY
        elif self == Edge.REFERENCE_IN_TYPE_TO_REWRITE:
            return Edge.REFERENCE_IN_TYPE
        else:
            raise ValueError(f"{self} --> normal!?")

    def normal_to_rewrite(self):
        if self == Edge.REFERENCE_IN_TYPE:
            return Edge.REFERENCE_IN_TYPE_TO_REWRITE
        elif self == Edge.REFERENCE_IN_BODY:
            return Edge.REFERENCE_IN_BODY_TO_REWRITE
        else:
            raise ValueError(f"{self} --> rewrite!?")

    def is_reference(self) -> bool:
        return self.value.startswith("REFERENCE")

    def is_normal_reference(self):
        return self == Edge.REFERENCE_IN_TYPE or self == Edge.REFERENCE_IN_BODY

    def is_normal(self):
        return (
            self.is_normal_reference() or self == Edge.DEFINES or self == Edge.CONTAINS
        )


class Node(Enum):
    # Agda and Lean
    ABSTRACT = ":abstract"
    APPLY = ":apply"
    AXIOM = ":axiom"
    CONSTRUCTOR = ":constructor"
    DATA = ":data"
    ENTRY = ":entry"
    FUNCTION = ":function"
    LAMBDA = ":lambda"
    LEVEL = ":level"
    LITERAL = ":literal"
    MAX = ":max"
    META = ":meta"
    MODULE = ":module"
    MODULE_NAME = ":module-name"
    NAME = ":name"
    PI = ":pi"
    PROJ = ":proj"
    SORT = ":sort"
    TYPE = ":type"
    VAR = ":var"

    # Agda specific
    ANONYMOUS = ":anonymous"
    ANONIMOUS = ":anonymous"  # just as a hack
    ARG = ":arg"
    ARG_NAME = ":arg-name"
    ARG_NONAME = ":arg-noname"
    ARG_NO_NAME = ":arg-noname"
    BODY = ":body"
    BOUND = ":bound"
    CASE_SPLIT = ":case-split"
    CLAUSE = ":clause"
    CONSTR = ":constr"
    DATA_OR_RECORD = ":data-or-record"
    DEF = ":def"
    DOT = ":dot"
    GENERALIZABLE_VAR = ":generalizable-var"
    HIDDEN = ":hidden"
    INSERTED = ":inserted"
    INSTANCE = ":instance"
    INTERNAL = ":internal"
    INTERVAL_APPLY = ":interval-apply"
    INTERVAL_ARG = ":interval-arg"
    IRRELEVANT = ":irrelevant"
    NO_BODY = ":no-body"
    NO_TYPE = ":no-type"
    NOT_HIDDEN = ":not-hidden"
    PATTERN = ":pattern"
    PATTERN_VAR = ":pattern-var"
    PLUS = ":plus"
    PRIMITIVE = ":primitive"
    RECORD = ":record"
    REFLECTED = ":reflected"
    SORT_DEF = ":sort-def"
    SORT_DUMMY = ":sort-dummy"
    SORT_FUN = ":sort-fun"
    SORT_INTERVAL = ":sort-interval"
    SORT_LOCK = ":sort-lock"
    SORT_META = ":sort-meta"
    SORT_PI = ":sort-pi"
    SORT_PROP = ":sort-prop"
    SORT_SET = ":sort-set"
    SORT_SETΩ = ":sort-setω"
    SORT_SET_OMEGA = ":sort-set-omega"
    SORT_SIZE = ":sort-size"
    SORT_SSET = ":sort-sset"
    SORT_UNIV = ":sort-univ"
    SUBSTITUTION = ":substitution"
    TELESCOPE = ":telescope"
    USER_WRITTEN = ":user-written"

    # Lean specific
    CONST = ":const"
    CTOR = ":ctor"
    DEFAULT = ":default"
    FVAR = ":fvar"
    IMAX = ":imax"
    IMPLICIT = ":implicit"
    IND = ":ind"
    INST_IMPLICIT = ":inst-implicit"
    LET = ":let"
    LIFT = ":lift"
    LSUCC = ":lsucc"
    LZERO = ":lzero"
    NODE = ":node"
    QUOT_INFO = ":quot-info"
    RECURSOR = ":recursor"
    REF = ":ref"
    REFERENCES = ":references"
    STRICT_IMPLICIT = ":strict-implicit"
    THEOREM = ":theorem"

    # special node types
    LIBRARY = ":library"
    MODULE_LIKE = ":module-like"

    EXTERNAL = ":external"
    EXTERNAL_MODULE = ":external-module"
    EXTERNAL_LIBRARY = ":external-library"

    # for testing
    FOO = ":foo"
    BAR = ":bar"
    BAZ = ":baz"

    def is_name(self):
        return self in [Node.NAME, Node.MODULE_NAME]

    def is_external(self):
        return self in [
            Node.EXTERNAL,
            Node.EXTERNAL_MODULE,
            Node.EXTERNAL_LIBRARY,
        ]

    def is_module(self):
        return self in [Node.MODULE, Node.EXTERNAL_MODULE]

    def is_definition_type(self):
        return self in [
            Node.FUNCTION,
            Node.CONSTRUCTOR,
            Node.RECORD,
            Node.DATA,
            Node.AXIOM,
            Node.PRIMITIVE,
            Node.SORT,
        ]

    def __str__(self):
        return self.value

    @staticmethod
    def get_theorem_like_tag(graph: nx.MultiDiGraph) -> "Node":
        """
        Returns THEOREM if any :theorem is present, and FUNCTION otherwise.
        """
        for node, props in graph.nodes(data=True):
            if not isinstance(props["label"], Node):
                raise ValueError(f"Should be node type, got {node}: {props}")
            if props["label"] == Node.THEOREM:
                return Node.THEOREM
        return Node.FUNCTION
