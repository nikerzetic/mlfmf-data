import pickle
import itertools
import os
import networkx as nx


def get_lib_files(directory: str, lib_name: str):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(lib_name + "_dataset_0.2")
    ]


def check_file(dataset_file):
    print("Checking file:", dataset_file)
    with open(dataset_file, "rb") as f:
        triple = pickle.load(f)
        if len(triple) != 3:
            print("Weird one: ", dataset_file)
            return
        train_graph: nx.MultiDiGraph = triple[0]
        defs = triple[1]
        edges = triple[2]

    train_defs, test_defs = defs
    positive_edges, negative_edges = edges

    # no test defs in train defs
    for t_def in test_defs:
        assert t_def not in train_defs, t_def
    # no test edges in train graph/test defs
    for source, sink, edge_type in positive_edges:
        assert source in test_defs, source
        # graph
        assert not train_graph.has_edge(source, sink, edge_type), (
            source,
            sink,
            edge_type,
        )
        # defs
        # a, b, c = sink
        # sink_string = f"{a} {b} {c.replace('.', ' ')}"


for lib_name in ["stdlib"]:
    for dataset_file in get_lib_files("dumps", lib_name):
        check_file(dataset_file)
