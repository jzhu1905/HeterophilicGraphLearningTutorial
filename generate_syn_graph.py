import importlib
import sys

sys.path.append("./ACM-GNN")
sys.path.append("./ACM-GNN/synthetic-experiments")

from collections import namedtuple

from Heterophilic_Benchmarks.generate_mixhop_data import generate_syn_graph

acm_repo_graph = importlib.import_module("ACM-GNN.synthetic-experiments.graph_generation")
generate_regular_syn = acm_repo_graph.generate_graph
acm_repo_feature = importlib.import_module("ACM-GNN.synthetic-experiments.feature_generation")
generate_regular_feature = acm_repo_feature.generate_feature

pa_args = namedtuple("PAArgs",
                     ["c", "h", "n", "plot", "num_graph"])
regular_graph_args = namedtuple("RegularArgs",
                                ["num_class", "num_node_total", "degree_intra", "num_graph", "graph_type",
                                 "edge_homos"])
regular_feature_args = namedtuple("RegularFeatureArgs",
                                  ["num_node_total", "base_dataset"])


def generate_pa_syn_graph(homo_lvl: float):
    print(f"Generating PA synthetic graph with homophily level {homo_lvl}")
    args = pa_args(5, homo_lvl, 2000, False, 1)
    generate_syn_graph(args)
    print("A regular synthetic graph has been generated. "
          "You can find it in the directory ./mixhop_syn-2000_5/.")


def generate_regular_syn_graph(homo_lvl: float):
    print(f"Generating regular synthetic graph with homophily level {homo_lvl}")
    # generate graph
    graph_args = regular_graph_args(5, 2000, 2, 1, "regular", [homo_lvl])
    generate_regular_syn(graph_args)
    # generate feature
    feature_args = regular_feature_args(2000, "cora")
    generate_regular_feature(feature_args)
    print(f"A PA synthetic graph has been generated. "
          f"You can find it in the directory ./synthetic_graphs/regular/{homo_lvl}")
