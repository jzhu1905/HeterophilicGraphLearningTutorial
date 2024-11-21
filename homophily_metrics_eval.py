import sys
sys.path.append("./Heterophilic_Benchmarks")

from collections import namedtuple

import torch

from constants import HETEROPHILY_METRICS
from Heterophilic_Benchmarks.hetero_metric_syn import compute_metrics_on_syn_graph

eval_args = namedtuple("EvalArgs",
                       ["mode", "metric", "graph_id", 
                       # for PA synthetic graphs
                       "mixhop_h", 
                       # for GenCat synthetic graphs
                       "base_dataset_gencat", "beta",
                       # for regular synthetic graphs
                       "num_edge_same", "homo_lvl", "base_dataset_rg"])   


def eval_gencat_syn_graph(beta: float, with_gpu: bool) -> dict:
    if with_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args = eval_args("Gencat", "edge", 0, 
                     .0, 
                     "cora", beta, 
                     800, 0.15, "cora")
    return _calculate_all_metrics_for_graph(device, args)


def eval_pa_syn_graph(mixhop_h: float, with_gpu: bool) -> dict:
    if with_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args = eval_args("PA", "edge", 0, 
                     mixhop_h, 
                     "cora", 0, 
                     800, 0.15, "cora")
    return _calculate_all_metrics_for_graph(device, args)


def eval_regular_syn_graph(homo_lvl: float, with_gpu: bool):
    if with_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    args = eval_args("RG", "edge", 0, 
                     .0, 
                     "cora", 0, 
                     800, homo_lvl, "cora")
    return _calculate_all_metrics_for_graph(device, args)


def _calculate_all_metrics_for_graph(device: torch.device, 
                                     args: namedtuple):
    res = {}
    for metric, label in HETEROPHILY_METRICS.items():
        print(f"Computing {label}...")
        args = args._replace(metric=metric)
        computed_metric = compute_metrics_on_syn_graph(args, device)
        res[metric] = computed_metric
    return res
