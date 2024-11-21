import sys

sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")
sys.path.append("../Heterophilic_Benchmarks/FAGCN")

from collections import namedtuple

import torch

from Heterophilic_Benchmarks.FAGCN.train_criticaldata_fagcn import train_criticaldata_fagcn
from Heterophilic_Benchmarks.FAGCN.train_largedata_fagcn import train_largedata_fagcn
from Heterophilic_Benchmarks.FAGCN.train_opengsldata_fagcn import train_opengsldata_fagcn
from Heterophilic_Benchmarks.FAGCN.train_pathnetdata_fagcn import train_pathnetdata_fagcn

fagcn_args = namedtuple("FAGCNArgs",
                        ["dataset", "cuda", "method", "run", "epoch_num",
                         "n_hid", "dropout", "lr", "weight_decay",
                         "remove_zero_in_degree_nodes", "eps", "layer_num", "patience"])


def train_fagcn(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = fagcn_args(dataset, cuda, "FAGCN", 10, 1000,
                      64, 0.5, 0.002, 0.0005,
                      False, 0.3, 2, 10000)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        args._replace(n_hid=512)
        train_criticaldata_fagcn(device, args)
    elif dataset_class.lower() == "large":
        train_largedata_fagcn(device, args)
    elif dataset_class.lower() == "opengsl":
        train_opengsldata_fagcn(device, args)
    elif dataset_class.lower() == "pathnet":
        train_pathnetdata_fagcn(device, args)
