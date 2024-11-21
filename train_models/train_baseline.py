import sys

sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")

from collections import namedtuple

import torch

from Heterophilic_Benchmarks.train_opengsldata_baseline import train_opengsldata_baseline
from Heterophilic_Benchmarks.train_pathnetdata_baseline import train_pathnetdata_basline

gcn_args = namedtuple("GCNArgs",
                      ["dataset", "cuda", "method", "run", "epoch_num",
                       "n_hid", "dropout", "lr", "weight_decay"])


def train_baseline(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = gcn_args(dataset, cuda, "GCN", 10, 1000,
                    64, 0.0, 0.01, 1e-3)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "opengsl":
        train_opengsldata_baseline(device, args)
    elif dataset_class.lower() == "pathnet":
        args._replace(weight_decay=5e-7)
        train_pathnetdata_basline(device, args)
