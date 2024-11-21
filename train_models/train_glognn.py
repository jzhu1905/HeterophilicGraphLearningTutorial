import sys
sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")
sys.path.append("../Heterophilic_Benchmarks/GloGNN_repo")

from collections import namedtuple

import torch

from Heterophilic_Benchmarks.GloGNN_repo.train_criticaldata_glognn import train_criticaldata_glognn
from Heterophilic_Benchmarks.GloGNN_repo.train_geomdata_glognn import train_geomdata_glognn

glognn_args = namedtuple("GloGNNArgs",
                         ["dataset", "cuda", "method", "run", "epoch_num",
                          "hidden_channels", "dropout", "lr", "weight_decay",
                          "early_stopping", "num_layers",
                          "alpha", "beta", "gamma", "delta", "norm_func_id", "norm_layers", "orders_func_id", "orders"])


def train_glognn(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = glognn_args(dataset, cuda, "mlpnorm", 10, 1000,
                       64, 0.5, 0.01, 0.0001,
                       10000, 2,
                       0.0, 1.0, 0.0, 0.0, 2, 1, 2, 1)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        train_criticaldata_glognn(device, args)
    elif dataset_class.lower() == "geom":
        train_geomdata_glognn(device, args)
