import sys

sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")
sys.path.append("../Heterophilic_Benchmarks/GBKGNN")

from collections import namedtuple

import torch

from Heterophilic_Benchmarks.GBKGNN.train_criticaldata_gbkgnn import train_criticaldata_gbkgnn
from Heterophilic_Benchmarks.GBKGNN.train_geomdata_gbkgnn import train_geomdata_gbkgnn
from Heterophilic_Benchmarks.GBKGNN.train_opengsldata_gbkgnn import train_opengsldata_gbkgnn
from Heterophilic_Benchmarks.GBKGNN.train_pathnetdata_gbkgnn import train_pathnetdata_gbkgnn

gbkgnn_args = namedtuple("GBKGNNArgs",
                         ["dataset_name", "cuda", "method", "run", "epoch_num",
                          "dim_size", "dropout", "lr", "weight_decay",
                          "split", "model_type", "aug", "lamda", "patience", "log_interval"])


def train_gbkgnn(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = gbkgnn_args(dataset, cuda, "GBKGCN", 10, 1000,
                       64, 0.0, 0.01, 5e-7,
                       [0.6, 0.2, 0.2], "GraphSage", True, 30, 10000, 100)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        args._replace(dim_size=512)
        train_criticaldata_gbkgnn(device, args)
    elif dataset_class.lower() == "geom":
        train_geomdata_gbkgnn(device, args)
    elif dataset_class.lower() == "opengsl":
        args._replace(dim_size=128)
        train_opengsldata_gbkgnn(device, args)
    elif dataset_class.lower() == "pathnet":
        train_pathnetdata_gbkgnn(device, args)
