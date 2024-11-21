import sys

sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")
sys.path.append("../Heterophilic_Benchmarks/GPRGNN")

from collections import namedtuple

import torch

from Heterophilic_Benchmarks.GPRGNN.train_criticaldata_gprgnn import train_criticaldata_gprgnn
from Heterophilic_Benchmarks.GPRGNN.train_opengsldata_gprgnn import train_opengsldata_gprgnn
from Heterophilic_Benchmarks.GPRGNN.train_pathnetdata_gprgnn import train_parthnetdata_gprgnn

gprgnn_args = namedtuple("GPRGNNArgs",
                         ["dataset", "cuda", "method", "run", "epoch_num",
                          "n_hid", "dropout", "lr", "weight_decay", "early_stopping",
                          "K", "alpha", "Init", "Gamma", "dprate", "ppnp"])


def train_gprgnn(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = gprgnn_args(dataset, cuda, "GPRGNN", 10, 1000,
                       512, 0.5, 0.002, 0.0005, 10000,
                       10, 0.9, "PPR", None, 0.5, "GPR_prop")
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        train_criticaldata_gprgnn(device, args)
    elif dataset_class.lower() == "opengsl":
        args._replace(n_hid=128)
        train_opengsldata_gprgnn(device, args)
    elif dataset_class.lower() == "pathnet":
        args._replace(n_hid=64)
        train_parthnetdata_gprgnn(device, args)
