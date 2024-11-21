import sys

sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")
sys.path.append("../Heterophilic_Benchmarks/BernNet")

from collections import namedtuple

import torch

from Heterophilic_Benchmarks.BernNet.train_criticaldata import train_criticaldata
from Heterophilic_Benchmarks.BernNet.train_largedata import train_largedata
from Heterophilic_Benchmarks.BernNet.train_opengsldata import train_opengsldata
from Heterophilic_Benchmarks.BernNet.train_pathnetdata import train_pathnetdata

bernet_args = namedtuple("BNArgs",
                         ["dataset", "cuda", "method", "run", "epoch_num",
                          "n_hid", "dropout", "lr", "weight_decay", "early_stopping",
                          "K", "dprate", "Bern_lr"])


def train_bernet(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = bernet_args(dataset, cuda, "BernNet", 10, 1000,
                       512, 0.5, 0.01, 0.0005, 10000,
                       10, 0.5, 0.01)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        train_criticaldata(device, args)
    elif dataset_class.lower() == "large":
        args._replace(n_hid=64)
        train_largedata(device, args)
    elif dataset_class.lower() == "opengsl":
        args._replace(n_hid=128)
        train_opengsldata(device, args)
    elif dataset_class.lower() == "pathnet":
        args._replace(n_hid=128)
        train_pathnetdata(device, args)
