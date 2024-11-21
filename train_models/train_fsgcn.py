import sys

sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")
sys.path.append("../Heterophilic_Benchmarks/FSGCN")

from collections import namedtuple

import torch

from Heterophilic_Benchmarks.FSGCN.train_criticaldata_fsgcn import train_criticaldata_fsgcn
from Heterophilic_Benchmarks.FSGCN.train_geomdata_fsgcn import train_geomdata_fsgcn
from Heterophilic_Benchmarks.FSGCN.train_largedata_fsgcn import train_largedata_fsgcn
from Heterophilic_Benchmarks.FSGCN.train_opengsldata_fsgcn import train_opengsldata_fsgcn
from Heterophilic_Benchmarks.FSGCN.train_pathdata_fsgcn import train_pathdata_fsgcn

fsgcn_args = namedtuple("FSGCNArgs",
                        ["dataset", "cuda", "method", "run", "epoch_num",
                         "n_hid", "dropout", "lr", "weight_decay",
                         "num_layers", "feat_type", "layer_norm", "patience"])


def train_fsgcn(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = fsgcn_args(dataset, cuda, "FSGCN", 10, 1000,
                      64, 0.0, 0.01, 5e-7,
                      3, "all", 1, 10000)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        args._replace(n_hid=512)
        args._replace(num_layers=8)
        train_criticaldata_fsgcn(device, args)
    elif dataset_class.lower() == "geom":
        train_geomdata_fsgcn(device, args)
    elif dataset_class.lower() == "large":
        train_largedata_fsgcn(device, args)
    elif dataset_class.lower() == "opengsl":
        args._replace(n_hid=128)
        train_opengsldata_fsgcn(device, args)
    elif dataset_class.lower() == "pathnet":
        train_pathdata_fsgcn(device, args)
