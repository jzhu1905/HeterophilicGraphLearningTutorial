import sys

sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")
sys.path.append("../Heterophilic_Benchmarks/GloGNN_repo")

from collections import namedtuple

import torch

from Heterophilic_Benchmarks.GloGNN_repo.train_criticaldata_othergnns import train_criticaldata_othergnns
from Heterophilic_Benchmarks.GloGNN_repo.train_largedata_othergnns import train_largedata_othergnns
from Heterophilic_Benchmarks.GloGNN_repo.train_opengsldata_othergnns import train_opengsldata_othergnns
from Heterophilic_Benchmarks.GloGNN_repo.train_pathnetdata_othergnns import train_pathnetdata_othergnns

appnp_args = namedtuple("APPNPArgs",
                        ["dataset", "cuda", "method", "run", "epoch_num",
                         "hidden_channels", "dropout", "lr", "weight_decay", "early_stopping", "num_layers",
                         "gpr_alpha"])
h2gcn_args = namedtuple("H2GCNArgs",
                        ["dataset", "cuda", "method", "run", "epoch_num",
                         "hidden_channels", "dropout", "lr", "weight_decay", "early_stopping", "num_layers",
                         "num_mlp_layers"])
linkx_args = namedtuple("LinkXArgs",
                        ["dataset", "cuda", "method", "run", "epoch_num",
                         "hidden_channels", "dropout", "lr", "weight_decay", "early_stopping", "num_layers",
                         "inner_activation", "inner_dropout", "link_init_layers_A", "link_init_layers_X"])


def train_appnp(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = appnp_args(dataset, cuda, "appnp", 10, 1000,
                      64, 0.5, 0.01, 0.0001,
                      10000, 2,
                      0.1)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        args._replace(hidden_channels=512)
        train_criticaldata_othergnns(device, args)
    elif dataset_class.lower() == "large":
        train_largedata_othergnns(device, args)
    elif dataset_class.lower() == "opengsl":
        train_opengsldata_othergnns(device, args)
    elif dataset_class.lower() == "pathnet":
        train_pathnetdata_othergnns(device, args)


def train_h2gcn(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = h2gcn_args(dataset, cuda, "h2gcn", 10, 1000,
                      64, 0.5, 0.01, 0.0001,
                      10000, 2,
                      1)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        args._replace(hidden_channels=512)
        train_criticaldata_othergnns(device, args)
    elif dataset_class.lower() == "large":
        train_largedata_othergnns(device, args)
    elif dataset_class.lower() == "opengsl":
        train_opengsldata_othergnns(device, args)
    elif dataset_class.lower() == "pathnet":
        train_pathnetdata_othergnns(device, args)


def train_linkx(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = linkx_args(dataset, cuda, "linkx", 10, 1000,
                      64, 0.5, 0.01, 0.0001,
                      10000, 2,
                      False, False, 1, 1)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        args._replace(hidden_channels=512)
        train_criticaldata_othergnns(device, args)
    elif dataset_class.lower() == "large":
        train_largedata_othergnns(device, args)
    elif dataset_class.lower() == "opengsl":
        train_opengsldata_othergnns(device, args)
    elif dataset_class.lower() == "pathnet":
        train_pathnetdata_othergnns(device, args)
