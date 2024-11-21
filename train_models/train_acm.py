import sys
sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")
sys.path.append("../Heterophilic_Benchmarks/GloGNN_repo")

from collections import namedtuple

import torch

from Heterophilic_Benchmarks.GloGNN_repo.train_criticaldata_acm import train_criticaldata_acm
from Heterophilic_Benchmarks.GloGNN_repo.train_opengsldata_acm import train_opengsldata_acm
from Heterophilic_Benchmarks.GloGNN_repo.train_pathnetdata_othergnns import train_pathnetdata_othergnns

acm_args = namedtuple("ACMArgs",
                      ["dataset", "cuda", "method", "run", "epoch_num",
                       "hidden_channels", "dropout", "lr", "weight_decay", "early_stopping", "num_layers",
                       "variant", "structure_info", "acm_method"])


def train_acm(dataset_class: str, dataset: str, cuda: int, with_gpu: bool):
    args = acm_args(dataset, cuda, "mlpnorm", 10, 1000,
                    64, 0.5, 0.01, 0.0001,
                    10000, 2,
                    1, 0, "acmgcnp")
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    if dataset_class.lower() == "critical":
        args._replace(hidden_channels=512)
        train_criticaldata_acm(device, args)
    elif dataset_class.lower() == "opengsl":
        train_opengsldata_acm(device, args)
    elif dataset_class.lower() == "pathnet":
        train_pathnetdata_othergnns(device, args)
