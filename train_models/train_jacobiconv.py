import sys

sys.path.append("..")
sys.path.append("../Heterophilic_Benchmarks")
sys.path.append("../Heterophilic_Benchmarks/JacobiConv")

from Heterophilic_Benchmarks.JacobiConv.train_alldata_jacobiconv import train_alldata_jacobiconv
from collections import namedtuple

import torch

jconv_args = namedtuple("JConvArgs",
                        ["dataset", "method", "run",
                         "optruns", "path", "name",
                         "detach", "savemodel", "power", "cheby", "legendre", "bern", "sole", "fixalpha", "multilayer",
                         "resmultilayer",
                         "lr", "wd", "alpha", "a", "b", "dpb", "dpt"])


def train_jacobiconv(dataset: str, cuda: int, with_gpu: bool):
    args = jconv_args(dataset, "Jaccobi", 10,
                      50, "", "opt",
                      False, False, False, False, False, False, False, False, False, False,
                      0.005, 1e-4, 0.2, 0.0, 0.0, 0.0, 0.0)
    if with_gpu:
        device = torch.device(f"cuda:{str(cuda)}")
    else:
        device = torch.device("cpu")

    train_alldata_jacobiconv(device, args)
