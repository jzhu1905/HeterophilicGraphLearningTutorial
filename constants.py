from ctypes import resize
import sys
import torch

from collections import defaultdict
from pathlib import Path

sys.path.append("./Heterophilic_Benchmarks")

GRAPH_MODELS = {
    "ACM": [
        ("Critical Look", "critical"),
        ("OpenGSL", "opengsl"),
        ("PathNet", "pathnet")
    ],
    "BernNet": [
        ("Critical Look", "critical"),
        ("Large Scale", "large"),
        ("OpenGSL", "opengsl"),
        ("PathNet", "pathnet")
    ],
    "FAGCN": [
        ("Critical Look", "critical"),
        ("Large Scale", "large"),
        ("OpenGSL", "opengsl"),
        ("PathNet", "pathnet")
    ],
    "FSGCN": [
        ("Critical Look", "critical"),
        ("Geometric", "geom"),
        ("Large Scale", "large"),
        ("OpenGSL", "opengsl"),
        ("PathNet", "pathnet")
    ],
    "GBKGNN": [
        ("Critical Look", "critical"),
        ("Geometric", "geom"),
        ("OpenGSL", "opengsl"),
        ("PathNet", "pathnet")
    ],
    "APPNP": [
        ("Critical Look", "critical"),
        ("Large Scale", "large"),
        ("OpenGSL", "opengsl"),
        ("PathNet", "pathnet")
    ],
    "H2GCN": [
        ("Critical Look", "critical"),
        ("Large Scale", "large"),
        ("OpenGSL", "opengsl"),
        ("PathNet", "pathnet")
    ],
    "LinkX": [
        ("Critical Look", "critical"),
        ("Large Scale", "large"),
        ("OpenGSL", "opengsl"),
        ("PathNet", "pathnet")
    ]
}

DATASETS = {
    "critical": [
        ('Actor', 'actor'),
        ('Amazon-Ratings', 'amazon-ratings'),
        ('Chameleon', 'chameleon'),
        ('Chameleon-Directed', 'chameleon-directed'),
        ('Chameleon-Filtered', 'chameleon-filtered'),
        ('Chameleon-Filtered-Directed', 'chameleon-filtered-directed'),
        ('Cornell', 'cornell'),
        ('Minesweeper', 'minesweeper'),
        ('Roman-Empire', 'roman-empire'),
        ('Questions', 'questions'),
        ('Squirrel', 'squirrel'),
        ('Squirrel-Directed', 'squirrel-directed'),
        ('Squirrel-Filtered', 'squirrel-filtered'),
        ('Squirrel-Filtered-Directed', 'squirrel-filtered-directed'),
        ('Texas', 'texas'),
        ('Texas-4-Classes', 'texas-4-classes'),
        ('Tolokers', 'tolokers'),
        ('Wisconsin', 'wisconsin')
    ],
    "geom": [
        ('Arxiv-Year', 'arxiv-year'),
        ('Chameleon', 'chameleon'),
        ('CiteSeer', 'CiteSeer'),
        ('Cornell', 'cornell'),
        ('Cora', 'Cora'),
        ('Deezer-Europe', 'deezer-europe'),
        ('Film', 'film'),
        ('Genius', 'genius'),
        ('Penn94', 'penn94'),
        ('Pokec', 'pokec'),
        ('PubMed', 'PubMed'),
        ('Snap-Patents', 'snap-patents'),
        ('Squirrel', 'squirrel'),
        ('Texas', 'texas'),
        ('Twitch-Gamers', 'twitch-gamers'),
        ('Wisconsin', 'wisconsin')
    ],
    "large": [
        ('Arxiv-Year', 'arxiv-year'),
        ('Chameleon', 'chameleon'),
        ('Citeseer', 'CiteSeer'),
        ('Cora', 'Cora'),
        ('Cornell', 'cornell'),
        ('Film', 'film'),
        ('Fb100', 'fb100'),
        ('Genius', 'genius'),
        ('Ogbn-Arxiv', 'ogbn-arxiv'),
        ('Ogbn-Products', 'ogbn-products'),
        ('Ogbn-Proteins', 'ogbn-proteins'),
        ('Pokec', 'pokec'),
        ('Pubmed', 'PubMed'),
        ('Snap-Patents', 'snap-patents'),
        ('Squirrel', 'squirrel'),
        ('Texas', 'texas'),
        ('Twitch-E', 'twitch-e'),
        ('Twitch-Gamer', 'twitch-gamer'),
        ('Wiki', 'wiki'),
        ('Wisconsin', 'wisconsin'),
        ('Yelp-Chi', 'yelp-chi')
    ],
    "opengsl": [
        ('Blogcatalog', 'blogcatalog'),
        ('Flickr', 'flickr'),
        ('Wiki-Cooc', 'wiki-cooc')
    ],
    "pathnet": [
        ('Bgp', 'Bgp'),
        ('Electronics', 'Electronics'),
        ('Nba', 'Nba')
    ]
}

HETEROPHILY_METRICS = {
    "node": "Node Homophily",
    "edge": "Edge Homophily",
    "class": "Class Homophily",
    "ge": "Generalized Edge Homophily",
    "agg": "Aggregation Homophily",
    "adjust": "Adjusted Homophily",
    "li": "Label informativeness",
    "ne": "Neighborhood Identifiability",
    "gnb": "Gaussian Naive Bayes",
    "kernel_reg0": "Kernel Linear Regression",
    "kernel_reg1": "Kernel Non-linear Regression",
}


def _load_all_metric_results() -> dict:
    RES_DIR_PATH = "./Heterophilic_Benchmarks/metrics_results"
    res = {
        "PA": torch.load(f'{RES_DIR_PATH}/PA.pt'),
        "RG": torch.load(f'{RES_DIR_PATH}/RG.pt'),
        "GenCat": torch.load(f'{RES_DIR_PATH}/Gencat.pt'),
    }
    return res

ALL_METRICS_RESULTS = _load_all_metric_results()
