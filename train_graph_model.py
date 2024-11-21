import os
import shutil
import zipfile

import gdown

from train_models.train_acm import train_acm
from train_models.train_bernnet import train_bernet
from train_models.train_fagcn import train_fagcn
from train_models.train_fsgcn import train_fsgcn
from train_models.train_gbkgnn import train_gbkgnn
from train_models.train_gprgnn import train_gprgnn
from train_models.train_jacobiconv import train_jacobiconv
from train_models.train_other_gnns import train_appnp, train_h2gcn, train_linkx


def train_graph_model(model_name: str, dataset_class: str, dataset_name: str, with_gpu: bool):
    cuda = 0

    print("Preparing dataset...")
    _get_dataset(dataset_class)

    print(f"Start training {model_name} on dataset: {dataset_name}")
    if model_name.lower() == "acm":
        train_acm(dataset_class, dataset_name, cuda, with_gpu)
    elif model_name.lower() == "bernnet":
        train_bernet(dataset_class, dataset_name, cuda, with_gpu)
    elif model_name.lower() == "fagcn":
        train_fagcn(dataset_class, dataset_name, cuda, with_gpu)
    elif model_name.lower() == "fsgcn":
        train_fsgcn(dataset_class, dataset_name, cuda, with_gpu)
    elif model_name.lower() == "gbkgnn":
        train_gbkgnn(dataset_class, dataset_name, cuda, with_gpu)
    elif model_name.lower() == "gprgnn":
        train_gprgnn(dataset_class, dataset_name, cuda, with_gpu)
    elif model_name.lower() == "jacobiconv":
        train_jacobiconv(dataset_name, cuda, with_gpu)
    elif model_name.lower() == "appnp":
        train_appnp(dataset_class, dataset_name, cuda, with_gpu)
    elif model_name.lower() == "h2gcn":
        train_h2gcn(dataset_class, dataset_name, cuda, with_gpu)
    elif model_name.lower() == "linkx":
        train_linkx(dataset_class, dataset_name, cuda, with_gpu)


def _get_dataset(dataset_class: str):
    if dataset_class == "critical":
        url = "https://drive.google.com/file/d/1_pysG3_l8w5F3GXvq6qTg-Ja2astBhxm/view?usp=sharing"
        output_path = "./Heterophilic_Benchmarks/critical_look_utils/data/"
        file_name = "critical_data"
        dest_dir = f"{output_path}critical_data/"
        _download_data_n_unzip(url, file_name, output_path, dest_dir)
    elif dataset_class == "opengsl":
        with zipfile.ZipFile("./Heterophilic_Benchmarks/Opengsl/data.zip", 'r') as zip_ref:
            zip_ref.extractall("./Heterophilic_Benchmarks/Opengsl/")
    elif dataset_class == "pathnet":
        with zipfile.ZipFile("./Heterophilic_Benchmarks/PathNet/other_data.zip", 'r') as zip_ref:
            zip_ref.extractall("./Heterophilic_Benchmarks/PathNet/other_data")
    elif dataset_class in ("geom", "large"):
        url = "https://drive.google.com/file/d/1pSRRd8skDieCy8pKYiATS9WGaK0gfDOs/view?usp=sharing"
        output_path = "./Heterophilic_Benchmarks/large_scale_data_utils/large_scale_data/"
        file_name = "large_scale_data"
        dest_dir = f"{output_path}{file_name}"
        _download_data_n_unzip(url, file_name, output_path, dest_dir)

        splits_url = "https://drive.google.com/file/d/1wCU40bgSm7XDXz_8o4ykxZEgK4swN_IR/view?usp=sharing"
        splits_output_path = "./Heterophilic_Benchmarks/splits/"
        splits_file_name = "splits"
        splits_dest_dir = ""
        _download_data_n_unzip(splits_url, splits_file_name, splits_output_path, splits_dest_dir)



def _download_data_n_unzip(url: str, file_name: str, output_path: str, dest_dir: str):
    gdown.download(url, output_path, quiet=False, fuzzy=True)
    file_path = f"{output_path}/{file_name}.zip"
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        if dest_dir:
            for file_name in os.listdir(dest_dir):
                shutil.move(os.path.join(dest_dir, file_name), output_path)
            shutil.rmtree(dest_dir)
        os.remove(file_path)
    except Exception:
        os.remove(file_path)
