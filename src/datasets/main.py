from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .pressure import PRESSURE_Dataset
import pandas as pd


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'pressure')
    assert dataset_name in implemented_datasets

    dataset = None
    path = '../dataset/feature/all.csv'

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'pressure':
        dataset = PRESSURE_Dataset(data_path, normal_class=normal_class)

    return dataset
