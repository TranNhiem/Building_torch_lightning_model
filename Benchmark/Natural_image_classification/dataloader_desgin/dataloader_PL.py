# This multi-Function dataloader for SSL Project using pl.LightningDataModule
# Author Tran Rick 03/2022
#

from ctypes import Union
from importlib.resources import path
import logging
from pathlib import Path
from this import d
from typing import Union, Optional

import pytorch_lighning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms
from torchvision.datasets import ImageFolder


class Natural_Img_dataloader(LightningDataModule):

    """
    Multi-Natural Natural images Dataloader.
        Args:
            data_path: root directory where to download the data
            num_workers: number of CPU or GPUs workers
            batch_size: number of sample in a batch
    """
    def __init__(self, dl_path: Union[str, Path] = "data",DATA_URL= None, num_workers: int = 0, batch_size: int = 8, ):
        """.
        Args:
            dl_path: root directory where to download the data
            DATA_URL: url downloads dataset
            num_workers: number of CPU workers
            batch_size: number of sample in a batch
        """
        super().__init__()

        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size
        self.DATA_URL=DATA_URL

    def prepare_data(self):
        """Download images and prepare images datasets."""
        if self.DATA_URL is not None: 
            download_and_extract_archive(url=self.DATA_URL, download_root=self._dl_path, remove_finished=True)
        

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("cats_and_dogs_filtered")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def test_val_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )
        
    @property
    def valid_transform(self):
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), self.normalize_transform])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        """Train/validation loaders."""
        if train:
            dataset = self.create_dataset(self.data_path.joinpath("train"), self.train_transform)
        else:
            dataset = self.create_dataset(self.data_path.joinpath("validation"), self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        log.info("Training data loaded.")
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info("Validation data loaded.")
        return self.__dataloader(train=False)


    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
