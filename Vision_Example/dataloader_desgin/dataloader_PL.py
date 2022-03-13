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


class ssl_dataloader(LightningDataModule):

    """Multi-Function SSL Dataloader.
        Args:
            data_path: root directory where to download the data
            num_workers: number of CPU or GPUs workers
            batch_size: number of sample in a batch
    """

    def __init__(self, train_transforms, val_transforms, test_transforms,
                 data_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8,
                 ):
        super().__init__()
        self.data_dir = data_path
        self.num_worker = num_workers
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def prepare_data(self):
        # Data not avalilable
        pass

    def set_up(self, stage: Optional[str] = None):
        '''
        setup is called from every Device (CPUs or GPUs). 
        args: stage (specify some runing stage --> If call this 
            train or val or test loader will be process)
        Default None (All will be process during trainer.fit)

        '''

        if stage in (None, "fit"):
            # Taking Prepared data and Transform corresponding
            # Under Develop
            self.train_ds = transforms = self.train_transforms
            # Under Develop
            self.val_ds = transforms = self.val_transforms

        if stage in (None, "test"):
            # Under Develop
            self.test_ds = transform = self.test_transform

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
