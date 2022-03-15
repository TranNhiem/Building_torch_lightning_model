# This linear and Fine-tune model for Natual Image Classification task
# Author Tran Nhiem 2022/03
# 

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple
from torchmetrics.functional import accuracy
from SSL_Frameworks.methods.base_structure import base_model
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)

## Fine Tuning With Certain Layers
class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:], optimizer=optimizer, train_bn=self.train_bn
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaining layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5], optimizer=optimizer, train_bn=self.train_bn
            )



class LinearModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr_rate: float,
        weight_decay: float,
        exclude_bias_n_norm: bool,
        extra_optimizer_args: dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        lr_decay_steps: Optional[Sequence[int]] = None,
        metric: str = 'accuracy',
        task: str = 'semi_supervised',
        **kwargs,
    ):
        """Implements linear evaluation.

        Args:
            backbone (nn.Module): backbone architecture for feature extraction.
            num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.
            batch_size (int): batch size.
            optimizer (str): optimizer to use.
            lars (bool): whether to use lars or not.
            lr (float): learning rate.
            weight_decay (float): weight decay.
            exclude_bias_n_norm (bool): whether to exclude bias and batch norm from weight decay
                and lars adaptation.
            extra_optimizer_args (dict): extra optimizer arguments.
            scheduler (str): learning rate scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            lr_decay_steps (Optional[Sequence[int]], optional): list of epochs where the learning
                rate will be decreased. Defaults to None.
        """
        super().__init__()
        

       
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.learning_rate = lr_rate
        self.weight_decay = weight_decay
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.lr_decay_steps = lr_decay_steps
        self.metric = metric
        self.task = task
        self.model = None

        # all the other parameters
        self.backbone= backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features
        
        self.classifier = nn.Linear(features_dim, num_classes)  # type: ignore
        if self.task == 'semi_supervised' or self.task == 'finetune':
            for param in self.backbone.parameters():
                param.requires_grad = True
            
            self.model = self.backbone
            self.model.new_module = self.classifier

        else:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        @staticmethod
        def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
            """
            Adds basic linear arguments.

            Args:
                parent_parser (ArgumentParser): argument parser that is used to create a
                    argument group.

            Returns:
                ArgumentParser: same as the argument, used to avoid errors.
            """

            parser = parent_parser.add_argument_group("linear")


            # general train
            parser.add_argument("--batch_size", type=int, default=128)
            parser.add_argument("--lr", type=float, default=0.3)
            parser.add_argument("--classifier_lr", type=float, default=0.3)
            parser.add_argument("--weight_decay", type=float, default=0.0001)
            parser.add_argument("--num_workers", type=int, default=4)

            # wandb
            parser.add_argument("--name")
            parser.add_argument("--project")
            parser.add_argument("--entity", default=None, type=str)
            parser.add_argument("--wandb", action="store_true")
            parser.add_argument("--offline", action="store_true")

            # optimizer
            SUPPORTED_OPTIMIZERS = ["sgd", "adam"]

            parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
            parser.add_argument("--lars", action="store_true")
            parser.add_argument("--exclude_bias_n_norm", action="store_true")

            # scheduler
            SUPPORTED_SCHEDULERS = [
                "reduce",
                "cosine",
                "warmup_cosine",
                "step",
                "exponential",
                "none",
            ]

            parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
            parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
            parser.add_argument("--min_lr", default=0.0, type=float)
            parser.add_argument("--warmup_start_lr", default=0.003, type=float)
            parser.add_argument("--warmup_epochs", default=10, type=int)

            return parent_parser
  


        
        def forward(self, X: torch.tensor) -> Dict[str, Any]:
            """
            Type 1: (Linear Classification) Performs forward pass of the frozen backbone and the linear layer for evaluation.
            Type 2: (Fine-Tune) Performs forward pass and update backbone and the linear layer parameter .

            Args:
                X (torch.tensor): a batch of images in the tensor format.

            Returns:
                Dict[str, Any]: a dict containing features and logits.
            """
            if self.task == 'semi_supervised' or self.task == 'finetune':
                feats = self.backbone(X)
            
            # This  will remove in future version 
            else: 
                with torch.no_grad():
                    feats = self.backbone(X)
                
            logits = self.classifier(feats)
            
            return {"logits": logits, "feats": feats}

        def shared_step(self, batch: Tuple, batch_idx: int
            ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Performs operations that are shared between the training nd validation steps.

            Args:
                batch (Tuple): a batch of images in the tensor format.
                batch_idx (int): the index of the batch.

            Returns:
                Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                    batch size, loss, accuracy @1 and accuracy @5.
            """

            X, target = batch
            batch_size = X.size(0)

            out = self(X)["logits"]

            loss = F.cross_entropy(out, target)

            if self.metric == 'accuracy':
                acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
                return batch_size, loss, acc1, acc5
            elif self.metric == 'mean per class accuracy':
                mean_per_class_accuracy = accuracy(out, target, average='macro', num_classes=self.num_classes)
            return batch_size, loss, mean_per_class_accuracy
        

        
        def configure_optimizers(self) -> Tuple[List, List]:
            """Configures the optimizer for the linear layer.

            Raises:
                ValueError: if the optimizer is not in (sgd, adam).
                ValueError: if the scheduler is not in not in (warmup_cosine, cosine, reduce, step,
                    exponential).

            Returns:
                Tuple[List, List]: two lists containing the optimizer and the scheduler.
            """

            if self.optimizer == "sgd":
                optimizer = torch.optim.SGD
            elif self.optimizer == "adam":
                optimizer = torch.optim.Adam
            else:
                raise ValueError(f"{self.optimizer} not in (sgd, adam)")

            if self.task == 'linear_eval':
                optimizer = optimizer(
                    self.classifier.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_args,
                )
            elif self.task == 'finetune' or self.task == 'semi_supervised':
                optimizer = optimizer(
                    self.model.parameters(),
                    lr= self.learning_rate,
                    weight_decay=self.weight_decay,
                    **self.extra_optimizer_args,
                )

            if self.lars:
                optimizer = LARSWrapper(optimizer, exclude_bias_n_norm=self.exclude_bias_n_norm)

            # select scheduler
            if self.scheduler == "none":
                return optimizer
            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs,
                    max_epochs=self.max_epochs,
                    warmup_start_lr=self.warmup_start_lr,
                    eta_min=self.min_lr,
                )
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.max_epochs)
            elif self.scheduler == "reduce":
                scheduler = ReduceLROnPlateau(optimizer)
            elif self.scheduler == "step":
                scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
            elif self.scheduler == "exponential":
                scheduler = ExponentialLR(optimizer, self.weight_decay)
            else:
                raise ValueError(
                    f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
                )

            return [optimizer], [scheduler]