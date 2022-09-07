"""
author: Frauke Albrecht
"""

import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models


class TreeClassification(pl.LightningModule):
    """
    model for classification
    """

    # pylint: disable=too-many-ancestors
    def __init__(self, args):
        super().__init__()
        self.args = args
        if "vgg" in args.backbone:
            self.model = models.__dict__[args.backbone](weights=args.weights)
            self.model.classifier = nn.Sequential(
                nn.Linear(25088, 4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(4096, 2048, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(2048, args.nr_classes),
            )
        self.modelname = args.backbone.replace("_", "-")
        print(f"training {self.modelname}")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def mse_loss(self, y_hat, y):
        # pylint: disable=missing-function-docstring
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        return loss

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        y_hat = torch.argmax(y_hat, dim=1)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        y_hat = torch.argmax(y_hat, dim=1)
