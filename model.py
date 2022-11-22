"""
author: Frauke Albrecht
"""

import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models

SUPPORTED_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnext50_32x4d",
    "resnext50_32x4d",
    "wide_resnet50_2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
    "alexnet",
    "squeezenet1_0",
]


class TreeClassification(pl.LightningModule):
    """
    model for classification
    """

    # pylint: disable=too-many-ancestors
    def __init__(self, args):
        super().__init__()
        assert (
            args.backbone in SUPPORTED_MODELS
        ), f"backbone model must be a supported model {SUPPORTED_MODELS}"
        self.args = args
        if "vgg" in args.backbone:
            self.model = models.__dict__[args.backbone](pretrained=args.pretrained)
            # adapt classifier
            self.model.classifier = nn.Sequential(
                nn.Linear(25088, 4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(4096, 2048, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(2048, args.nr_classes),
            )

        elif "resnet" in args.backbone or "resnext" in args.backbone:
            self.model = models.__dict__[args.backbone](pretrained=args.pretrained)
            # replace output features with nr of classes
            self.model.fc.out_features = args.nr_classes

        elif "mobilenet_v3" in args.backbone:
            self.model = models.__dict__[args.backbone](pretrained=args.pretrained)
            # adapt classifier
            in_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, args.nr_classes),
            )
        elif "alexnet" in args.backbone:
            self.model = models.__dict__[args.backbone](pretrained=args.pretrained)
            # adapt classifier
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(9216, 2048, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(2048, args.nr_classes),
            )
        # elif "squeezenet" in backbone:
        #    self.model = models.__dict__[backbone](pretrained=pretrained)
        #    self.model.features[0] = nn.Conv2d(input_dim, 96, kernel_size=(7, 7), stride=(2, 2))
        #    self.output_dim = self.model.classifier[1].out_channels

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
