"""
author: Frauke Albrecht
"""
import argparse
import os

import mlflow.pytorch
import pytorch_lightning as pl

# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import TreeClassification


class TreeDataModule(pl.LightningDataModule):
    """
    create data module
    """

    def __init__(self, args):
        super().__init__()
        datadir_train = os.path.join(args.data, "train_set")
        datadir_val = os.path.join(args.data, "val_set")
        print(datadir_train)
        print(datadir_val)

        self.args = args
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.train_dataset = datasets.ImageFolder(
            datadir_train, transform=self.train_transforms
        )
        self.val_dataset = datasets.ImageFolder(
            datadir_val, transform=self.val_transforms
        )

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        print(f"train_dataloader: {next(iter(train_dataloader))[0].shape}")
        print(f"train_dataloader: {next(iter(train_dataloader))[1].shape}")
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size)
        print(f"val_dataloader: {next(iter(val_dataloader))[0].shape}")
        print(f"val_dataloader: {next(iter(val_dataloader))[1].shape}")
        return val_dataloader


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--nr-classes", type=int, default=2)
    parser.add_argument("--backbone", type=str, default="vgg16")
    parser.add_argument("--weights", type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    params = vars(args)
    print("argparse arguments:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print()

    model = TreeClassification(args)
    data_module = TreeDataModule(args)
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    trainer = pl.Trainer.from_argparse_args(args, max_epochs=1)

    mlflow.set_experiment("test")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.pytorch.autolog()
        trainer.fit(model, train_loader, val_loader)
