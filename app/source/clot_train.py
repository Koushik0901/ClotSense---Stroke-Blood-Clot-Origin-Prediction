import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torchmetrics
import wandb
from clot_data import ClotDataModule
from pytorch_lightning.loggers import WandbLogger
from torch import nn


def seed_everything(seed: int = 0) -> None:
    pl.seed_everything(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LitClotModel(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 3e-4,
        model_name: str = "swinv2_tiny_window16_256",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.lr = lr
        self.model = timm.create_model(model_name, pretrained=True, num_classes=2)
        self.criterion = nn.CrossEntropyLoss()

        self.test_preds, self.test_labels = [], []

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        imgs, labels = batch
        logits = self(imgs).squeeze()
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        acc = torchmetrics.functional.accuracy(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        precision = torchmetrics.functional.precision(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        recall = torchmetrics.functional.recall(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        f1 = torchmetrics.functional.f1_score(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        self.log_dict(
            {
                "train_f1": f1,
                "train_acc": acc,
                "train_precision": precision,
                "train_recall": recall,
                "train_loss": loss,
            },
            on_epoch=True,
            prog_bar=True,
            on_step=True,
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        imgs, labels = batch
        logits = self(imgs)

        loss = self.criterion(logits, labels)
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        acc = torchmetrics.functional.accuracy(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        precision = torchmetrics.functional.precision(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        recall = torchmetrics.functional.recall(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        f1 = torchmetrics.functional.f1_score(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        self.log_dict(
            {
                "eval_f1": f1,
                "eval_acc": acc,
                "eval_precision": precision,
                "eval_recall": recall,
                "eval_loss": loss,
            },
            on_epoch=True,
            prog_bar=True,
            on_step=True,
        )
        return f1

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        imgs, labels = batch
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)

        self.test_preds.extend(preds.cpu().tolist())
        self.test_labels.extend(labels.cpu().tolist())

        acc = torchmetrics.functional.accuracy(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        precision = torchmetrics.functional.precision(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        recall = torchmetrics.functional.recall(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )
        f1 = torchmetrics.functional.f1_score(
            preds, labels, task="multiclass", average="weighted", num_classes=2
        )

        self.log_dict(
            {
                "test_f1": f1,
                "test_acc": acc,
                "test_precision": precision,
                "test_recall": recall,
                "test_loss": loss,
            },
            on_epoch=True,
        )

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optimizer]


def main(opt: argparse.Namespace) -> None:
    seed_everything(opt.seed)
    wandb_logger = WandbLogger(
        project="ClotSense", name=opt.model_name, log_model="all"
    )
    clotdatamodule = ClotDataModule(opt)
    model = LitClotModel(
        batch_size=opt.batch_size, lr=opt.lr, model_name=opt.model_name
    )

    #### Train ####
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=opt.epochs,
        logger=wandb_logger,
        precision=opt.precision,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="eval_f1", mode="max", patience=3),
            pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
            pl.callbacks.BatchSizeFinder(
                mode="power",
                steps_per_trial=3,
                init_val=2,
                max_trials=25,
                batch_arg_name="batch_size",
            ),
        ],
        deterministic=True,
        gradient_clip_val=1.0,
    )
    trainer.tune(model, datamodule=clotdatamodule)
    trainer.fit(model, datamodule=clotdatamodule)
    trainer.test(model, clotdatamodule.test_dataloader())
    wandb.finish()

    #### Optimize ####
    scripted = model.to_torchscript()
    optimized = torch.utils.mobile_optimizer.optimize_for_mobile(scripted)
    optimized._save_for_lite_interpreter(
        "./checkpoints/optmizied_scripted_lite.ptl"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="swinv2_tiny_window16_256")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="../dataset")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--precision", type=int, default=16)

    opt = parser.parse_args()
    main(opt)
