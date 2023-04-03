import argparse
import glob
import os

import pytorch_lightning as pl
import timm
import torch
import torchmetrics
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple

class BGDataset(Dataset):
    def __init__(self, data: List, transforms =None) -> None:
        self.data = data
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.tensor(label, dtype=torch.float32)


class BGDataModule(pl.LightningDataModule):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.train_transforms = T.Compose([
            T.Resize((opt.img_size, opt.img_size)),
            T.RandomHorizontalFlip(p=opt.prob),
            T.RandomVerticalFlip(p=opt.prob),
            T.RandomRotation(degrees=(0, 90)),
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.RandomAutocontrast(),
            T.RandomApply([
                T.Grayscale(num_output_channels=3),
                T.ColorJitter(brightness=.5, hue=.3),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                T.RandomEqualize()
            ], p=opt.prob),
            T.ToTensor(),
        ])
        self.eval_transforms = T.Compose([
            T.Resize((opt.img_size, opt.img_size)),
            T.ToTensor(),
        ])
    
    def setup(self) -> None:
        files = glob.glob(f"./{self.opt.data_dir}/*/*")
        data = []
        classes = ["negative", "positive"]
        for filepath in files:
            if filepath.endswith(".jpg"):
                label = classes.index(filepath.split("/")[3])
                data.append((filepath, label))

        train_data, eval_data = train_test_split(data, test_size=0.3, shuffle=True)
        eval_data, test_data = train_test_split(eval_data, test_size=0.5, shuffle=True)

        self.train_dataset = BGDataset(train_data, self.train_transforms)
        self.eval_dataset = BGDataset(eval_data, self.eval_transforms)
        self.test_dataset = BGDataset(test_data, self.eval_transforms)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )
    
    


class LitBGModel(pl.LightningModule):
    def __init__(self, batch_size: int = 64, lr: float = 3e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.lr = lr
        self.model = timm.create_model("mobilenetv3_small_050", pretrained=False, num_classes=1)
        self.criterion = nn.BCEWithLogitsLoss()

        self.test_preds, self.test_labels = [], []
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)

    def training_step(self, batch: List[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        imgs, labels = batch
        logits = self(imgs).squeeze()
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        acc = torchmetrics.functional.accuracy(preds, labels, task="binary", average="weighted")
        auroc = torchmetrics.functional.auroc(preds, labels.to(torch.long), task="binary", average="weighted")
        f1 = torchmetrics.functional.f1_score(preds, labels, task="binary", average="weighted")
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        self.log("train_auroc", auroc, prog_bar=True, on_epoch=True)
        self.log("train_f1", f1, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        imgs, labels = batch
        logits = self(imgs).squeeze()
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        acc = torchmetrics.functional.accuracy(preds, labels, task="binary", average="weighted")
        auroc = torchmetrics.functional.auroc(preds, labels.to(torch.long), task="binary", average="weighted")
        f1 = torchmetrics.functional.f1_score(preds, labels, task="binary", average="weighted")
        self.log("eval_acc", acc, prog_bar=True, on_epoch=True)
        self.log("eval_auroc", auroc, prog_bar=True, on_epoch=True)
        self.log("eval_f1", f1, prog_bar=True, on_epoch=True)
        self.log("eval_loss", loss, prog_bar=True, on_epoch=True)

    def test_step(self, batch: List[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        imgs, labels = batch
        logits = self(imgs).squeeze()
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)

        self.test_preds.extend(preds.cpu().tolist())
        self.test_labels.extend(labels.cpu().tolist())

        acc = torchmetrics.functional.accuracy(preds, labels, task="binary", average="weighted")
        auroc = torchmetrics.functional.auroc(preds, labels.to(torch.long), task="binary", average="weighted")
        f1 = torchmetrics.functional.f1_score(preds, labels, task="binary", average="weighted")
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        self.log("test_auroc", auroc, prog_bar=True, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)
        self.log("test_loss", loss, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    

def main(opt: argparse.Namespace) -> None:
    datamodule = BGDataModule(opt)
    model = LitBGModel(batch_size=opt.batch_size, lr=opt.lr)
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1, 
        callbacks=[pl.callbacks.EarlyStopping(monitor="eval_f1", mode="max", patience=3, verbose=1)],
        max_epochs=opt.epochs,
        precision=opt.precision,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    
    #### Optimize ####
    scripted = model.to_torchscript()
    optimized = torch.utils.mobile_optimizer.optimize_for_mobile(scripted)
    optimized.save(
        "./checkpoints/bg_classifier_optmizied_scripted.pt"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--precision", type=int, default=16)
    opt = parser.parse_args()
    main(opt)
