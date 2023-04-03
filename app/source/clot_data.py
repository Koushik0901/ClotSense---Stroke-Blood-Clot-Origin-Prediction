import glob
import os
import random
from typing import List, Tuple

import albumentations as A
import cv2
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ClotDataset(Dataset):
    def __init__(self, data: List[Tuple], transforms=None) -> None:
        self.data = data
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.data[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)


def get_dataset(opt):
    train_transforms = A.Compose(
        [
            A.Resize(
                height=opt.img_size,
                width=opt.img_size,
            ),
            A.HorizontalFlip(p=opt.prob),
            A.VerticalFlip(p=opt.prob),
            A.Sharpen(p=opt.prob),
            A.ColorJitter(brightness=0.2, hue=0.5, saturation=0.5, p=opt.prob),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    eval_transforms = A.Compose(
        [
            A.Resize(
                height=opt.img_size,
                width=opt.img_size,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    LAA_images = [(filepath, 0) for filepath in glob.glob(f"{opt.data_dir}/LAA/*.png")]
    CE_images = [(filepath, 1) for filepath in glob.glob(f"{opt.data_dir}/CE/*.png")]

    random.shuffle(CE_images)
    CE_images = CE_images[: len(LAA_images)]

    data = LAA_images + CE_images
    train_data, eval_data = train_test_split(data, test_size=0.3, shuffle=True)
    eval_data, test_data = train_test_split(eval_data, test_size=0.5, shuffle=True)

    train_dataset = ClotDataset(train_data, transforms=train_transforms)
    eval_dataset = ClotDataset(eval_data, transforms=eval_transforms)
    test_dataset = ClotDataset(test_data, transforms=eval_transforms)
    return train_dataset, eval_dataset, test_dataset


class ClotDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self):
        self.train_dataset, self.eval_dataset, self.test_dataset = get_dataset(self.opt)

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
