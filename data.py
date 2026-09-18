import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = transforms.ToTensor()

        self.train_ds = datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transform,
        )

        self.val_ds = datasets.MNIST(
            "data",
            train=False,
            download=True,
            transform=transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
