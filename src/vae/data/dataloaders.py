import os
import torch
from vae.data.dataloaders_config import NISTLoaderConfig

from pathlib import Path
from torchvision import datasets, transforms

def get_data(config:NISTLoaderConfig):
    # Load MNIST dataset
    if config.data_set == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(config.dataloader_data_dir, train=True, download=True, transform=transform),
            batch_size=config.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(config.dataloader_data_dir, train=True, download=True, transform=transform),
            batch_size=config.batch_size, shuffle=True)
    else:
        raise Exception("Data Loader Not Found!")

    return train_loader,test_loader

class NISTLoader:

    name_ = "NISTLoader"

    def __init__(self, config:NISTLoaderConfig):
        self.config = config

        self.batch_size = config.batch_size
        self.delete_data = config.delete_data

        self.dataloader_data_dir = config.dataloader_data_dir
        self.dataloader_data_dir_path = Path(self.dataloader_data_dir)
        self.dataloader_data_dir_file_path = Path(config.dataloader_data_dir_file)

        self.train_loader,self.test_loader = get_data(self.config)

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader


