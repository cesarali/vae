import os
import sys
from dataclasses import dataclass, asdict

# torch stuff
import torch
from torch import nn

# load models
from vae.data.dataloader_utils import load_dataloader
from vae.models.models_utils import load_encoder,load_decoder
from vae.trainers.trainers_utils import load_experiments_configuration

# configs
from vae.configs.vae_config import VAEConfig

# models
from vae.models.decoder import Decoder
from vae.models.encoder import Encoder
from vae.data.dataloaders import NISTLoader

EPSILON = 1e-12

class VAE(nn.Module):
    """

    """
    encoder: Encoder = None
    decoder: Decoder = None
    dataloader: NISTLoader = None

    def __init__(self,
                 config:VAEConfig=None,
                 experiment_name='vae',
                 experiment_type='mnist',
                 experiment_indentifier="test",
                 checkpoint=None,
                 device=torch.device("cpu")):

        self.config = config
        if self.config is not None:
            self.create_new_from_config()
        else:
            self.load_results_from_directory(experiment_name='vae',
                                             experiment_type='mnist',
                                             experiment_indentifier="test",
                                             checkpoint=None)

    def forward(self,image):
        z, mu, logvar = self.encoder(image)
        return self.decoder(z),mu,logvar

    def create_new_from_config(self, config:VAEConfig, device=torch.device("cpu")):
        self.config = config
        self.config.initialize_new_experiment()

        self.dataloader = load_dataloader(self.config)

        self.encoder = load_encoder(self.config)
        self.encoder.to(device)

        self.encoder = load_decoder(self.config)
        self.encoder.to(device)

    def load_results_from_directory(self,
                                    experiment_name='vae',
                                    experiment_type='mnist',
                                    experiment_indentifier="test",
                                    checkpoint=None,
                                    device=torch.device("cpu")):

        self.encoder, self.dataloader = load_experiments_configuration(experiment_name,
                                                                       experiment_type,
                                                                       experiment_indentifier,
                                                                       checkpoint)
        self.encoder.to(device)

    def generate(self, number_of_samples=64):
        # Generating samples from the trained VAE
        self.eval()
        with torch.no_grad():
            z_sample = torch.randn(number_of_samples, 20)
            sample = self.decoder(z_sample).cpu()
        return sample

