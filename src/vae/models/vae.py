import os
import sys
import torch
from dataclasses import dataclass, asdict

# load models
from vae.data.dataloader_utils import load_dataloader
from vae.models.models_utils import load_binary_classifier
from vae.trainers.trainers_utils import load_experiments_configuration

# configs
from vae.configs.vae_config import VAEConfig

# models
from vae.models.encoder import BaseBinaryClassifier
from vae.data.dataloaders import ContrastiveMultivariateGaussianLoader

EPSILON = 1e-12

@dataclass
class VAE:
    """

    """
    encoder: BaseBinaryClassifier = None
    dataloader: ContrastiveMultivariateGaussianLoader = None

    def create_new_from_config(self, config:VAEConfig, device=torch.device("cpu")):
        self.config = config
        self.config.initialize_new_experiment()

        self.dataloader = load_dataloader(self.config)
        self.encoder = load_binary_classifier(self.config)
        self.encoder.to(device)

    def load_results_from_directory(self,
                                    experiment_name='mi',
                                    experiment_type='multivariate_gaussian',
                                    experiment_indentifier="test",
                                    checkpoint=None,
                                    device=torch.device("cpu")):
        self.encoder, self.dataloader = load_experiments_configuration(experiment_name,
                                                                       experiment_type,
                                                                       experiment_indentifier,
                                                                       checkpoint)
        self.encoder.to(device)

    def MI_Estimate(self):
        if self.config.trainer.loss_type == "contrastive":
            log_q = 0.
            number_of_pairs = 0
            for databath in self.dataloader.train():
                #select data
                x_join = databath["join"]

                #calculate probability
                q = self.encoder(x_join)
                assert torch.isnan(q).any() == False
                assert torch.isinf(q).any() == False

                #average
                log_q = torch.log(q)
                where_inf = torch.isinf(log_q)
                log_q[where_inf] = 0.

                log_q = log_q.sum()
                where_inf = ~where_inf
                number_of_pairs += where_inf.int().sum()

            log_q_av = log_q/number_of_pairs

            return log_q,log_q_av
        elif self.config.trainer.loss_type == "mine":
            return None
        else:
            raise Exception("Not implemented yet")

