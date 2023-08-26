import torch
import unittest
from pprint import pprint
from dataclasses import asdict

import torch
import unittest
from pprint import pprint
from dataclasses import asdict
from vae.configs.vae_config import get_config_from_file
from vae.configs.vae_config import VAEConfig
from vae.data.dataloaders import ContrastiveMultivariateGaussianLoaderConfig
from vae.trainers.vae_trainer_config import VAETrainerConfig
from vae.models.encoder_config import EncoderConfig

from vae.data.dataloader_utils import load_dataloader
from vae.models.models_utils import load_binary_classifier

from vae.trainers.vae_trainer import VAETrainer
from vae.trainers.trainers_utils import load_experiments_configuration
from vae.trainers.trainers_utils import load_experiments_results
from vae.models.vae import VAE

class TestMITrainer(unittest.TestCase):

    read_config = VAEConfig

    def setUp(self):
        self.experiment_indentifier = "mi_contrastive2"
        self.config = VAEConfig(experiment_name='mi',
                                experiment_type='multivariate_gaussian',
                                experiment_indentifier=self.experiment_indentifier,
                                delete=True)
        self.config.dataloader = ContrastiveMultivariateGaussianLoaderConfig(sample_size=1000,
                                                                             batch_size=32,
                                                                             data_set="example_big",
                                                                             delete_data=False)
        self.config.encoder = EncoderConfig(hidden_size=100)
        self.config.trainer = VAETrainerConfig(number_of_epochs=1000,
                                               save_model_epochs=250,
                                               loss_type="contrastive")

        self.config.initialize_new_experiment()
        self.contrastive_dataloader = load_dataloader(self.config)
        self.binary_classifier = load_binary_classifier(self.config)


    def test_trainer_setup(self):
        MIT = VAETrainer(self.config,
                         self.contrastive_dataloader,
                         self.binary_classifier)
        MIT.train()

        binary_classifier, dataloader = load_experiments_configuration(experiment_name='mi',
                                                                       experiment_type='multivariate_gaussian',
                                                                       experiment_indentifier=self.experiment_indentifier)

        databatch = next(dataloader.train().__iter__())
        x_join = databatch["join"]
        forward_classifier = binary_classifier(x_join)
        print(forward_classifier.shape)


if __name__=="__main__":
    unittest.main()
