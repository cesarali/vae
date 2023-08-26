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
from vae.models.encoder_config import BaseBinaryClassifierConfig

from vae.data.dataloader_utils import load_dataloader
from vae.models.models_utils import load_binary_classifier

from vae.trainers.vae_trainer import VAETrainer
from vae.trainers.trainers_utils import load_experiments_configuration
from vae.trainers.trainers_utils import load_experiments_results
from vae.models.vae import VAE

class TestMITrainer(unittest.TestCase):

    read_config = VAEConfig

    def setUp(self):
        self.config = VAEConfig(experiment_name='mi',
                                experiment_type='multivariate_gaussian',
                                experiment_indentifier="mi_unittest",
                                delete=True)
        self.config.dataloader = ContrastiveMultivariateGaussianLoaderConfig(sample_size=1000,
                                                                             batch_size=32,
                                                                             data_set="example_big",
                                                                             delete_data=False)
        self.config.encoder = BaseBinaryClassifierConfig(hidden_size=40)
        self.config.trainer = VAETrainerConfig(number_of_epochs=5,
                                               save_model_epochs=2)
        self.MI = VAE()


    def test_trainer_setup(self):
        self.MI.create_new_from_config(self.config)

#    @unittest.skip
    def test_load(self):
        MIE = VAE()
        MIE.load_results_from_directory(experiment_name='mi',
                                        experiment_type='multivariate_gaussian',
                                        experiment_indentifier="mi_trainer_big",
                                        checkpoint=None)
        databath = next(MIE.dataloader.train().__iter__())
        x_join = databath["join"]
        x_independent = databath["independent"]
        p_join = MIE.encoder(x_join)
        p_independent = MIE.encoder(x_independent)

        print("p join: {0} p independent: {1}".format(p_join.mean(),p_independent.mean()))

        estimate = MIE.MI_Estimate()
        real_value = MIE.dataloader.mutual_information()

        print("estimate: {0} real: {1}".format(estimate.item(),real_value.item()))


if __name__=="__main__":
    unittest.main()

