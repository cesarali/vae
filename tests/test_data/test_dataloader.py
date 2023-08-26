import torch
import unittest
from pprint import pprint
from dataclasses import asdict
from vae.data.dataloader_utils import load_dataloader
from vae.data.dataloaders_config import NISTLoaderConfig
from vae.configs.vae_config import get_config_from_file
from vae.configs.vae_config import VAEConfig

class TestMIDataloader(unittest.TestCase):


    def setUp(self):
        self.batch_size = 23
        self.config = VAEConfig(experiment_name='vae',
                                experiment_type='mnist',
                                experiment_indentifier="vae_unittest",
                                delete=True)

        self.config.dataloader = NISTLoaderConfig(batch_size=128)

    def test_dataloader(self):
        dataloader = load_dataloader(self.config)
        databatch = next(dataloader.train().__iter__())
        data,_ = databatch
        print(data.shape)

if __name__=="__main__":
    unittest.main()
