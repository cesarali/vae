import torch
import unittest

# configs
from vae.configs.vae_config import VAEConfig
from vae.models.encoder_config import EncoderConfig

#load
from vae.data.dataloader_utils import load_dataloader
from vae.models.models_utils import load_encoder

class TestEncoder(unittest.TestCase):

    def test_encoder(self):
        config = VAEConfig()
        dataloader = load_dataloader(config)
        databatch = next(dataloader.train().__iter__())
        data,_ = databatch

        encoder = load_encoder(config)
        self.assertIsNotNone(encoder)



if __name__=="__main__":
    unittest.main()