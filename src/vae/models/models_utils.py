from typing import Union
from vae.configs.vae_config import VAEConfig
from vae.models.encoder import Encoder
from vae.models.decoder import Decoder

def load_encoder(config:VAEConfig):
    if config.encoder.name == "Encoder":
        encoder = Encoder(config)
    else:
        raise Exception("No Classifier")
    return encoder

def load_decoder(config:VAEConfig):
    if config.encoder.name == "Encoder":
        decoder = Decoder(config)
    else:
        raise Exception("No Classifier")
    return decoder