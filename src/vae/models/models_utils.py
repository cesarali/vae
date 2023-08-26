from typing import Union
from vae.configs.vae_config import VAEConfig
from vae.models.encoder import BaseBinaryClassifier
from vae.models.encoder_config import BaseBinaryClassifierConfig

def load_binary_classifier(config:Union[BaseBinaryClassifierConfig,VAEConfig]):
    if isinstance(config, VAEConfig):
        config_ = config.encoder
    elif isinstance(config, BaseBinaryClassifierConfig):
        config_ = config
    else:
        raise Exception("No Classifier Config Found")

    if config_.name == "BaseBinaryClassifier":
        binary_classifier = BaseBinaryClassifier(config_)
    else:
        raise Exception("No Classifier")
    return binary_classifier