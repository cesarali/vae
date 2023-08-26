import torch
from vae.configs.vae_config import VAEConfig
from vae.configs.vae_config import get_config_from_file

def load_experiments_results(experiment_name, experiment_type, experiment_indentifier, checkpoint:int = None):
    config: VAEConfig
    config = get_config_from_file(experiment_name, experiment_type, experiment_indentifier)
    if checkpoint is None:
        results = torch.load(config.experiment_files.best_model_path)
    else:
        results = torch.load(config.experiment_files.best_model_path_checkpoint.format(checkpoint))
    return config, results

def load_experiments_configuration(experiment_name, experiment_type, experiment_indentifier, checkpoint:int = None):
    from vae.data.dataloader_utils import load_dataloader

    config: VAEConfig
    config, results = load_experiments_results(experiment_name, experiment_type, experiment_indentifier, checkpoint)
    binary_classifier = results["binary_classifier"].to(torch.device("cpu"))
    dataloader = load_dataloader(config)

    return binary_classifier,dataloader