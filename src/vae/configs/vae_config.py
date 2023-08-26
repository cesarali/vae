import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from vae.configs.files_config import ExperimentFiles
from vae.models.encoder_config import BaseBinaryClassifierConfig
from vae.trainers.vae_trainer_config import VAETrainerConfig
from vae.data.dataloaders_config import NISTLoaderConfig

all_dataloaders_configs = {"NISTLoader":NISTLoaderConfig}
all_encoders_configs = {"BaseBinaryClassifier":BaseBinaryClassifierConfig}
all_decoders_configs = {"BaseBinaryClassifier":BaseBinaryClassifierConfig}
all_trainers_configs = {"MutualInformationTrainer":VAETrainerConfig}


@dataclass
class VAE_ExperimentsFiles(ExperimentFiles):
    best_model_path_checkpoint:str = None
    best_model_path:str = None
    plot_path:str = None

    def __post_init__(self):
        super().__post_init__()
        self.best_model_path_checkpoint = os.path.join(self.results_dir, "model_checkpoint_{0}.tr")
        self.best_model_path = os.path.join(self.results_dir, "best_model.tr")
        self.plot_path = os.path.join(self.results_dir, "plot.png")

@dataclass
class VAEConfig:

    config_path : str = ""

    # files, directories and naming ---------------------------------------------
    delete :bool = True
    experiment_name :str = 'mi'
    experiment_type :str = 'multivariate_gaussian'
    experiment_indentifier :str  = None
    init_model_path = None

    # all configs ---------------------------------------------------------------
    encoder: BaseBinaryClassifierConfig = BaseBinaryClassifierConfig()
    decoder: BaseBinaryClassifierConfig = BaseBinaryClassifierConfig()

    dataloader: NISTLoaderConfig = NISTLoaderConfig()
    trainer: VAETrainerConfig = VAETrainerConfig()
    experiment_files:VAE_ExperimentsFiles = None

    def __post_init__(self):
        self.experiment_files = VAE_ExperimentsFiles(delete=self.delete,
                                                     experiment_name=self.experiment_name,
                                                     experiment_indentifier=self.experiment_indentifier,
                                                     experiment_type=self.experiment_type)

        if isinstance(self.encoder, dict):
            self.encoder = all_encoders_configs[self.encoder["name"]](**self.encoder)
        if isinstance(self.decoder, dict):
            self.decoder = all_encoders_configs[self.decoder["name"]](**self.decoder)
        if isinstance(self.dataloader,dict):
            self.dataloader = all_dataloaders_configs[self.dataloader["name"]](**self.dataloader)
        if isinstance(self.trainer,dict):
            self.trainer = all_trainers_configs[self.trainer["name"]](**self.trainer)

    def initialize_new_experiment(self,
                                  experiment_name: str = None,
                                  experiment_type: str = None,
                                  experiment_indentifier: str = None):
        if experiment_name is not None:
            self.experiment_name = experiment_name
        if experiment_type is not None:
            self.experiment_type = experiment_type
        if experiment_indentifier is not None:
            self.experiment_indentifier = experiment_indentifier

        self.align_configurations()
        self.experiment_files.create_directories()
        self.config_path = self.experiment_files.config_path
        self.save_config()

    def align_configurations(self):
        pass
        #self.encoder.input_size = self.dataloader.dimensions_per_variable * self.dataloader.number_of_variables

    def save_config(self):
        config_as_dict = asdict(self)
        with open(self.experiment_files.config_path, "w") as file:
            json.dump(config_as_dict, file)

def get_config_from_file(experiment_name, experiment_type, experiment_indentifier) -> VAEConfig:
    from vae import results_path

    experiment_dir = os.path.join(results_path, experiment_name)
    experiment_type_dir = os.path.join(experiment_dir, experiment_type)
    results_dir = os.path.join(experiment_type_dir, experiment_indentifier)
    results_dir_path = Path(results_dir)
    if results_dir_path.exists():
        config_path = os.path.join(results_dir, "config.json")
        config_path_json = json.load(open(config_path, "r"))
        config_path_json["delete"] = False
        config = VAEConfig(**config_path_json)
        return config
    else:
        raise Exception("Folder Does Not Exist")
