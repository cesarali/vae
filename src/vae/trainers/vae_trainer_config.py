from dataclasses import dataclass

@dataclass
class VAETrainerConfig:
    name:str = "VAETrainer"
    learning_rate: float = 1e-3
    number_of_epochs: int = 1000
    save_model_epochs:int = 100
    loss_type:str = "vae_loss" #contrastive,mine
    experiment_class: str = "mnist"
    device:str = "cuda:0"
