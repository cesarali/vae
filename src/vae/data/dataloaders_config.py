import os
from dataclasses import dataclass

@dataclass
class NISTLoaderConfig:
    name:str = "NISTLoader"
    data_set:str = "mnist"
    dataloader_data_dir:str = None

    input_dim: int = 784
    batch_size: int = 32
    delete_data:bool = False

    def __post_init__(self):
        from vae import data_path
        self.dataloader_data_dir = os.path.join(data_path,"raw",self.name)
        self.dataloader_data_dir_file = os.path.join(self.dataloader_data_dir,self.data_set+".tr")