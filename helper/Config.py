import yaml
from dataclasses import dataclass

@dataclass
class Config:
    """
    To use the class:
    config=Config.load_from_file(path) or Config(dict_with_values)
    """
   
    description: str
    name_exp : str
    parent_folder: str
    video_path: str
    landmarks_path : str
    csv_file: str
    labels_path: str
    data_path: str
    edges_path: str
    filter_idx_90: str
    filter_idx_80: str
    weight_decay: float
    momentum:float
    L2:float
    step_decay: int
    optimizer_name: str
    batch_size: int
    continue_training: bool  
    pretrain_model: str 
    Temporal_downsampling: bool
    model_name: str 
    embed_dim: int
    gru: int
    bn: bool
    lr: float
    num_subset: int
    num_features: int
    hidden_size: int
    TS: int
    n_joints: int
    idx_train:  str
    idx_test: str
    LOG_DIR: str
    t_kernel_size: int
    strid: int
    normalize_labels: bool
    adaptive: bool
    attention: bool
    drop_out: float
    protocol: str 
    maxMinNormalization: bool
    augmentaion: bool
    prop: float
    Aug_type: str
    concatenate: bool
    log_name: str
    gpu: int
    num_classes: int
    num_epoch: int
    project_name: str

    @staticmethod
    def load_from_file(file_path: str):
        with open(file_path, 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        return Config(**config_dict)
