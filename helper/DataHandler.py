import torch
from helper.Config import Config
from helper.dataloader import  DataLoader,Rotate,FlipV,TranslateX,TranslateY
from torchvision.transforms import RandomApply,RandomChoice,Compose


class DataHandler:
    def __init__(self, config: Config):
        self.config = config
        self._set_transfrom()
        self._set_shared_config()
        self.train_dataset = self._load_dataset(idx_path=config.parent_folder+config.idx_train,
                                                transform=self.transform,
                                                min_max_values=None,
                                                **self.shared_config)
        self.train_dataset_for_test = self._load_dataset(idx_path=config.parent_folder+config.idx_train,
                                                        transform=None,
                                                        min_max_values=None,
                                                        **self.shared_config)
        self.test_dataset = self._load_dataset(idx_path=config.parent_folder+config.idx_test,
                                                transform=None,
                                                min_max_values=self.train_dataset.min_max_values,
                                                **self.shared_config)

        self.train_loader = self._create_data_loader(dataset=self.train_dataset, 
                                                    batch_size=config.batch_size, 
                                                    shuffle=True,
                                                    drop_last=True)
        self.train_loader_for_test= self._create_data_loader(dataset=self.train_dataset_for_test, 
                                                            batch_size=config.batch_size, 
                                                            shuffle=False,
                                                            drop_last=False)
        self.test_loader = self._create_data_loader(dataset=self.test_dataset, 
                                                   batch_size=config.batch_size, 
                                                   shuffle=False,
                                                   sampler=torch.utils.data.SequentialSampler(self.test_dataset),
                                                   drop_last=False)

    def _load_dataset(self, **kwords):
        # Load and possibly preprocess the dataset
        dataset = DataLoader(**kwords)  # Replace with actual dataset loading logic
        return dataset

    def _create_data_loader(self,**kwords):
        return torch.utils.data.DataLoader(**kwords)
    
    def _set_transfrom(self):
        if self.config.augmentaion:
            augmentation_factory={ "r": lambda: RandomApply([Rotate()],p=self.config.prop),
                                "f":lambda: RandomApply([FlipV()],p=self.config.prop),
                                "r+f": lambda: RandomApply([RandomChoice([Rotate(),FlipV()])],p=self.config.prop), 
                                "all": lambda:RandomApply([RandomChoice([Rotate(),FlipV(),TranslateY(),TranslateX()])],p=self.config.prop)
                                }
            if self.config.Aug_type in augmentation_factory:
                self.transform=augmentation_factory[self.config.Aug_type]
            else:
                raise(ValueError("No such augmentation type is defind. Available Augmentation : [r, f, r+f, all]"))
        else:
            self.transform=None
    def _set_shared_config(self):
        self.shared_config={"data_path":self.config.parent_folder+self.config.data_path,
                "labels_path":self.config.parent_folder+self.config.labels_path,
                "edges_index_path":self.config.parent_folder+self.config.edges_path,
                "model_name":self.config.model_name,
                "num_features": self.config.num_features,
                "num_nodes":self.config.n_joints,
                "num_classes":self.config.num_classes,
                "contantenat":self.config.concatenate,
                "maxMinNormalization":self.config.maxMinNormalization,
                "normalize_labels":self.config.normalize_labels
                }