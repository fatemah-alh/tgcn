
import numpy as np
from tqdm import tqdm
import torch
import yaml

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path,labels_path,edges_index_path,data_shape=[(0, 3, 1, 2),(8700, 6,137,51)],normalize_labels=True,idx_path=None,reshape_data=True,expand_dim=True,model_name="aagcn",augmentation=True):
        super(DataLoader, self).__init__()

        self.data_path = data_path
        self.labels_path=labels_path
        self.edges_index_path=edges_index_path
        self.idx_path=idx_path
        self.normalize_labels=normalize_labels
        self.expand_dim=expand_dim
        self.model_name=model_name
        self.reshape_data=reshape_data
        self.augmentation=augmentation
        if self.model_name=="a3tgcn":
            self.data_shape=[(0, 2, 3, 1),(8700,51,4,137)]
            self.expand_dim=False
        elif self.model_name=="aagcn":
            self.data_shape=[(0, 3, 1, 2),(8700, 4,137,51)]
        else:
            self.data_shape=data_shape
        self._read_data()
        
        #self.get_shapes()
        print(self.features.shape)
        
    def _read_data(self):
        print("Loading Dataset")#(8700,137,51,6 )
        self.X=np.load(self.data_path) #
        
        #self.X=self.X[:,:,:,:2]
        self.X=np.concatenate( (self.X[:,:,:,:2],self.X[:,:,:,3:5]),axis=3)
        print(self.X.shape)
        self._reshape_data()
        self.labels=np.load(self.labels_path)#0,..,4
        #Normalize label between 0,1.
        if self.normalize_labels :
            self.labels=self.labels/np.max(self.labels)
        #split data set with idx
        if self.idx_path!=None:
            self.split_data()
        print("Data is loaded!")
       
    def _reshape_data(self):
        if self.reshape_data:
             
             #reshape (8700,137,51,6 ) to (8700, 6,137,51)aagcn (0, 3, 1, 2)
             #reshape (8700,137,51,6 ) to(8700,51,6,137) a3gcn(0, 2, 3, 1)
             reshaped_tensor = np.transpose(self.X, self.data_shape[0])  # Transpose dimensions from 8700,137,469,4) to 8600, 469, 4, 137
             self.features= torch.tensor(np.reshape(reshaped_tensor, self.data_shape[1]))  # Reshape to desired shape 
             
        else:
            self.features=self.X
        if self.expand_dim:
            self.features=np.expand_dims(self.features,axis=-1)#unsqueeze, view
        
    def __len__(self):
        return self.features.shape[0]
    def get_shapes(self):
        print("featuers: ",self.features.shape, "labels:", self.labels.shape)
        print("assert featuers:",self.features[0][0])
        print("assert X:",self.X[0][0])
        print("assert label",np.unique(self.labels))
        
        return self.features.shape

    def __getitem__(self, index):

        x = self.features[index]
        if self.augmentation:
            pass
        y=self.labels[index]
        return x, y
    
    def split_data(self):
        idx=np.load(self.idx_path) 
        idx=np.array(idx,dtype=np.int32)
        self.features=self.features[idx]
        self.labels=self.labels[idx]

    def binary_classification(self):
        values = [0, 4]
        indices = np.where(np.isin(self.labels, values))[0]
        self.labels=self.labels[indices]
        self.features=self.features[indices]
        indices_4 = np.where(self.labels == 4)
        self.labels[indices_4] = 1
        print(len(self.labels))
        print("number of samples with 4 class ",len(indices_4))



        
   
if __name__=="__main__":
    name_exp="open_face"
   
    parent_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
    config_file=open(parent_folder+"config/"+name_exp+".yml", 'r')
    config = yaml.safe_load(config_file)
    data_path=parent_folder+config['data_path']
    labels_path=parent_folder+config['labels_path']
    edges_path=parent_folder+config['edges_path']
    idx_train= parent_folder+config['idx_train']
    idx_test=parent_folder+config['idx_test']
    
    loader=DataLoader(data_path,labels_path,edges_path)
    train_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_train)
    test_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_test)
    
    for sample in train_dataset:
        x,y=sample
       #print (x,x.shape,np.max(x))
        break