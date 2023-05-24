import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric_temporal.signal import StaticGraphTemporalSignal,StaticGraphTemporalSignalBatch
import torch_geometric
from torch_geometric.utils import dense_to_sparse,add_self_loops,to_undirected
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import yaml

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path,labels_path,edges_index_path,name_exp,idx_path=None,mode=None):
        super(DataLoader, self).__init__()

        self.data_path = data_path
        self.labels_path=labels_path
        self.edges_index_path=edges_index_path
        self.idx_path=idx_path
        self.mode=mode
        self.name_exp=name_exp
        self._read_data()
        #split data set with idx
        if self.mode!=None:
            if self.idx_path==None:
                raise(ValueError("No idx_path has been found"))
            else:
                self.split_data()
        self.get_shapes()
    def _read_data(self):
        print("Loading Dataset")
        self.X=np.load(self.data_path)
        
        if self.name_exp=="mediapipe":
            self.reshape_media_pipe(self.X)
            self.load_label_mediapip()
        elif self.name_exp=="dlib":
            self.reshape_data(self.X)
            self.load_label_dlib()
        else:
            raise ValueError("No such name_exp!")
       
        self.load_edges()
        self.values=np.ones(self.edges_index.shape[1],dtype=np.float32)
        #Normalize label between 0,1.
        self.labels=self.labels/np.max(self.labels)
       
    def reshape_data(self,data):
        reshaped_tensor = np.transpose(data, (0, 3, 1, 2))  # 
        reshaped_tensor = np.reshape(reshaped_tensor, (8600, 51, 4, 137))  # Reshape to desired shape
        self.features=reshaped_tensor
        
       
    def reshape_media_pipe(self,data):
        reshaped_tensor = np.transpose(data, (0, 2, 3, 1))  # Transpose dimensions from 8700,137,469,4) to 8600, 469, 4, 137
        reshaped_tensor = np.reshape(reshaped_tensor, (8700, 469, 4, 137))  # Reshape to desired shape
        self.features=reshaped_tensor
        
    def load_label_dlib(self):
        label_file= open(self.labels_path,'rb')
        labels=pickle.load(label_file)
        labels=labels[1]
        self.labels= [[labels[i]] for i in range(len(labels))]
        self.labels=np.array(self.labels)
        
    def load_label_mediapip(self):
        self.labels=np.load(self.labels_path)
       
    def load_edges(self):
        self.edges_index=np.load(self.edges_index_path)
        self.edges_index=to_undirected(torch.tensor(self.edges_index),num_nodes=self.features.shape[1])
        self_loops=add_self_loops(self.edges_index,num_nodes=self.features.shape[1])
        self.edges_index=torch.stack((self_loops[0][0],self_loops[0][1]))
       
    def __len__(self):
        return self.features.shape[0]
    def get_shapes(self):
        print("featuers: ",self.features.shape, "labels:", self.labels.shape,"edges:",self.edges_index.shape)
        print("assert featuers:",self.features[0][0],np.max(self.features[0]))
        print("assert X:",self.X[0][0],np.max(self.X[0]))
        print("assert label",self.labels,np.unique(self.labels))
        print("assert edges",self.edges_index)
        return self.features.shape, self.edges_index.shape

    def __getitem__(self, index):
        x = self.features[index]
        y=self.labels[index]
        return x, y, self.edges_index,self.values
    
    def split_data(self):
        idx=np.load(self.idx_path) 
        idx=np.array(idx,dtype=np.int32)
        self.features=self.features[idx]
        self.labels=self.labels[idx]
        
   
if __name__=="__main__":

    name_exp = 'mediapipe'
    #name_exp = 'dlib'

    config_file=open("./config/"+name_exp+".yml", 'r')
    config = yaml.safe_load(config_file)
    data_path=config['data_path']
    labels_path=config['labels_path']
    edges_path=config['edges_path']
    idx_train= config['idx_train']
    idx_test=config['idx_test']
    loader=DataLoader(data_path,labels_path,edges_path,name_exp)
    train_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_train,mode="train")
    #test_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_test,mode="test")
    
   # print(next(iter(train_dataset)))
  #  print(next(iter(test_dataset)))
    for sample in train_dataset:
        x,y,edges_index,edges_attr=sample
       # print (x,x.shape,np.max(x))
        break