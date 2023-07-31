
import numpy as np
from tqdm import tqdm
import torch
import yaml

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path,labels_path,edges_index_path,name_exp,data_shape=(8700, 51, 6, 137),normalize_labels=True,idx_path=None):
        super(DataLoader, self).__init__()

        self.data_path = data_path
        self.labels_path=labels_path
        self.edges_index_path=edges_index_path
        self.idx_path=idx_path
        self.name_exp=name_exp
        self.normalize_labels=normalize_labels
        self.data_shape=data_shape
        self._read_data()
        self.get_shapes()
        
    def _read_data(self):
        print("Loading Dataset")
        self.X=np.load(self.data_path)
        self.reshape_data()
        self.edges_index=np.load(self.edges_index_path)
        self.labels=np.load(self.labels_path)
        #Normalize label between 0,1.
        if self.normalize_labels :
            self.labels=self.labels/np.max(self.labels)
        #split data set with idx
        if self.idx_path!=None:
            self.split_data()
        print("Data is loaded!")
       
    def reshape_data(self):
        reshaped_tensor = np.transpose(self.X, (0, 2, 3, 1))  # Transpose dimensions from 8700,137,469,4) to 8600, 469, 4, 137
        self.features= np.reshape(reshaped_tensor, self.data_shape)  # Reshape to desired shape 
        
    def __len__(self):
        return self.features.shape[0]
    def get_shapes(self):
        print("featuers: ",self.features.shape, "labels:", self.labels.shape,"edges:",self.edges_index.shape)
        print("assert featuers:",self.features[0][0],np.max(self.features[0]))
        print("assert X:",self.X[0][0],np.max(self.X),np.min(self.X))
        print("assert label",self.labels,np.unique(self.labels))
        print("assert edges",self.edges_index)
        return self.features.shape, self.edges_index.shape

    def __getitem__(self, index):
        x = self.features[index]
        y=self.labels[index]
        return x, y, self.edges_index
    
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
    #name_exp = 'mediapipe'
    #name_exp = 'dlib'

    config_file=open("./config/"+name_exp+".yml", 'r')
    config = yaml.safe_load(config_file)
    data_path=config['data_path']
    labels_path=config['labels_path']
    edges_path=config['edges_path']
    idx_train= config['idx_train']
    idx_test=config['idx_test']
    data_shape=config['data_shape']
    loader=DataLoader(data_path,labels_path,edges_path,name_exp)
    train_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_train)
    test_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_test)
    
    for sample in train_dataset:
        x,y,edges_index=sample
       #print (x,x.shape,np.max(x))
        break