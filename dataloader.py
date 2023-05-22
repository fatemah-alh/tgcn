import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric_temporal.signal import StaticGraphTemporalSignal,StaticGraphTemporalSignalBatch
import torch_geometric
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import temporal_signal_split
import torch


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path,labels_path,edges_index_path,idx_path=None,mode=None):
        super(DataLoader, self).__init__()
        self.data_path = data_path
        self.labels_path=labels_path
        self.edges_index_path=edges_index_path
        self.idx_path=idx_path
        self.mode=mode
        self._read_data()
        if self.mode!=None:
            if self.idx_path==None:
                raise(ValueError("No idx_path has been found"))
            else:
                self.split_data()

    def _read_data(self):
        self.edges_index=np.load(self.edges_index_path)
        self.X=np.load(self.data_path)
        self.X=self.reshape_data(self.X)
        self.features=self.X
        label_file= open(self.labels_path,'rb')
        labels=pickle.load(label_file)
        labels=labels[1]
        self.labels= [[labels[i]] for i in range(len(labels))]
        self.labels=np.array(self.labels)

    def reshape_data(self,data):
        reshaped_tensor = np.transpose(data, (0, 3, 1, 2))  # Transpose dimensions
        reshaped_tensor = np.reshape(reshaped_tensor, (8600, 51, 4, 137))  # Reshape to desired shape
        return reshaped_tensor
    def __len__(self):
        return self.features.shape[0]
    def get_shapes(self):
        print("featuers_len",len(self.features), "one sample", self.features[0].shape, "labels", len(self.labels))
        return self.X.shape, self.edges_index.shape
   
    def __getitem__(self, index):
        x = self.features[index]
        y=self.labels[index]
        return x, y, self.edges_index
    
    def split_data(self):
        idx=np.load(self.idx_path)#
        idx=np.array(idx,dtype=np.int32)
        self.features=self.features[idx]
        self.labels=self.labels[idx]
        print("len dataset ", self.mode ,self.__len__())
    def get_dataset(self) -> StaticGraphTemporalSignal:
        """Returns data iterator for NTU-RGBD dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* 
        """
       # StaticGraphTemporalSignalBatch
        dataset = StaticGraphTemporalSignal(
           edge_index= self.edges_index,edge_weight= None ,features= self.features, targets= self.labels
        )
        """
       
        batched_dataset=StaticGraphTemporalSignalBatch(edge_index= self.edges_index,
                                                       edge_weight= None ,
                                                       features= self.features, 
                                                       targets= self.labels
                                                      
        )
         """
        return dataset


if __name__=="__main__":

    data_path="/home/falhamdoosh/tgcn/Painformer/dataset_data_biovid.npy"
    labels_path="/home/falhamdoosh/tgcn/Painformer/dataset_label_biovid.pkl"
    edges_path="/home/falhamdoosh/tgcn/Painformer/edge_index_51biovid.npy"
    idx_train= "/home/falhamdoosh/tgcn/Painformer/idx_train.npy"
    idx_test="/home/falhamdoosh/tgcn/Painformer/idx_test.npy"
    TS=137

    loader=DataLoader(data_path,labels_path,edges_path)
    
    train_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_train,mode="train")
    test_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_test,mode="test")
    
    #print(next(iter(train_dataset)))
    #print(next(iter(test_dataset)))

    # i should get a Data object at each iteration
    # apply manual dataset for batch and for filtred train and test
    #apply function to add ids training and id test
    #apply random sampling.
