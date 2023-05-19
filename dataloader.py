import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric_temporal.signal import StaticGraphTemporalSignal,StaticGraphTemporalSignalBatch
import torch_geometric
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import temporal_signal_split
import torch
class DataLoader(object):
    def __init__(self, data_path,labels_path,edges_index_path):
        super(DataLoader, self).__init__()
        self.data_path = data_path
        self.labels_path=labels_path
        self.edges_index_path=edges_index_path
        self._read_data()

    def _read_data(self):
        self.edges_index=np.load(self.edges_index_path)
        #print(self.edges_index.shape, self.edges_index[0].shape, self.edges_index[0])
        """
       
        #filter edges
        filtered_edges=[[],[]]
        for i in range(edges.shape[1]):
            if edges[0][i]<51 and edges[1][i]<51:
                    filtered_edges[0].append(edges[0][i])
                    filtered_edges[1].append(edges[1][i])
        self.edges_index=np.array(filtered_edges)
        """
       
        print("index shape ",self.edges_index.shape)
        #self.edge_weights=
        self.X=np.load(self.data_path)
        self.X=self.reshape_data(self.X)
        self.features= [self.X[i] for i in range(self.X.shape[0])]
        label_file= open(self.labels_path,'rb')
        labels=pickle.load(label_file)
        labels=labels[1]
        self.labels= [[labels[i]] for i in range(len(labels))]
        self.labels=np.array(self.labels)
        
        """
        X = np.load(datapath+"AllDSSkeleton.npy").transpose((1, 2, 0))
        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)
        """


    def reshape_data(self,data):
       
        reshaped_tensor = np.transpose(data, (0, 3, 1, 2))  # Transpose dimensions
        reshaped_tensor = np.reshape(reshaped_tensor, (8600, 51, 4, 137))  # Reshape to desired shape
        return reshaped_tensor
    def __len__(self):
        return self.X.shape[0]
    def get_shapes(self):
        print("featuers_len",len(self.features), "one sample", self.features[0].shape, "labels", len(self.labels))
        return self.X.shape, self.edges_index.shape
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
    edges_path="/home/falhamdoosh/tgcn/data/edges_indx_dlib68.npy"
    TS=137

    loader=DataLoader(data_path,labels_path,edges_path)
    
    print(loader.get_shapes())
    dataset=loader.get_dataset()
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
    
    print(next(iter(train_dataset)))

    # i should get a Data object at each iteration
    # apply manual dataset for batch and for filtred train and test
    #apply function to add ids training and id test
    #apply random sampling.

    for i in tqdm(dataset):
        print (i.x[:,:,1].shape,i.y)
        if i>10:
            break
 