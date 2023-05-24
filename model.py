#%%
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN,A3TGCN2
import torch
from tqdm import tqdm
from dataloader import DataLoader
#%%
"""
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features=4,output_features=1,num_nodes=51,embed_dim=32, periods=137):
        super(TemporalGNN, self).__init__()
        self.node_features = node_features
        self.embed_dim=embed_dim
        self.periods=periods
        self.output_features=output_features
        self.num_nodes=num_nodes
        self.tgnn = A3TGCN(in_channels=self.node_features,
                           out_channels=self.embed_dim,
                           periods=self.periods)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear_1= torch.nn.Linear(self.embed_dim, self.output_features)
        self.linear_2=torch.nn.Linear(self.num_nodes, self.output_features)


    def forward(self, x, edge_index):
     
        x = Node features for T time steps [51,4,137]
        edge_index = Graph edge indices [2,num_edges]
        
        h = self.tgnn(x, edge_index)
        #h = F.relu(h)
        h=self.dropout(h)
        h = self.linear_1(h)
        h = F.relu(h)
        h=h.view(-1,self.num_nodes)
       
        h= self.linear_2(h)
       
        #h = F.relu(h)
       # h=5.0 * torch.sigmoid(h)
        h=h.view(-1)
        return h



"""
class TemporalGNNBatch(torch.nn.Module):
    def __init__(self, node_features=4,output_features=1,num_nodes=51,embed_dim=32, periods=137,batch_size=32):
        super(TemporalGNNBatch, self).__init__()
        self.node_features = node_features
        self.embed_dim=embed_dim
        self.periods=periods
        self.output_features=output_features
        self.num_nodes=num_nodes
        self.batch_size=batch_size
        self.tgnn = A3TGCN2(in_channels=self.node_features,
                           out_channels=self.embed_dim,
                           periods=self.periods,batch_size=self.batch_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear_1= torch.nn.Linear(self.embed_dim, 1)
        # input[32,[51*32]] uscita come [batch 32, 51*1] per ogni nodo ha un imbed di dimensione 32 invece del 137
        
        self.linear_2=torch.nn.Linear(self.num_nodes, self.output_features)# batch [32, 51] [32,1]


    def forward(self, x, edge_index,edge_attr=None):
        """
        x = Node features for T time steps [51,4,137]
        edge_index = Graph edge indices [2,num_edges]
        """
        
        if edge_attr!=None:
            h = self.tgnn(x, edge_index,edge_attr)
        else:
            h = self.tgnn(x, edge_index) #batch,
        h = F.relu(h)
        h=self.dropout(h)
        h = self.linear_1(h)
        h = F.relu(h)
        h=h.view(-1,self.num_nodes)
        h=self.linear_2(h)
       # h = F.relu(h) #same loss with 0 values or 2.25
        h=5.0 * torch.sigmoid(h)
        # a volte l'output è sempre zero, a volte 2.45
        h=h.view(-1)
        
        
        return h

if __name__=="__main__":
    model=TemporalGNNBatch(node_features= 4,periods= 137,batch_size=32)
    model.cuda()
    print(model)
    print(model.parameters)
    data_path="/home/falhamdoosh/tgcn/Painformer/dataset_data_biovid.npy"
    labels_path="/home/falhamdoosh/tgcn/Painformer/dataset_label_biovid.pkl"
    edges_path="/home/falhamdoosh/tgcn/Painformer/edge_index_51biovid.npy"
    idx_train= "/home/falhamdoosh/tgcn/Painformer/idx_train.npy"
    idx_test="/home/falhamdoosh/tgcn/Painformer/idx_test.npy"
    TS=137

    loader=DataLoader(data_path,labels_path,edges_path)
    
    train_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_train,mode="train")
    test_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_test,mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=32, 
                                                   shuffle=False,
                                                   sampler=torch.utils.data.SequentialSampler(test_dataset),
                                                   drop_last=False)
    
    tq=tqdm(test_loader)
    for i in tq:
        x,y,edge=i
        print(x.shape,y.shape,edge[0].shape)
        x=x.cuda()
        edge=edge.cuda()
        y_hat=model(x,edge[0])
        break