#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN,A3TGCN2
import numpy as np
from tqdm import tqdm
import sys
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from dataloader import DataLoader
import yaml

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

class A3TGCN2_network(nn.Module):
    def __init__(self,edge_index, node_features=4,output_features=1,num_nodes=51, periods=137,batch_size=32):
        super(A3TGCN2_network, self).__init__()
        self.node_features = node_features
       
        self.periods=periods
        self.output_features=output_features
        self.num_nodes=num_nodes
        self.batch_size=batch_size
        self.edge_index=edge_index
        self.tgnn1 = A3TGCN2(in_channels=self.node_features,out_channels=256,periods=self.periods,batch_size=self.batch_size)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear1= torch.nn.Linear(256, 128)
        self.linear3=torch.nn.Linear(128, 1)#
        self.linear4=torch.nn.Linear(self.num_nodes, 1)# batch [32, 51] [32,1]
        


    def forward(self, x):
        """
        x = Node features for T time steps [B,51,4,137]
        edge_index = Graph edge indices [2,num_edges]
        """
        
        
        h = self.tgnn1(x, self.edge_index) #torch.Size([32, 51, 256]) eliminate temporal dimension
        h=self.dropout(h)
        h = self.linear1(h)#[32, 51, 128]
        h=self.linear3(h)#[32, 51, 1]
        h=h.view(-1,self.num_nodes)#[32, 51]
        h=self.linear4(h)
        h = F.relu(h) 
        h=h.view(-1)
       
        
        return h

if __name__=="__main__":
    name_exp = 'open_face'
    
    parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
    config_file=open(parent_folder+"config/"+name_exp+".yml", 'r')
    config = yaml.safe_load(config_file)

    data_path=parent_folder+config['data_path']
    labels_path=parent_folder+config['labels_path']
    edges_path=parent_folder+config['edges_path']
    
    idx_train= parent_folder+config['idx_train']
    idx_test=parent_folder+config['idx_test']
    TS=config['TS']
    batch_size=config['batch_size']
    embed_dim=config['embed_dim']
    num_features=config['num_features']
    num_nodes=config['n_joints'] 
    gpu=config['gpu']
    model_name=config['model_name']
    
    if torch.cuda.is_available():
        print("set cuda device")
        device="cuda"
        torch.cuda.set_device(gpu)
    else:
        device="cpu"
        print('Warning: Using CPU')
    edge_index=torch.LongTensor(np.load(edges_path)).to(device)
    model = A3TGCN2_network(edge_index,node_features=num_features,num_nodes=num_nodes,periods=TS,batch_size=batch_size)
    model.cuda()
    print(model)
    

    

     #[Num_sample,Num_nodes,Num_features, Timsetep][8700,51,6,137]
    train_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_train,model_name=model_name)
    test_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_test,model_name=model_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=False,
                                                   sampler=torch.utils.data.SequentialSampler(test_dataset),
                                                   drop_last=False)
    
    tq=tqdm(test_loader)
    for i in tq:
        x,y=i
        x=x.to(device)
        y_hat=model(x)
        break
        