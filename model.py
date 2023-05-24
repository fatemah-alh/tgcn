#%%
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN,A3TGCN2
import torch
from tqdm import tqdm
from dataloader import DataLoader
import yaml
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
        self.embed_dim=embed_dim #64 ,128,..
        self.periods=periods
        self.output_features=output_features
        self.num_nodes=num_nodes
        self.batch_size=batch_size
        self.tgnn = A3TGCN2(in_channels=self.node_features,
                           out_channels=self.embed_dim,
                           periods=self.periods,
                           batch_size=self.batch_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.linear_1= torch.nn.Linear(self.embed_dim, 32)
        self.linear_2=torch.nn.Linear(32, 1)#
        self.linear_3=torch.nn.Linear(self.num_nodes, self.output_features)# batch [32, 51] [32,1]


    def forward(self, x, edge_index,edge_attr=None):
        """
        x = Node features for T time steps [B,51,4,137]
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
        h=self.linear_2(h)
        h=h.view(-1,self.num_nodes)
        h=self.linear_3(h)
        #h = F.relu(h) 
        h=torch.sigmoid(h)
        #h=5.0 * torch.sigmoid(h)
        h=h.view(-1)
        
        
        return h

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
    TS=config['TS']
    batch_size=config['batch_size']
    embed_dim=config['embed_dim']
    num_features=config['num_features']
    num_nodes=config['n_joints'] 
    gpu=config['gpu']
    model = TemporalGNNBatch(node_features=num_features,
                                      num_nodes=num_nodes,
                                      embed_dim=embed_dim, 
                                      periods=TS,
                                      batch_size=batch_size)
    if torch.cuda.is_available():
        print("set cuda device")
        device="cuda"
        torch.cuda.set_device(gpu)
    else:
        device="cpu"
        print('Warning: Using CPU')
    model.cuda()
    print(model)
    

    

    loader=DataLoader(data_path,labels_path,edges_path,name_exp)
    train_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_train,mode="train")
    test_dataset=DataLoader(data_path,labels_path,edges_path,name_exp,idx_path=idx_test,mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=False,
                                                   sampler=torch.utils.data.SequentialSampler(test_dataset),
                                                   drop_last=False)
    
    tq=tqdm(test_loader)
    for i in tq:
        x,y,edge,attr=i
        #print("assert data ",x,x.shapeedge[0].shape)
        print("assert label ",y,y.shape)
        x=x.to(device)
        attr=attr.to(device)
        edge=edge.to(device)
        y_hat=model(x,edge[0],attr[0])
        