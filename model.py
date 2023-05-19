#%%
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
import torch
#%%
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
        self.linear_1= torch.nn.Linear(self.embed_dim, self.output_features)
        self.linear_2=torch.nn.Linear(self.num_nodes, self.output_features)


    def forward(self, x, edge_index):
        """
        x = Node features for T time steps [51,4,137]
        edge_index = Graph edge indices [2,num_edges]
        """
        h = self.tgnn(x, edge_index)
        #print("output tgnn",h.shape)
        h = F.relu(h)
        h = self.linear_1(h)
        #print("output linear_1",h.shape)
        h=h.view(self.output_features,self.num_nodes)
        h= self.linear_2(h)
        h=h.view(-1)
        #print("output linear_2",h.shape)
        
        return h

if __name__=="__main__":
    model=TemporalGNN(4,137)
    print(model)
    print(model.parameters)