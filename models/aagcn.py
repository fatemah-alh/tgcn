import torch
import torch.nn.functional as F
import torch
from tqdm import tqdm
import sys
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from helper.dataloader import DataLoader
import yaml
import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.utils.to_dense_adj import to_dense_adj
from torch_geometric.utils import add_self_loops,to_undirected
import torch.nn.functional as F
from torch.nn import GRU,Linear,LSTM
from torch_geometric_temporal.nn.recurrent import A3TGCN2

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class GraphAAGCN:
    r"""
    Defining the Graph for the Two-Stream Adaptive Graph Convolutional Network.
    It's composed of the normalized inward-links, outward-links and
    self-links between the nodes as originally defined in the
    `authors repo  <https://github.com/lshiwjx/2s-AGCN/blob/master/graph/tools.py>`
    resulting in the shape of (3, num_nodes, num_nodes).
    Args:
        edge_index (Tensor array): Edge indices
        num_nodes (int): Number of nodes
    Return types:
            * **A** (PyTorch Float Tensor) - Three layer normalized adjacency matrix
    """

    def __init__(self, edge_index: list, num_nodes: int,num_subset:int):
        self.num_nodes = num_nodes
        self.edge_index = edge_index #261 #274 edges
        self.num_subset=num_subset
        self.A = self.get_spatial_graph()

    def get_spatial_graph(self):
        
        if self.num_subset==3:
            return self.get_three_adj()
        elif self.num_subset==2:
            return self.get_two_adj()
        elif self.num_subset==1:
            return self.get_one_adj()
        else:
            return ValueError("Not supported subset")
    def get_one_adj(self):
        #edges_index=torch.LongTensor(self.edge_index)
        edges_index=to_undirected(self.edge_index,num_nodes=self.num_nodes)
        edges_index_with_loops=add_self_loops(edges_index)
        edges_index=edges_index_with_loops[0]
        adj_mat=torch.squeeze(to_dense_adj(edges_index,max_num_nodes=self.num_nodes))
        adj_mat=torch.unsqueeze(adj_mat, dim=0)
        print(adj_mat.shape)
        return adj_mat
    def get_two_adj(self):
        self_mat = torch.eye(self.num_nodes).to(self.edge_index.device)
        #edges_index=torch.LongTensor(self.edge_index)
        edges_index=to_undirected(self.edge_index,num_nodes=self.num_nodes)
        adj_1=torch.squeeze(to_dense_adj(edges_index,max_num_nodes=self.num_nodes))
        adj_mat = torch.stack((self_mat,adj_1))
        print(adj_mat.shape)
        return adj_mat
    def get_three_adj(self):
        self_mat = torch.eye(self.num_nodes).to(self.edge_index.device)
        inward_mat = torch.squeeze(to_dense_adj(self.edge_index,max_num_nodes=self.num_nodes))
        inward_mat_norm = F.normalize(inward_mat, dim=0, p=1)
        outward_mat = inward_mat.transpose(0, 1)
        outward_mat_norm = F.normalize(outward_mat, dim=0, p=1)
        adj_mat = torch.stack((self_mat, inward_mat_norm, outward_mat_norm)) 
        print(adj_mat.shape)
        return adj_mat


class UnitTCN(nn.Module):
    r"""
    Temporal Convolutional Block applied to nodes in the Two-Stream Adaptive Graph
    Convolutional Network as originally implemented in the
    `Github Repo <https://github.com/lshiwjx/2s-AGCN>`. For implementational details
    see https://arxiv.org/abs/1805.07694
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size. (default: :obj:`9`)
        stride (int): Temporal Convolutional kernel stride. (default: :obj:`1`)
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1,bn :bool=True
    ):
        super(UnitTCN, self).__init__()
        
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            padding_mode='replicate')
        
        self.do_bn=bn
        self.relu = nn.ReLU(inplace=True)
        self._conv_init(self.conv)
        if self.do_bn:
            self.bn = nn.BatchNorm2d(out_channels)
            self._bn_init(self.bn, 1)

    def _bn_init(self, bn, scale):
        nn.init.constant_(bn.weight, scale)
        nn.init.constant_(bn.bias, 0)

    def _conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
        nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        #print(x.shape)
        #print(self.conv(x).shape)
        x=self.conv(x) 
        if self.do_bn:
            x = self.bn(x)
        return x



class UnitGCN(nn.Module):
    r"""
    Graph Convolutional Block applied to nodes in the Two-Stream Adaptive Graph Convolutional
    Network as originally implemented in the `Github Repo <https://github.com/lshiwjx/2s-AGCN>`.
    For implementational details see https://arxiv.org/abs/1805.07694.
    Temporal attention, spatial attention and channel-wise attention will be applied.
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        A (Tensor array): Adaptive Graph.
        coff_embedding (int, optional): Coefficient Embeddings. (default: :int:`4`)
        num_subset (int, optional): Subsets for adaptive graphs, see
        :math:`\mathbf{A}, \mathbf{B}, \mathbf{C}` in https://arxiv.org/abs/1805.07694
        for details. (default: :int:`3`)
        adaptive (bool, optional): Apply Adaptive Graph Convolutions. (default: :obj:`True`)
        attention (bool, optional): Apply Attention. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.FloatTensor,
        coff_embedding: int = 4,
        num_subset: int = 3,
        adaptive: bool = True,
        attention: bool = True,
        kernel_size: int = 9,
        bn:bool=True,
        L_name="l"
    ):
        super(UnitGCN, self).__init__()
        self.inter_c = out_channels // coff_embedding
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        self.A = A
        self.num_jpts = A.shape[-1]
        self.attention = attention
        self.adaptive = adaptive
        self.kernel_size=kernel_size
        self.conv_d = nn.ModuleList()
        self.do_bn=bn
        self.L_name=L_name

        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self._init_adaptive_layers()
        else:
            self.A = Variable(self.A, requires_grad=False)

        if self.attention:
            self._init_attention_layers()

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

       
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        if self.do_bn:
            self.bn = nn.BatchNorm2d(out_channels)
            self._init_conv_bn()

    def _bn_init(self, bn, scale):
        nn.init.constant_(bn.weight, scale)
        nn.init.constant_(bn.bias, 0)

    def _conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
        nn.init.constant_(conv.bias, 0)

    def _conv_branch_init(self, conv, branches):
        weight = conv.weight
        n = weight.size(0)
        k1 = weight.size(1)
        k2 = weight.size(2)
        nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
        nn.init.constant_(conv.bias, 0)

    def _init_conv_bn(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                self._bn_init(m, 1)
        self._bn_init(self.bn, 1e-6)

        for i in range(self.num_subset):
            self._conv_branch_init(self.conv_d[i], self.num_subset)

    def _init_attention_layers(self):
        # temporal attention
        pad_t = int((self.kernel_size - 1) / 2)
       
        self.conv_ta = nn.Conv1d(self.out_c, 1, self.kernel_size, padding=pad_t,padding_mode='replicate')
        
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        # s attention
        ker_jpt = self.num_jpts - 1 if not self.num_jpts % 2 else self.num_jpts
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(self.out_c, 1, ker_jpt, padding=pad,padding_mode='replicate') 
       
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        # channel attention
        rr = 2
        self.fc1c = nn.Linear(self.out_c, self.out_c // rr)
        self.fc2c = nn.Linear(self.out_c // rr, self.out_c)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

    def _init_adaptive_layers(self):
        self.PA = nn.Parameter(self.A)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(self.in_c, self.inter_c, 1))
            self.conv_b.append(nn.Conv2d(self.in_c, self.inter_c, 1))

    def _attentive_forward(self, y):
        #y_shape=(N, C, T, V)
        # spatial attention
    
        se = y.mean(-2)  # N C V
        #se1 = self.sigmoid(self.conv_sa(se)) # l'importanza di ogni nodo 
        se1 = self.sigmoid(self.conv_sa(se)) # l'importanza di ogni nodo 
        #se1_save=se1.cpu().numpy()
        #np.save(f"Attention_nodes_{self.L_name}.npy",se1_save)
        y = y * se1.unsqueeze(-2) + y # moltiplica l'input per il vettore di pesi o attention se1.
  
        # temporal attention
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        #np.save("temporal_attention.npy",se1)
        y = y * se1.unsqueeze(-1) + y
        # channel attention
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        #np.save("channel_attention.npy",se2)
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        return y

    def _adaptive_forward(self, x, y):
        N, C, T, V = x.size()

        A = self.PA
        for i in range(self.num_subset):
            A1 = (
                self.conv_a[i](x)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(N, V, self.inter_c * T)
            )
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V # The C matrix
            #makeit_symmetric
            #A1=torch.matmul(torch.transpose(A1,1,2),A1)/ A1.size(-1)
            #print(A1.shape,A1==torch.transpose(A1,1,2))
            #A1_save=A1.cpu().numpy()
            #np.save("Adaptive_matrix.npy",A1_save)
            #print(self.alpha)
            A1 = A[i] + A1 * self.alpha 
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        return y

    def _non_adaptive_forward(self, x, y):
        N, C, T, V = x.size()
        for i in range(self.num_subset):
            A1 = self.A[i]
            A2 = x.view(N, C * T, V) 
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        return y

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            y = self._adaptive_forward(x, y)
        else:
            y = self._non_adaptive_forward(x, y)
        if self.do_bn:
            y = self.bn(y)
        y += self.down(x)
        y = self.relu(y) # (N, self.out_c, T, V)
        if self.attention:
            y = self._attentive_forward(y)
        return y



class AAGCN(nn.Module):
    r"""Two-Stream Adaptive Graph Convolutional Network.

    For details see this paper: `"Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition." <https://arxiv.org/abs/1805.07694>`_.
    This implementation is based on the authors Github Repo https://github.com/lshiwjx/2s-AGCN.
    It's used by the author for classifying actions from sequences of 3D body joint coordinates.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        edge_index (PyTorch LongTensor): Graph edge indices.
        num_nodes (int): Number of nodes in the network.
        stride (int, optional): Time strides during temporal convolution. (default: :obj:`1`)
        residual (bool, optional): Applying residual connection. (default: :obj:`True`)
        adaptive (bool, optional): Adaptive node connection weights. (default: :obj:`True`)
        attention (bool, optional): Applying spatial-temporal-channel-attention.
        (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_index: torch.LongTensor,
        num_nodes: int,
        num_subset: int=3,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True,
        attention: bool = True,
        kernel_size:int= 9,
        bn:bool=True,
        L_name="l"
    ):
        super(AAGCN, self).__init__()
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        self.graph = GraphAAGCN(self.edge_index, self.num_nodes,num_subset=num_subset)
        self.A = self.graph.A

        self.gcn1 = UnitGCN(
            in_channels, out_channels, self.A,num_subset=num_subset, adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name=L_name
        )
        self.tcn1 = UnitTCN(out_channels, out_channels, stride=stride,kernel_size=kernel_size,bn=bn)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = UnitTCN(
                in_channels, out_channels, kernel_size=1, stride=stride  
            )

    def forward(self, x):
        """
        Making a forward pass.

        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods,
            with shape (B, F_in, T_in, N_nodes).

        Return types:
            * **X** (PyTorch FloatTensor)* - Sequence of node features,
            with shape (B, out_channels, T_in//stride, N_nodes).
        """
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y
    
class aagcn_network(nn.Module):
    def __init__(self, num_person=1, 
                 graph=None,
                 num_nodes=51, 
                 in_channels=4,
                 drop_out=0.5, 
                 num_subset=3,
                 kernel_size=9,
                 stride=1,
                 hidden_size=1,
                 gru_layer=2,
                 adaptive=False, 
                 attention=True,
                 bn=True,
                 embed=False,
                 return_all_outputs=False):
        super(aagcn_network, self).__init__()
        #graph:edges_index
        if graph is None:
            raise ValueError("No edges_index is found!")
        self.edge_index=graph
        self.embed=embed
        self.return_all_outputs=return_all_outputs
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)#TODO senza 
        self.l1 = AAGCN(in_channels, 64, graph, num_subset=num_subset,num_nodes=num_nodes, residual=False, adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l1")
       # self.l2 = AAGCN(64, 64, graph,num_subset=num_subset, num_nodes=num_nodes,stride=1,adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l2")
        #self.l3 = AAGCN(64, 64, graph,num_subset=num_subset, num_nodes=num_nodes,stride=1, adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l3")
        #self.l4 = AAGCN(64, 64, graph,num_subset=num_subset, num_nodes=num_nodes,stride=1 ,adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l4")
        self.l5 = AAGCN(64, 128, graph,num_subset=num_subset, num_nodes=num_nodes,stride=stride, adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l5")
        self.l6 = AAGCN(128,128, graph,num_subset=num_subset, num_nodes=num_nodes,stride=stride,adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l6")
        
       # self.l7 = AAGCN(128, 128, graph,num_subset=num_subset, num_nodes=num_nodes,stride=stride,adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l7")
       # self.l8 = AAGCN(128, 256, graph,num_subset=num_subset, num_nodes=num_nodes,stride=3, adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l8")
        #self.l9 = AAGCN(256, 256, graph,num_subset=num_subset, num_nodes=num_nodes,stride=stride,adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l9")
        #self.l10 = AAGCN(256, 256, graph,num_subset=num_subset, num_nodes=num_nodes,stride=1,adaptive=adaptive, attention=attention,kernel_size=kernel_size,bn=bn,L_name="l10")
        self.gru=GRU(input_size=128*num_nodes, hidden_size=128,num_layers=gru_layer,batch_first=True)
        #self.fc_1=Linear(in_features=128*num_nodes,out_features= 128) 
        self.fc=Linear(in_features=128,out_features= 1) 
        nn.init.kaiming_normal_(self.fc.weight)
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x = self.l1(x)
        #x = self.l2(x)#
        #x = self.l3(x)#
        #x = self.l4(x)#
        x = self.l5(x)
        x = self.l6(x)
        #x = self.l7(x)#
        #x = self.l8(x)#
        #x = self.l9(x)#
        #x = self.l10(x) #
        x = self.drop_out(x)
       
        t_new=x.size(2)
        c_new=x.size(1)
        x=x.permute(0, 2, 1, 3).contiguous().view(N,t_new,c_new,V) #(32,35,128,51)
        x=x.view(N,t_new,-1)
      
        embed_graph=x # Try with tow type of embeddings
        x,h=self.gru(x)
        #x=self.fc_1(x)
        embed_gru=x 
        x = self.drop_out(x)
        x=self.fc(x)
        all_outputs= x
        all_outputs=all_outputs.view(N,-1)
       
        #Take the last output
        x=x[:,-1,:]
        x=x.view(-1)
        
        #x=F.relu(x)
       
        if self.embed:
            return x,embed_graph
        elif self.return_all_outputs:
            return x, all_outputs
        else:
            return x,embed_gru[:,-1,:] #embed_graph[:,-1,:]


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
    
    if torch.cuda.is_available():
        print("set cuda device")
        device="cuda"
        torch.cuda.set_device(gpu)
    else:
        device="cpu"
        print('Warning: Using CPU')
    
    edges_index=torch.LongTensor(np.load(edges_path)).to(device)
    print(edges_index.device)
    model = aagcn_network(num_person=1,
                           graph=edges_index,
                          num_nodes=num_nodes,
                           in_channels=num_features,
                           drop_out=0.5, 
                           adaptive=config['adaptive'],
                             attention=True,num_subset=2)
    model.to(device)
    print(model)

   
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate the total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    train_dataset=DataLoader(data_path,labels_path,
                             edges_path,
                             idx_path=idx_train,
                             num_features= num_features,
                             num_nodes=num_nodes)
    test_dataset=DataLoader(data_path,labels_path,
                            edges_path,idx_path=idx_test,
                            num_features= num_features,
                            num_nodes=num_nodes)
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
        