import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN,A3TGCN2
import torch
from tqdm import tqdm
from dataloader import DataLoader
import yaml


if __name__=="__main__":

    name_exp = 'open_face'
    #name_exp = 'mediapipe'
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
    

    

    loader=DataLoader(data_path,labels_path,edges_path)
    train_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_train)
    test_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=False,
                                                   sampler=torch.utils.data.SequentialSampler(test_dataset),
                                                   drop_last=False)
    
    tq=tqdm(test_loader)
    for i in tq:
        x,y,edge=i
        #print("assert data ",x,x.shapeedge[0].shape)
        #print("assert label ",y,y.shape)
        x=x.to(device)
        edge=edge.to(device)
        
        y_hat=model(x,edge[0])
        