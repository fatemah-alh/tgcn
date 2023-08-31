#%%
import numpy as np
from tqdm import tqdm
import torch
import yaml
import random
from utiles import rotation_matrix_2d
import math
from torchvision.transforms import RandomChoice,RandomApply,Compose

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path,labels_path,edges_index_path,data_shape=[(0, 3, 1, 2),(8700, 6,137,51)],normalize_labels=True,idx_path=None,reshape_data=True,expand_dim=True,model_name="aagcn",num_features=6,num_nodes=51,class_3=False,transform=None):
        super(DataLoader, self).__init__()

        self.data_path = data_path
        self.labels_path=labels_path
        self.edges_index_path=edges_index_path
        self.idx_path=idx_path
        self.normalize_labels=normalize_labels
        self.expand_dim=expand_dim
        self.model_name=model_name
        self.reshape_data=reshape_data
        self.num_features=num_features
        self.num_nodes=num_nodes
        self.class_3=class_3
        self.transform = transform
        if self.model_name=="a3tgcn":
            self.data_shape=[(0, 2, 3, 1),(8700,num_nodes,num_features,137)]
            self.expand_dim=False
        elif self.model_name=="aagcn":
            self.data_shape=[(0, 3, 1, 2),(8700, num_features,137,num_nodes)]
        else:
            self.data_shape=data_shape
        self._read_data()
        
    def _read_data(self):
        print("Loading Dataset")#(8700,137,51,6 )
        self.X=np.load(self.data_path) #
        if self.num_nodes==43:
            print("Fal expirment")
            self.X=np.concatenate((self.X[:,:,:11,:],self.X[:,:,19:,:]),axis=2)
        if self.num_features==4:
             self.X=self.X[:,:,:,:4]
            #self.X=np.concatenate( (self.X[:,:,:,:2],self.X[:,:,:,3:5]),axis=3)
        elif self.num_features==2:
            self.X=self.X[:,:,:,:2]
        #self.preprocess()
        self._reshape_data()
        print("Contains Nan values",np.isnan(np.array(self.features)).any())
        self.labels=np.load(self.labels_path)#0,..,4
        #Normalize label between 0,1.
        if self.class_3:
            self.three_classes()
            print("Three class expirment")
        if self.normalize_labels :
            self.labels=self.labels/np.max(self.labels)
        
        #split data set with idx
        self.split_data()
        print("Data is loaded!")
        print(self.features.shape)
        print(np.unique( self.labels))
       
    def _reshape_data(self):
        if self.reshape_data:
             
             #reshape (8700,137,51,6 ) to (8700, 6,137,51)aagcn (0, 3, 1, 2)
             #reshape (8700,137,51,6 ) to(8700,51,6,137) a3gcn(0, 2, 3, 1)
             reshaped_tensor = np.transpose(self.X, self.data_shape[0])  # Transpose dimensions from 8700,137,469,4) to 8600, 469, 4, 137
             self.features= torch.tensor(np.reshape(reshaped_tensor, self.data_shape[1]))  # Reshape to desired shape 
             
        else:
            self.features=self.X
        if self.expand_dim:
            self.features=np.expand_dims(self.features,axis=-1)#unsqueeze, view
    def preprocess(self):
        for i in range(0,self.X.shape[0]):
            for j in range(0,self.X.shape[1]):
                x_std=np.std(self.X[i,j,:,0]) 
                y_std=np.std(self.X[i,j,:,1])
                if x_std!=0:
                    self.X[i,j,:,0]=self.X[i,j,:,0]/x_std
                if y_std!=0:
                    self.X[i,j,:,1]=self.X[i,j,:,1]/y_std

    def __len__(self):
        return self.features.shape[0]
    def get_shapes(self):
        print("featuers: ",self.features.shape, "labels:", self.labels.shape)
        print("assert featuers:",self.features[0][0])
        print("assert X:",self.X[0][0])
        print("assert label",np.unique(self.labels))
        
        return self.features.shape

    def __getitem__(self, index):

        x = self.features[index]
        y=self.labels[index]
        if self.transform:
            x,y=self.transform((x,y))
        return x, y
    
    def split_data(self):
        if self.idx_path==None:
            idx_train=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/idx_train_filterd.npy")
            idx_test=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/data/PartA/idx_test_filterd.npy")
            idx=np.concatenate((idx_train,idx_test),axis=0,dtype=np.int32)
        else:
            idx=np.load(self.idx_path) 
            idx=np.array(idx,dtype=np.int32)
        self.features=self.features[idx]
        self.labels=self.labels[idx]
    

    def three_classes(self):
        indices_1 = np.where(self.labels==1)[0]
        self.labels[indices_1]= 0

        indices_2 = np.where(self.labels==2)[0]
        self.labels[indices_2]=1

        indices_3 = np.where(self.labels==3)[0]
        self.labels[indices_3]= 2
        
        indices_4 = np.where(self.labels==4)[0]
        self.labels[indices_4]= 2


    

    def binary_classification(self):
        values = [0, 4]
        indices = np.where(np.isin(self.labels, values))[0]
        self.labels=self.labels[indices]
        self.features=self.features[indices]
        indices_4 = np.where(self.labels == 4)
        self.labels[indices_4] = 1
        print(len(self.labels))
        print("number of samples with 4 class ",len(indices_4))
    

class Rotate(object):
    """
    Rotate sample of N frame of 2d points
    """

    def __init__(self, angles=[-5,5,-10,10,-15,15]):
        self.angles=angles

    def __call__(self, sample):
        x,y = sample
        
        C,T,V,M=x.shape
        angle=random.choice(self.angles)
        angle = math.radians(angle)
        Rotation=rotation_matrix_2d(angle)
        for i in range(0,T):
            landmarks= torch.tensor(x[:2,i,:,0])
            landmarks=landmarks.permute(1,0).contiguous().view(V,-1) 
            rotated_l=np.dot(landmarks,Rotation.T)
            x[:2,i,:,0]=rotated_l.reshape((2,V))
        
        #print("rotate:",angle)
        return x,y
        
class FlipV(object):
    """
    Flip landmarks with respect to vertical axis
    """
    def __call__(self, sample):
        x,y = sample
        x[:,:,0]=-x[:,:,0]
       # print("Flip")
        return x,y
if __name__=="__main__":
    name_exp="open_face"
   
    parent_folder="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
    config_file=open(parent_folder+"config/"+name_exp+".yml", 'r')
    config = yaml.safe_load(config_file)
    data_path=parent_folder+config['data_path']
    labels_path=parent_folder+config['labels_path']
    edges_path=parent_folder+config['edges_path']
    idx_train= parent_folder+config['idx_train']
    idx_test=parent_folder+config['idx_test']

    batch_size=config['batch_size']
    embed_dim=config['embed_dim']
    num_features=config['num_features']
    num_nodes=config['n_joints'] 
    class_3=config['class_3']
    transform=RandomApply([RandomChoice([Rotate(),FlipV(),Compose([FlipV(),Rotate()])])],p=0.5)
    #loader=DataLoader(data_path,labels_path,edges_path,num_features= num_features,num_nodes=num_nodes)
    train_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_train,num_features= num_features,num_nodes=num_nodes,class_3= class_3,transform=transform)
    test_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_test,num_features= num_features,num_nodes=num_nodes,class_3= class_3)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True,
                                                   drop_last=True)
    for sample in train_loader:
        x,y=sample
       
        print(y)