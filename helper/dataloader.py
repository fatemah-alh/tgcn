#%%
import numpy as np
from tqdm import tqdm
import torch
import yaml
import random
import sys
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)

from utiles import rotation_matrix_2d
import math
from torchvision.transforms import RandomChoice,RandomApply,Compose

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path,
                    labels_path,
                    edges_index_path,
                    data_shape=[(0, 3, 1, 2),(8700, 6,137,51)],
                    normalize_labels=True,
                    idx_path=None,
                    reshape_data=True,
                    expand_dim=True,
                    model_name="aagcn",
                    num_features=6,
                    num_nodes=51,
                    N_sample=8700,
                    TS=137,
                    num_classes=5,
                    transform=None,
                    contantenat=False,
                    maxMinNormalization=False,
                    min_max_values=None):
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
        self.num_classes=num_classes
        self.transform = transform
        self.concatenate=contantenat
        self.maxMinNormalization=maxMinNormalization
        self.min_max_values=min_max_values

        if self.model_name=="a3tgcn":
            self.data_shape=[(0, 2, 3, 1),(N_sample,num_nodes,num_features,TS)]
            self.expand_dim=False
        elif self.model_name=="aagcn":
            self.data_shape=[(0, 3, 1, 2),(N_sample, num_features,TS,num_nodes)]
        else:
            self.data_shape=data_shape
        self._read_data()
        
    def _read_data(self):
        
        print("Loading Dataset")
        self.labels=np.load(self.labels_path)#0,..,4
        self.X=np.load(self.data_path)
        print(np.min(self.X[:,:,:,0]),np.max(self.X[:,:,:,0]))
        print("Contains Nan values",np.isnan(self.X).any())
        if self.maxMinNormalization:
            self.maxMinNorm()
        #Select featurs 
        print("Contains Nan values",np.isnan(self.X).any())
        if self.num_nodes==43:
            print("Fal expirment...")
            self.X=np.concatenate((self.X[:,:,:11,:],self.X[:,:,19:,:]),axis=2)
        if self.num_features==4:
             self.X=self.X[:,:,:,:4]
            #self.X=np.concatenate( (self.X[:,:,:,:2],self.X[:,:,:,3:5]),axis=3)
        elif self.num_features==2:
            self.X=self.X[:,:,:,:2] #position
            #self.X=self.X[:,:,:,2:4]#Velocity
           # self.X=self.X[:,:,:,4:]#headMotion
        
        self._reshape_data()
        self.split_data()
        #Select the expirment:
        if self.num_classes==3:
            self.three_classification()
            print("Three class expirment...")
        elif self.num_classes==2:
            self.binary_classification()
            print("binary classification expirment...")
        
        #Normalize label between 0,1.
        if self.normalize_labels :
            self.labels=self.labels/np.max(self.labels)

        print(self.features.shape)
        print(np.min(self.X[:,:,:,0]),np.max(self.X[:,:,:,0])) 
        print(np.unique( self.labels))
    def augment_data(self):
        augmented_data=np.empty_like(self.features)
        for i in range(0,len(self.features)):
            x,y=self.transform(self.features[i],self.labels[i])
            augmented_data[i]=x
        self.features=np.concatenate((self.features,augmented_data),axis=0)
        return augmented_data
    def _reshape_data(self):
        """
        reshape (8700,137,51,6 ) to (8700, 6,137,51) aagcn (0, 3, 1, 2)
        reshape (8700,137,51,6 ) to(8700,51,6,137) a3gcn(0, 2, 3, 1)
        """
       
        if self.reshape_data:
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
        print("assert label",np.unique(self.labels))
        return self.features.shape

    def __getitem__(self, index):
        #C,T,V,M
        x = self.features[index]
        y=self.labels[index]
        if self.transform:
            if self.concatenate:
                x_t,y_t=self.transform((x,y))
                x=np.stack((x_t,x),axis=0)
                y=np.stack((y_t,y),axis=0)
            else:
                x,y=self.transform((x,y))
       
        return x, y
    def apply_maxMinNorm(self,val,min_val,max_val):
         if max_val == min_val:
           
        # Handle the case when the denominator is zero (division by zero)
            return 0.0  # or any other appropriate value
      #  elif any(math.isnan(x) or math.isinf(x) for x in [val, min_val, max_val]):
            # Handle the case when any of the values is NaN or inf
       #     return 0.0  # or any other appropriate value
         else:
            # Perform the division if everything is valid
            return (2 * (val - min_val) / (max_val - min_val)) - 1
        #return np.where(max_val == min_val, 0.0, 2 * (val - min_val) / (max_val - min_val) - 1)
  
    def maxMinNorm(self):
        print("min_max X tensor", self.X.shape)
        for i in range(0,len(self.X)):
            sample = self.X[i]
            minX, maxX = np.min(sample[:,:,0]), np.max(sample[:,:,0])
            minY, maxY = np.min(sample[:,:,1]), np.max(sample[:,:,1])
            sample[:,:,0]= self.apply_maxMinNorm(sample[:,:,0],minX,maxX) 
            sample[:,:,1]=self.apply_maxMinNorm(sample[:,:,1],minY,maxY) 
            self.X[i]=sample
           
           
    def split_data(self):
        if self.idx_path!=None:
            idx=np.load(self.idx_path) 
            idx=np.array(idx,dtype=np.int32)
            self.features=self.features[idx]
            self.labels=self.labels[idx]
            for sample in self.features:
               
                min_s=np.min(sample[:,:,:])
                max_s=np.max(sample[:,:,:])
               
                if (min_s==max_s):
                    print ("found_zero_sample")
            """
            if self.min_max_values==None:
                self.min_max_values=[np.min(self.features[:,2,:,:,:]),
                            np.max(self.features[:,2,:,:,:]),
                            np.min(self.features[:,3,:,:,:]),
                            np.max(self.features[:,3,:,:,:]),
                            np.min(self.features[:,4,:,:,:]),
                            np.max(self.features[:,4,:,:,:]),
                            np.min(self.features[:,5,:,:,:]),
                            np.max(self.features[:,5,:,:,:])]
            print("max_min_used dynamics before norm:",self.min_max_values )
            
            
            if self.maxMinNormalization:
                for i in range(0,len(self.features)):
                    sample=self.features[i]
                    sample[2,:,:,:]= self.apply_maxMinNorm(sample[2,:,:,:],self.min_max_values[0],self.min_max_values[1]) 
                    sample[3,:,:,:]=self.apply_maxMinNorm(sample[3,:,:,:],self.min_max_values[2],self.min_max_values[3]) 
                    sample[4,:,:,:]=self.apply_maxMinNorm(sample[4,:,:,:],self.min_max_values[4],self.min_max_values[5]) 
                    sample[5,:,:,:]=self.apply_maxMinNorm(sample[5,:,:,:],self.min_max_values[6],self.min_max_values[7]) 
                    self.features[i]=sample
                print("max_min_dynamics after norm:",[np.min(self.features[:,2,:,:,:]),
                            np.max(self.features[:,2,:,:,:]),
                            np.min(self.features[:,3,:,:,:]),
                            np.max(self.features[:,3,:,:,:]),
                            np.min(self.features[:,4,:,:,:]),
                            np.max(self.features[:,4,:,:,:]),
                            np.min(self.features[:,5,:,:,:]),
                            np.max(self.features[:,5,:,:,:])])
                """
    def three_classification(self):
        values = [0, 2 , 4]
        indices = np.where(np.isin(self.labels, values))[0]
        self.labels=self.labels[indices]
        self.features=self.features[indices]
    def binary_classification(self):
        values = [0, 4]
        indices = np.where(np.isin(self.labels, values))[0]
        self.labels=self.labels[indices]
        self.features=self.features[indices]  
        self.labels=self.labels/4 


class Rotate(object):
    """
    Rotate sample of N frame of 2d points
    """

    def __init__(self, angles=[-5,5,-3,3]):
        self.angles=angles

    def __call__(self, sample):
        x,y = sample
        #print("rotate")
        C,T,V,M=x.shape
        angle=random.choice(self.angles)
        angle = math.radians(angle)
        Rotation=rotation_matrix_2d(angle)
        
        for i in range(0,T):
            landmarks= torch.tensor(x[:2,i,:,0])
            landmarks=landmarks.permute(1,0).contiguous().view(V,-1) # [51,2]
            rotated_l=np.dot(landmarks,Rotation.T)
            rotated_l=np.transpose(rotated_l, (1,0))
            x[:2,i,:,0]=np.reshape(rotated_l, (2,V))
        return x,y
        
class FlipV(object):
    """
    Flip landmarks with respect to vertical axis
    """
    def __init__(self,re_index=None):
        if re_index!=None:
            self.re_index=re_index
        if re_index==None:
            self.re_index= [9,8,7,6,5,4,3,2,1,0, #sopraciglia
            10,11,12,13,# vertical Nois
            18,17,16,15,14, #Bottom nois
            28,27,26,25,30,29, #left eye
            22,21,20,19,24,23,#Right eye
            37,36,35,34,33,32,31,# upper lip
            42,41,40,39,38, #down lip
            47,46,45,44,43,#up inside lip
            50,49,48] 
    def __call__(self, sample):
        x,y = sample #C,T,V,M
        x[0]=x[0][:,self.re_index,:]
        x[1]=x[1][:,self.re_index,:]
        x[2]=-x[2][:,self.re_index,:]
        x[3]=x[3][:,self.re_index,:]
        x[4]=-x[4][:,self.re_index,:]
        #print("Flip")
        return x,y
class TranslateX(object):
    def __init__(self,t=[-3,3]):
        self.t=t
    def __call__(self, sample):
        x,y = sample #C,T,V,M
        val=random.randint(self.t[0], self.t[1])
        x[0]=x[0]+val
        return x,y
class TranslateY(object):
    def __init__(self,t=[-3,3]):
        self.t=t
    def __call__(self, sample):
        x,y = sample #C,T,V,M
        val=random.randint(self.t[0], self.t[1])
        x[1]=x[1]+val
        return x,y
if __name__=="__main__":
    name_exp="open_face_PartB"
   
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
    num_classes=config['num_classes']
    N_sample=config['N_sample']
    TS=config['TS']
    #transform=RandomApply([RandomChoice([Rotate(),FlipV()])],p=0.5)
    transform=None
    train_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_train,
                             num_features= num_features,
                             num_nodes=num_nodes,
                             num_classes= num_classes,
                            N_sample=N_sample,
                            TS=TS,
                             transform=transform,
                             maxMinNormalization=True,
                             min_max_values=None)
    
    test_dataset=DataLoader(data_path,
                            labels_path,
                            edges_path,
                            idx_path=idx_test,
                            num_features= num_features,
                            num_nodes=num_nodes,
                            N_sample=N_sample,
                            TS=TS,
                            num_classes= num_classes,
                            transform=transform,
                            maxMinNormalization=True,
                            min_max_values=train_dataset.min_max_values
                            )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True,
                                                   drop_last=True)
    for sample in train_loader:
        x,y=sample
        print(x.shape,y.shape)
        break
# %%
