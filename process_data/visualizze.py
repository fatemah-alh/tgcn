
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import  ConfusionMatrixDisplay
import numpy as np
from tqdm import tqdm 
import yaml
from PIL import Image
import imageio
import sys
from scipy.spatial import Delaunay
import torch

parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)

from dataloader import DataLoader,Rotate,FlipV
from torchvision.transforms import RandomApply,RandomChoice,Compose
import wandb
from utiles import rotation_matrix_2d

def visualize_landmarks(data,label_data,edges=[],time_steps=137,vis_edges=False,vis="2d",vis_index=False,save=False,path_vis=None):
    """
    this function will visualize landmarks of data tensor, just the f
    """
    
    figures=[]
    origin=[0,0,0]
    for time_step in tqdm(range(time_steps)):
        # Create a figure and axes for subplots
        if vis=="2d":
            fig, axs = plt.subplots(5, 4, figsize=(30, 30))
        else:
            fig, axs = plt.subplots(5, 4, figsize=(30, 30),subplot_kw=dict(projection='3d'))
        
        # Flatten the axes array
        axs = axs.flatten()
        for d in range(0,20):
            fram=data[d,time_step,:,:]
            axs[d].set_title(f"VAS level: {label_data[d]}")
            if vis_index:
                for index in range(len(fram)):
                    axs[d].annotate(index,(fram[index][0],fram[index][1]))
            if vis=="2d":
                axs[d].scatter(fram[:,0], fram[:,1], alpha=0.8,s=1)
            else:
                axs[d].scatter(fram[:,0], fram[:,1],fram[:,2], alpha=0.8)
                #axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,0],autovettori[1,0],autovettori[2,0],color="r")
                #axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,1],autovettori[1,1],autovettori[2,1],color="g")
                #axs[d].quiver(origin[0],origin[1],origin[2], autovettori[0,2],autovettori[1,2],autovettori[2,2],color="b")
                #axs[d].quiver(origin[0],origin[1],origin[2], n[0],n[1],n[2],color="fuchsia")
                axs[d].quiver(origin[0],origin[1],origin[2],0,0,1,color="y")
                axs[d].text(0,0,1,"z")
                axs[d].quiver(origin[0],origin[1],origin[2], 0,1,0,color="y")
                axs[d].text(0,1,0,"y")
                axs[d].quiver(origin[0],origin[1],origin[2], 1,0,0,color="y")
                axs[d].text(1,0,0,"x")
                #axs[d].view_init(-30,60)
            
            if vis_edges:
                axs[d].plot([fram[edges[0],0],fram[edges[1],0]],[fram[edges[0],1],fram[edges[1],1]],"blue")
        
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        figures.append(image)
    if save:
        imageio.mimsave(path_vis+'vis_landmarks_3D_openface_absolut_eigenvectors_without_contur.gif', figures, duration=50)

def visualize_sample(data,label_data,edges=[],time_steps=137,vis_edges=False,vis="2d",vis_index=False,save=False,path_vis=None):
    figures=[]
    origin=[0,0,0]
    for time_step in tqdm(range(time_steps)):
        # Create a figure and axes for subplots
        fig = plt.figure()
        if vis=="2d":
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(projection='3d')
        # Flatten the axes array
        fram=data[time_step,:,:]
        print(fram.shape)
        ax.set_title(f"VAS level: {label_data}")
        if vis_index:
            for index in range(len(fram)):
                ax.annotate(index,(fram[index][0],fram[index][1]))
        if vis=="2d":
            ax.scatter(fram[:,0], fram[:,1], alpha=0.8,s=1)
        else:
            ax.scatter(fram[:,0], fram[:,1],fram[:,2], alpha=0.8)
            #ax.quiver(origin[0],origin[1],origin[2], autovettori[0,0],autovettori[1,0],autovettori[2,0],color="r")
            #ax.quiver(origin[0],origin[1],origin[2], autovettori[0,1],autovettori[1,1],autovettori[2,1],color="g")
            #ax.quiver(origin[0],origin[1],origin[2], autovettori[0,2],autovettori[1,2],autovettori[2,2],color="b")
            #ax.quiver(origin[0],origin[1],origin[2], n[0],n[1],n[2],color="fuchsia")
            ax.quiver(origin[0],origin[1],origin[2],0,0,1,color="y")
            ax.text(0,0,1,"z")
            ax.quiver(origin[0],origin[1],origin[2], 0,1,0,color="y")
            ax.text(0,1,0,"y")
            ax.quiver(origin[0],origin[1],origin[2], 1,0,0,color="y")
            ax.text(1,0,0,"x")
            #axs.view_init(-30,60)
        
        if vis_edges:
            ax.plot([fram[edges[0],0],fram[edges[1],0]],[fram[edges[0],1],fram[edges[1],1]],"blue")
        
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        figures.append(image)
    if save:
        imageio.mimsave(path_vis+'vis_landmarks_3D_openface_absolut_eigenvectors_without_contur.gif', figures, duration=50)


def visualize_cm(confusion_matrices,path_vis=None,time_steps=None):
    figures= [] # for storing the generated images
    for i,cm_k in enumerate(confusion_matrices):
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_k)
        disp.plot(ax=ax)
        ax.set_title("Epoche: {}".format(i))
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        figures.append(image)
        
    imageio.mimsave(path_vis+'cm.gif', figures, duration=100)

         
#%%
path=parent_folder+"/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/08-23-16:47/"
name_file = 'minidata'
config_file=open(parent_folder+"config/"+name_file+".yml", 'r')
config = yaml.safe_load(config_file)
data_path=parent_folder+config['data_path']
labels_path=parent_folder+config['labels_path']
edges_path=parent_folder+config['edges_path']
idx_train= parent_folder+config['idx_train']
idx_test=parent_folder+config['idx_test']
TS=config['TS']
#transform=RandomApply([RandomChoice([Rotate(),FlipV(),Compose([FlipV(),Rotate()])])],p=0.01)
transform=RandomApply([Rotate()],p=0)
#transform=RandomApply([FlipV(),Rotate()],p=0.5)
#transform=None
train_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_train,num_features=6,num_nodes=51,reshape_data=True,expand_dim=True,normalize_labels=False,transform=transform)
#test_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_test,mode="test")
#%%
data=np.zeros((20,6,137,51,1))
labels=[]

for i,sample in enumerate(train_dataset):
    data[i]=sample[0]
    labels.append(sample[1])
print(data.shape,len(labels))
#%%
reshaped_tensor = np.transpose(data, (0, 2, 3, 1,4))  # Transpose dimensions from 8700,137,469,4) to 8600, 469, 4, 137
features= torch.tensor(np.reshape(reshaped_tensor, (20, 137,51,6,1)))  
print(features.shape)
# Reshape to desired shape 
#%%
features=features.view(20, 137,51,-1) 
print(features.shape)  
#%%
import math
angle=10
angle = math.radians(angle)
Rotation=rotation_matrix_2d(angle)
rotated_sample=np.array(features[0])
rotated_sample[0,:,:2]=np.dot(rotated_sample[0,:,:2],Rotation.T)

#%%
edges=np.load(edges_path)
#%%
edges.shape
#%%
reindex= [9,8,7,6,5,4,3,2,1,0, #sopraciglia
            10,11,12,13,# vertical Nois
            18,17,16,15,14, #Bottom nois
            28,27,26,25,30,29, #left eye
            22,21,20,19,24,23,#Right eye
            37,36,35,34,33,32,31,# upper lip
            42,41,40,39,38, #down lip
            47,46,45,44,43,#up inside lip
            50,49,48] 
#%%
print(rotated_sample[0,0,0],rotated_sample[0,9,0])
#%%

rotated_sample=rotated_sample[:,reindex,:]
#%%
visualize_landmarks(features[:,:,:,:2],labels,edges,time_steps=1,vis_index=True,vis_edges=False)
#visualize_sample(rotated_sample,labels[0],edges,time_steps=1,vis_index=True,vis_edges=True)

#%%
import math
sample=features[0,0,:,:2]
#%%
print(sample.shape)
angle=10
angle = math.radians(angle)
Rotation=rotation_matrix_2d(angle)
rotated=np.dot(sample,Rotation.T)
print(rotated.shape)
#%%
#%%
data[0,0,:,:2]=rotated
#%%
features[0,0,:,0]=-features[0,0,:,0]
#%%
visualize_sample(features[0],labels[0],edges,time_steps=1,vis_index=True,vis_edges=True)
# %%

wandb.init()
def visualize_one_cm(cm,path_vis=None,time_steps=None,title="Confusion_matrix"):

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    ax.set_title(title)
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    image_wand = wandb.Image(image, caption=title)
    wandb.log({title: image_wand})

 #%%   
path="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/08-23-20:18/"
confusion_matrices=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/08-23-20:18/cm.npy")
#%%
confusion_matrices[70]
#%%

cm =np.array([[  0, 198,  92,  79,  23],
                  [  0, 205,  83,  85,  14],
                  [  0, 161,  96, 108,  29],
                  [  0, 134,  85, 120,  39],
                  [  0,  82,  75, 124,  82]])


visualize_one_cm(cm,title="128,2gru,15+7")

# %%
