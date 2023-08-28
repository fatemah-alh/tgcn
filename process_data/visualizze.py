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

from dataloader import DataLoader
import wandb
#%%

#from dataloader import DataLoader

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
                axs[d].scatter(fram[:,0], fram[:,1], alpha=0.8)
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
        ax.set_title(f"VAS level: {label_data}")
        if vis_index:
            for index in range(len(fram)):
                ax.annotate(index,(fram[index][0],fram[index][1]))
        if vis=="2d":
            ax.scatter(fram[:,0], fram[:,1], alpha=0.8)
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
train_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_train,reshape_data=False,expand_dim=False,normalize_labels=False)
#test_dataset=DataLoader(data_path,labels_path,edges_path,idx_path=idx_test,mode="test")
data=np.zeros((20,137,43,6))
labels=[]
edges=np.load(edges_path)
for i,sample in enumerate(train_dataset):
    data[i]=np.concatenate((sample[0][:,:10,:],sample[0][:,19:]),axis=1)
    
    labels.append(sample[1])
print(data.shape,len(labels))

#%%
#visualize_landmarks(data,labels,edges,time_steps=2,vis_index=False,vis_edges=True)
# %%
def get_edges(landmarks):
    tri = Delaunay(landmarks[:,:2])
    edges = []
    for simplex in tri.simplices:
        for i in range(3):
            edges.append((simplex[i], simplex[(i+1)%3]))
    edges_index=[[],[]]
    for e in edges:
        edges_index[0].append(e[0])
        edges_index[1].append(e[1])
    edges_index=torch.LongTensor(edges_index)
    
    print("number of index before adding symmetric edges:",edges_index.shape)
    print(len(edges_index[0]))
    
    print("number of index after adding symmetric edges:",edges_index.shape)
    np.save("FAL_edges.npy",edges_index)
    return edges_index
edges=get_edges(data[0][0,:,:2],)
visualize_sample(data[0],labels[0],edges,time_steps=1,vis_index=True,vis_edges=True)
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
