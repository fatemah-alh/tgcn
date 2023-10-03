#%%
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
from sklearn.manifold import TSNE
parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from main import Trainer
name_file = 'open_face'
config_file=open(parent_folder+"config/"+name_file+".yml", 'r')
config = yaml.safe_load(config_file)

def get_embeddings(config,path_pretrained_model):
    trainer=Trainer(config=config)
    return trainer.get_embedding(path_pretrained_model)

def get_TSNE(embeddings,
             classes,
             predicted,
             predicted_label=4,
             vis="2d",
             title="Embeddings visualising predicted 4",
             path_save=parent_folder+"data/PartA/vis/"):
    
    labels=np.stack((classes,predicted),axis=1)
    pain_class=1 #4
    
    if predicted_label==4:
        idx_0_4=np.where((labels[:,0]==0) & (labels[:,1]==pain_class))[0] #Red
        idx_4_4=np.where((labels[:,0]==pain_class) & (labels[:,1]==pain_class))[0] #Green
        labels_4=np.concatenate((idx_0_4,idx_4_4))
        filtred_idx=labels_4
        filterd_classes=classes[filtred_idx]
        filtred_embedings=embeddings[filtred_idx]
        cdict = {0:'red',pain_class:'green'}
    elif predicted_label==0:
        idx_4_0=np.where((labels[:,0]==pain_class) & (labels[:,1]==0))[0] #Red
        idx_0_0=np.where((labels[:,0]==0) & (labels[:,1]==0))[0] #Green
        labels_0=np.concatenate((idx_4_0,idx_0_0))
        filtred_idx=labels_0
        filterd_classes=classes[filtred_idx]
        filtred_embedings=embeddings[filtred_idx]
        cdict = {0:'green',pain_class:'red'}

    elif predicted_label=="all":
        filterd_classes=classes
        filtred_embedings=embeddings
        cdict = {0:'green', 1: 'blue', 2: 'gold', 3:'orange',4:'red'}
    elif predicted_label=="true":
        idx_0_0=np.where((labels[:,0]==0) & (labels[:,1]==0))[0] #Green
        idx_4_4=np.where((labels[:,0]==pain_class) & (labels[:,1]==pain_class))[0] #Green
        labels_true=np.concatenate((idx_0_0,idx_4_4))
        filtred_idx=labels_true
        filterd_classes=classes[filtred_idx]
        filtred_embedings=embeddings[filtred_idx]
        cdict = {0:'blue',pain_class:'orange'}
    print(classes.shape,predicted.shape,labels.shape,embeddings.shape)
    #print(idx_0_4.shape,idx_4_0.shape,idx_0_0.shape,idx_4_4.shape,labels_0.shape,labels_4.shape)
    print(filterd_classes.shape,filtred_embedings.shape)

    tsn_embedded = TSNE(n_components=2, 
                        learning_rate='auto',
                        init='random',
                        perplexity=30).fit_transform(filtred_embedings)
    

    
    if vis=="2d":
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    for g in np.unique(filterd_classes):
        ix = np.where(filterd_classes == g)
        if vis=="2d":
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1], c = cdict[g], label = g,s=1)
        else:
            ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1],tsn_embedded[ix,2], c = cdict[g], label = g,s=1)
        # ax.view_init(-30,60)

    ax.legend()
    ax.set_title(title)
    plt.show()
    plt.savefig(path_save+title+".png")


def get_adaptive_matrix():
    pass

#%%
embeddings,classes,predicted,initial_label= get_embeddings(config,
#path_pretrained_model="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/1s+15k+filterlow_react+binary_class_regression/best_model.pkl"
path_pretrained_model="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/1s+15k+filterlow_react+binary_class_regression+adaptiv/best_model.pkl"
   )
#%%
adaptive=np.load("/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/process_data/Adaptive_matrix.npy")

#%%
adaptive.shape
#%%
get_TSNE(embeddings,
         classes,
         predicted,
         predicted_label="true",
         vis="2d",
         title="Embeddings visualising predicted true - Daptive model",
         path_save=parent_folder+"./data/PartA/vis/")

# %%
def get_ids(labels,predicted=4,pain_class=1):
    idx_0_4=np.where((labels[:,0]==0) & (labels[:,1]==pain_class))[0]
    idx_4_0=np.where((labels[:,0]==pain_class) & (labels[:,1]==0))[0] #Red
   
    pass