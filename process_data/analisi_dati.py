#%%
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
from sklearn.manifold import TSNE

parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from main import Trainer

path_save=parent_folder+"data/PartA/vis/"
name_file = 'open_face'
config_file=open(parent_folder+"config/"+name_file+".yml", 'r')
config = yaml.safe_load(config_file)
path_pretrained_model="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/1s+15k/best_model.pkl"
cdict = {0:'green', 1: 'blue', 2: 'gold', 3:'orange',4:'red'}

trainer=Trainer(config=config)
embeddings,classes,predicted=trainer.get_embedding(path_pretrained_model)
#%%
print(classes.shape,predicted.shape,embeddings.shape)
labels=np.stack((classes,predicted),axis=1)
print(labels.shape,embeddings.shape)

#%%

idx_4_0=np.where((labels[:,0]==4) & (labels[:,1]==0))[0] #rosso
idx_0_0=np.where((labels[:,0]==0) & (labels[:,1]==0))[0] #verde
labels_0=np.concatenate((idx_4_0,idx_0_0))

idx_0_4=np.where((labels[:,0]==0) & (labels[:,1]==4))[0] #rosso
idx_4_4=np.where((labels[:,0]==4) & (labels[:,1]==4))[0] #verde
labels_4=np.concatenate((idx_0_4,idx_4_4))

print(idx_0_4.shape,idx_4_0.shape,idx_0_0.shape,idx_4_4.shape,labels_0.shape,labels_4.shape)
#%%
filtred_idx=labels_4
classes_option=None

if classes_option=="all":
    filterd_classes=classes
    filtred_embedings=embeddings
else:
    #filtred_idx=np.where(classes=classes_option)
    
    filterd_classes=classes[filtred_idx]
    filtred_embedings=embeddings[filtred_idx]
    print(filterd_classes.shape,filtred_embedings.shape)
#%%
tsn_embedded = TSNE(n_components=2, 
                    learning_rate='auto',
                    init='random',
                    perplexity=20).fit_transform(filtred_embedings)
#%%

cdict = {0:'red',4:'green'}
vis="2d"
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
ax.set_title(f"Embeddings visualising predicted 4")
plt.show()
plt.savefig(path_save+"tsn_4_2d.png")


#%%
"""
g=2
fig, ax = plt.subplots()
ix = np.where(classes == g)
ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1], c = cdict[g], label = g)
ax.legend()
ax.set_title(f"Embeddings visualising ")
plt.show()

"""