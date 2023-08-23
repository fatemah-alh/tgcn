#%%
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
from sklearn.manifold import TSNE

parent_folder= "/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/"
sys.path.append(parent_folder)
from main import Trainer
path=parent_folder+"data/PartA/vis/"
name_file = 'open_face'
config_file=open(parent_folder+"config/"+name_file+".yml", 'r')
config = yaml.safe_load(config_file)
#%%
config
#%%
trainer=Trainer(config=config)
path_pretrained_model="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/08-16-15:53/best_model.pkl"
trainer.calc_accuracy(path_pretrained_model)
#%%
embeddings,classes,predicted=trainer.get_embedding(path_pretrained_model)
#%%
embeddings.shape
#%%

#%%
idx_0_4=np.where((classes==4 )| (classes==0))

#%%
tsn_embedded = TSNE(n_components=3, learning_rate='auto',init='random', perplexity=30).fit_transform(embeddings[idx_0_4])

cdict = {0:'green', 1: 'blue', 2: 'gold', 3:'orange',4:'red'}
#%%

idx_o_4_classes=classes[idx_0_4]
#%%
vis="3d"
if vis=="2d":
    fig, ax = plt.subplots()
else:
    fig, ax = plt.subplots(figsize=(30, 30),subplot_kw=dict(projection='3d'))

for g in np.unique(idx_o_4_classes):
    ix = np.where(idx_o_4_classes == g)
    if vis=="2d":
        ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1], c = cdict[g], label = g)
    else:
        ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1],tsn_embedded[ix,2], c = cdict[g], label = g)
       # ax.view_init(-30,60)

ax.legend()
ax.set_title(f"Embeddings visualising ")
plt.show()
#%%
g=2
fig, ax = plt.subplots()
ix = np.where(classes == g)
ax.scatter(tsn_embedded[ix,0], tsn_embedded[ix,1], c = cdict[g], label = g)
ax.legend()
ax.set_title(f"Embeddings visualising ")
plt.show()
#plt.savefig(path+"tsn.png")

# %%
