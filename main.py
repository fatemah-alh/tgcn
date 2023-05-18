import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from model import TemporalGNN 
from dataloader import DataLoader
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm 

data_path="/home/falhamdoosh/tgcn/Painformer/dataset_data_biovid.npy"
labels_path="/home/falhamdoosh/tgcn/Painformer/dataset_label_biovid.pkl"
edges_path="/home/falhamdoosh/tgcn/data/edges_indx_dlib68.npy"
TS=137
loader = DataLoader(data_path,labels_path,edges_path)
dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
if torch.cuda.is_available():
  device = 'cuda'  
else:
  device='cpu'
print('Using %s'%(device))

# Create model and optimizers
model = TemporalGNN(node_features=4, periods=TS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()

print("Running training...")
for epoch in range(100): 
    loss = 0
    step = 0
    tq=tqdm(train_dataset)
    for snapshot in tqdm(tq):
        snapshot = snapshot.to(device)
        # Get model predictions

        y_hat = model(snapshot.x, snapshot.edge_index)
        # Mean squared error
        label = snapshot.y
       # label = label.contiguous().view(25, -1)
        loss = loss + torch.mean((y_hat-label)**2) 
        step += 1
        del snapshot
        if step>31:
            loss = loss / (step + 1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            tq.set_description("train: Loss batch , value loss: {l}".format(l=loss.item()))
            loss=0
            step=0
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))

#%%
model.eval()
print("Evaluation...")
loss = 0
step = 0
#horizon = 200

# Store for analysis
predictions = []
labels = []


inpSeq = []
pr = []
la = []
for snapshot in test_dataset:
    snapshot = snapshot.to(device)
    # Get predictions
    y_hat = model(snapshot.x, snapshot.edge_index)
    # Mean squared error
    label = snapshot.y
    #label = label.contiguous().view(25, -1)
    loss = loss + torch.mean((y_hat-label)**2)
    # Store for analysis below
    inpSeq.append(snapshot.x)
    labels.append(label)
    predictions.append(y_hat)

    pr.append(y_hat)
    la.append(snapshot.y)
    step += 1
    del snapshot
loss = loss / (step+1)
loss = loss.item()
print("Test MSE: {:.4f}".format(loss))
"""

#%%
# X coordinate prediction over time
joint = 3 #joint number for visualizzation 
ax = 0  #select the axis for visualizzation --> 0=x 1=y 2=z
SEQ = 10 #select the temporal sequence in the horizon (from 0 to 200)
#print(predictions[0].shape)
#print(labels[0].shape)

Xlabel = [] #Xlabels axis shifted after the input sequence
for i in range(TS):
  Xlabel.append(i+TS-1)


iSeq = np.asarray([iseq[joint][ax].detach().cpu().numpy() for iseq in inpSeq])
preds = np.asarray([pred[joint][ax].detach().cpu().numpy() for pred in pr])
labs  = np.asarray([label[joint][ax].cpu().numpy() for label in la])

print("Data points:,", preds.shape)
print(iSeq.shape)

"""