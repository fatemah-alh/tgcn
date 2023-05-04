#%%
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
print ("hello world")

import torch
from IPython.display import clear_output
import os, sys
pt_version = torch.__version__
print(pt_version)
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

import os
import urllib
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

#%%

TS = 10
class SkeletonDataLoader(object):


    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data")):
        super(SkeletonDataLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self._read_data()

    def _read_data(self):
        A = np.load("/content/drive/Shareddrives/GeometricLearning/Datasets/AD.npy")
        X = np.load("/content/drive/Shareddrives/GeometricLearning/Datasets/AllDSSkeleton.npy").transpose(
            (1, 2, 0)
        )
        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = TS, num_timesteps_out: int = TS):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting joints position using num_timesteps_in to predict the
        join position in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, :, i + num_timesteps_in : j]).numpy()) #[:,0,i+num...:j] se voglio predirre solo la x

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = TS, num_timesteps_out: int = TS
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for NTU-RGBD dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* 
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset

loader = SkeletonDataLoader()
dataset = loader.get_dataset(num_timesteps_in=TS, num_timesteps_out=TS)

#%%
next(iter(dataset))
#%%
from torch_geometric_temporal.signal import temporal_signal_split
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.node_features = node_features
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=32, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods*3)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

TemporalGNN(node_features=3, periods=TS)

#%%
# GPU support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using %s'%(device))

subset = 2000
from tqdm import tqdm 
# Create model and optimizers
model = TemporalGNN(node_features=3, periods=TS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

print("Running training...")
for epoch in range(20): 
    loss = 0
    step = 0
    for snapshot in tqdm(train_dataset):
        snapshot = snapshot.to(device)
        # Get model predictions

        y_hat = model(snapshot.x, snapshot.edge_index)
        # Mean squared error
        label = snapshot.y
        label = label.contiguous().view(25, -1)
        loss = loss + torch.mean((y_hat-label)**2) 
        step += 1
        if step > subset:
          break

    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))

#%%
model.eval()
loss = 0
step = 0
horizon = 200

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
    label = label.contiguous().view(25, -1)
    loss = loss + torch.mean((y_hat-label)**2)
    # Store for analysis below
    inpSeq.append(snapshot.x)
    labels.append(label)
    predictions.append(y_hat)

    pr.append(y_hat.view(25,3,TS))
    la.append(snapshot.y)
    step += 1
    if step > horizon:
          break

loss = loss / (step+1)
loss = loss.item()
print("Test MSE: {:.4f}".format(loss))

#%%
import numpy as np
# X coordinate prediction over time
joint = 3 #joint number for visualizzation 
ax = 1  #select the axis for visualizzation --> 0=x 1=y 2=z
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

#%%
