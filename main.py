import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from model import TemporalGNN 
from dataloader import DataLoader
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm 
import torch.optim.lr_scheduler as lr_scheduler
import datetime
import os
class Trainer():
    def __init__(self,gpu,
                 data_loader_train,
                 data_loader_test,
                 model,loss,
                 weight_decay=0.5,
                 step_decay=20,
                 num_epochs=20,
                 lr=0.001) -> None:
        self.model=model
        #self.batch_size=batch_size
        self.gpu=gpu
        self.data_loader_train=data_loader_train
        self.data_loader_test=data_loader_test
        self.loss=loss
        self.lr=lr
        self.step_decay=step_decay
        self.weight_decay=weight_decay
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr,betas=(0.9, 0.999))
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,self.step_decay, self.weight_decay)
        self.num_epochs=num_epochs
        self.date=datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.log_dir=os.path.join("/log/",self.date)
        os.makedirs(self.log_dir)
        if torch.cuda.is_available():
            print("set cuda device")
            self.device="cuda"
        else:
            self.device="cpu"
            print('Warning: Using CPU')
        self.model.to(self.device)
        self.loss.to(self.device)
    def train(self):
        print("Running training...")
        avg_loss = 0
        loss = 0
        step = 0
        self.model.train()
        tq=tqdm(self.data_loader_train)
        for num_index,snapshot in enumerate(tq):
            snapshot = snapshot.to(self.device)
            y_hat = model(snapshot.x, snapshot.edge_index)
            label = snapshot.y   
            loss += self.loss(y_hat,label)
            tq.set_description("train: y hat and label  {} {}".format(y_hat,label))
            avg_loss += loss.item()
            del snapshot
            if step>31:
                loss = loss / (step + 1)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss=0
                step=0
                tq.set_description("train: Loss batch , value loss: {l}".format(l=loss.item()))
            step += 1
        return avg_loss/(num_index+1)

    def eval(self):
        model.eval()
        print("Evaluation...")
        loss = 0
        step = 0
        # Store for analysis
        predictions = []
        labels = []


        inpSeq = []
        pr = []
        la = []
        tq=tqdm(test_dataset)
        for num_index,snapshot in enumerate(tq):
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
            tq.set_description("Test Loss  , value loss: {}".format(torch.mean((y_hat-label)**2)))
                
            pr.append(y_hat)
            la.append(snapshot.y)
            step += 1
            del snapshot
        loss = loss / (step+1)
        loss = loss.item()
        print("Test MSE: {:.4f}".format(loss))
        return 
    
    def run(self):
        previous_best_avg_test_acc = 0.0
        previous_best_avg_loss=1000000
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            f.write("parameter of model:\nDataset:{data} N:{n},num of layer:{layer},CREATE_ZERO_MANIP_ONLY :{crea},MOVE_ZERO_MANIP_LAST:{move_zer},VAL_ORIGINAL:{val_orig},MODEL_EVAL:{model_eval}, Eval_variable_legnth:{eval_varia},Train_variable_legnth:{train_var},\n num_epoch:{epoch} ,lr:{lr},step_decay:{s},weight_decay:{dec},cont_training:{cont},pretrainde_model:{pretraind},".format(pretraind= par.pretrain_model,cont=par.contin_training,layer=par.NUM_LAYER,
                     n=par.N,crea=par.CREATE_ZERO_MANIP_ONLY,data=par.name_data_set ,move_zer=par.MOVE_ZERO_MANIP_LAST,
                     epoch=par.NUM_EPOCH,lr=par.LR,
                     val_orig=par.VAL_ORIGINAL,model_eval=par.MODEL_EVAL,
                     eval_varia=par.Eval_variable_legnth,train_var=par.Train_variable_legnth,
                     s=par.step_decay,dec=par.weight_decay))
            
        for epoch in range(self.num_epochs):
            avg_train_loss=self.train()
            if (epoch%5==0):
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "ckpt_%d.pkl" % (epoch + 1)))
                print('Saved new checkpoint  ckpt_{epoch}.pkl , avr loss {l}'.format( epoch=epoch+1, l=avg_train_loss))
            avg_test_loss= self.eval()
            result="Epoch {e}, Train_loss: {l}, test_acc:{a} ,lr:{lr}\n".format(lr= self.lr_scheduler.get_lr(),e=epoch + 1,l=avg_train_loss,a=avg_test_acc)
            with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
                f.write(result)
            print(result)

        
            if avg_test_loss < previous_best_avg_loss:
                with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
                    f.write("saved_a new best_model\n ")
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best_model.pkl"))
                print('Best model in {dir}/best_model.pkl'.format(dir=self.log_dir))
                #previous_best_avg_test_acc = avg_test_acc
                previous_best_avg_loss=avg_test_loss

            self.lr_scheduler.step()


torch.manual_seed(100)
TS=137
lr=0.001
data_path="/home/falhamdoosh/tgcn/Painformer/dataset_data_biovid.npy"
labels_path="/home/falhamdoosh/tgcn/Painformer/dataset_label_biovid.pkl"
edges_path="/home/falhamdoosh/tgcn/data/edges_indx_dlib68.npy"

loader = DataLoader(data_path,labels_path,edges_path)
dataset = loader.get_dataset()
train_dataset,test_dataset = temporal_signal_split(dataset, train_ratio=0.7)

# Create model and optimizers
model = TemporalGNN(node_features=4, periods=TS)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss=torch.nn.MSELoss()
if(contin_training):
        path_pretrained_model=os.path.join(par.LOG_DIR,"{pretrain_model}/best_model.pkl".format(pretrain_model=par.pretrain_model))
        model.load_state_dict(torch.load(path_pretrained_model))
        print("Pre trained model is loaded...")
trainer=Trainer(gpu=1,
                data_loader_train=train_dataset, 
                data_loader_test=test_dataset,
                loss=loss,
                model=model,
                num_epochs=par.NUM_EPOCH,
                lr=par.LR)
trainer.run()
#%%
"""
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