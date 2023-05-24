import torch
import torch.nn.functional as F
import numpy as np
from model import TemporalGNNBatch
from dataloader import DataLoader
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm 
import torch.optim.lr_scheduler as lr_scheduler
import datetime
import os
import yaml
from tensorboardX import SummaryWriter


class Trainer():
    def __init__(self, config) -> None:
        self.config=config
        self.lr=config['lr']
        self.LOG_DIR=config['LOG_DIR']
        self.num_epoch=config['num_epoch']
        self.gpu=config['gpu']
        self.weight_decay=config['weight_decay']
        self.step_decay=config['step_decay']
        self.name_exp=config['name_exp']
        #model parameters
        self.num_features=config['num_features']
        self.embed_dim=config['embed_dim']
        self.output_features=config['output_features']
        self.TS=config['TS']
        self.continue_training=config['continue_training']
        self.pretrain_model=config['pretrain_model']
        #data set parametrs 
        self.data_path=config['data_path']
        self.labels_path=config['labels_path']
        self.edges_path=config['edges_path']
        self.idx_train=config['idx_train']
        self.idx_test=config['idx_test']
        self.train_ratio=config['train_ratio']
        self.batch_size=config['batch_size']
        self.num_nodes=config['n_joints']
        self.set_log_dir()
        self.set_device()
        self.load_datasets()
        self.load_model()
        self.load_optimizer()
        self.load_loss()
        self.init_writer()

    def set_log_dir(self):
        self.date=datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.log_dir=os.path.join(self.LOG_DIR,self.date)
        os.makedirs(self.log_dir,exist_ok=True)
    def set_device(self):
        if torch.cuda.is_available():
            print("set cuda device")
            self.device="cuda"
            torch.cuda.set_device(self.gpu)
        else:
            self.device="cpu"
            print('Warning: Using CPU')
    def init_writer(self):
        self.writer = SummaryWriter('./writer/' + self.name_exp+self.date)
    def load_datasets(self):
        self.train_dataset=DataLoader(self.data_path,self.labels_path,self.edges_path,self.name_exp,idx_path=self.idx_train,mode="train")
        self.test_dataset=DataLoader(self.data_path,self.labels_path,self.edges_path,self.name_exp,idx_path=self.idx_test,mode="test")
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                   batch_size=self.batch_size, 
                                                   shuffle=True,
                                                   drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                                                   batch_size=self.batch_size, 
                                                   shuffle=False,
                                                   sampler=torch.utils.data.SequentialSampler(self.test_dataset),
                                                   drop_last=False)
    def load_model(self):
        self.model = TemporalGNNBatch(node_features=self.num_features,
                                      num_nodes=self.num_nodes,
                                      embed_dim=self.embed_dim, 
                                      periods=self.TS,
                                      batch_size=self.batch_size)
        
        if(self.continue_training):
                path_pretrained_model=os.path.join(self.LOG_DIR,"{}/best_model.pkl".format(self.pretrain_model))
                self.model.load_state_dict(torch.load(path_pretrained_model))
                print("Pre trained model is loaded...")
        print(self.model.parameters)
        self.model.to(self.device)

    def load_optimizer(self):
        #self.optimizer = torch.optim.SGD(list(self.model.parameters()),lr = self.lr,momentum = 0.9)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr)
        for var_name in self.optimizer.state_dict():
            print(var_name, '\t', self.optimizer.state_dict()[var_name])
      
       # self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,self.step_decay, self.weight_decay)
    def load_loss(self):
        self.loss=torch.nn.MSELoss().to(self.device)
        #self.loss=torch.nn.L1Loss().to(self.device)
        #torch.mean((y_hat-label)**2)
    def train(self):
        avg_loss = 0.0
        self.model.train()
        tq=tqdm(self.train_loader)
        for i,snapshot in enumerate(tq):
            x,y,edge_index,edg_attr=snapshot
            #Move tensors to device
            x=x.to(self.device)
            label = y.to(self.device) 
            edge_index=edge_index.to(self.device)
            edg_attr=edg_attr.to(self.device)
            #forward
            y_hat = self.model(x, edge_index[0],edg_attr[0])
            loss=self.loss(y_hat,label.float())
            #calc gradient and backpropagation
            loss.backward()
            self.optimizer.step()
            #check gradient, if some tensors has been detached from CG
            for name, param in self.model.named_parameters():
               #print(name, param.grad)
               if param.grad==None:
                   raise RuntimeError(f"Gradient is None {name}" )
            self.optimizer.zero_grad()
            for j in range(len(y_hat)):
                self.writer.add_scalars('Training y-hat and label', {"y_hat":y_hat[j],"label":label[j]}, (i*x.shape[0])+j)
            self.writer.add_scalars('train loss batch', {"loss":loss}, i)
            tq.set_description("train: Loss batch: {}".format(loss))
            avg_loss += loss
        return avg_loss/(i+1)

    def eval(self):
        self.model.eval()
        avg_loss = 0.0
        tq=tqdm(self.test_loader)
        with torch.no_grad():
            for i,snapshot in enumerate(tq):
                x,y,edge_index,edg_attr=snapshot
                x=x.to(self.device)
                edge_index=edge_index.to(self.device)
                label = y.to(self.device) 
                edg_attr=edg_attr.to(self.device) 
                y_hat = self.model(x, edge_index[0],edg_attr[0]) 
                loss=self.loss(y_hat,label.float())
                avg_loss += loss
                for j in range(len(y_hat)):
                    self.writer.add_scalars('Eval y_hat,label', {"y_hat":y_hat[j],"label":label[j]}, (i*x.shape[0])+j)
                self.writer.add_scalars('Eval loss batch', {"loss":loss}, i)
                tq.set_description("Test Loss batch: {}".format(loss))
               
        avg_loss = avg_loss / (i+1)
        return avg_loss
    
    def run(self):
        #previous_best_avg_test_acc = 0.0
        previous_best_avg_loss=1000000
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            str_par=""
            for i in config.keys():
                str_par+=i +" : "+ str(config[i])+"\n"
                
            f.write(" Parametrs:\n {}".format(str_par))
            
        for epoch in range(self.num_epoch):
            avg_train_loss=self.train()
        
            if (epoch%5==0):
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "ckpt_%d.pkl" % (epoch + 1)))
                print('Saved new checkpoint  ckpt_{epoch}.pkl , avr loss {l}'.format( epoch=epoch+1, l=avg_train_loss))
          
            avg_test_loss= self.eval()
            self.writer.add_scalars("Loss training and evaluating",{'train_loss': avg_train_loss,'eval_loss': avg_test_loss}, epoch)

            result="Epoch {}, Train_loss: {} , eval loss {}  \n".format(epoch + 1,avg_train_loss,avg_test_loss)
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

            #self.lr_scheduler.step()

if __name__=="__main__":
    torch.manual_seed(100)

    #name_exp = 'mediapipe'
    name_exp = 'dlib'
    config_file=open("./config/"+name_exp+".yml", 'r')
    config = yaml.safe_load(config_file)

    trainer=Trainer(config=config)
    trainer.run()