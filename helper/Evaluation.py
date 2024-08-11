import torch
import numpy as np 
from helper.Config import Config
from sklearn.metrics import confusion_matrix
from torcheval.metrics.functional import multiclass_f1_score
import re


class Evaluation:
    def __init__(self, config:Config) -> None:
        self.config=config

    def calc_errors(self,targets,unrounded_predicted,max_classes):
        mse_loss=torch.nn.MSELoss()
        mea_loss=torch.nn.L1Loss(reduction="mean")
        unrounded_predicted=torch.clamp(unrounded_predicted, min=0, max=max_classes)
        mse_err =mse_loss(targets,unrounded_predicted)
        mea_err=mea_loss(targets,unrounded_predicted)
        rmse_err= torch.sqrt(mse_err)
        return mse_err.item(),rmse_err,mea_err
    def round_values(self,values,normalized_labels,max_classes):
        if normalized_labels:
            values=[x * max_classes  for x in values]
        if max_classes>1:
            values= torch.round(values) #torch.ceil(values)
        else:
            values=self.custom_round(values)
        values=torch.clamp(values, min=0, max=max_classes)
        return values
    def custom_round(self, x):
        return torch.floor(x + 0.5)

    def calc_acc(self,targets,unrounded_predicted,classes,normalized_labels=False):
        max_classes=np.max(classes)
        predicted= self.round_values(unrounded_predicted,normalized_labels,max_classes)
        targets=self.round_values(targets,normalized_labels,max_classes)
        cm=confusion_matrix( targets,predicted,labels=classes)
        #if  micro is like accuracy? 
        #if macro is the mean 
        #Nel caso Multi it's same as accuracy
        #f1_micro=multiclass_f1_score(predicted,targets,num_classes=len(classes),average="micro")
        #The precision is the ratio tp / (tp + fp)
        #The recall is the ratio tp / (tp + fn)
        acc_class=100*cm.diagonal()/cm.sum(1)
        print("acc_class",acc_class)
        #Accuracy is calculated as the number of correct predictions 
        #divided by the total number of predictions made by the model.
        acc= 100*cm.diagonal().sum()/np.sum(cm)
        
        p,r,f1_macro=self.get_precision_recall(cm)
    
        return acc,f1_macro,p,r,cm,acc_class

    
    def get_precision_recall(self,cm):
        #Precision is calculated as the number of true positives divided by the total number of
        #positive predictions made by the model.
        # The F1 score is a measure of a model’s accuracy that takes both precision and 
        # recall into account. It is the harmonic mean of precision and recall.
        true_pos = np.diag(cm)
        false_pos = np.sum(cm, axis=0) #sum of rows
        false_neg = np.sum(cm, axis=1)#sum of col

        precision=np.zeros(cm.shape[0])
        recall=np.zeros(cm.shape[0])
        f1=np.zeros(cm.shape[0]) #calculate precision and recall for each classe and then take the mean
        for i in range(0,len(true_pos)):
            if false_pos[i]!=0:
                precision[i]=100*(true_pos[i]/false_pos[i])
            if false_neg[i]!=0:
                recall[i]=100*(true_pos[i]/false_neg[i])
            if recall[i]!=0 and precision[i]!=0:
                f1[i]=(2*recall[i]*precision[i])/(recall[i]+precision[i])
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1=np.mean(f1)
        return precision,recall,f1
    def calc_accuracy_loso(self,path):
        pass
        

    def extract_loso_results(self,path="/andromeda/shared/reco-pomigliano/tempo-gnn/tgcn/log/loso_ME87_new/log_loso_ME87_multi.txt"):
        # Initialize an empty dictionary to store the data
        data_dict = {}
        # Regular expression pattern to match the subject, accuracy, and loss
        pattern = r"Subject:\s*(\d+)\s*,\s*acc:\s*([0-9.]+)\s*,\s*f1:\s*([0-9.]+)\s*,\s*rmse:\s*([0-9.]+)\s*,\s*mae:\s*([0-9.]+)"
        #pattern = r"Subject:\s*(\d+)\s*,\s*acc:\s*([0-9.]+)\s*,\s*rmse:\s*([0-9.]+)\s*"

        # Read the file (replace 'your_file.txt' with the actual file name)
        with open(path, 'r') as file:
            for line in file:
                # Search for the pattern in each line
                match = re.search(pattern, line)
                if match:
                    # Extract subject, accuracy, and loss
                    subject = int(match.group(1))
                    acc = float(match.group(2))
                    f1 = float(match.group(3))
                    rmse= float(match.group(4))
                    mae = float(match.group(5))
                    
                    # Store the extracted values in the dictionary
                    data_dict[subject] = {'acc': acc, 'rmse':rmse,'f1': f1,'mae':mae}
                    
        acc=[]
        rmse=[]
        mae=[]
        f1=[]
        subject=[]
        for i in data_dict.keys():
            acc.append(data_dict[i]["acc"])
            rmse.append(data_dict[i]['rmse'])
            mae.append(data_dict[i]['mae'])
            f1.append(data_dict[i]['f1'])
            subject.append(i)
        print(np.mean(acc),np.mean(rmse),np.mean(f1),np.mean(mae))
        return np.mean(acc),np.mean(rmse),np.mean(f1),np.mean(mae)

            
    
   
        
    


