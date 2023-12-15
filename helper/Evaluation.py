import torch
import numpy as np 
from helper.Config import Config
from sklearn.metrics import confusion_matrix
from torcheval.metrics.functional import multiclass_f1_score



class Evaluation:
    def __init__(self, config:Config) -> None:
        self.config=config

    def calc_errors(self,targets,unrounded_predicted):
        mse_loss=torch.nn.MSELoss()
        mea_loss=torch.nn.L1Loss(reduction="mean")
        mse_err =mse_loss(targets,unrounded_predicted)
        mea_err=mea_loss(targets,unrounded_predicted)
        rmse_err= torch.sqrt(mse_err)
        return mse_err.item(),rmse_err,mea_err
    def round_values(self,values,normalized_labels,max_classes):
        if normalized_labels:
            values=[x * max_classes  for x in values]
        values=torch.round(values)
        values=torch.clamp(values, min=0, max=max_classes)
        return values
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


    
    
    
   
        
    


