## PyTorch dependencies
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

#External libaries
import pdb


class Linear_Model(nn.Module):
    
    def __init__(self,num_ftrs,num_classes,device, feature_layer,reconstruction=False,
                aggregation_type='GAP', preprocess_layer = None):
        
        #inherit nn.module
        super(Linear_Model,self).__init__()

        self.num_ftrs = num_ftrs
        self.num_classes = num_classes
        self.reconstruction = reconstruction
        self.device = device
        self.feature_layer = feature_layer
        self.aggregation_type = aggregation_type
        
        if reconstruction:
            self.fc = torch.nn.Sequential()
        else:
            self.fc = nn.Linear(num_ftrs, num_classes)


        ###
        if preprocess_layer is None:
            self.preprocess_layer = torch.nn.Sequential()
        else:
            self.preprocess_layer = preprocess_layer
        ###
        
        
    def forward(self,x):
        # Preprocess
        x = self.preprocess_layer(x)

        #Extract features from base feature (aggregation completed in feature layer
        #and pass to fully connected layer
        feats = self.feature_layer(x)
        
        #if reconstructon experiments, do not flatten tensor
        #if classification, flatten tensor
        if self.reconstruction:
            pass
        else:
            x = torch.flatten(feats,start_dim=1)

        output = self.fc(x)
 
        return feats, output
    

        
        
        
        
        
        