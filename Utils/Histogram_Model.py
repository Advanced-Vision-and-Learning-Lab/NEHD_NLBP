## PyTorch dependencies
import torch.nn as nn
import torch
import pdb

class HistogramNetwork(nn.Module):
    def __init__(self,histogram_layer,num_ftrs,num_classes,reconstruction=False,preprocess_layer = None):
        
        #inherit nn.module
        super(HistogramNetwork,self).__init__()
        self.num_ftrs = num_ftrs
        self.num_classes = num_classes
        self.reconstruction = reconstruction
        self.in_channels = histogram_layer.in_channels

        ###
        if preprocess_layer is None:
            self.preprocess_layer = torch.nn.Sequential()
        else:
            self.preprocess_layer = preprocess_layer
        ###
            
        
        #Define histogram layer and fc
        self.histogram_layer = histogram_layer
        
        if reconstruction:
            self.fc = torch.nn.Sequential()
        else:
            self.fc = nn.Linear(num_ftrs, num_classes)
        
        
    def forward(self,x):
        # Preprocess
        x = self.preprocess_layer(x)
        #Extract features from histogram layer and pass to fully connected layer
        feats = self.histogram_layer(x)

        #if reconstructon experiments, do not flatten tensor
        #else classification, flatten tensor
        if self.reconstruction:
            pass
        else:
            x = torch.flatten(feats,start_dim=1)
    
        output = self.fc(x)
 
        return feats, output
    

        
        
        
        
        
        