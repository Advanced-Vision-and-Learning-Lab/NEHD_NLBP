# -*- coding: utf-8 -*-
"""
Baseline LBP feature as Pytorch layer
@author: luke saleh
"""
import torch
import torch.types
import torch.nn as nn
import numpy as np
from skimage.feature import local_binary_pattern
import dask
import warnings
    
class LocalBinaryLayer(nn.Module):
    def __init__(self, in_channels, radius = 1, n_points = 8, method = 'default',
                 num_bins = 8, density = True, return_encoding = False, device = 'cpu'):
        super(LocalBinaryLayer, self).__init__()
        self.in_channels = in_channels
        self.radius = radius
        self.method = method
        self.n_points = n_points
        self.num_bins = num_bins
        self.density = density
        self.LBP_max = 2 ** (self.n_points) - 1
        self.device = device
        self.return_encoding = return_encoding
    
    @dask.delayed
    def LBP(self,x):
        feature = []
        for channel in range(0,x.shape[0]):
            feature.append(local_binary_pattern(x[channel], self.n_points, self.radius, method = self.method))
        feature = np.stack(feature,axis=0)
        return feature
    
    @dask.delayed
    def histogram(self,x):
        
        if self.return_encoding:
            return x
        else:
            hist = []
            for channel in range(0,x.shape[0]):
                temp_hist = np.histogram(x[channel],bins=self.num_bins,density=self.density,
                                    range=(0,self.LBP_max))
                hist.append(temp_hist[0])
            hist = np.stack(hist, axis=0)
            
        return hist
        
    def forward(self,x):
       
        #Convert representation to integer to disable warning (look into disabling warning)
        x_npy = x.cpu().numpy()
        
        batch, channel, dimx, dimy = x.shape
        lbp_newarr = np.empty([batch,channel,dimx,dimy])
        lbp_list = []
        
        #Compute LBP bit map and histogram representation
        # Works currently for single channel (multiple channels will need additional for loop)
        for sample in range(0,batch):
            lbp_temparr = self.LBP(x_npy[sample])
            lbp_temparr = self.histogram(lbp_temparr)
            lbp_list.append(lbp_temparr)
            
        lbp_list = dask.compute(*lbp_list)
        
        lbp_newarr = np.asarray(lbp_list)
        lbp = torch.from_numpy(lbp_newarr)
        
        #Change array from N x C x B to N x (CB) (match neural feature)
        #This will need to be updated if doing reconstruction
        if not(self.return_encoding):
            lbp = torch.flatten(lbp,start_dim=1)
            
        lbp = lbp.type(torch.float32).to(x.device)
        
        return lbp
    

