# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:20:07 2021
Implementation of histogram layer with only convolutional layer and global average pooling
@author: jpeeples
"""

import torch
import torch.nn as nn
from Utils.Compute_EHD import EHD_Layer
from Utils.RBFHistogramPooling import HistogramLayer
import numpy as np
import pdb

class NEHDLayer(nn.Module):
    def __init__(self,in_channels,window_size=[3,3],aggregation_type = 'GAP',
                 mask_size=3,num_bins=4, stride=1,padding=0,normalize_count=True,normalize_bins = True,
                 count_include_pad=False,ceil_mode=False,EHD_init=True,
                 learn_no_edge=True,threshold=.9,angle_res=45,normalize_kernel=False,
                 dilation=1):

        # inherit nn.module
        super(NEHDLayer, self).__init__()

        # define layer properties
        # histogram bin data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.window_size = window_size
        self.mask_size = mask_size
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        self.learn_no_edge= learn_no_edge
        self.threshold = threshold
        self.angle_res = angle_res
        self.normalize_kernel = normalize_kernel
        self.dilation = dilation
        self.aggregation_type = aggregation_type
        self.EHD_init = EHD_init
        
        #For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
       
        # Image Data, only dimension implemented with sobel
        #Learn no edge through convolution filter
        if self.learn_no_edge:
            if self.in_channels == 1:
                self.edge_responses = nn.Conv2d(self.in_channels,self.numBins+1,
                                                  self.mask_size,groups=self.in_channels,
                                                  bias=False,dilation=self.dilation)

            else:
                self.edge_responses = nn.Conv2d(self.in_channels,(self.numBins+1)*self.in_channels,
                                                  self.mask_size,groups=self.in_channels,
                                                  bias=False,dilation=self.dilation)
            bin_number = self.edge_responses.out_channels 
        #Concatenate binary no edge/edge map
        else:
            if self.in_channels == 1: #Gray scale (independently)
                self.edge_responses = nn.Conv2d(self.in_channels,self.numBins,
                                                  self.mask_size,groups=self.in_channels,
                                                  bias=False,dilation=self.dilation)
                

            else: #Multichannel, aggregate
                self.edge_responses = nn.Conv2d(self.in_channels,self.numBins*self.in_channels,
                                                  self.mask_size,groups=self.in_channels,
                                                  bias=False,dilation=self.dilation)
                
            bin_number = self.edge_responses.out_channels + self.in_channels    
            
        #Intialize histogram layer for aggregation of edge responses
        self.histogram_layer = HistogramLayer(self.edge_responses.out_channels, self.mask_size, dim=2,
                                              num_bins= (self.numBins + 1), stride=self.stride,
                                              padding= self.padding, normalize_count= self.normalize_count,
                                              normalize_bins= self.normalize_bins,
                                              count_include_pad= self.count_include_pad,
                                              ceil_mode= self.ceil_mode)
        
        #Change histogram layer convolution to apply bin width and center to each individual feature map
        self.histogram_layer.bin_centers_conv = nn.Conv2d(bin_number,bin_number,1,
                                        groups=bin_number,bias=True)
        self.histogram_layer.bin_centers_conv.weight.data.fill_(1)
        self.histogram_layer.bin_centers_conv.weight.requires_grad = False
        self.histogram_layer.bin_widths_conv = nn.Conv2d(bin_number,
                                         bin_number,1, groups=bin_number,
                                         bias=False)
          
        #Initialize edge kernels to correct values and intial bins to 0 and widths to 1
        if self.EHD_init:
            masks = EHD_Layer.Generate_masks(mask_size=self.mask_size,angle_res=self.angle_res,
                                   normalize= self.normalize_kernel,rotate=False)
            masks = torch.tensor(masks).float().unsqueeze(1)
            
            #Repeat along channel dimension
            masks = masks.repeat(1,self.in_channels,1,1)
            
            self.bin_init = 1
            self.bin_init = sum(2*self.bin_init*torch.topk(torch.flatten(masks[0,0]),self.mask_size[0])[0])
            masks.requires_grad = True
        
            if self.in_channels == 1: #Gray
                self.edge_responses.weight.data[0:self.numBins] = masks
            else:
                
                #Generate indices for multichannel (treat each channel independently)
                if self.learn_no_edge:
                    indices = np.arange(0,self.in_channels*(self.numBins+1))
                    indices = indices.reshape(-1,self.numBins+1)[:,1:].flatten() - 1
                else:
                    indices = np.arange(0,self.in_channels*(self.numBins))
                
                #Split into equal parts to match corresponding kernels
                indices = np.array_split(indices,self.in_channels)
                
               
                for channel in range(0,self.in_channels):
                    self.edge_responses.weight.data[indices[channel]]= masks[:,channel].unsqueeze(1)
          
            self.histogram_layer.bin_centers_conv.bias.data.fill_(-self.bin_init.clone().detach().requires_grad_(True))
            self.histogram_layer.bin_centers_conv.weight.data.fill_(1)
            self.histogram_layer.bin_centers_conv.weight.requires_grad = False
            self.histogram_layer.bin_widths_conv.weight.data.fill_(1)
     
        #Set values for analysis later
        self.centers = self.histogram_layer.bin_centers_conv.bias
        self.widths = self.histogram_layer.bin_widths_conv.weight
        self.edge_kernels = self.edge_responses.weight
        
        #Change aggregation type for histogram layer
        self.histogram_layer = self.set_histogram_layer()
     
    def set_histogram_layer(self):
        #Perform global or local average pooling for feature
        if self.aggregation_type == 'GAP':
            self.histogram_layer.hist_pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.histogram_layer.hist_pool = nn.AvgPool2d(self.window_size,stride=self.stride,
                                             padding=self.padding,ceil_mode=self.ceil_mode,
                                             count_include_pad=self.count_include_pad)
        return self.histogram_layer
    
    def forward(self,xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window
        #Learn sobel responses
        xx = self.edge_responses(xx)
        
        if not(self.learn_no_edge):
            xx_no_edge = self.get_no_edge(xx)
            xx = torch.cat((xx,xx_no_edge),dim=1)
        
        #Pass through histogram layer (binning)
        xx = self.histogram_layer(xx)
      
        return xx
    
    #Will need to change for multi-channel, similar to constrain bins
    def get_no_edge(self,xx):
        # This function has been modified up due to an issue
        # When there is one function it does not work since
        # xx.shape = torch.Size([1, 8, 110, 110])
        # self.get_no_edge(xx).shape = torch.Size([1, 1, 1, 110, 110])
        # Therefore:
        # xx = torch.cat((xx,xx_no_edge),dim=1) produces an error saying function mismatch
        # One change was the [value, index] definition


        #Reshape tensor to be N x C x B x M x N 
        n,c,h,w = xx.size()
        xx_max = xx.reshape(n, c//self.numBins, self.numBins, h, w)
        
        #Find max response for each channel dimension
        if self.in_channels == 1:
            [value,index] = torch.max(xx,dim=1)
        else:
            [value,index] = torch.max(xx_max,dim=2)
            
        #Set edge responses to "no edge" if not larger than threshold
        if self.in_channels == 1:
            xx_no_edge = (value<self.threshold).float().unsqueeze(1)
        else:
            xx_no_edge = (value<self.threshold).float()
       
        return xx_no_edge
    
    
 
    
    
        
        
        
        
        
    
