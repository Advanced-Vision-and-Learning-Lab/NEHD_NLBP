# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:59:35 2023
Implementation of histogram layer as LBP feature
@author: jpeeples
"""
import torch
import torch.nn as nn
from Utils.RBFHistogramPooling import HistogramLayer
import pdb

class NLBPLayer(nn.Module):
    def __init__(self,in_channels, R = 1, P = 8, aggregation_type = 'GAP',
                 window_size= [3,3], num_bins= 4,
                 stride= 1, padding= 0, normalize_count= False,  normalize_bins= True,
                 count_include_pad= False, ceil_mode= False, LBP_init= True, learn_kernel= True,
                 learn_base= True, learn_hist=True, normalize_kernel= False, padding_mode= 'reflect',
                 dilation= 1, threshold= nn.ReLU()): #Todo: Please make the threshold a string as opposed to function

        #Function: nn.Sigmoid()
        # inherit nn.module
        super(NLBPLayer, self).__init__()

        # define layer properties
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.window_size = window_size
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        self.learn_base = learn_base
        self.normalize_kernel = normalize_kernel
        self.dilation = dilation
        self.LBP_init = LBP_init
        self.learn_kernel = learn_kernel
        self.padding_mode = padding_mode
        self.threshold = threshold
        self.R = R
        self.P = P
        self.aggregation_type = aggregation_type
        self.learn_hist = learn_hist
        
        #Approach to have the same output of LBP with a lot of background information (e.g., FashionMNIST)
        # if self.LBP_init:
        #     self.threshold = nn.Sequential(nn.Threshold(0, 1),nn.ReLU())
        
        #Compute kernel size and out channels
        self.kernel_size = int(2*self.R + 1)
        self.out_channels = int(self.kernel_size**2 - 1)

        #Define convolution layer to select samples for test
        self.edge_responses = nn.Conv2d(self.in_channels, self.out_channels*self.in_channels,
                                          self.kernel_size, stride= self.stride,
                                          padding= 'same', # This variable is very important and changes the output shape but is not modular
                                          padding_mode= self.padding_mode, # This here is not modular but yields different values, worth investigating
                                          dilation= self.dilation, bias= False)
        
        #Intialize histogram layer for aggregation of bit maps
        self.histogram_layer = HistogramLayer(self.in_channels, self.window_size, dim=2,
                                              num_bins= self.numBins, stride=self.stride,
                                              padding= self.padding, normalize_count= self.normalize_count,
                                              normalize_bins= self.normalize_bins,
                                              count_include_pad= self.count_include_pad,
                                              ceil_mode= self.ceil_mode)

        # Set the vars that needed helper funcs
        self.weighted_sum = self.set_weighted_sum()

        self.histogram_layer = self.set_histogram_layer()
        
        if self.LBP_init:
            self.masks = self.handle_base_learning()[0]
            self.histogram_layer = self.handle_base_learning()[1]

        # Set the learning parameters for the kernel
        if self.learn_kernel:
            self.edge_responses.weight.requires_grad = True
        else:
            self.edge_responses.weight.requires_grad = False
            
        # Set the learning parameters for the histogram layer
        if self.learn_hist:
            self.histogram_layer.centers.requires_grad = True
            self.histogram_layer.widths.requires_grad = True
        else:
            self.histogram_layer.centers.requires_grad = False
            self.histogram_layer.widths.requires_grad = False

        #Set values for layers for analysis later            
        self.centers = self.histogram_layer.bin_centers_conv.bias
        self.widths = self.histogram_layer.bin_widths_conv.weight
        self.edge_kernels = self.edge_responses.weight
        
        

    # Helper functions
    def set_weighted_sum(self):
        if self.learn_base:
            self.weighted_sum = nn.Conv2d(self.in_channels*self.P, self.in_channels, kernel_size= 1,
                                        stride= self.stride, groups= self.in_channels, bias= False)
        else:
            self.weighted_sum = nn.Conv2d(self.in_channels*self.P, self.in_channels, 1,
                                            stride= self.stride, groups= self.in_channels, bias=False)
            self.weighted_sum.weight.requires_grad = False
        return self.weighted_sum
    
    def set_histogram_layer(self):
        #Perform global or local average pooling for feature
        if self.aggregation_type == 'GAP':
            self.histogram_layer.hist_pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.histogram_layer.hist_pool = nn.AvgPool2d(self.window_size,stride=self.stride,
                                             padding=self.padding,ceil_mode=self.ceil_mode,
                                             count_include_pad=self.count_include_pad)
        return self.histogram_layer
    
    def initialize_edge_kernels(self):
        # Set base weights to be powers of 2
        bases = 2 ** torch.arange(self.P)
        bases = bases.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bases = bases.repeat(self.in_channels, 1, 1, 1).float()
        self.weighted_sum.weight.data = bases

        # Set all values in kernel to zero
        self.edge_responses.weight.data.fill_(0)

        # Set center kernel value to 1
        self.edge_responses.weight.data[:, :, self.R, self.R] = torch.tensor(-1)

        # Get indices of weights surrounding center value
        index = torch.arange(0, self.kernel_size)
        neighbors = torch.cartesian_prod(index, index)

        # Remove center value from consideration
        center = neighbors.shape[0] // 2
        neighbors = torch.cat((neighbors[:center], neighbors[center + 1:]))

        # Expand neighbors to account for multidimensional inputs
        neighbors = neighbors.repeat(self.in_channels, 1)

        # Set neighbors to have weight of negative one for norm and one for cosine
        sim_space = torch.arange(0, self.out_channels * self.in_channels)
        self.edge_responses.weight.data[sim_space, :, neighbors[:, 0], neighbors[:, 1]] = 1

        # Keep weights fixed to keep meaning of feature, may allow update later (done outside layer)
        self.edge_responses.weight.requires_grad = False
        masks = self.edge_responses.weight

        # Repeat along channel dimension
        masks = masks.repeat(1, self.in_channels, 1, 1)

        return masks


    def initialize_histogram_bins(self): 
        # Calculate bin centers based on the number of bins
        bin_center_init = -torch.arange(0, 256, 256 / self.numBins) # This is the same as just 4 evenly spaced ones
        
        # Set the bias data of bin_centers_conv to the calculated bin centers
        self.histogram_layer.bin_centers_conv.bias.data = bin_center_init.repeat(self.in_channels)
        
        # Set the weight data of bin_widths_conv to have initial widths of 0.01 (wider bin width)
        self.histogram_layer.bin_widths_conv.weight.data = torch.ones(
            self.histogram_layer.bin_widths_conv.weight.shape
        )
        return self.histogram_layer



    def handle_base_learning(self):
        # Signature: returns the mask and sets the histogram layer
        masks = self.initialize_edge_kernels()
        histogram_bins = self.initialize_histogram_bins()
        return masks, histogram_bins

        
    def forward(self,xx):
        ## xx is the input and is a torch.tensor
        ##each element of output is the frequency for the bin for that window
        
        #Compute difference maps
        xx = self.edge_responses(xx)
        
        #Apply "threshold" function on difference maps to generate bit maps
        xx = self.threshold(xx)
        
        #Multiply differences by base (second convolution)
        xx = self.weighted_sum(xx)

        #Pass through histogram layer (binning)
        xx = self.histogram_layer(xx)
        
        return xx
      
