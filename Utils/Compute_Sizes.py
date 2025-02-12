# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 20:15:10 2021

@author: jpeeples
"""
import numpy as np
from Utils.NLBP import NLBPLayer
from Utils.NEHD import NEHDLayer
import pdb

def get_feat_size(parameters, dataloaders, preprocess_layer = None, histogram_layer=None):
    
    #If base model, define histogram layer based on features
    if histogram_layer is None:
        
        if parameters['feature'] == 'LBP':
            
            histogram_layer = NLBPLayer(parameters['in_channels'], 
                                        P=parameters['P'], 
                                        R=parameters['R'], 
                                        window_size = parameters['window_size'],
                                        num_bins = parameters['numBins'],
                                        stride=parameters['stride'],
                                        normalize_count=parameters['normalize_count'],
                                        normalize_bins=parameters['normalize_bins'],
                                        LBP_init=parameters['feature_init'],
                                        learn_base = parameters['learn_transform'],
                                        normalize_kernel=parameters['normalize_kernel'],
                                        dilation=parameters['dilation'],
                                        aggregation_type=parameters['aggregation_type'])
            
            # Base model is global histogram layer for now (update later to local)
            out_size = parameters['out_channels'] * parameters['in_channels']
            # Converting input to single channel. Output will be only number of bins
            # May possibly compute LBP for each channel independently
            out_size = parameters['out_channels'] * parameters['in_channels'] ## May cause issues with conv, going to have to look into it
            
            return out_size
            
        #Update linear for dilation
        elif parameters['feature'] == 'EHD': 
            histogram_layer = NEHDLayer(parameters['in_channels'],
                                      parameters['window_size'],
                                      mask_size=parameters['mask_size'],
                                      num_bins=parameters['numBins'],
                                      stride=parameters['stride'],
                                      normalize_count=parameters['normalize_count'],
                                      normalize_bins=parameters['normalize_bins'],
                                      EHD_init=parameters['feature_init'],
                                      learn_no_edge=parameters['learn_transform'],
                                      threshold=parameters['threshold'],
                                      angle_res=parameters['angle_res'],
                                      normalize_kernel=parameters['normalize_kernel'],
                                      aggregation_type=parameters['aggregation_type'])
        
        elif parameters['feature'] in ["DSA","MSDCNN"]:
            out_size = 0
            return out_size
        else:
            raise RuntimeError('Invalid type for histogram layer')
        
    
    #Get single example of data to compute shape (use CPU instead of GPU)
    for idx, (inputs, labels, index) in enumerate(dataloaders['train']):
        

        if len(inputs.shape) < 4: #If no channel dimension (grayscale), add dimension
            inputs = inputs.unsqueeze(1)
        else:
            pass
        break
    
    #Forward pass to compute feature size
    feats = preprocess_layer(inputs[0].unsqueeze(0))
    feats = histogram_layer(feats)

    
    #Compute out size
    out_size = feats.flatten(1).shape[-1]
    
    return out_size

    
