# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:26:08 2019

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import scipy.stats

## PyTorch dependencies
import torch
import torch.nn as nn
from math import floor, log
import torch.nn.functional as F
from torchvision import models
from torchvision.utils import make_grid

## Local external libraries
from Utils.Histogram_Model import HistogramNetwork
from Utils.Base_Model import Linear_Model
from barbar import Bar
from Utils.Compute_EHD import EHD_Layer
from Utils.NLBP import NLBPLayer
from Utils.Compute_LBP import LocalBinaryLayer
from Utils.Compute_Sizes import get_feat_size
from Utils.Generate_Plots import *
from .pytorchtools import EarlyStopping
import pdb

from Utils.DSAnet import NMNet
from Utils.MSDCNN import MSDCNN

plt.ioff()

def train_model(model, dataloaders, criterion, optimizer, device,parameters, 
                          split,saved_bins=None, saved_widths=None,histogram=True,
                          num_epochs=25,scheduler=None,num_params=0):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    train_error_history = []
    val_error_history = []
    epoch_weights = []
    best_epoch = 0
    early_stopping = EarlyStopping(patience=20, verbose=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    valid_loss = best_loss
    
    #Generate masks
    feature_masks = EHD_Layer.Generate_masks(mask_size=parameters['mask_size'],
                                    angle_res=parameters['angle_res'],
                                    normalize=parameters['normalize_kernel'])
   
    #Plot initial bin centers and widths
    if saved_bins is not None:
        plot_histogram(saved_bins[0,:],saved_widths[0,:],-1,'train',parameters,split)
        
    #Generate feature layers for reconstruction
    if parameters['reconstruction']:
        if parameters['feature'] == 'EHD': 
           feature_layer = EHD_Layer(parameters['in_channels'], parameters['angle_res'], 
                                     parameters['normalize_kernel'],
                                     parameters['dilation'], parameters['threshold'], 
                                     parameters['window_size'], parameters['stride'], 
                                     parameters['normalize_count'],
                                     parameters['aggregation_type'], 
                                     kernel_size= parameters['mask_size'],device = device)
        elif parameters['feature'] == 'LBP':
            feature_layer = LocalBinaryLayer(parameters['in_channels'],
                                             radius=parameters['R'],
                                            n_points = parameters['P'],
                                            method = parameters['LBP_method'],
                                            num_bins = parameters['numBins'],
                                            density = parameters['normalize_count'],
                                            device = device)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and testidation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode 
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idx, (inputs, labels, index) in enumerate(Bar(dataloaders[phase])):
                
                if len(inputs.shape) < 4: #If no channel dimension (grayscale), add dimension
                    inputs = inputs.to(device).unsqueeze(1)
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                        
                    # Get model outputs and calculate loss
                    feats, outputs = model(inputs)
    
                    if parameters['reconstruction']:
                        # pdb.set_trace()
                    
                        feature_outputs = feature_layer(inputs)

                        #If NLBP (local), take average over spatial to get vector
                        if parameters['aggregation_type'] == 'Local':
                            feats = nn.AdaptiveAvgPool2d(1)(feats)
                       
                        feats = torch.flatten(feats,start_dim=1) 
                        loss = criterion(feats,feature_outputs)
                    else:
                        loss = criterion(outputs,labels.long())
                        _, preds = torch.max(outputs, 1)
                 
                    if torch.isnan(loss):
                      pdb.set_trace()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if num_params==0:
                            pass
                        else:
                            loss.backward()
                            optimizer.step()
                            
                            if scheduler is not None:
                                scheduler.step()
                                
            
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                if not(parameters['reconstruction']):
                    running_corrects += torch.sum(preds == labels.data)
        
            epoch_loss = running_loss / (len(dataloaders[phase].sampler))
            if parameters['reconstruction']:
                epoch_acc = 0.0
            else:
                epoch_acc = running_corrects.float() / (len(dataloaders[phase].sampler))
            
            if phase == 'train':
                train_error_history.append(epoch_loss)
                
                train_acc_history.append(epoch_acc)
                
                if(histogram):
                    #save bins and widths
                    saved_bins[epoch+1,:] = model.neural_feature.histogram_layer.centers.detach().cpu().numpy()
                    saved_widths[epoch+1,:] = model.neural_feature.histogram_layer.widths.reshape(-1).detach().cpu().numpy()
                    if parameters['aggregation_type'] == 'GAP':
                        epoch_weights.append(model.neural_feature.edge_kernels.data)
                        plot_kernels(feature_masks,model.neural_feature.edge_kernels.data,phase,
                                 epoch,parameters,split)
                    else:
                        epoch_weights.append(model.neural_feature.edge_kernels.data)
                        plot_kernels(feature_masks,model.neural_feature.edge_kernels.data,phase,
                                 epoch,parameters,split)
                    plot_histogram(saved_bins[epoch+1,:],saved_widths[epoch+1,:],
                                epoch,phase,parameters,split)
            

            # deep copy the model
            if parameters['reconstruction']:
                if phase == 'val' and epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    valid_loss = best_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                if phase == 'val' and epoch_acc > best_acc:
                    best_epoch = epoch
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    valid_loss = best_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_error_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                
            print()
            if parameters['reconstruction']:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            else:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print()
            
        #Check validation loss
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print()
            print("Early stopping")
            print()
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    if parameters['reconstruction']:
        print('Best val loss: {:4f}'.format(best_loss))
    else:
        print('Best val Acc: {:4f}'.format(best_acc))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    #Returning error (unhashable), need to fix
    train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                  'val_error_track': val_error_history,'train_acc_track': train_acc_history, 
                  'train_error_track': train_error_history,'best_epoch': best_epoch, 
                  'saved_bins': saved_bins, 'saved_widths': saved_widths,
                  'epoch_weights': epoch_weights}
    
    return train_dict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
          
def test_model(dataloader,model,device,parameters,split):
    
    #Initialize and accumalate ground truth, predictions, and image indices
    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    running_loss = 0
    test_acc = 0
    model.eval()
    
    #Generate feature layers for reconstruction
    if parameters['reconstruction']:
        if parameters['feature'] == 'EHD': 
           feature_layer = EHD_Layer(parameters['in_channels'], parameters['angle_res'], 
                                     parameters['normalize_kernel'],
                                     parameters['dilation'], parameters['threshold'], 
                                     parameters['window_size'], parameters['stride'], 
                                     parameters['normalize_count'],
                                     parameters['aggregation_type'], 
                                     kernel_size= parameters['mask_size'],device = device)
        elif parameters['feature'] == 'LBP':
            feature_layer = LocalBinaryLayer(parameters['in_channels'],
                                             radius=parameters['R'],
                                            n_points = parameters['P'],
                                            method = parameters['LBP_method'],
                                            num_bins = parameters['numBins'],
                                            density = parameters['normalize_count'],
                                            device = device)
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels,index) in enumerate(Bar(dataloader)):
            
            if len(inputs.shape) < 4: #If no channel dimension (grayscale), add dimension
                inputs = inputs.to(device).unsqueeze(1)
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
          
            # forward
            feats, outputs = model(inputs)
            
            if parameters['reconstruction']:
              if parameters['aggregation_type'] == 'Local':
                feats = nn.AdaptiveAvgPool2d(1)(feats)
               
              feats = torch.flatten(feats,start_dim=1) 
              feature_outputs = feature_layer(inputs)
              loss = F.mse_loss(feats, feature_outputs)
               
            else:
                _, preds = torch.max(outputs, 1)
        
                #If test, accumulate labels for confusion matrix
                GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
                Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
               
            if parameters['reconstruction']:
                running_loss += loss.item() * inputs.size(0)
            else:
                running_corrects += torch.sum(preds == labels.data)
                test_acc = running_corrects.float() / (len(dataloader.sampler))

        
    test_loss = running_loss / len(dataloader.sampler)
   
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], 'Index':Index,
                    'test_acc': np.round(test_acc.cpu().numpy()*100,2),
                    'test_loss': np.round(test_loss,2)}
    
    return test_dict
       
def initialize_model(parameters,dataloaders_dict,device,num_classes, in_channels,
                     reconstruction=True, histogram_layer=None,fusion_method=None): 
    
    if fusion_method == "conv":
        preprocess_layer = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        in_channels = 1
    else:
        preprocess_layer = nn.Sequential()

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    #Compute size out of output layer
    num_ftrs = get_feat_size(parameters, dataloaders_dict, preprocess_layer= preprocess_layer,
                             histogram_layer=histogram_layer)

    if histogram_layer is not None and parameters["feature"] in ["EHD", "LBP"]:

        model_ft = HistogramNetwork(histogram_layer,num_ftrs,num_classes,
                                    reconstruction=reconstruction, 
                                    preprocess_layer=preprocess_layer)
    
    elif parameters['feature'] == 'DSA':
        model_ft = NMNet(num_channels=in_channels, img_size=dataloaders_dict['img_size'], num_classes=num_classes)
        
    elif parameters['feature'] == 'MSDCNN':
        model_ft = MSDCNN(num_channels=in_channels, img_size=dataloaders_dict['img_size'], num_classes=num_classes)
    # Baseline model
    else:
        if parameters['feature'] == 'EHD': 
           feature_layer = EHD_Layer(in_channels, parameters['angle_res'], 
                                     parameters['normalize_kernel'],
                                     parameters['dilation'], parameters['threshold'], 
                                     parameters['window_size'], parameters['stride'], 
                                     parameters['normalize_count'],
                                     parameters['aggregation_type'], 
                                     kernel_size= parameters['mask_size'],device = device)
        elif parameters['feature'] == 'LBP':
            feature_layer = LocalBinaryLayer(in_channels,radius=parameters['R'],
                                            n_points = parameters['P'],
                                            method = parameters['LBP_method'],
                                            num_bins = parameters['numBins'],
                                            density = parameters['normalize_count'],
                                            device = device)

        model_ft = Linear_Model(num_ftrs,num_classes,device, feature_layer,reconstruction=reconstruction,
                       aggregation_type=parameters['aggregation_type'], preprocess_layer=preprocess_layer)
    
    return model_ft
