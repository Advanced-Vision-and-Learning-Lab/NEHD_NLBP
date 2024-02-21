# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 20:49:45 2023
Function to plot results during training, validation and testing
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

#Pytorch libraries
import torch.nn as nn
import torch

## Local external libraries
from Utils.Save_Results import generate_filename

import pdb

def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_histogram(centers,widths,epoch,phase,Network_parameters,split):
    
    # Plot some images
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    plt.subplots_adjust(right=.75)
    angles = np.arange(0,360,Network_parameters['angle_res'])
    
    num_bins = len(centers)
    handles = []
    for temp_ang in range(0,num_bins):
        
        toy_data = np.linspace(centers[temp_ang] - 6*abs(widths[temp_ang]),
                              centers[temp_ang] + 6*abs(widths[temp_ang]), 300)
        y = scipy.stats.norm.pdf(toy_data,centers[temp_ang],abs(widths[temp_ang]))
        y = y/y.max()
        plt.plot(toy_data,y)
        handles.append('Bin {}: \u03BC = {:.2f}, \u03C3 = {:.2f}'.format(temp_ang+1,centers[temp_ang],widths[temp_ang]))
    plt.legend(handles,bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.suptitle(('{}-Bin Histogram for {} '+
                  'Feature Maps Epoch {}').format(num_bins,len(angles),epoch+1))

    filename = generate_filename(Network_parameters,split)
    filename = filename + phase + '/' + 'Histograms/'
    
    if not os.path.exists(filename):
        os.makedirs(filename)
    
    try:
        if epoch is not None:
            plt.suptitle('Epoch {} during {} phase'.format(epoch+1,phase))
            plt.savefig(filename+'Epoch_{}_Phase_{}.png'.format(epoch+1,phase),dpi=fig.dpi)
        else:
            plt.suptitle('Best Epoch for {} phase'.format(phase))
            plt.savefig(filename+'Best_Epoch_Phase_{}.png'.format(phase),dpi=fig.dpi)
    except:
        pass
    plt.close(fig=fig)

def plot_kernels(EHD_masks,Hist_masks,phase,epoch,Network_parameters,split):
    
    # Plot some images
    fig, ax = plt.subplots(3,Hist_masks.size(0),figsize=(24,12))
    fig.subplots_adjust(hspace=.4,wspace=.4)
    angles = np.arange(0,360,Network_parameters['angle_res'])
    plot_min = np.amin(EHD_masks)
    if np.amin(Hist_masks.detach().cpu().numpy()) < plot_min:
        plot_min = np.amin(Hist_masks.detach().cpu().numpy())
    plot_max = np.amax(EHD_masks)
    if np.amax(Hist_masks.detach().cpu().numpy()) > plot_max:
        plot_max = np.amax(Hist_masks.detach().cpu().numpy())
    
    #Remove extra dimension on histogram masks tensor
    Hist_masks = Hist_masks.squeeze(1)
    num_orientations = Hist_masks.size(0)//Network_parameters['in_channels'] 
    
    #^^ Ax size is 8 when running debug (previously had -1)
    for temp_ang in range(0, num_orientations):
        
        if temp_ang == (num_orientations-1):
             ax[0,temp_ang].set_title('No Edge (N/A)')
             ax[1,temp_ang].set_title('Bin {}'.format(temp_ang+1))
             ax[2,temp_ang].set_title('No Edge')
             ax[0,temp_ang].set_yticks([])
             ax[1,temp_ang].set_yticks([])
             ax[2,temp_ang].set_yticks([])
             ax[0,temp_ang].set_xticks([])
             ax[1,temp_ang].set_xticks([])
             ax[2,temp_ang].set_xticks([])
             im = ax[0,temp_ang].imshow(np.zeros(EHD_masks[temp_ang-1].shape))
             im2 = ax[1,temp_ang].imshow(Hist_masks[temp_ang].detach().cpu().numpy())
             im3 = ax[2,temp_ang].imshow(abs(Hist_masks[temp_ang].detach().cpu().numpy()-
                                            np.zeros(EHD_masks[temp_ang-1].shape))) 
        else:
            ax[0,temp_ang].set_title(str(angles[temp_ang])+u'\N{DEGREE SIGN}')
            ax[1,temp_ang].set_title('Bin {}'.format(temp_ang+1))
            ax[2,temp_ang].set_title(str(angles[temp_ang])+u"\N{DEGREE SIGN}")
            ax[0,temp_ang].set_yticks([])
            ax[1,temp_ang].set_yticks([])
            ax[2,temp_ang].set_yticks([])
            ax[0,temp_ang].set_xticks([])
            ax[1,temp_ang].set_xticks([])
            ax[2,temp_ang].set_xticks([])
            im = ax[0,temp_ang].imshow(EHD_masks[temp_ang])
            im2 = ax[1,temp_ang].imshow(Hist_masks[temp_ang].detach().cpu().numpy())
            diff = abs(Hist_masks[temp_ang].detach().cpu().numpy()-EHD_masks[temp_ang])
            im3 = ax[2,temp_ang].imshow(diff)
            
        plt.colorbar(im,ax=ax[0,temp_ang],fraction=0.046, pad=0.04)
        plt.colorbar(im2,ax=ax[1,temp_ang],fraction=0.046, pad=0.04)
        plt.colorbar(im3,ax=ax[2,temp_ang],fraction=0.046,pad=0.04)
   
    
    ax[0,0].set_ylabel('EHD True Masks',rotation=90,size='small')
    ax[1,0].set_ylabel('Hist Layer Masks',rotation=90,size='small')
    ax[2,0].set_ylabel('Absolute Difference',rotation=90,size='small')
    plt.tight_layout()

    filename = generate_filename(Network_parameters,split)
    filename = filename + phase + '/' + 'Masks/'
    
    if not os.path.exists(filename):
        os.makedirs(filename)
    
    try:
        if epoch is not None:
            plt.suptitle('Epoch {} during {} phase'.format(epoch+1,phase))
            plt.savefig(filename +'Epoch_{}_Phase_{}.png'.format(epoch+1,phase),dpi=fig.dpi)
        else:
            plt.suptitle('Best Epoch for {} phase'.format(phase))
            plt.savefig(filename +'Best_Epoch_Phase_{}.png'.format(phase),dpi=fig.dpi)
    except:
        pass
    plt.close(fig=fig)
    
def plot_FMS_GAP_EHD(images,EHD_outputs,outputs,phase,epoch,parameters,split,
             img_max=5):
    
    #Take difference between estimated and true outputs 
    diff_outputs = abs((EHD_outputs-outputs)).detach().cpu().numpy()
    img_count = np.arange(0,img_max,1)
    
    #Change to pink for data
    if images.shape[1] == 3:
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        cmap = None
    else:
        mean = (.5,)
        std = (.5,)
        cmap = 'gray'
        
    angles = np.arange(0,360,parameters['angle_res'])
    angle_names = []
    bin_names = []
    bin_count = 0
    for angle in angles:
        angle_names.append(u'{}\N{DEGREE SIGN}'.format(angle))
        bin_names.append('Bin {}'.format(bin_count+1))
        bin_count += 1
    angle_names.append('No Edge')
    bin_names.append('Bin {}'.format(bin_count+1))
    
    for img in img_count:
        fig, ax = plt.subplots(1,4,figsize=(12,6))
        plt.subplots_adjust(wspace=.4,hspace=.4)

        #Change to pink for SAS data
        if images.shape[1] == 3:
            temp_img = inverse_normalize(images[img],mean=mean,std=std).permute(1,2,0).detach().cpu().numpy()
        else:
            temp_img = inverse_normalize(images[img],mean=mean,std=std).permute(1,2,0)[:,:,0].detach().cpu().numpy()
            
        im = ax[0].imshow(temp_img,cmap=cmap,aspect="auto")
        ax[0].set_box_aspect(1)
        
    
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        
        #Compute avg count of feature values
        y_pos = np.arange(len(angle_names))
        rects = ax[1].bar(y_pos,EHD_outputs[img,:,0,0].detach().cpu().numpy())
        ax[1].set_box_aspect(1)
        ax[1].set_ylabel('Avg Feature Count')
        ax[1].set_title('{}'.format(parameters['feature']))
        
        rects = ax[2].bar(y_pos,outputs[img,:,0,0].detach().cpu().numpy())
        ax[2].set_box_aspect(1)
        ax[2].set_ylabel('Avg Feature Count')
        ax[2].set_title('N{}'.format(parameters['feature']))
        
        rects = ax[3].bar(y_pos,diff_outputs[img,:,0,0])
        ax[3].set_box_aspect(1)
        ax[3].set_ylabel('Avg Feature Count')
        ax[3].set_title('Absolute Differences')
       
        
        plt.tight_layout()
    
        filename = generate_filename(parameters,split)
        filename = filename + phase + '/'
        
        if not os.path.exists(filename):
            try:
                os.makedirs(filename)
            except:
                pass
        
        if epoch is not None:
            plt.suptitle('Epoch {} during {} phase'.format(epoch+1,phase))
            try:
                fig.savefig(filename+'Image_{}_Epoch_{}_Phase_{}.png'.format(img,epoch+1,phase),dpi=fig.dpi)
            except:
                pass
        else:
            plt.suptitle('Best Epoch for {} phase'.format(phase))
            try:
                plt.savefig(filename+'Image_{}_Best_Epoch_Phase_{}.png'.format(img,phase),dpi=fig.dpi)
            except:
                pass
        plt.close(fig=fig)

def plot_FMS_EHD(images,EHD_outputs,outputs,phase,epoch,parameters,split,
             img_max=5):
    
    #Take difference between estimated and true outputs 
    diff_outputs = (EHD_outputs-outputs).detach().cpu().numpy()
    
    plot_min = np.amin(EHD_outputs.detach().cpu().numpy())
    if np.amin(outputs.detach().cpu().numpy()) < plot_min:
        plot_min = np.amin(outputs.detach().cpu().numpy())
    plot_max = np.amax(EHD_outputs.detach().cpu().numpy())
    if np.amax(outputs.detach().cpu().numpy()) > plot_max:
        plot_max = np.amax(outputs.detach().cpu().numpy())
        
    if images.shape[1] == 3:
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        cmap = None
    else:
        mean = (.5,)
        std = (.5,)
        cmap = 'gray'
    
    # Plot some images
    img_count = np.arange(0,img_max,1)
    for img in img_count:
        fig, ax = plt.subplots(3,EHD_outputs.size(1)+1,figsize=(24,12))
        plt.subplots_adjust(wspace=.4,hspace=.4)
        angles = np.arange(0,360, parameters['angle_res'])
       
        if images.shape[1] == 3:
            temp_img = inverse_normalize(images[img],mean=mean,std=std).permute(1,2,0).detach().cpu().numpy()
        else:
            temp_img = inverse_normalize(images[img],mean=mean,std=std).permute(1,2,0)[:,:,0].detach().cpu().numpy()
            
        im = ax[0,0].imshow(temp_img,cmap=cmap)
        im = ax[1,0].imshow(temp_img,cmap=cmap)
        im = ax[2,0].imshow(temp_img,cmap=cmap)
        
        ax[0,0].set_yticks([])
        ax[1,0].set_yticks([])
        ax[2,0].set_yticks([])
        ax[0,0].set_xticks([])
        ax[1,0].set_xticks([])
        ax[2,0].set_xticks([])
        
    
        for temp_ang in range(0,EHD_outputs.size(1)):
            im = ax[0,temp_ang+1].imshow(EHD_outputs[img][temp_ang].detach().cpu().numpy(),vmin=0,vmax=1,cmap='coolwarm')
            im2 = ax[1,temp_ang+1].imshow(outputs[img][temp_ang].detach().cpu().numpy(),vmin=0,vmax=1,cmap='coolwarm')
            im3 = ax[2,temp_ang+1].imshow(abs(diff_outputs[img][temp_ang]),vmin=0,vmax=1,cmap='coolwarm')
            if temp_ang == EHD_outputs.size(1)-1:
                 ax[0,temp_ang+1].set_title('No Edge', fontsize=16)
                 ax[1,temp_ang+1].set_title('Bin {}'.format(temp_ang+1),fontsize=16)
                 ax[2,temp_ang+1].set_title('No Edge',fontsize=16)
                 ax[0,temp_ang+1].set_yticks([])
                 ax[1,temp_ang+1].set_yticks([])
                 ax[2,temp_ang+1].set_yticks([])
                 ax[0,temp_ang+1].set_xticks([])
                 ax[1,temp_ang+1].set_xticks([])
                 ax[2,temp_ang+1].set_xticks([])
            else:
                ax[0,temp_ang+1].set_title(str(angles[temp_ang])+u'\N{DEGREE SIGN}',fontsize=16)
                ax[1,temp_ang+1].set_title('Bin {}'.format(temp_ang+1),fontsize=16)
                ax[2,temp_ang+1].set_title(str(angles[temp_ang])+u"\N{DEGREE SIGN}",fontsize=16)
                ax[0,temp_ang+1].set_yticks([])
                ax[1,temp_ang+1].set_yticks([])
                ax[2,temp_ang+1].set_yticks([])
                ax[0,temp_ang+1].set_xticks([])
                ax[1,temp_ang+1].set_xticks([])
                ax[2,temp_ang+1].set_xticks([])
                
            plt.colorbar(im,ax=ax[0,temp_ang+1],fraction=0.046, pad=0.04)
            plt.colorbar(im2,ax=ax[1,temp_ang+1],fraction=0.046, pad=0.04)
            plt.colorbar(im3,ax=ax[2,temp_ang+1],fraction=0.046, pad=0.04)
       
        
        ax[0,1].set_ylabel('{} Outputs'.format(parameters['feature']),rotation=90,fontsize=16)
        ax[1,1].set_ylabel('N{} Outputs'.format(parameters['feature']),rotation=90,fontsize=16)
        ax[2,1].set_ylabel('Absolute Differences',rotation=90,fontsize=16)
        plt.tight_layout()
    
        filename = generate_filename(parameters,split)
        filename = filename + phase + '/'
        
        if not os.path.exists(filename):
            try:
                os.makedirs(filename)
            except:
                pass
        
        if epoch is not None:
            plt.suptitle('Epoch {} during {} phase'.format(epoch+1,phase))
            try:
                fig.savefig(filename+'Image_{}_Epoch_{}_Phase_{}.png'.format(img,epoch+1,phase),dpi=fig.dpi)
            except:
                pass
        else:
            plt.suptitle('Best Epoch for {} phase'.format(phase))
            try:
                plt.savefig(filename+'Image_{}_Best_Epoch_Phase_{}.png'.format(img,phase),dpi=fig.dpi)
            except:
                pass
            
        plt.close(fig=fig)
  
    
def plot_FMS_GAP_LBP(images,LBP_outputs,outputs,phase,epoch,feature_layer,model,
                     parameters,split, img_max=5):

    #Take difference between estimated and true outputs 
    diff_outputs = abs((LBP_outputs-outputs)).detach().cpu().numpy()
    img_count = np.arange(0,img_max,1)
    
    if images.shape[1] == 3:
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        cmap = None
    else:
        mean = (0,)
        std = (1,)
        cmap = 'gray'
        
    bins = np.arange(1,parameters['numBins'])
    bin_names = []
    bin_count = 0
    for hist_bin in bins:
        bin_names.append('Bin {}'.format(bin_count+1))
        bin_count += 1
    bin_names.append('Bin {}'.format(bin_count+1))
    
    for img in img_count:
        fig, ax = plt.subplots(2,3,figsize=(14,7))
        plt.subplots_adjust(wspace=.4,hspace=.4)
        
        if images.shape[1] == 3:
            temp_img = inverse_normalize(images[img],mean=mean,std=std).permute(1,2,0).detach().cpu().numpy()
        else:
            temp_img = inverse_normalize(images[img],mean=mean,std=std).permute(1,2,0)[:,:,0].detach().cpu().numpy()
            
        im = ax[0,0].imshow(temp_img,cmap=cmap,aspect="auto")
        im = ax[1,0].imshow(temp_img,cmap=cmap,aspect="auto")
        ax[0,0].set_title('Input Image',fontsize=24)
        ax[1,0].set_title('Input Image',fontsize=24)
        
        ax[0,0].set_yticks([])
        ax[0,0].set_xticks([])
        ax[1,0].set_yticks([])
        ax[1,0].set_xticks([])
        
        #Compute avg count of feature values
        y_pos = np.arange(len(bin_names))
        rects = ax[0,2].bar(y_pos,LBP_outputs[img,:].detach().cpu().numpy())
        max_value = LBP_outputs[img,:].max().detach().cpu().numpy() + .01
        ax[0,2].set_ylim([0,max_value])
        ax[0,2].set_box_aspect(1)
        ax[0,2].set_ylabel('Normalized Count',fontsize=16)
        ax[0,2].set_title('{} Histogram'.format(parameters['feature']),fontsize=24)
        
        rects = ax[1,2].bar(y_pos,outputs[img,:].detach().cpu().numpy())
        ax[1,2].set_ylim([0,max_value])
        ax[1,2].set_box_aspect(1)
        ax[1,2].set_ylabel('Normalized Count',fontsize=16)
        ax[1,2].set_title('N{} Histogram'.format(parameters['feature']),fontsize=24)
        
        #Get encodings
        LBP_encoding = feature_layer((images[img].unsqueeze(0)))
        
        #Revisit for RGB
        im = ax[0,1].imshow(LBP_encoding[0,0].detach().cpu().numpy(),cmap='gist_gray',aspect="auto")
        ax[0,1].set_title('{} Encoding'.format(parameters['feature']),fontsize=24)
        plt.colorbar(im,ax=ax[0,1],fraction=0.046, pad=0.04)
        
        #Remove histogram layer to get encoding
        model.neural_feature.histogram_layer = nn.Sequential()
        NEHD_encoding, _ = model(images[img].unsqueeze(0))
        
        im = ax[1,1].imshow(NEHD_encoding[0,0].detach().cpu().numpy(),cmap='gist_gray',aspect="auto")
        ax[1,1].set_title('N{} Encoding'.format(parameters['feature']),fontsize=24)
        plt.colorbar(im,ax=ax[1,1],fraction=0.046, pad=0.04)
        
        ax[0,1].set_yticks([])
        ax[0,1].set_xticks([])
        ax[1,1].set_yticks([])
        ax[1,1].set_xticks([])
        
        plt.tight_layout()
    
        filename = generate_filename(parameters,split)
        filename = filename + phase + '/'
        
        if not os.path.exists(filename):
            try:
                os.makedirs(filename)
            except:
                pass
        
        if epoch is not None:
            try:
                fig.savefig(filename+'Image_{}_Epoch_{}_Phase_{}.png'.format(img,epoch+1,phase),dpi=fig.dpi)
            except:
                pass
        else:
            try:
                plt.savefig(filename+'Image_{}_Best_Epoch_Phase_{}.png'.format(img,phase),dpi=fig.dpi)
            except:
                pass
        plt.close(fig=fig)
        
def plot_FMS_LBP(images,LBP_outputs,outputs,phase,epoch,feature_layer,
                     parameters,split, img_max=5):
    
    import pdb
    pdb.set_trace()
    #Take difference between estimated and true outputs 
    diff_outputs = abs((LBP_outputs-outputs)).detach().cpu().numpy()
    img_count = np.arange(0,img_max,1)
    
    #Change to pink for data
    if images.shape[1] == 3:
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        cmap = None
    else:
        mean = (.5,)
        std = (.5,)
        cmap = 'gray'
        
    angles = np.arange(0,360,parameters['angle_res'])
    angle_names = []
    bin_names = []
    bin_count = 0
    for angle in angles:
        angle_names.append(u'{}\N{DEGREE SIGN}'.format(angle))
        bin_names.append('Bin {}'.format(bin_count+1))
        bin_count += 1
    
    for img in img_count:
        fig, ax = plt.subplots(1,4,figsize=(12,6))
        plt.subplots_adjust(wspace=.4,hspace=.4)
        
        if images.shape[1] == 3:
            temp_img = inverse_normalize(images[img],mean=mean,std=std).permute(1,2,0).detach().cpu().numpy()
        else:
            temp_img = inverse_normalize(images[img],mean=mean,std=std).permute(1,2,0)[:,:,0].detach().cpu().numpy()
            
        im = ax[0].imshow(temp_img,cmap=cmap,aspect="auto")
        ax[0].set_box_aspect(1)
        if not(images.shape[1] == 3):
            plt.colorbar(im,ax=ax[0],fraction=0.046, pad=0.04)
        
    
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        
        #Compute avg count of feature values
        y_pos = np.arange(len(angle_names))
        rects = ax[1].bar(y_pos,LBP_outputs[img,:,0,0].detach().cpu().numpy())
        ax[1].set_box_aspect(1)
        ax[1].set_ylabel('Avg Feature Count')
        ax[1].set_title('{}'.format(parameters['feature']))
        
        rects = ax[2].bar(y_pos,outputs[img,:,0,0].detach().cpu().numpy())
        ax[2].set_box_aspect(1)
        ax[2].set_ylabel('Avg Feature Count')
        ax[2].set_title('N{}'.format(parameters['feature']))
        
        rects = ax[3].bar(y_pos,diff_outputs[img,:,0,0])
        ax[3].set_box_aspect(1)
        ax[3].set_ylabel('Avg Feature Count')
        ax[3].set_title('Absolute Differences')
       
        
        plt.tight_layout()
    
        filename = generate_filename(parameters,split)
        filename = filename + phase + '/'
        
        if not os.path.exists(filename):
            try:
                os.makedirs(filename)
            except:
                pass
        
        if epoch is not None:
            plt.suptitle('Epoch {} during {} phase'.format(epoch+1,phase))
            try:
                fig.savefig(filename+'Image_{}_Epoch_{}_Phase_{}.png'.format(img,epoch+1,phase),dpi=fig.dpi)
            except:
                pass
        else:
            plt.suptitle('Best Epoch for {} phase'.format(phase))
            try:
                plt.savefig(filename+'Image_{}_Best_Epoch_Phase_{}.png'.format(img,phase),dpi=fig.dpi)
            except:
                pass
        plt.close(fig=fig)
