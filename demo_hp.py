# -*- coding: utf-8 -*-
"""
Demo for histogram layer networks (HistRes_B)
Current script is only for experiments on
single cpu/gpu. If you want to run the demo
on multiple gpus (two were used in paper), 
please contact me at jpeeples@ufl.edu 
for the parallel version of 
demo.py
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import argparse
from itertools import product
import matplotlib.pyplot as plt
import random

## PyTorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim

## Local external libraries
from Utils.Network_functions import initialize_model, train_model,test_model
from Utils.NLBP import NLBPLayer
from Utils.NEHD import NEHDLayer
from Utils.Compute_LBP import LocalBinaryLayer
from Utils.Compute_EHD import EHD_Layer
from Utils.Save_Results import save_results
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders
import pdb

plt.ioff()

def main(args,params):
    #Change mode of batch script
    mode = params['mode']
    
    #Set 16 different settings for a) initialization and b) parameter learning
    if mode == 'config':
        settings = list(product((True, False), repeat=4))
    
    elif mode == 'kernel':
    #Kernel Settings
        settings = [[5,5]]
    else:
    #Dilation
        settings = [2, 4, 8, 16]
    
    print('Starting Batch Experiments...')
    setting_count = args.setting_cnt
    single_setting = args.single_setting
    
    #Append base model (histogram_layer = None)
    settings.append(settings[-1])
    
    data_parameters = Parameters(args)
    Dataset = data_parameters['Dataset']
    # Create training and validation dataloaders
    dataloaders_dict = Prepare_DataLoaders(data_parameters)

    
    setting_n = 0 if args.feature == 'EHD' else 2



    settings_params_dict = {}
    for setting in [settings[setting_n]]: 

        #Set initial parameters
        if mode == 'config':
            Network_parameters = Parameters(args,learn_hist=setting[0],learn_edge_kernels=setting[1],
                                            feature_init=setting[2],learn_transform=setting[3],
                                            dilation=1)
    
        elif mode == 'kernel':
            #Kernel experiments
            Network_parameters = Parameters(args,learn_hist=False,learn_edge_kernels=True,
                                            feature_init=True,learn_transform=False,
                                            dilation=1,mask_size=setting)
        else:
            #Dilation experiments
            Network_parameters = Parameters(args,learn_hist=False,learn_edge_kernels=True,
                                            feature_init=True,learn_transform=False,
                                            dilation=setting,mask_size=[3,3])
        
        #Check for base model
        if setting_count == (len(settings) - 1):
            Network_parameters['histogram'] = False
        
        #Name of dataset
        # Dataset = Network_parameters['Dataset']
                                         
        #Number of runs and/or splits for dataset
        numRuns = Network_parameters['Splits'][Dataset]
        
        #Number of bins and input convolution feature maps after channel-wise pooling
        num_feature_maps = Network_parameters['out_channels']
        
        # Detect if we have a GPU available
        #if torch.cuda.is_available():
        #    device = torch.device("cuda")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        for split in range(0, numRuns):
            # Set seed for reproducibility
            torch.manual_seed(split)
            np.random.seed(split)
            random.seed(split)
            torch.cuda.manual_seed(split)
            torch.cuda.manual_seed_all(split)
            print("Initializing Datasets and Dataloaders...")
            
            # Create training and validation dataloaders
            #dataloaders_dict = Prepare_DataLoaders(Network_parameters,split,
            #                                   mean=Network_parameters['mean'][Dataset],
            #                                   std=Network_parameters['std'][Dataset])
            
            #Keep track of the bins and widths as these values are updated each
            #epoch ##########
            if (Network_parameters['learn_transform'] or Network_parameters['feature'] == 'LBP'):
                saved_bins = np.zeros((Network_parameters['num_epochs']+1, int(Network_parameters['in_channels']) *int(num_feature_maps)))
                saved_widths =  np.zeros((Network_parameters['num_epochs']+1,int(Network_parameters['in_channels']) * int(num_feature_maps)))
            else:
                saved_bins = np.zeros((Network_parameters['num_epochs']+1,int(Network_parameters['in_channels']) * int(num_feature_maps)))
                saved_widths =  np.zeros((Network_parameters['num_epochs']+1,int(Network_parameters['in_channels']) *int(num_feature_maps))) 
            
            #Initialize histogram layer based on type
            if Network_parameters['histogram']:
                if Network_parameters['feature'] == 'LBP':
                    
                    histogram_layer = NLBPLayer(Network_parameters['in_channels'], 
                                                P=Network_parameters['P'], 
                                                R=Network_parameters['R'], 
                                                window_size = Network_parameters['window_size'],
                                                num_bins = Network_parameters['numBins'],
                                                stride=Network_parameters['stride'],
                                                normalize_count=Network_parameters['normalize_count'],
                                                normalize_bins=Network_parameters['normalize_bins'],
                                                LBP_init=Network_parameters['feature_init'],
                                                learn_base = Network_parameters['learn_transform'], ###
                                                learn_hist = Network_parameters['learn_hist'],
                                                normalize_kernel=Network_parameters['normalize_kernel'],
                                                dilation=Network_parameters['dilation'],
                                                learn_kernel = Network_parameters['learn_edge_kernels'],
                                                aggregation_type=Network_parameters['aggregation_type'])
                    
                #Update linear for dilation
                elif Network_parameters['feature'] == 'EHD': 
                    histogram_layer = NEHDLayer(Network_parameters['in_channels'],
                                              Network_parameters['window_size'],
                                              mask_size=Network_parameters['mask_size'],
                                              num_bins=Network_parameters['numBins'],
                                              stride=Network_parameters['stride'],
                                              normalize_count=Network_parameters['normalize_count'],
                                              normalize_bins=Network_parameters['normalize_bins'],
                                              EHD_init=Network_parameters['feature_init'],
                                              learn_no_edge=Network_parameters['learn_transform'],
                                              learn_kernel = Network_parameters['learn_edge_kernels'],
                                              learn_hist = Network_parameters['learn_hist'],
                                              threshold=Network_parameters['threshold'],
                                              angle_res=Network_parameters['angle_res'],
                                              normalize_kernel=Network_parameters['normalize_kernel'],
                                              aggregation_type=Network_parameters['aggregation_type'],
				              dilation=Network_parameters['dilation'])
                else:
                    raise RuntimeError('Invalid type for histogram layer')
            else:
               histogram_layer = None
            
        
            # model_ft = histogram_layer
            model_ft = initialize_model(Network_parameters,dataloaders_dict,device,
                                        Network_parameters['num_classes'][Dataset],
                                        reconstruction=Network_parameters['reconstruction'], 
                                        in_channels=Network_parameters['in_channels'],
                                        histogram_layer=histogram_layer, fusion_method=Network_parameters['fusion_method'])
            
            if Network_parameters['Parallelize_model'] and False:
                if torch.cuda.device_count() > 1:
                  print("Using", torch.cuda.device_count(), "GPUs!")
                  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                  model_ft = nn.DataParallel(model_ft)
    
            # Send the model to GPU if available
            model_ft = model_ft.to(device)
            
            #Save initial bin widths and centers
            if Network_parameters['histogram']:
                saved_bins[0,:] = model_ft.neural_feature.histogram_layer.centers.reshape(-1).detach().cpu().numpy() 
                saved_widths[0,:] = model_ft.neural_feature.histogram_layer.widths.reshape(-1).detach().cpu().numpy()
            else:
                saved_bins = None
                saved_widths = None
            
           
            #Print number of trainable parameters (if not learnable (base model),
            # only show parameters from fully connected layer)
            
            #Verify parameter learning settings (Salim, remove from updated code base)
            setting_params = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad:
                    # print(name)
                    setting_params.append(name)
                    
            settings_params_dict['Setting {}'.format(setting)] = setting_params
            ###################################################################
            
            try:
                num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
            except:
                num_params = sum(p.numel() for p in model_ft.fc.parameters() if p.requires_grad)
           
            print("Number of parameters: %d" % (num_params))    
            
            # Setup the loss fxn
            if Network_parameters['reconstruction']:
                criterion = nn.MSELoss()
            else:
                criterion = nn.CrossEntropyLoss()
            

            optimizer_ft = optim.Adam(model_ft.parameters(),lr=Network_parameters['lr'])
            scheduler = optim.lr_scheduler.StepLR(optimizer_ft,
                                                step_size=Network_parameters['step_size'],
                                                gamma= Network_parameters['gamma'])
            
            # Train and evaluate
            train_dict = train_model(
                    model_ft, dataloaders_dict, criterion, optimizer_ft, device,
                    Network_parameters,split,saved_bins=saved_bins,saved_widths=saved_widths,
                    histogram=Network_parameters['histogram'],
                    num_epochs=Network_parameters['num_epochs'],scheduler=scheduler,
                    num_params=num_params)
            
            test_dict = test_model(dataloaders_dict['test'],model_ft,device,
                                    Network_parameters,split)
            
            # Save results
            if(Network_parameters['save_results']):
                save_results(train_dict,test_dict,split,Network_parameters,num_params)
                del train_dict,test_dict
                if device == torch.device("cuda"):
                    torch.cuda.empty_cache()
                elif device == torch.device("mps"):
                    torch.mps.empty_cache()
                else:
                    pass
             
            if(Network_parameters['histogram']):
                print('**********Run ' + str(split + 1) + ' For ' + Network_parameters['hist_model'] + ' Finished**********') 
            else:
                print('**********Run ' + str(split + 1) + ' For ' + Network_parameters['base_model_name'] + ' Finished**********')
        
        setting_count += 1
        
        print('Finished setting {} of {}'.format(setting_count,len(settings)))

        if single_setting:
            break
    
      
    
def parse_args():
    parser = argparse.ArgumentParser(description='Run neural handcrafted experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/', # Default is Saved_Models/
                        help='Location to save models')
    parser.add_argument('--feature', type=str, default='EHD', 
                        help='Select feature to evaluate (EHD or LBP)')
    parser.add_argument('--mode', type=str, default='config',
                        help='Mode for experiments: ‘config’, ‘kernel’, dilation')
    parser.add_argument('--reconstruction', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to reconstruction or classification, --no-reconstruction (classification) or --reconstruction')
    parser.add_argument('--histogram', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram model or baseline global average pooling (GAP), --no-histogram (GAP) or --histogram')
    parser.add_argument('--data_selection', type=int, default=1, # Data Config
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('--numBins', type=int, default=16, # Reduced to accomodate memory
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 256)')
    parser.add_argument('--angle_res', type=int, default=45,
                        help='Number of angle resolutions (controls number of bins). Recommended value is 45 for 8 edge orientations (default: 45)')
    parser.add_argument('-R', type=int, default=1,
                        help='Radius of neighborhood for LBP. Recommended value is 1 for 3 by 3 window (default: 1)')
    parser.add_argument('-P', type=int, default=8,
                        help='Number of neighborhood for LBP. Recommended value is 8 for 3 by 3 window (default: 8)')
    parser.add_argument('--LBP_method', type=str, default='default',
                        help='Select LBP method for baseline method to evaluate (‘default’, ‘ror’, ‘uniform’, ‘nri_uniform’, ‘var’)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=128, # Reduced to accomodate memory
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=512, # Reduced to accomodate memory
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=256, # Reduced to accomodate memory
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=128,
                        help='Resize the image before center crop. (default: 128)')
    parser.add_argument('--center_size', type=int, default=112,
                        help='Center crop size. (default: 112)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for histogram feature. (default: 1)')
    parser.add_argument('--num_workers', type=int, default=3, ################
                        help='Number of workers for dataloader. (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--parallelize_model', default=True, action=argparse.BooleanOptionalAction,
                        help='enables training on mulitiple GPUs')
    parser.add_argument('--setting_cnt', default=0, type=int, # Only do the last 2 settings for speed
                        help='Setting min is 0 and max is 16')
    parser.add_argument('--fusion_method', type=str, default=None,
                        help='Fusion method for n>1 channels (default: None); Options: None, grayscale, conv')
    parser.add_argument('--single_setting', default=True, action=argparse.BooleanOptionalAction,
                        help='Run a single setting')
    parser.add_argument('--kernel_size', nargs='+', type=int, default=[3, 3],
                        help='Mask size controls structural hyper param')
    parser.add_argument('--window_size', nargs='+', type=int, default=[5, 5],
                        help='Controls the aggregation kernel hyper param')
    parser.add_argument('--dilation', default=1, type=int,
                        help='control lbp structural')
    args = parser.parse_args()
    print('got args')
    return args

if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available() and  args.use_cuda:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print('Using device: {}'.format(device))
    
    args.kernel_size = list(args.kernel_size)
    args.window_size = list(args.window_size)
    params = Parameters(args, mask_size=args.kernel_size)
    main(args,params)
