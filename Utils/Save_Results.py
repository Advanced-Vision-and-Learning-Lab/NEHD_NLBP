# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:39:28 2020
Save results from training/testing model
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import os
import pickle

## PyTorch dependencies
import torch

def generate_filename(Network_parameters,split):
    
    if Network_parameters['feature'] in ['DSA', 'MSDCNN']:
    	filename = f"{Network_parameters['folder']}/{Network_parameters['feature']}/{Network_parameters['Dataset']}/Run_{str(split + 1)}"
    	if not os.path.exists(filename):
                try:
                	os.makedirs(filename)
                except:
                	pass
    	return filename

    if Network_parameters['feature'] == 'EHD':
        if Network_parameters['learn_transform']:
            transform = 'Conv'
        else:
            transform = 'Thresh'
    elif Network_parameters['feature'] == 'LBP':
        if Network_parameters['learn_transform']:
            transform = 'Learn'
        else:
            transform = 'Fixed'
	    
        
    if(Network_parameters['histogram']):
        if(Network_parameters['feature_init']):
            
            #Removed params settings due to long filename
        
            filename = '{}/{}/{}/{}_{}_{}/init_{}/Run_{}/'.format(Network_parameters['folder'],
                                         Network_parameters['feature'],
                                         Network_parameters['hist_model'],
                                         Network_parameters['fusion_method'],
                                         Network_parameters['aggregation_type'],
                                         Network_parameters['feature'],
                                         transform,str(split + 1))
        else:
            filename = '{}/{}/{}/{}_{}_{}/Rand_init_{}/Run_{}/'.format(Network_parameters['folder'],
                                       Network_parameters['feature'],
                                       Network_parameters['Dataset'],
                                       Network_parameters['hist_model'],
                                       Network_parameters['fusion_method'],
                                       Network_parameters['aggregation_type'],
                                       transform,str(split + 1))
    #Baseline model
    else:
        if Network_parameters['feature'] == 'LBP':
            filename = '{}/{}/{}/Baseline_{}_{}_{}/{}/Run_{}/'.format(Network_parameters['folder'],
                                       Network_parameters['feature'], 
                                       Network_parameters['Dataset'],
                                       Network_parameters['LBP_method'],
                                       Network_parameters['fusion_method'],
                                       Network_parameters['feature'],
                                       Network_parameters['base_model_name'],
                                       str(split + 1))
        else:
            filename = '{}/{}/{}/Baseline_{}_{}/{}/Run_{}/'.format(Network_parameters['folder'],
                                       Network_parameters['feature'],
                                       Network_parameters['Dataset'],
                                       Network_parameters['fusion_method'],
                                       Network_parameters['feature'],
                                       Network_parameters['base_model_name'],
                                       str(split + 1))
        
    if not os.path.exists(filename):
        try:
            os.makedirs(filename)
        except:
            pass
        
    return filename

def save_results(train_dict,test_dict,split,Network_parameters,num_params):
    
    filename = generate_filename(Network_parameters,split)        
        
    #Save training and testing dictionary, save model using torch
    torch.save(train_dict['best_model_wts'], filename + 'Best_Weights.pt')
    torch.save(train_dict['epoch_weights'], filename + 'epoch_weights.pt')
    
    #Remove model from training dictionary
    train_dict.pop('best_model_wts')
    train_dict.pop('epoch_weights')
    output_train = open(filename + 'train_dict.pkl','wb')
    pickle.dump(train_dict,output_train)
    output_train.close()
    
    output_test = open(filename + 'test_dict.pkl','wb')
    pickle.dump(test_dict,output_test)
    output_test.close()
