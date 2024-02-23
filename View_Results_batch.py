# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef
from itertools import product
import pickle
import argparse

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Generate_TSNE_visual import Generate_TSNE_visual
from Utils.Network_functions import initialize_model
from Prepare_Data import Prepare_DataLoaders
from Utils.Confusion_mats import plot_confusion_matrix,plot_avg_confusion_matrix
from Utils.Generate_Learning_Curves import Plot_Learning_Curves
from Demo_Parameters_batch import Parameters
from Utils.Crisp_Histogram_visual import Generate_Histogram_visual
from Utils.Save_Results import generate_filename
from Utils.NLBP import NLBPLayer
from Utils.NEHD import NEHDLayer

def save_metric(sub_dir,metric_name,value):
    #Function to save metric values
    with open((sub_dir + '{}.txt'.format(metric_name)), "w") as output:
        output.write(str(value))
        
def save_avg_std_metric(directory,metric_name,values,axis=0):
        with open((directory + 'Overall_{}.txt'.format(metric_name)), "w") as output:
            output.write('Average {}: {} u"\u00B1" {}'.format(metric_name,
                                                              str(np.mean(values, axis=axis)),
                                                              str(np.std(values,axis=axis)))) 
        
def main(args, params):
    
    #Location of experimental results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    plt.ioff()
    
    #Change mode of batch script
    mode = params['mode']
    
    #Set 16 different settings for a) initialization and b) parameter learning
    if mode == 'config':
        settings = list(product((True, False), repeat=4))
    
    elif mode == 'kernel':
    #Kernel Settings
        settings = [[3,3],[7,7],[15,15],[31,31],[63,63]]
    else:
    #Dialation
        settings = [2, 4, 8, 16]
    
    print('Generating Batch Experiments Results...')
    setting_count = args.setting_cnt
    single_setting = args.single_setting
    TSNE_visual = args.TSNE_visual
    #Append base model (histogram_layer = None)
    settings.append(settings[-1])
    
    for setting in settings[setting_count:]: 
    
        #Set initial parameters
        if mode == 'config':
            Results_parameters = Parameters(args,learn_hist=setting[0],learn_edge_kernels=setting[1],
                                            feature_init=setting[2],learn_transform=setting[3],
                                            dilation=1)
    
        elif mode == 'kernel':
            #Kernel experiments
            Results_parameters = Parameters(args,learn_hist=False,learn_edge_kernels=True,
                                            feature_init=True,learn_transform=False,
                                            dilation=1,mask_size=setting)
        else:
            #Dialation experiments
            Results_parameters = Parameters(args,learn_hist=False,learn_edge_kernels=True,
                                            feature_init=True,learn_transform=False,
                                            dilation=setting,mask_size=[3,3])
        #Check for base model
        if setting_count == (len(settings) - 1):
            Results_parameters['histogram'] = False

        #Figure Sizes
        fig_size = Results_parameters['fig_size']
        font_size = Results_parameters['font_size']
        
        #Name of dataset
        Dataset = Results_parameters['Dataset']
        
     
        #Initial variables for saving
        NumRuns = Results_parameters['Splits'][Results_parameters['Dataset']]
        plot_name = Results_parameters['Dataset'] + ' Test Confusion Matrix'
        avg_plot_name = Results_parameters['Dataset'] + ' Test Average Confusion Matrix'
        class_names = Results_parameters['Class_names'][Results_parameters['Dataset']]
        cm_stack = np.zeros((len(class_names),len(class_names)))
        cm_stats = np.zeros((len(class_names),len(class_names),NumRuns))
        FDR_scores = np.zeros((len(class_names),NumRuns))
        log_FDR_scores = np.zeros((len(class_names),NumRuns))
        accuracy = np.zeros(NumRuns)
        train_acc = np.zeros(NumRuns)
        val_acc = np.zeros(NumRuns)
        MCC = np.zeros(NumRuns)
        loss = np.zeros(NumRuns)
        
        
        #Name of dataset
        Dataset = Results_parameters['Dataset']
        
      
        # Parse through files and plot results
        for split in range(0, NumRuns):
            
            #Set directory location for experiments
            sub_dir = generate_filename(Results_parameters, split)
            print(sub_dir) 
            #Load files
            train_pkl_file = open(sub_dir+'train_dict.pkl','rb')
            train_dict = pickle.load(train_pkl_file)
            train_pkl_file.close()
            
            test_pkl_file = open(sub_dir+'test_dict.pkl','rb')
            test_dict = pickle.load(test_pkl_file)
            test_pkl_file.close()
            
            
            #Initialize histogram layer based on type
            if Results_parameters['histogram']:
                if Results_parameters['feature'] == 'LBP':
                    
                    histogram_layer = NLBPLayer(Results_parameters['in_channels'], 
                                                P=Results_parameters['P'], 
                                                R=Results_parameters['R'], 
                                                window_size = Results_parameters['window_size'],
                                                num_bins = Results_parameters['numBins'],
                                                stride=Results_parameters['stride'],
                                                normalize_count=Results_parameters['normalize_count'],
                                                normalize_bins=Results_parameters['normalize_bins'],
                                                LBP_init=Results_parameters['feature_init'],
                                                learn_base = Results_parameters['learn_transform'],
                                                normalize_kernel=Results_parameters['normalize_kernel'],
                                                dilation=Results_parameters['dilation'],
                                                aggregation_type=Results_parameters['aggregation_type'])
                    
                #Update linear for dialation
                elif Results_parameters['feature'] == 'EHD': 
                    histogram_layer = NEHDLayer(Results_parameters['in_channels'],
                                              Results_parameters['window_size'],
                                              mask_size=Results_parameters['mask_size'],
                                              num_bins=Results_parameters['numBins'],
                                              stride=Results_parameters['stride'],
                                              normalize_count=Results_parameters['normalize_count'],
                                              normalize_bins=Results_parameters['normalize_bins'],
                                              EHD_init=Results_parameters['feature_init'],
                                              learn_no_edge=Results_parameters['learn_transform'],
                                              threshold=Results_parameters['threshold'],
                                              angle_res=Results_parameters['angle_res'],
                                              normalize_kernel=Results_parameters['normalize_kernel'],
                                              aggregation_type=Results_parameters['aggregation_type'])
                else:
                    raise RuntimeError('Invalid type for histogram layer')
            else:
                histogram_layer = None
            
        

            # Prepare dataloaders
            dataloaders_dict = Prepare_DataLoaders(Results_parameters)

            model = initialize_model(Results_parameters,dataloaders_dict, device,
                                        num_classes= Results_parameters['num_classes'][Dataset],
                                        reconstruction=Results_parameters['reconstruction'],
                                        in_channels=Results_parameters['in_channels'],
                                        histogram_layer=histogram_layer, fusion_method=Results_parameters['fusion_method'])

            
            #Load best weight to analyze model
            device_loc = torch.device(device)
            best_weights = torch.load(sub_dir + 'Best_Weights.pt',map_location=device_loc) #map_location=device_loc
            
            #If parallelized, need to set change model
            if Results_parameters['Parallelize_model']:
                if torch.cuda.device_count() > 1:
                  print("Using", torch.cuda.device_count(), "GPUs!")
                  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                  model = nn.DataParallel(model)
            
            model.load_state_dict(best_weights)
            model = model.to(device)

            if (TSNE_visual):
                print("Initializing Datasets and Dataloaders...")
                
                dataloaders_dict = Prepare_DataLoaders(Results_parameters,split,
                                                       mean=Results_parameters['mean'][Dataset],
                                                       std=Results_parameters['std'][Dataset])
                print('Creating TSNE Visual...')
                
                #Remove fully connected layer
                if Results_parameters['Parallelize_model']:
                    if torch.cuda.device_count() > 1:
                        model.module.fc = nn.Sequential()
                else:
                    model.fc = nn.Sequential()
                
                #Generate TSNE visual
                FDR_scores[:,split], log_FDR_scores[:,split] = Generate_TSNE_visual(
                                      dataloaders_dict,
                                      model,sub_dir,device,class_names,
                                      Num_TSNE_images=Results_parameters['Num_TSNE_images'])
           
            #Create CM for testing data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            cm = confusion_matrix(test_dict['GT'],test_dict['Predictions'])
            
            #Create classification report
            report = classification_report(test_dict['GT'],test_dict['Predictions'],
                                           target_names=class_names,output_dict=True)
            
            #Convert to dataframe and save as .CSV file
            df = pd.DataFrame(report).transpose()
            
            #Save to CSV
            df.to_csv((sub_dir+'Classification_Report.csv'))

            # visualize results
            #Generate learning curves
            Plot_Learning_Curves(train_dict['train_acc_track'],
                                  train_dict['train_error_track'],
                                  train_dict['val_acc_track'],
                                  train_dict['val_error_track'],
                                  train_dict['best_epoch'],
                                  sub_dir)

                
            print("Done!") ###
            
            # Confusion Matrix
            np.set_printoptions(precision=2)
            fig4, ax4= plt.subplots(figsize=(fig_size, fig_size))
            plot_confusion_matrix(cm, classes=class_names, title=plot_name,ax=ax4,
                                  fontsize=font_size)
            fig4.savefig((sub_dir + 'Confusion Matrix.png'), dpi=fig4.dpi)
            plt.close()
            cm_stack = cm + cm_stack
            cm_stats[:, :, split] = cm
            
            loss[split] = test_dict['test_loss']
            accuracy[split] = test_dict['test_acc']
            train_acc[split] = train_dict['train_acc_track'][train_dict['best_epoch']]
            val_acc[split] = train_dict['val_acc_track'][train_dict['best_epoch']]
            MCC[split] =  matthews_corrcoef(test_dict['GT'], test_dict['Predictions'])


            
            #Save metrics
            save_metric(sub_dir,'Test_Loss',test_dict['test_loss'])
            save_metric(sub_dir,'Test_Accuracy',test_dict['test_acc'])
            save_metric(sub_dir,'Train_Accuracy',train_acc[split])
            save_metric(sub_dir, 'Val_Accuracy', val_acc[split])
            save_metric(sub_dir, 'MCC', MCC[split])                    
        
            print('**********Run ' + str(split+1) + ' Finished**********')
        
        
        directory = os.path.dirname(os.path.dirname(sub_dir)) + '/'   
        np.set_printoptions(precision=2)
        fig5, ax5 = plt.subplots(figsize=(fig_size, fig_size))
        plot_avg_confusion_matrix(cm_stats, classes=class_names, 
                                  title=avg_plot_name,ax=ax5,fontsize=font_size)
        fig5.savefig((directory + 'Average Confusion Matrix.png'), dpi=fig5.dpi)
        plt.close()
        
        save_avg_std_metric(directory, 'Loss', loss)
        save_avg_std_metric(directory, 'Test_Accuracy', accuracy)
        save_avg_std_metric(directory, 'Train_Accuracy', train_acc)
        save_avg_std_metric(directory, 'Val_Accuracy', val_acc)
        save_avg_std_metric(directory, 'FDR', FDR_scores, axis=1)
        save_avg_std_metric(directory, 'Log_FDR', FDR_scores, axis=1)
  
        
        np.savetxt((directory+'List_Loss.txt'),loss.reshape(-1,1),fmt='%.2f')
        
        # Write list of accuracies and MCC for analysis
        np.savetxt((directory + 'List_Accuracy.txt'), accuracy.reshape(-1, 1), fmt='%.2f')
        np.savetxt((directory + 'List_MCC.txt'), MCC.reshape(-1, 1), fmt='%.2f')
        np.savetxt((directory + 'test_List_FDR_scores.txt'), FDR_scores, fmt='%.2E')
        np.savetxt((directory + 'test_List_log_FDR_scores.txt'), log_FDR_scores, fmt='%.2f')
        plt.close("all")
        
        setting_count += 1
       
        print('Finished setting {} of {}'.format(setting_count,len(settings)))

        if single_setting:
            break
    
def parse_args():
    parser = argparse.ArgumentParser(description='Run neural handcrafted experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/', # Default is Saved_Models/
                        help='Location to save models') # Results_Test in the future
    parser.add_argument('--feature', type=str, default='EHD', ###
                        help='Select feature to evaluate (EHD or LBP)')
    parser.add_argument('--mode', type=str, default='config',
                        help='Mode for experiments: ‘config’, ‘kernel’, ‘dilation’')
    parser.add_argument('--reconstruction', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to reconstruction or classification, --no-reconstruction (classification) or --reconstruction')
    parser.add_argument('--histogram', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram model or baseline global average pooling (GAP), --no-histogram (GAP) or --histogram')
    parser.add_argument('--data_selection', type=int, default=1, # Data Config
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=256, # Data Config
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('-angle_res', type=int, default=45,
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
    parser.add_argument('--num_epochs', type=int, default=50, # Intentionally slowed from 30
                        help='Number of epochs to train each model for (default: 30)')
    parser.add_argument('--resize_size', type=int, default=128,
                        help='Resize the image before center crop. (default: 126)')
    parser.add_argument('--center_size', type=int, default=112,
                        help='Center crop size. (default: 112)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for histogram feature. (default: 1)')
    parser.add_argument('--num_workers', type=int, default=0, ########
                        help='Number of workers for dataloader. (default: 1)')
    parser.add_argument('--lr', type=float, default=.01, # Increased to accomodate speed
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--parallelize_model', default=True, action=argparse.BooleanOptionalAction,
                        help='enables training on mulitiple GPUs')
    parser.add_argument('--setting_cnt', default=0, type=int, # Only do the last 2 settings for speed
                        help='Setting min is 0 and max is 16')
    parser.add_argument('--fusion_method', type=str, default=None,
                    help='Fusion method for n>1 channels (default: None); Options: None, grayscale, conv')
    parser.add_argument('--single_setting', default=False, action=argparse.BooleanOptionalAction,
                        help='Run a single setting')
    parser.add_argument('--TSNE_visual', default=False, action=argparse.BooleanOptionalAction,
                        help='Generates the TSNE visual')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    main(args,params)
