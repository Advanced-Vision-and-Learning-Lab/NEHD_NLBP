# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:31:02 2020
Create visual to show distribution of features for handcrafted and neural feature
@author: jpeeples
"""
from barbar import Bar
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from Utils.Compute_EHD import EHD_Layer as Get_EHD
import matplotlib.ticker as mticker
import pdb
import os
from Utils.Compute_LBP import LocalBinaryLayer


def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def Generate_Histogram_visual(dataloaders_dict,model,sub_dir,device,class_names,
                              parameters,img_max=3,mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225),feature='EHD'): 
    
      if feature == 'EHD':
          angles = np.arange(0,360,parameters['angle_res'])
          angle_names = []
          bin_names = []
          bin_count = 0
          for angle in angles:
              angle_names.append(u'{}\N{DEGREE SIGN}'.format(angle))
              bin_names.append('Bin {}'.format(bin_count))
              bin_count += 1
          angle_names.append('No Edge')
          bin_names.append('Bin {}'.format(bin_count+1))
      else:
         angles = np.arange(0, 256, 256 / model.histogram_layer.numBins)
         angle_names = []
         bin_names = []
         bin_count = 0
         for angle in angles:
             angle_names.append(u'{}'.format(int(angle)))
             bin_names.append(u'{}'.format(int(angle)))
             bin_count += 1
         
      #TSNE visual of validation data
      #Works for test data, need to fix for train/validation if sampler is used
      for phase in ['test']:
            #Get labels and outputs
            GT_val = np.array(0)
            img_indices = np.array(0)
            model.eval()
            model.to(device)
            features_extracted = []
            saved_imgs = []
          
            for idx, (inputs, classes,index)  in enumerate(Bar(dataloaders_dict[phase])):
                images = inputs.to(device)
                labels = classes.to(device, torch.long)
                indices  = index.to(device).cpu().numpy()
                
                GT_val = np.concatenate((GT_val, labels.cpu().numpy()),axis = None)
                img_indices = np.concatenate((img_indices,indices),axis = None)
                
    
                saved_imgs.append(images)
         
      
            saved_imgs = torch.cat(saved_imgs,dim=0)
            
            #Compute FDR scores
            GT_val = GT_val[1:]
            img_indices = img_indices[1:]
     
        
            for temp_class in range(0,len(class_names)):
                
                #For each class, plot image and features as histograms
                temp_indices = img_indices[GT_val==temp_class]
                sel_index = np.partition(temp_indices,img_max)[:img_max]
                selected_imgs = saved_imgs[sel_index]
                if len(selected_imgs.shape) < 4: #If no channel dimension (grayscale), add dimension
                    selected_imgs = selected_imgs.to(device).unsqueeze(1)
                
                if feature == 'EHD':
                    selected_feat = Get_EHD(selected_imgs,parameters,device=device)
                else:
                    selected_feat = LocalBinaryLayer(inputs.shape[1],radius=parameters['R'],
                                                       n_points = parameters['P'],
                                                       method = parameters['LBP_method'],
                                                       num_bins = parameters['numBins'],
                                                       density = parameters['normalize_count'])(selected_imgs)
    
                selected_Nfeat, _ =  model(selected_imgs)
                
                #Create subdir for histogram features
                #hist_dir = sub_dir + 'Hist_Vis/{}/{}/'.format(phase,class_names[temp_class])
                hist_dir = sub_dir + 'Hist_Vis/{}/'.format(phase) ## Debug
                
                if not os.path.exists(hist_dir):
                    os.makedirs(hist_dir)
                
                
                #Plot images and resulting histogram
                fig, ax = plt.subplots(img_max,3,figsize=(12,6))
                plt.subplots_adjust(wspace=.4,hspace=.4)
                
                for img in range(0,img_max):
                    if saved_imgs.shape[1] == 3:
                        temp_img = inverse_normalize(selected_imgs[img],mean=mean,std=std).permute(1,2,0).detach().cpu().numpy()
                    else:
                        temp_img = inverse_normalize(selected_imgs[img],mean=mean,std=std).permute(1,2,0)[:,:,0].detach().cpu().numpy()
                    
                    im = ax[img,0].imshow(temp_img,cmap='gray')
                    ax[img,0].axis('off')
                    
                    if saved_imgs.shape[1] == 3:
                        pass
                    else:
                        plt.colorbar(im,ax=ax[img,0],fraction=.046,pad=.04)
                    
                    #Compute avg count of feature values
                    y_pos = np.arange(len(angle_names))
                    rects = ax[img,1].bar(y_pos,torch.mean(selected_feat[img],dim=(1,2)).detach().cpu().numpy())
                    ax[img,1].xaxis.set_major_locator(mticker.MaxNLocator(len(angle_names),min_n_ticks=len(angle_names)-2))
                    if feature == 'EHD':
                        ticks_loc = ax[img,1].get_xticks().tolist()[1:len(angle_names)+1]
                    else:
                        ticks_loc = ax[img,1].get_xticks().tolist()[2:len(angle_names)+2]
                    ax[img,1].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax[img,1].set_xticklabels(angle_names,rotation=45)
                    ax[img,1].set_ylabel('Avg Pixel Count')
                    ax[img,1].set_title('{}'.format(feature))
                    
                    rects = ax[img,2].bar(y_pos,torch.mean(selected_Nfeat[img],dim=(1,2)).detach().cpu().numpy())
                    ax[img,2].xaxis.set_major_locator(mticker.MaxNLocator(len(angle_names),min_n_ticks=len(angle_names)-2))
                    if feature == 'EHD':
                        ticks_loc = ax[img,2].get_xticks().tolist()[1:len(angle_names)+1]
                    else:
                        ticks_loc = ax[img,2].get_xticks().tolist()[2:len(angle_names)+2]
                    ax[img,2].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                    ax[img,2].set_xticklabels(bin_names,rotation=45)
                    ax[img,2].set_ylabel('Avg Pixel Count')
                    ax[img,2].set_title('N{}'.format(feature))
                
                plt.tight_layout()
                temp_class_name = re.sub(r'\W+', '', class_names[temp_class])
                fig.savefig((hist_dir + '{}.png').format(temp_class_name,dpi=fig.dpi))
                plt.close()
    

        
            del features_extracted
            torch.cuda.empty_cache()
            
