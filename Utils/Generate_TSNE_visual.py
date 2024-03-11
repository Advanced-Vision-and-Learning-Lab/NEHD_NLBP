# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:31:02 2020

@author: jpeeples
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import numpy as np
import torch
from matplotlib import offsetbox
from Utils.Compute_FDR import Compute_Fisher_Score
from sklearn.model_selection import train_test_split
import pdb

def Generate_TSNE_visual(dataloaders_dict,model,sub_dir,device,class_names,
                         Num_TSNE_images=None):

      # Turn interactive plotting off, don't show plots
        plt.ioff()
        
      #TSNE visual of (all) data
      #Get labels and outputs
        for phase in ['test']:
            GT_val = np.array(0)
            data_indices = np.array(0)
            model.eval()
            model.to(device)
            features_extracted = []
            saved_imgs = []
            for idx, (inputs, classes,index)  in enumerate(dataloaders_dict[phase]):
                images = inputs.to(device)
                labels = classes.to(device, torch.long)
                indices  = index.to(device).cpu().numpy()
                
                GT_val = np.concatenate((GT_val, labels.cpu().numpy()),axis = None)
                data_indices = np.concatenate((data_indices,indices),axis = None)
                
                features, _ = model(images)
                
                features = torch.flatten(features, start_dim=1)
                
                features = features.cpu().detach().numpy()
                
                features_extracted.append(features)
                saved_imgs.append(images.cpu().permute(0,2,3,1).numpy())
         
            #Sub select images to view using TSNE
            GT_val = GT_val[1:]
            data_indices = data_indices[1:]
            
    
            if (Num_TSNE_images is None) or (Num_TSNE_images > len(GT_val)):
                Num_TSNE_images = len(GT_val)
                temp_indices = np.arange(len(dataloaders_dict[phase].sampler))
            
            else:
                indices = np.arange(len(dataloaders_dict[phase].sampler))
                _,_,_,_,_,temp_indices = train_test_split(GT_val,GT_val,indices,
                                                          stratify=GT_val,
                                                          test_size = Num_TSNE_images,
                                                          random_state=42)
            
            features_extracted = np.concatenate(features_extracted,axis=0)
            saved_imgs = np.concatenate(saved_imgs,axis=0)
            
            #Compute FDR scores using all data
            num_classes = len(np.unique(GT_val))
            FDR_scores, log_FDR_scores = np.zeros(num_classes), np.zeros(num_classes)
            np.savetxt((sub_dir+'{}_FDR.txt'.format(phase)),FDR_scores,fmt='%.2E')
            np.savetxt((sub_dir+'{}_log_FDR.txt'.format(phase)),log_FDR_scores,fmt='%.2f')
    
            features_embedded = TSNE(n_components=2,verbose=1,init='pca',random_state=42).fit_transform(features_extracted[temp_indices])
            GT_val = GT_val[temp_indices]
            fig6, ax6 = plt.subplots()
            colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
            for texture in range (0, len(class_names)):
                x = features_embedded[[np.where(GT_val==texture)],0]
                y = features_embedded[[np.where(GT_val==texture)],1]
                
                ax6.scatter(x, y, color = colors[texture,:],label=class_names[texture])
            
            box = ax6.get_position()
            ax6.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            ax6.legend(loc='upper center',bbox_to_anchor=(.5,-.05),fancybox=True,ncol=len(class_names) // 2)
            plt.axis('off')
            
            fig6.savefig((sub_dir + 'TSNE_Visual_{}_Data.png'.format(phase.capitalize())), dpi=fig6.dpi)
            plt.close()
    
        # del dataloaders_dict,features_embedded
        torch.cuda.empty_cache()
        
        return FDR_scores, log_FDR_scores
