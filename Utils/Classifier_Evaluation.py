# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:31:02 2020

@author: jpeeples
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier




def Classifier_Evaluation(dataloaders_dict,model,sub_dir,device,class_names):

  # Turn interactive plotting off, don't show plots
  plt.ioff()
    
  #TSNE visual of (all) data
  #Get labels and outputs
  features_dict = {}
  labels_dict = {}
  for phase in ['train', 'val', 'test']:
      GT_val = np.array(0)
      data_indices = np.array(0)
      model.eval()
      model.to(device)
      features_extracted = []
      
      for idx, (inputs, classes,index)  in enumerate(dataloaders_dict[phase]):
          images = inputs.to(device)
          labels = classes.to(device, torch.long)
          indices  = index.to(device).cpu().numpy()
          
          GT_val = np.concatenate((GT_val, labels.cpu().numpy()),axis = None)
          data_indices = np.concatenate((data_indices,indices),axis = None)
          
          features, _ = model(images)
          
          features = torch.flatten(features, start_dim=1)
          
          features = features.cpu().detach().numpy()
          # Perform PCA on the features
          # Adjust n by looking at the explained variance ratio
          #features = PCA(n_components=50,random_state=42).fit_transform(features)
          
          features_extracted.append(features)
          
    
      #Sub select images to view using TSNE
      GT_val = GT_val[1:]
      data_indices = data_indices[1:]
      
      features_extracted = np.concatenate(features_extracted,axis=0)

      #Save feature and labels for each datas split
      features_dict[phase] = features_extracted
      labels_dict[phase] = GT_val
      
  #Combine training and validation to form new training set
  features_dict['train'] = np.concatenate((features_dict['train'],features_dict['val']),axis=0)
  labels_dict['train'] = np.concatenate((labels_dict['train'],labels_dict['val']),axis=0)
  #For each split, perform PCA (fit based on training data)
  pca = PCA(n_components=50,random_state=42, whiten=True).fit(features_dict['train'])
  # Apply PCA on training data
  features_dict['train'] = pca.transform(features_dict['train'])
  # Apply PCA on test data
  features_dict['test'] = pca.transform(features_dict['test'])
  #Train classifier on training data


  # First KNNs
  KNN_classifier = KNeighborsClassifier(n_neighbors=5)
  KNN_classifier.fit(features_dict['train'],labels_dict['train'])
  #Evaluate on test data
  KNN_preds = KNN_classifier.predict(features_dict['test'])
  # Generates reports
  cm = confusion_matrix(labels_dict['test'], KNN_preds)
  report = classification_report(labels_dict['test'], KNN_preds,
                                  target_names=class_names,output_dict=True)
  # Make it into a csv
  df = pd.DataFrame(report).transpose()
  # Label run iteration
  df.to_csv((sub_dir+'KNN_Classifier_Report.csv'))

  # Train a SVM
  SVM_classifier = SVC(kernel='rbf',gamma='scale')
  SVM_classifier.fit(features_dict['train'],labels_dict['train'])
  #Evaluate on test data
  SVM_preds = SVM_classifier.predict(features_dict['test'])
  # Generates reports
  cm = confusion_matrix(labels_dict['test'], SVM_preds)
  report = classification_report(labels_dict['test'], SVM_preds,
                                    target_names=class_names,output_dict=True)
  # Make it into a csv
  df = pd.DataFrame(report).transpose()
  # Label run iteration
  df.to_csv((sub_dir+'SVM_Classifier_Report.csv'))

  # Train XGBoost
  XGB_classifier = XGBClassifier()
  XGB_classifier.fit(features_dict['train'],labels_dict['train'])
  #Evaluate on test data
  XGB_preds = XGB_classifier.predict(features_dict['test'])
  # Generates reports
  cm = confusion_matrix(labels_dict['test'], XGB_preds)
  report = classification_report(labels_dict['test'], XGB_preds,
                                    target_names=class_names,output_dict=True)
  # Make it into a csv
  df = pd.DataFrame(report).transpose()
  # Label run iteration
  df.to_csv((sub_dir+'XGB_Classifier_Report.csv'))
  

  # End the process
  del dataloaders_dict
  torch.cuda.empty_cache()

  # Save accuracies
  accuracies = [KNN_classifier.score(features_dict['test'],labels_dict['test']),
                SVM_classifier.score(features_dict['test'],labels_dict['test']),
                XGB_classifier.score(features_dict['test'],labels_dict['test'])]   
  # Return accuracies 
  return accuracies
