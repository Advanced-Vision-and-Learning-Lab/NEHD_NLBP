# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:07:33 2019
Load datasets for models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import itertools
import pdb

## PyTorch dependencies
import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler #added
import torchvision.transforms as T

## Local external libraries
from Datasets.KTH_TIPS_2b import KTH_TIPS_2b_data
from sklearn.model_selection import train_test_split
from Datasets.Pytorch_Datasets import MNIST_Index, FashionMNIST_Index, SVHN_Index, CIFAR10_Index, UCMerced_Index, EUROSAT_Index
from Datasets.PRMIDataset import PRMIDataset
from Datasets.Pytorch_Datasets import DermaMNIST, BloodMNIST

###
# Added functions to get the mean, std for one channel
###
def get_mean_1Ch(img_dataset):
    '''
    This function returns the mean for 1 channel
    '''
    # Initialize variables to accumulate pixel values and count
    total_sum = 0

    # Iterate through the dataset to calculate the sum of pixel values
    for data, _, _ in img_dataset:
        # Accumulate the sum of pixel values
        total_sum += torch.mean(data)  

    # Calculate the mean
    mean = total_sum / len(img_dataset)
    
    return (mean.item(),)

def get_std_1Ch(img_dataset):
    '''
    This function returns the std for 1 channel
    '''
    # Iterate through the dataset again to calculate the sum of squared differences
    std_sum = 0
    for data, _, _ in img_dataset:
        std_sum += data.std()

    # Calculate the variance
    std = std_sum / len(img_dataset)
    return (std.item(),)


###
# Get 3 channel mean and std
###

def get_mean_3Ch(img_dataset):
    '''
    This function returns the mean for 3 channels
    '''
    # Initialize variables to accumulate pixel values and count
    total_sum = torch.zeros(3)

    # Iterate through the dataset to calculate the sum of pixel values
    for data, _, _ in img_dataset:
        # Ensure that data is in the correct shape (3, height, width) (below line was added by chatgpt after asking for debugging help)
        data = data.permute(1, 2, 0)  # Change to (height, width, 3) format

        # Calculate the mean and accumulate for each channel
        channel_means = torch.mean(data, dim=(0, 1))
        total_sum += channel_means

    # Calculate the mean for each channel
    mean = total_sum / len(img_dataset)
    print((mean[0].item(), mean[1].item(), mean[2].item(),))
    return ((mean[0].item(), mean[1].item(), mean[2].item(),))

def get_std_3Ch(img_dataset):
    '''
    This function returns the std for 3 channels
    '''
    total_std_sum = torch.zeros(3)
    # Iterate through the dataset again to calculate the sum of squared differences
    for data, _, _ in img_dataset:
        # Ensure that data is in the correct shape (3, height, width)
        data = data.permute(1, 2, 0)  # Change to (height, width, 3) format

        # Calculate the standard deviation and accumulate for each channel
        channel_stds = data.std(dim=(0, 1))
        total_std_sum += channel_stds

    # Calculate the standard deviation for each channel
    std = total_std_sum / len(img_dataset)
    print((std[0].item(), std[1].item(), std[2].item(),))
    return ((std[0].item(), std[1].item(), std[2].item(),))

def Prepare_DataLoaders(Network_parameters, split=None,
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                        val_percent=0.1, random_state=42):
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']

    fusion_method = Network_parameters['fusion_method']
    
    dataloaders_dict = None
    if Dataset == 'MNIST': #(0.1307,), (0.3081,)
        
        if fusion_method is not None:
            raise RuntimeError('Fusion not implemented for MNIST')
        
        initial_transform = transforms.Compose([transforms.ToTensor()])
        img_dataset = MNIST_Index('./Datasets/',train=True,transform=initial_transform,
                                       download=True)

        # Get the targets with no major transforms to prevent data leakage
        y = img_dataset.targets
        indices = np.arange(len(y))
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                          test_size=val_percent, 
                                                          stratify=y, random_state=random_state)
        



        train_dataset = torch.utils.data.Subset(img_dataset, train_indices)
        # Now get the mean, std for the train only dataset // Expecting 0.1307, 0.3081
        mean = get_mean_1Ch(train_dataset)
        std = get_std_1Ch(train_dataset)
        print(mean, std)
        ####

        validation_dataset = torch.utils.data.Subset(img_dataset, val_indices)
        
        test_dataset = MNIST_Index(data_dir,train=False,transform=initial_transform,
                                       download=True)
        
        # Now apply the transforms to the train, val, test datasets
        transform=transforms.Compose([transforms.Normalize(mean, std)])
        train_dataset.dataset.transform = transform
        validation_dataset.dataset.transform = transform
        test_dataset.transform = transform
        

    elif Dataset == 'Fashion_MNIST': #See people also use .5, .5 for normalization
        if fusion_method is not None:
            raise RuntimeError('Fusion not implemented for Fashion MNIST')
        
        initial_transform = transforms.Compose([transforms.ToTensor()])
  
        img_dataset = FashionMNIST_Index(data_dir,train=True,transform=initial_transform,
                                       download=True)
        
        # Get the targets with no major transforms to prevent data leakage
        y = img_dataset.targets
        indices = np.arange(len(y))
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                          test_size=val_percent, 
                                                          stratify=y, random_state=random_state)
        train_dataset = torch.utils.data.Subset(img_dataset, train_indices)
        # Now get the mean, std for the train only dataset // Expecting 0.1307, 0.3081
        mean = get_mean_1Ch(train_dataset)
        std = get_std_1Ch(train_dataset)
        print(mean, std)
        ####
        validation_dataset = torch.utils.data.Subset(img_dataset, val_indices)
        
        test_dataset = FashionMNIST_Index(data_dir,train=False,transform=initial_transform,
                                       download=True)
        
        # Now apply the transforms to the train, val, test datasets
        transform=transforms.Compose([transforms.Normalize(mean, std)])
        train_dataset.dataset.transform = transform
        validation_dataset.dataset.transform = transform
        test_dataset.transform = transform

    elif Dataset == 'KTH-TIPS-2b':
        samples = ['a', 'b', 'c', 'd']
        # Set to 1 to train on 1, test 3; set to 2 to train on 2, test on 2;
        # set to 3 to train on 3 and test 1
        setting = 3

        # Define the transforms for the training and testing sets
        if fusion_method == "grayscale":
            print("Implementing KTH as grayscale")
            initial_transform = data_transforms = {
                'train': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(Network_parameters['resize_size']),
                    transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'test': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(Network_parameters['resize_size']),
                    transforms.CenterCrop(Network_parameters['center_size']),
                    transforms.ToTensor(),
                ]),
            }
            
            sample_combos = list(itertools.combinations(samples, setting))
            train_setting = []
            test_setting = []
            for ii in range(0, len(sample_combos)):
                train_setting.append(list(sample_combos[ii]))
                test_setting.append(list(sorted(set(samples) - set(sample_combos[ii]))))
            
            train_val_dataset = KTH_TIPS_2b_data(data_dir,train=True,
                                            img_transform=data_transforms['train'],
                                            train_setting=train_setting[0]) 
            test_dataset = KTH_TIPS_2b_data(data_dir,train=False,
                                            img_transform=data_transforms['test'],
                                            test_setting=test_setting[0])
            indices = np.arange(len(train_val_dataset))
            y = [sub['label'] for sub in train_val_dataset.files]

            # Use stratified split to balance training validation splits, set random state to be same for each encoding method
            _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, stratify=y, test_size=val_percent,
                                                                    random_state= random_state)

            train_dataset = torch.utils.data.Subset(train_val_dataset,train_indices)
            # Now get the mean, std for the train only dataset // Expecting TBD
            mean = get_mean_1Ch(train_dataset)
            std = get_std_1Ch(train_dataset)
            print(mean, std)
        else:
            print("Implementing KTH as Conv Fusion or Indepedently")
            initial_transform = data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(Network_parameters['resize_size']),
                    transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'test': transforms.Compose([
                    transforms.Resize(Network_parameters['resize_size']),
                    transforms.CenterCrop(Network_parameters['center_size']),
                    transforms.ToTensor(),
                ]),
            }

            sample_combos = list(itertools.combinations(samples, setting))
            train_setting = []
            test_setting = []
            for ii in range(0, len(sample_combos)):
                train_setting.append(list(sample_combos[ii]))
                test_setting.append(list(sorted(set(samples) - set(sample_combos[ii]))))
            
            train_val_dataset = KTH_TIPS_2b_data(data_dir,train=True,
                                            img_transform=data_transforms['train'],
                                            train_setting=train_setting[0]) 
            test_dataset = KTH_TIPS_2b_data(data_dir,train=False,
                                            img_transform=data_transforms['test'],
                                            test_setting=test_setting[0])
            indices = np.arange(len(train_val_dataset))
            y = [sub['label'] for sub in train_val_dataset.files]

            # Use stratified split to balance training validation splits, set random state to be same for each encoding method
            _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, stratify=y, test_size=val_percent,
                                                                    random_state= random_state)

            train_dataset = torch.utils.data.Subset(train_val_dataset,train_indices)
            # Now get the mean, std for the train only dataset // Expecting TBD
            mean = get_mean_3Ch(train_dataset)
            std = get_std_3Ch(train_dataset)
            print(mean, std)
        ####
        validation_dataset = torch.utils.data.Subset(train_val_dataset,val_indices) 
        # TO DO: Verify there is no test dataset in KTH
        # Now apply the transforms to the train, val datasets
        transform=transforms.Compose([transforms.Normalize(mean, std)])
        train_dataset.dataset.transform = transform
        validation_dataset.dataset.transform = transform
        # Done?

    
    elif Dataset == 'SVHN': #See people also use .5, .5 for normalization

        if fusion_method == "grayscale":
            print("Implementing SVHN as grayscale")
            initial_transform = {
                        'train': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Grayscale(num_output_channels=1)
                        ]),
                        'test': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.CenterCrop(Network_parameters['center_size']),
                            transforms.ToTensor(),
                            transforms.Grayscale(num_output_channels=1)
                        ]),
                    }
            # Call train and test
            train_dataset = SVHN_Index(data_dir,train=True,transform=initial_transform['train'],
                                        download=True)
            test_dataset = SVHN_Index('./Datasets/', train=False, transform=initial_transform['test'], download=True)

            y = train_dataset.targets
            indices = np.arange(len(y))
            _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                            test_size=0.1, 
                                                            stratify=y, random_state=42)
            
            # Creating PT data samplers and loaders:

            # Now get the mean, std for the train only dataset // Expecting TBD
            mean = get_mean_1Ch(train_dataset)
            std = get_std_1Ch(train_dataset)
            print("Mean and std for SVHN")
            print(mean, std)
            ####
        else: 
            print("Implementing SVHN as Conv Fusion or Indepedently")
            initial_transform = {
                        'train': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),
                        'test': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.CenterCrop(Network_parameters['center_size']),
                            transforms.ToTensor(),
                        ]),
                    }
            # Call train and test
            train_dataset = SVHN_Index(data_dir,train=True,transform=initial_transform['train'],
                                        download=True)
            test_dataset = SVHN_Index('./Datasets/', train=False, transform=initial_transform['test'], download=True)

            y = train_dataset.targets
            indices = np.arange(len(y))
            _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                            test_size=0.1, 
                                                            stratify=y, random_state=42)
            
            # Creating PT data samplers and loaders:

            # Now get the mean, std for the train only dataset // Expecting TBD
            mean = get_mean_3Ch(train_dataset)
            std = get_std_3Ch(train_dataset)
            print("Mean and std for SVHN")
            print(mean, std)
            ####

        # Create the transform to normalize the data
        normalize_transform = transforms.Normalize(mean, std)

        # Define validation dataset
        validation_dataset = torch.utils.data.Subset(train_dataset, val_indices)

        # Apply the transforms to the datasets
        train_dataset.transform = transforms.Compose([initial_transform['train'], normalize_transform])
        validation_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])
        test_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])

        # Ensure this works as expected
        train_dataset = SVHN_Index('./Datasets/', train=True, transform=train_dataset.transform, download=False)
        test_dataset = SVHN_Index('./Datasets/', train=False, transform=test_dataset.transform, download=False)

        
    ## Added CiFar # See if my SVHN Code works first
    elif Dataset == 'CIFAR10': #See people also use .5, .5 for normalization
        ###
        # Implementing strictly as NN to grayscale for now
        ###
        # print("Implementing CiFar as NN to grayscale")
        initial_transform = {
                    'train': transforms.Compose([
                        transforms.Resize(Network_parameters['resize_size']),
                        transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]),
                    'test': transforms.Compose([
                        transforms.Resize(Network_parameters['resize_size']),
                        transforms.CenterCrop(Network_parameters['center_size']),
                        transforms.ToTensor(),
                    ]),
                }
        
        # Call train and test
        train_dataset = CIFAR10_Index(data_dir,train=True,transform=initial_transform['train'],
                                       download=True)
        test_dataset = CIFAR10_Index(data_dir,train=False,transform=initial_transform['test'],
                                       download=True)
        
        y = train_dataset.targets
        
        indices = np.arange(len(y))
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                          test_size=0.1, 
                                                          stratify=y, random_state=42)
        

        # Now get the mean, std for the train only dataset // Expecting TBD
        #mean = get_mean_3Ch(train_dataset)
        #std = get_std_3Ch(train_dataset)
        print("Mean and std for CiFar") 
        mean = (0.4905622899532318, 0.4792354702949524, 0.44345641136169434)
        std = (0.19373157620429993, 0.19092957675457, 0.19207505881786346)
        # Expecting (0.49139968, 0.48215827 ,0.44653124) // (0.24703233, 0.24348505, 0.26158768)
        print(mean, std)
        ####

        # Create the transform to normalize the data
        normalize_transform = transforms.Normalize(mean, std)

        # Define validation dataset
        validation_dataset = torch.utils.data.Subset(train_dataset, val_indices)

        # Apply the transforms to the datasets
        train_dataset.transform = transforms.Compose([initial_transform['train'], normalize_transform])
        validation_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])
        test_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])

        # Apply the transforms to the datasets
        train_dataset = CIFAR10_Index('./Datasets/', train=True, transform=train_dataset.transform, download=False)
        test_dataset = CIFAR10_Index('./Datasets/', train=False, transform=train_dataset.transform, download=False)

    elif Dataset == 'PRMI':
        print("Implementing PRMI")
        if fusion_method == "grayscale":
            print("Implementing PRMI as grayscale")
            initial_transform = {
                        'train': transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),
                        'test': transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.CenterCrop(Network_parameters['center_size']),
                            transforms.ToTensor(),
                        ]),
                    }
            # Call train and test
            train_dataset = PRMIDataset(data_dir,subset='train',transform=initial_transform['train'])
            test_dataset = PRMIDataset(data_dir, subset='test', transform=initial_transform['test'])
            validation_dataset = PRMIDataset(data_dir, subset='val', transform=initial_transform['test'])

            # Now get the mean, std for the train only dataset // Expecting TBD
            mean = get_mean_1Ch(train_dataset)
            std = get_std_1Ch(train_dataset)
            print("Mean and std for PRMI")
            print(mean, std)
            ####
        else: 
            print("Implementing PRMI as Conv Fusion or Indepedently")
            initial_transform = {
                        'train': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),
                        'test': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.CenterCrop(Network_parameters['center_size']),
                            transforms.ToTensor(),
                        ]),
                    }
            # Call train and test
            train_dataset = PRMIDataset(data_dir,subset='train',transform=initial_transform['train'])
            test_dataset = PRMIDataset(data_dir, subset='test', transform=initial_transform['test'])
            validation_dataset = PRMIDataset(data_dir, subset='val', transform=initial_transform['test'])
            
            # Creating PT data samplers and loaders:

            # Now get the mean, std for the train only dataset // Expecting TBD
            mean = get_mean_3Ch(train_dataset)
            std = get_std_3Ch(train_dataset)
            print("Mean and std for PRMI")
            print(mean, std)
            ####

        # Create the transform to normalize the data
        normalize_transform = transforms.Normalize(mean, std)

        # Apply the transforms to the datasets
        train_dataset.transform = transforms.Compose([initial_transform['train'], normalize_transform])
        validation_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])
        test_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])

        # Ensure this works as expected
        train_dataset = PRMIDataset(data_dir, subset='train', transform=train_dataset.transform)
        validation_dataset = PRMIDataset(data_dir, subset='val', transform=validation_dataset.transform)
        test_dataset = PRMIDataset(data_dir, subset='test', transform=test_dataset.transform)

    elif Dataset == 'Derma_MNIST':
        print("Implementing DermaMNIST as color")
        initial_transform = {
            'train': transforms.Compose([
                #transforms.Grayscale(num_output_channels=1),
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                #transforms.Grayscale(num_output_channels=1),
                transforms.Resize(Network_parameters['resize_size']),
                transforms.CenterCrop(Network_parameters['center_size']),
                transforms.ToTensor(),
            ]),
        }
        # Call train and test
        train_dataset = DermaMNIST(data_dir, split='train', transform = initial_transform['train'], target_transform=None)
        test_dataset = DermaMNIST(data_dir, split='test', transform = initial_transform['test'], target_transform=None)
        validation_dataset = DermaMNIST(data_dir, split='val', transform = initial_transform['test'], target_transform=None)

        # Now get the mean, std for the train only dataset // Expecting TBD
        mean = get_mean_3Ch(train_dataset)
        std = get_std_3Ch(train_dataset)
        print("Mean and std for DermaMNIST")
        print(mean, std)
        ####
       

        # Create the transform to normalize the data
        normalize_transform = transforms.Normalize(mean, std)

        # Apply the transforms to the datasets
        train_dataset.transform = transforms.Compose([initial_transform['train'], normalize_transform])
        validation_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])
        test_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])

        # Ensure this works as expected
        train_dataset = DermaMNIST(data_dir, split='train', transform = train_dataset.transform, target_transform= None)
        test_dataset = DermaMNIST(data_dir, split='test', transform =test_dataset.transform, target_transform= None)
        validation_dataset = DermaMNIST(data_dir, split='val', transform = test_dataset.transform, target_transform= None)

        # Flatten the labels
        train_dataset.label = train_dataset.label.flatten()
        test_dataset.label = test_dataset.label.flatten()
        validation_dataset.label = validation_dataset.label.flatten()

    elif Dataset == 'BloodMNIST':
        print("Implementing BloodMNIST as color")
        initial_transform = {
            'train': transforms.Compose([
                #transforms.Grayscale(num_output_channels=1),
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                #transforms.Grayscale(num_output_channels=1),
                transforms.Resize(Network_parameters['resize_size']),
                transforms.CenterCrop(Network_parameters['center_size']),
                transforms.ToTensor(),
            ]),
        }
        # Call train and test
        train_dataset = BloodMNIST(data_dir, split='train', transform = initial_transform['train'], target_transform=None)
        test_dataset = BloodMNIST(data_dir, split='test', transform = initial_transform['test'], target_transform=None)
        validation_dataset = BloodMNIST(data_dir, split='val', transform = initial_transform['test'], target_transform=None)

        # Now get the mean, std for the train only dataset // Expecting TBD
        mean = get_mean_3Ch(train_dataset)
        std = get_std_3Ch(train_dataset)
        print("Mean and std for BloodMNIST")
        print(mean, std)
        ####
       

        # Create the transform to normalize the data
        normalize_transform = transforms.Normalize(mean, std)

        # Apply the transforms to the datasets
        train_dataset.transform = transforms.Compose([initial_transform['train'], normalize_transform])
        validation_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])
        test_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])

        # Ensure this works as expected
        train_dataset = BloodMNIST(data_dir, split='train', transform = train_dataset.transform, target_transform= None)
        test_dataset = BloodMNIST(data_dir, split='test', transform =test_dataset.transform, target_transform= None)
        validation_dataset = BloodMNIST(data_dir, split='val', transform = test_dataset.transform, target_transform= None)

        # Flatten the labels
        train_dataset.label = train_dataset.label.flatten()
        test_dataset.label = test_dataset.label.flatten()
        validation_dataset.label = validation_dataset.label.flatten()

    elif Dataset == 'UCMerced_Index':
        print("Implementing UCMerced_Index")
        if fusion_method == "grayscale":
            print("Implementing UCMerced_Index as grayscale")
            initial_transform = {
                        'train': transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),
                        'test': transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.CenterCrop(Network_parameters['center_size']),
                            transforms.ToTensor(),
                        ]),
                    }
            # Call train and test
            train_dataset = UCMerced_Index(root = data_dir,split='train', transform = initial_transform['train'], download=True)
            test_dataset = UCMerced_Index(data_dir,split='test', transform = initial_transform['test'], download=True)
            validation_dataset = UCMerced_Index(data_dir,split='val', transform = initial_transform['test'], download=True)

            # Now get the mean, std for the train only dataset // Expecting TBD
            mean = get_mean_1Ch(train_dataset)
            std = get_std_1Ch(train_dataset)
            print("Mean and std for PRMI")
            print(mean, std)
            ####
        else: 
            print("Implementing UCMerced_Index as Conv Fusion or Indepedently")
            initial_transform = {
                        'train': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),
                        'test': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.CenterCrop(Network_parameters['center_size']),
                            transforms.ToTensor(),
                        ]),
                    }
            # Call train and test
            train_dataset = UCMerced_Index(root = data_dir,split='train', transform = initial_transform['train'], download=True)
            test_dataset = UCMerced_Index(data_dir,split='test', transform = initial_transform['test'], download=True)
            validation_dataset = UCMerced_Index(data_dir,split='val', transform = initial_transform['test'], download=True)
            
            # Creating PT data samplers and loaders:

            # Now get the mean, std for the train only dataset // Expecting TBD
            mean = get_mean_3Ch(train_dataset)
            std = get_std_3Ch(train_dataset)
            print("Mean and std for PRMI")
            print(mean, std)
            ####

        # Create the transform to normalize the data
        normalize_transform = transforms.Normalize(mean, std)

        # Apply the transforms to the datasets
        train_dataset.transform = transforms.Compose([initial_transform['train'], normalize_transform])
        validation_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])
        test_dataset.transform = transforms.Compose([initial_transform['test'], normalize_transform])

        # Ensure this works as expected
        train_dataset = UCMerced_Index(root = data_dir,split='train', transform = train_dataset.transform, download=True)
        test_dataset = UCMerced_Index(data_dir,split='test', transform = test_dataset.transform, download=True)
        validation_dataset = UCMerced_Index(data_dir,split='val', transform = validation_dataset.transform, download=True)

    elif Dataset == 'Eurosat': #See people also use .5, .5 for normalization
        print("Implementing Eurosat")
        if fusion_method == "grayscale":
            print("Implementing Eurosat as grayscale")
            initial_transform = {
                        'train': transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),
                        'test': transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.CenterCrop(Network_parameters['center_size']),
                            transforms.ToTensor(),
                        ]),
                    }
  
            img_dataset = EUROSAT_Index(data_dir, transform=initial_transform,
                                        download=True)
            
            # Get the targets with no major transforms to prevent data leakage
            indices = np.arange(len(y))
            _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                            test_size=val_percent, 
                                                            stratify=y, random_state=random_state)

            # Further split the training set into training and test sets
            train_indices, test_indices = train_test_split(train_indices, test_size=val_percent, 
                                                        stratify=y[train_indices], random_state=random_state)

            # Create datasets using the indices
            train_dataset = torch.utils.data.Subset(img_dataset, train_indices)

            # Now get the mean, std for the train only dataset
            mean = get_mean_1Ch(train_dataset)
            std = get_std_1Ch(train_dataset)
            print(mean, std)
            ####
        else: 
            print("Implementing Eurosat as Conv Fusion or Indepedently")
            initial_transform = {
                        'train': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.RandomResizedCrop(Network_parameters['center_size'],scale=(.8,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]),
                        'test': transforms.Compose([
                            transforms.Resize(Network_parameters['resize_size']),
                            transforms.CenterCrop(Network_parameters['center_size']),
                            transforms.ToTensor(),
                        ]),
                    }
            img_dataset = EUROSAT_Index(data_dir,transform=initial_transform,
                                        download=True)
            
            # Get the targets with no major transforms to prevent data leakage
            indices = np.arange(len(y))
            _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                            test_size=val_percent, 
                                                            stratify=y, random_state=random_state)

            # Further split the training set into training and test sets
            train_indices, test_indices = train_test_split(train_indices, test_size=val_percent, 
                                                        stratify=y[train_indices], random_state=random_state)

            # Create datasets using the indices
            train_dataset = torch.utils.data.Subset(img_dataset, train_indices)
            # Now get the mean, std for the train only dataset // Expecting 0.1307, 0.3081
            mean = get_mean_3Ch(train_dataset)
            std = get_std_3Ch(train_dataset)
            print(mean, std)
            ####
            
        
        # Now apply the transforms to the train, val, test datasets
        validation_dataset = torch.utils.data.Subset(img_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(img_dataset, test_indices)
        transform=transforms.Compose([transforms.Normalize(mean, std)])
        train_dataset.dataset.transform = transform
        validation_dataset.dataset.transform = transform
        test_dataset.transform = transform

    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset))
        
    if dataloaders_dict is None:
        image_datasets = {'train': train_dataset, 'val': validation_dataset,
                          'test': test_dataset}
        
       
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                           batch_size=Network_parameters['batch_size'][x],
                                                           shuffle=False, 
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory']) for x in ['train', 'val','test']}            
    
    return dataloaders_dict