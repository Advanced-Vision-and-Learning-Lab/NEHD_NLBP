# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:25:39 2021
Return index of built in Pytorch datasets (MNIST/FashionMNIST)
@author: jpeeples
"""
import PIL
from PIL import Image
import os
from medmnist.info import INFO
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as T
import ssl
    

ssl._create_default_https_context = ssl._create_unverified_context


class FashionMNIST_Index(Dataset):
    def __init__(self,directory,transform=None,train=True,download=True):  
        
        self.images = datasets.FashionMNIST(directory,train=train,transform=transform,
                                       download=download)

        self.targets = self.images.targets
        
    def __getitem__(self, index):
        data, target = self.images[index]
        
        return data, target, index

    def __len__(self):
        return len(self.images)
    
    
class PRMI_Index(Dataset):
    def __init__(self,directory,transform=None,train=True,download=True):  
        
        self.transform = transform
        if train:
            self.split = 'train'
        else:
            self.split = 'test'
        self.images = datasets.ImageFolder(directory,transform=transform)

        self.targets = self.images.targets
        
        self.classes = self.images.classes
        
    def __getitem__(self, index):
        data, target = self.images[index]
        
        return data, target, index

    def __len__(self):
        return len(self.images)
    
# Code added from Akshatha
class MedMNIST(Dataset):

    flag = ...

    def __init__(self,
                 root,
                 split='train',
                 as_rgb = False,
                 transform=None,
                 target_transform=None,
                 download=True,
                 ):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
    
        '''
        self.c=5
        self.info = INFO[self.flag]
        self.as_rgb = as_rgb
        self.root = root
        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(self.info["label"].values())

        if self.split == 'train':
            self.img = npz_file['train_images']
            self.label = npz_file['train_labels']
        elif self.split == 'val':
            self.img = npz_file['val_images']
            self.label = npz_file['val_labels']
        elif self.split == 'test':
            self.img = npz_file['test_images']
            self.label = npz_file['test_labels']

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(np.uint8(img))
        #img = img.convert('L')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return self.img.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.
        '''
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__

        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        body.append("Split: {}".format(self.split))
        body.append("Task: {}".format(self.info["task"]))
        body.append("Number of channels: {}".format(self.info["n_channels"]))
        body.append("Meaning of labels: {}".format(self.info["label"]))
        body.append("Number of samples: {}".format(self.info["n_samples"]))
        body.append("Description: {}".format(self.info["description"]))
        body.append("License: {}".format(self.info["license"]))

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=self.info["url"],
                         root=self.root,
                         filename="{}.npz".format(self.flag),
                         md5=self.info["MD5"])
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               'https://github.com/MedMNIST/MedMNIST')

class MedMNIST2D(MedMNIST):

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(img)
        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def save(self, folder, postfix="png", write_csv=True):

        from medmnist.utils import save2d

        save2d(imgs=self.img,
               labels=self.label,
               img_folder=os.path.join(folder, self.flag),
               split=self.split,
               postfix=postfix,
               csv_path=os.path.join(folder, f"{self.flag}.csv") if write_csv else None)

    def montage(self, length=20, replace=False, save_folder=None):
        from medmnist.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(imgs=self.img,
                                n_channels=self.info['n_channels'],
                                sel=sel)

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(os.path.join(save_folder,
                                          f"{self.flag}_{self.split}_montage.jpg"))

        return montage_img


class PathMNIST(MedMNIST):
    flag = "pathmnist"

class BloodMNIST(MedMNIST):
    flag = "bloodmnist"

class OCTMNIST(MedMNIST):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST):
    flag = "chestmnist"


class DermaMNIST(MedMNIST):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST):
    flag = "retinamnist"


class BreastMNIST(MedMNIST):
    flag = "breastmnist"


class OrganMNISTAxial(MedMNIST):
    flag = "organamnist"


class OrganMNISTCoronal(MedMNIST):
    flag = "organcmnist"


class OrganMNISTSagittal(MedMNIST):
    flag = "organsmnist"
