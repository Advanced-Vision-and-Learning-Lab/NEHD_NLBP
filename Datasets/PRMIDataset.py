# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:18:40 2022

@author: weihuang.xu
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
from sklearn import preprocessing

class PRMIDataset(Dataset):
    def __init__(self, root, subset, species=None, transform=None):   
        assert subset in ['train', 'val', 'test'], \
            "Subset can only be 'train','val' or 'test'."
        self.root = root
        self.subset = subset
        self.transform = transform
        self.files = []
        self.img_lt = []
        self.lab_lt = []
        # default species: cotton, papaya, sunflower, switchgrass
        if species == None:
            self.species = ['Cotton_736x552_DPI150',
                            'Papaya_736x552_DPI150',
                            'Sunflower_640x480_DPI120',
                            'Switchgrass_720x510_DPI300']
        else:
            self.species = species
        
        # get the list of image file that contains root
        for item in self.species:
            lab_json = os.path.join(self.root, self.subset, 'labels_image_gt',
                                    (item+'_'+self.subset+'.json'))
            with open(lab_json) as f:
              data = json.load(f)
              for img in data:
                  if img['has_root'] == 1:
                      img_dir = os.path.join(self.root, self.subset, 'images',
                                             item, img['image_name'])
                      lab = img['crop']
                      self.img_lt.append(img_dir)
                      self.lab_lt.append(lab)
          
        # encode the species type into class label
        label_encoder = preprocessing.LabelEncoder()
        self.lab_lt = label_encoder.fit_transform(self.lab_lt)

        for item in zip(self.img_lt, self.lab_lt):
            self.files.append({
                    "img": item[0],
                    "label": item[1]
                    })

    def __len__(self):     
        return len(self.files)
    
    def __getitem__(self, index):      
        datafiles = self.files[index]
        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        label = datafiles["label"]
        if self.transform is not None:
            img = self.transform(img)        
        return img, label, index

if __name__ == '__main__':
    from torchvision.transforms import Compose, ToTensor
    import matplotlib.pyplot as plt
    input_transform = Compose([ToTensor(),])
    path = '../../PRMI_official'
    RootsData = PRMIDataset(root=path, subset='test', transform=input_transform)
    print('The length of dataloader:%d'%len(RootsData))
    
    for i,(img, label) in enumerate(RootsData):
        img, gt = RootsData[i]
        # show example image in the dataset
        plt.figure()
        plt.imshow(img.permute(1,2,0))
        break