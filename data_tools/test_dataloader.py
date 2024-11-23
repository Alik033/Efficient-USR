#!/usr/bin/env python3

import os
import glob
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms as T

class TestDataset:
    def __init__(   self,
                    dataset_name,
                    data_root,
                    batch_size  = 16,
                    image_scale = 4,
                    subffix='png'):
        """Initialize and preprocess the dataset."""
        self.data_root      = data_root
        self.image_scale    = image_scale
        self.dataset_name   = dataset_name
        self.subffix        = subffix
        self.dataset        = []
        self.pointer        = 0
        self.batch_size     = batch_size
        self.__preprocess__()
        self.num_images = len(self.dataset)

        if self.dataset_name.lower() == "usr248":
            self.dataset_name = "USR248"
        elif self.dataset_name.lower() == "ufo120":
            self.dataset_name = "UFO120"
        elif self.dataset_name.lower() == "div2k":
            self.dataset_name = "DIV2K"
        elif self.dataset_name.lower() == "custom":
            self.dataset_name = "Custom"

        c_transforms  = []
        c_transforms.append(T.ToTensor())
        c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.img_transform = T.Compose(c_transforms)
    
    def __preprocess__(self):
        """Preprocess the dataset."""
        if self.dataset_name == "DIV2K":
            #change
            usr248hr_path  = os.path.join(self.data_root, "hr")
            usr248lr_path  = os.path.join(self.data_root, "lr_%dx"%self.image_scale)
        else:
            usr248hr_path  = os.path.join(self.data_root, "hr")
            usr248lr_path  = os.path.join(self.data_root, "lr_%dx"%self.image_scale)
        
        print("processing %s images..."%self.dataset_name)
        # import pdb; pdb.set_trace()
        # temp_path   = os.path.join(usr248hr_path,'*.%s'%(self.subffix))
        temp_path   = os.path.join(usr248hr_path,'*.%s'%(self.subffix))
        images      = glob.glob(temp_path)
        for item in images:
            file_name   = os.path.basename(item)
            lr_name     = os.path.join(usr248lr_path, file_name)
            self.dataset.append([item,lr_name])
        # self.dataset = images
        print('Finished preprocessing the %s dataset, total image number: %d...'%(self.dataset_name,len(self.dataset)))

    def __call__(self):
        """Return one batch images."""
        if self.pointer>=self.num_images:
            self.pointer = 0
            a = "The end of the story!"
            raise StopIteration(print(a))
        filename= self.dataset[self.pointer][0]
        image   = Image.open(filename)
        hr      = self.img_transform(image)
        filename= self.dataset[self.pointer][1]
        image   = Image.open(filename)
        lr      = self.img_transform(image)
        file_name   = os.path.basename(filename)
        file_name   = os.path.splitext(file_name)[0]
        hr_ls   = hr.unsqueeze(0)
        lr_ls   = lr.unsqueeze(0)
        nm_ls   = [file_name,]

        self.pointer += 1
        return hr_ls, lr_ls, nm_ls
    
    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.data_root + ')'