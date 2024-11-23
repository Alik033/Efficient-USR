#!/usr/bin/env python3


import os
import cv2
import glob
import torch
from tqdm import tqdm

class EvalDataset:
    def __init__(   self,
                    dataset_name,
                    data_root,
                    batch_size  = 1,
                    image_scale = 4,
                    subffix='png'):
        """Initialize and preprocess the usr248 dataset."""
        self.data_root      = data_root
        self.image_scale    = image_scale
        self.dataset_name   = dataset_name
        self.subffix        = subffix
        self.dataset        = []
        self.pointer        = 0
        self.batch_size     = 1

        if self.dataset_name.lower() == "usr248":
            self.dataset_name = "USR248"
        elif self.dataset_name.lower() == "ufo120":
            self.dataset_name = "UFO120"
        elif self.dataset_name.lower() == "div2k":
            self.dataset_name = "DIV2K"
        else:
            raise FileNotFoundError
        print("%s dataset is used!"%self.dataset_name)

        self.__preprocess__()
        self.num_images     = len(self.dataset)

        
    
    def __preprocess__(self):
        """Preprocess the USR248 dataset."""
        if self.dataset_name == "DIV2K":
            #change
            dataset_hr_path = os.path.join(self.data_root, "hr")
            dataset_lr_path = os.path.join(self.data_root, "lr_%dx"%self.image_scale)
        else:
            dataset_hr_path = os.path.join(self.data_root, "hr")
            dataset_lr_path = os.path.join(self.data_root, "lr_%dx"%self.image_scale)
        print("Evaluation dataset HR path: %s"%dataset_hr_path)
        print("Evaluation dataset LR path: %s"%dataset_lr_path)
        assert os.path.exists(dataset_hr_path)
        assert os.path.exists(dataset_lr_path)
        data_paths  = []    
        print("processing %s images..."%self.dataset_name)
        temp_path   = os.path.join(dataset_hr_path,'*.%s'%(self.subffix))
        images      = glob.glob(temp_path)
        print(images)
        for item in images:
            file_name   = os.path.basename(item)
            lr_name     = os.path.join(dataset_lr_path, file_name)
            data_paths.append([item,lr_name])

        for item_pair in tqdm(data_paths):
            hr_img      = cv2.imread(item_pair[0])
            hr_img      = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
            hr_img      = hr_img.transpose((2,0,1))
            hr_img      = torch.from_numpy(hr_img)
            
            lr_img      = cv2.imread(item_pair[1])
            lr_img      = cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
            lr_img      = lr_img.transpose((2,0,1))
            lr_img      = torch.from_numpy(lr_img)
            
            self.dataset.append((hr_img,lr_img))
        # self.dataset = images
        print('Finished preprocessing the Validation dataset, total image number: %d...'%len(self.dataset))

    def __call__(self):
        """Return one batch images."""
        if self.pointer>=self.num_images:
            self.pointer = 0
        hr = self.dataset[self.pointer][0]
        lr = self.dataset[self.pointer][1]
        H,W = lr.shape[1:]
        hr = hr[:,:H*self.image_scale,:W*self.image_scale]
        hr = (hr/255.0 - 0.5) * 2.0
        lr = (lr/255.0 - 0.5) * 2.0
        hr = hr.unsqueeze(0)
        lr = lr.unsqueeze(0)
        self.pointer += 1
        return hr, lr
    
    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.data_root + ')'