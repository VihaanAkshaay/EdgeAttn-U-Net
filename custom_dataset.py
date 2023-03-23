'''
This file helps in building dataloaders from the dataset folder to be used by segmentation models
'''


## Making custom dataset
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    '''
    Obtains the dataset directory name as input and outputs splits as image and labels that can be used by dataloaders directly
    '''
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.label_dir = os.path.join(data_dir, "labels")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        assert len(self.image_files) == len(self.label_files), "Number of images and labels should be same"

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])
        image = torch.from_numpy(np.load(image_path))
        temp = np.load(image_path).astype(np.float32)
        r = temp[:,:,3]
        g = temp[:,:,2]
        b = temp[:,:,1]
        image = np.ndarray(shape=(3,256,256), dtype=float)
        image[0] = r
        image[1] = g
        image[2] = b
        image = image.astype(np.float32)*0.0001
        label = torch.from_numpy(np.load(label_path).astype(np.float32))
        return image, label

    def __len__(self):
        return len(self.image_files)
