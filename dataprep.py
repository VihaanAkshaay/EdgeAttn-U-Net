import os
import numpy as np
import cv2
from random import randint
import matplotlib.pyplot as plt

'''
This function is used to load images from SWED_sample folder
'''

from PIL import Image

import torch

import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader


def DataPrep(batch_size=1):
    '''
    Reads through SWED_sample folder and retuns images and labels/masks
    '''

    #Getting into SWED_sample folder
    os.chdir('SWED_sample/')

    #Entering each image and labels folder for training
    image_path = os.path.join(os.getcwd(),"train/images/")
    mask_path = os.path.join(os.getcwd(),"train/labels/")

    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)

    #Dictionary for images and labels: dict[image name] = image
    image_dict = {}
    mask_dict = {}

    #Saving all train images (rgb channel) in image_list
    for i in range(len(image_list)):
    
        temp = np.load(os.path.join(image_path,image_list[i])).astype(np.float32)
        r = temp[:,:,3]
        g = temp[:,:,2]
        b = temp[:,:,1]

        temp = np.ndarray(shape=(3,256,256), dtype=float)
        temp[0] = r
        temp[1] = g
        temp[2] = b

    
        image_dict[image_list[i]] = temp.astype(np.float32)*0.0001


#print(image_dict[image_list[0]].shape) == (3,256,256)

#Saving all labels similar to above
    for i in range(len(mask_list)):
        mask_dict[mask_list[i]] = np.load(os.path.join(mask_path,mask_list[i])).astype(np.float32)
    
    image = []
    mask = []
    for img, msk in zip(image_dict.values(), mask_dict.values()):
    #print(len(img))
    #print(len(msk))
        image.append(img)
        mask.append(msk)
    

    #Converting list to np array
    image = np.array(image)
    mask = np.array(mask)


    return image,mask
#print(mask_dict[mask_list[0]].shape) == (1,256,256)

#X_train,Y_train = DataPrep()

#train_dataloader = DataLoader(X_train, batch_size=1, shuffle=False)
#label_dataloader = DataLoader(Y_train, batch_size=1, shuffle=False)
#input_train = next(iter(train_dataloader))
#label_train = next(iter(label_dataloader))
