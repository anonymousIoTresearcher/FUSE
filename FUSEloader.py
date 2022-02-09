# -*- coding: utf-8 -*-
"""
Created on Mon May 10 00:06:28 2021

@author: sizhean
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import os
import scipy.io
import numpy as np
import random

import torch
import torch.utils.data as data
from torchvision import transforms, datasets


torch.set_default_tensor_type(torch.DoubleTensor)


torch.manual_seed(1)  # reproducible
 
transform = transforms.Compose([
    transforms.ToTensor(), 
])

path = os.getcwd()

os.chdir(path)


class MyDataset():
    def __init__(self, data, label):
        self.data = np.load(data) #
        self.label = np.load(label)
        self.transforms = transform #
    def __getitem__(self, index):
        datapoint= self.data[index, :, :, :]  # 
        datapoint = np.squeeze(datapoint)  # 
        labelpoint = self.label[index, :]  # 
        # labelpoint= self.transforms(labelpoint)  #
        datapoint= self.transforms(datapoint)  #
        return datapoint, labelpoint #
    def __len__(self):
        return self.data.shape[0] #

class MyDataset1():
    def __init__(self, data, label):
        self.data = np.load(data) #
        self.label = np.load(label)
        self.transforms = transform #转
    def __getitem__(self, index):
        datapoint= self.data[index, :, :, :]  # 
        datapoint = np.squeeze(datapoint)  # 
        labelpoint = self.label[index, :]  # 
        # labelpoint= self.transforms(labelpoint)  #
        datapoint= self.transforms(datapoint)  #
        return datapoint, labelpoint #
    def __len__(self):
        return self.data.shape[0] #



def get_data():


    print('load from FUSE npy.')

    # [1623, 20, 84, 84, 1]
    # 1623 classes, written by 20 different users, 84*84 size, grey channel
    # TODO: can not shuffle here, we must keep training and test set distinct!
    
    # [39, each movement]
    # each_movement = [frames, 5, 14, 14] number of frames, channels, size, size 
    # User1-3: 1-10; User4 - 1-8, 10

    
    
    
    
    
    
    # single frame
    dataset = MyDataset("FUSE/bigUC_data.npy", "FUSE/bigUC_labels.npy")
    # accumulated frames
    dataset1 = MyDataset1("FUSE/smallUC_data.npy", "FUSE/smallUC_labels.npy")

    
    # for mixed frames: 64, single frames: 128
    batch_size = 128
    test_split = .4
    shuffle_dataset = True
    random_seed= 42
    
    

    # Creating data indices for training and test splits:
#    dataset_size = len(dataset)
#    indices = list(range(dataset_size))
#    split = int(np.floor(test_split * dataset_size))
#    if shuffle_dataset :
#        np.random.seed(random_seed)
#        np.random.shuffle(indices)
#    train_indices, test_indices = indices[split:], indices[:split]
#    
    
    # Creating PT data samplers and loaders:
#    train_sampler = data.SubsetRandomSampler(train_indices)
#    test_sampler = data.SubsetRandomSampler(test_indices)
    
#    # specify the multiframe ratio
#    multi_frame_ratio = args.multi_ratio
#    frame_split = int(np.floor(multi_frame_ratio * len(train_indices)))
#    single_frame_indices, multi_frame_indices = train_indices[frame_split:], train_indices[:frame_split]
#    
#    single_sampler = data.SubsetRandomSampler(single_frame_indices)
#    multi_sampler = data.SubsetRandomSampler(multi_frame_indices)
#    
#    train_loader_single =  data.DataLoader(dataset, batch_size=batch_size, sampler = single_sampler)
#    train_loader_multi = data.DataLoader(dataset1, batch_size=batch_size, sampler = multi_sampler)
    
    # few shot training
#    few_split = int(np.floor(train_ratio * len(train_indices)))
#    train_few_indices = train_indices[:few_split]
#    few_sampler = data.SubsetRandomSampler(train_few_indices)
#    train_few_loader =  data.DataLoader(dataset1, batch_size=batch_size, sampler = few_sampler)
#    
#    train_loader =  data.DataLoader(dataset1, batch_size=batch_size, sampler = train_sampler)
#    test_loader =  data.DataLoader(dataset1, batch_size=batch_size, sampler = test_sampler)
#    
    
    # debug dataset
    dataset_size_big = len(dataset)
    total_indices_big = list(range(len(dataset)))
    split_big = int(np.floor(test_split * dataset_size_big))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(total_indices_big)
    train_indices_big, test_indices_big = total_indices_big[split_big:], total_indices_big[:split_big]
    
    train_sampler_big = data.SubsetRandomSampler(train_indices_big)
    test_sampler_big = data.SubsetRandomSampler(test_indices_big)
    
    train_loader_big =  data.DataLoader(dataset, batch_size=batch_size, sampler = train_sampler_big)
    test_loader_big =  data.DataLoader(dataset, batch_size=batch_size, sampler = test_sampler_big)    


    dataset1_size_small = len(dataset1)
    total_indices_small = list(range(len(dataset1)))
    split_small = int(np.floor(test_split * dataset1_size_small))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(total_indices_small)
    train_indices_small, test_indices_small = total_indices_small[split_small:], total_indices_small[:split_small]
    
    train_sampler_small = data.SubsetRandomSampler(train_indices_small)
    test_sampler_small = data.SubsetRandomSampler(test_indices_small)
    
    train_loader_small =  data.DataLoader(dataset1, batch_size=batch_size, sampler = train_sampler_small)
    test_loader_small =  data.DataLoader(dataset1, batch_size=batch_size, sampler = test_sampler_small)
    
    # for mix frames
    # return train_loader_single, train_loader_multi, test_loader, args
    
    # for one kind frames
    return train_loader_big, test_loader_big, train_loader_small, test_loader_small
