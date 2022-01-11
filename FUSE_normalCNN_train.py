# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 22:27:33 2021

@author: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import numpy as np
from MARS_CNN_model import CNN
from MARSloader import get_data



# if torch.cuda.is_available():
    # model = model.cuda()

train_loader_big, test_loader_big, train_loader_small, test_loader_small = get_data()

# load the final test and train all data

# predict the user 4 movement 10 using the data
final_test_data = np.load("smallUC_data.npy")
final_test_label = np.load("smallUC_labels.npy")
final_test_data = final_test_data.reshape((final_test_data.shape[0], final_test_data.shape[3], final_test_data.shape[2], final_test_data.shape[1]))

final_train_data = np.load("bigUC_data.npy")
final_train_label = np.load("bigUC_labels.npy")
final_train_data = final_train_data.reshape((final_train_data.shape[0], final_train_data.shape[3], final_train_data.shape[2], final_train_data.shape[1]))


#
device = torch.device('cuda:0')
model = CNN(5,57).to(device)
    
# Declaring Loss and Optimizer
creterion = F.l1_loss
optimizer = optim.Adam(model.parameters(), betas=(0.5, 0.999))


#train
total_epochs = 200
batch_iter = 0
best_epoch, best_mae = 0, 99999
finetuning_steps = 20


                   
#def reshape_data(data):
#    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
#    return data
#    
#def reshape_label(label):
#    label = label.reshape((label.shape[0]*label.shape[1], label.shape[2]))
#    return label
    
# for the main training
for epoch_count in range(total_epochs):
    model.train()
    


    for batch_data, batch_labels in train_loader_big:
    
        batch_data = batch_data.cuda()
        batch_labels = batch_labels.cuda()
        # forward
        batch_preds = model(batch_data)
        loss = creterion(batch_preds, batch_labels)
        # print("the current loss is: ", loss)
        # set the gradient to zero
        optimizer.zero_grad()
        # do backward prop
        loss.backward()
        # update the gradient
        optimizer.step()
        
        batch_iter += 1
        
        if batch_iter % 200 == 0:
            print('iter num: {} \t loss: {:.2f}'.format(batch_iter, loss.item()))
        
    
    val_loss = 0
    
    #for testing
    
    model.eval()
    total, mae, cnt = 0, 0, 0

    for batch_data, batch_labels in test_loader_big:
        batch_data = batch_data.cuda()
        # testing we don't want the gradient change
        with torch.no_grad():
            batch_preds = model(batch_data)
            mae = mean_absolute_error(batch_labels, batch_preds.cpu())
            #mae = torch.abs(batch_labels - batch_preds.cpu()).sum().item()/(batch_preds.shape[0]*57)
            total += mae
            cnt += 1
            
    model.train()
    
    test_error = total/cnt*100
    print("final train set test mae for epoch %d is: %.2f " % (epoch_count, test_error))


# see error for all train data after training
model.eval()
temp_mae = 0
with torch.no_grad():
    preds = model(torch.from_numpy(final_train_data).cuda())
    temp_mae = mean_absolute_error(final_train_label, preds.cpu().detach().numpy())
    # temp_mae = torch.abs(torch.from_numpy(final_train_label).cpu() - preds.cpu()).sum().item()/(final_train_data.shape[0]*57)
    
final_train_error = temp_mae * 100
# np.save('final_predict_result.npy', preds.cpu().numpy())
print("final train mae after training is: %.2f " % (final_train_error))


model.eval()
temp_mae = 0
# see all data's error
with torch.no_grad():
    preds = model(torch.from_numpy(final_test_data).cuda())
    temp_mae = mean_absolute_error(final_test_label, preds.cpu().detach().numpy())
#temp_mae = torch.abs(torch.from_numpy(final_test_label.cpu() - preds.cpu()).sum().item()/(final_test_data.shape[0]*57)
print("all new users' mae after training is: %.2f " % (temp_mae*100))





