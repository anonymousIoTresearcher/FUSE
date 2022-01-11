import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import numpy as np
from MARS_CNN_model import CNN
from MARSloader import get_data
from copy import deepcopy
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--finetune_step', default= 200, type = int)
    parser.add_argument('--finetune_layers', default= 'last')
    
    return parser

parser = get_parser()
args = parser.parse_args()

# predict the user 4 movement 10 using the data
final_test_data = np.load("smallUC_data.npy")
final_test_label = np.load("smallUC_labels.npy")
#final_test_data = final_test_data.reshape((final_test_data.shape[0], final_test_data.shape[3], final_test_data.shape[2], final_test_data.shape[1]))
final_test_data = final_test_data.transpose(0,3,1,2)

final_train_data = np.load("bigUC_data.npy")
final_train_label = np.load("bigUC_labels.npy")
#final_train_data = final_train_data.reshape((final_train_data.shape[0], final_train_data.shape[3], final_train_data.shape[2], final_train_data.shape[1]))
final_train_data = final_train_data.transpose(0,3,1,2)




device = torch.device('cuda:0')
maml = torch.load('model/model.pth')
maml = maml.to(device)

if args.finetune_layers == 'last':

    for param in maml.parameters():
        param.requires_grad = False
    

    #for param in maml.fc.parameters():
    #    param.requires_grad = True
    #for param in maml.batchnorm2.parameters():
    #    param.requires_grad = True    
    ##for param in model.drop3.parameters():
    ##    param.requires_grad = True    
    for param in maml.regression.parameters():
        param.requires_grad = True
else:
    pass

# Declaring Loss and Optimizer
creterion = F.l1_loss
optimizer = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), betas=(0.5, 0.999))

# see error for all train data after training
maml.eval()
temp_mae = 0
with torch.no_grad():
    preds = maml(torch.from_numpy(final_train_data).cuda())
    temp_mae = mean_absolute_error(final_train_label, preds.cpu().detach().numpy())
    # temp_mae = torch.abs(torch.from_numpy(final_train_label).cpu() - preds.cpu()).sum().item()/(final_train_data.shape[0]*57)
    
final_train_error = temp_mae * 100
# np.save('final_predict_result.npy', preds.cpu().numpy())
print("final train mae before finetune is: %.2f " % (final_train_error))


maml.eval()
temp_mae = 0
# see all data's error
preds = maml(torch.from_numpy(final_test_data).cuda())
temp_mae = mean_absolute_error(final_test_label, preds.cpu().detach().numpy())
#temp_mae = torch.abs(torch.from_numpy(final_test_label.cpu() - preds.cpu()).sum().item()/(final_test_data.shape[0]*57)
print("all new users' mae before finetune is: %.2f " % (temp_mae*100))







finetuning_steps = args.finetune_step

# for finetune and testing with the fine-tune test set
test_mae_list = np.zeros((finetuning_steps,5))
train_mae_list = np.zeros((finetuning_steps,5))

for iterations in range(5):
    model = deepcopy(maml)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.5, 0.999))
    random_tune_index = np.random.choice(len(final_test_data), 200, False)
    random_test_index = np.setdiff1d(np.arange(len(final_test_data)), random_tune_index)
    finetune_data = final_test_data[random_tune_index]
    finetune_label = final_test_label[random_tune_index]
    
    finetest_data = final_test_data[random_test_index]
    finetest_label = final_test_label[random_test_index]
    #print("current iteration is: ", iterations)
    for step_count in range(finetuning_steps):
        #print("current step is: ", step_count)
        model.train()
        
        batch_data = torch.from_numpy(finetune_data).cuda()
        # print("shape of tune data is: ", batch_data.shape)
        batch_labels = torch.from_numpy(finetune_label).cuda()
        # forward
        batch_preds = model(batch_data)
        loss = creterion(batch_preds, batch_labels)
        # print("current loss is: ", loss)
        
        # set the gradient to zero
        optimizer.zero_grad()
        # do backward prop
        loss.backward()
        # update the gradient
        optimizer.step()


            
        model.eval()
        temp_mae = 0
        # see all data's error
        with torch.no_grad():
            preds = model(torch.from_numpy(finetest_data).cuda())
            temp_mae = mean_absolute_error(finetest_label, preds.cpu().detach().numpy())*100
        test_mae_list[step_count][iterations] = temp_mae
        #temp_mae = torch.abs(torch.from_numpy(final_test_label.cpu() - preds.cpu()).sum().item()/(final_test_data.shape[0]*57)
        # print("new users' test mae is: %.2f " % (temp_mae))
    
    
    
    
    
        # see error for all train data after finetune
        model.eval()
        temp_mae = 0
        with torch.no_grad():
            preds = model(torch.from_numpy(final_train_data).cuda())
            temp_mae = mean_absolute_error(final_train_label, preds.cpu().detach().numpy())*100
            # temp_mae = torch.abs(torch.from_numpy(final_train_label).cpu() - preds.cpu()).sum().item()/(final_train_data.shape[0]*57)
    
        train_mae_list[step_count][iterations] = temp_mae
        # print("final train mae after training is: %.2f " % (temp_mae))
    
    del model
    
#avg_test_mae_list  = np.array(test_mae_list)/iterations
#avg_train_mae_list  = np.array(train_mae_list)/iterations
#print("avg new users' mae for %d iter is: %.2f " % (iterations, test_mae_list[iterations-1]))
#print("avg old users' mae for %d iter is: %.2f " % (iterations, train_mae_list[iterations-1]))

np.save('result/cnn_test_mae_' + args.finetune_layers + '.npy', test_mae_list)
np.save('result/cnn_train_mae_' + args.finetune_layers + '.npy', train_mae_list)

