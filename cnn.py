import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

from time import time

class StarNet(nn.Module):
    '''
    Neural network class.
    Architecture:
        Input         (1x7214)
        Convolution_1 (4  @ 1x7214)     [ReLU]
        Convolution_2 (16 @ 1x7214)     [ReLU]
        Max Pooling   (16 @ 1x1803)
        FC_1          (1x256)           [ReLU]
        FC_2          (1x128)           [ReLU]
        FC_3          (1x3)             [Linear]
    '''
    
    def __init__(self):
        super(StarNet, self).__init__()
        
        self.conv1    = nn.Conv1d(in_channels=1, out_channels=4,  kernel_size=8, stride=1, padding=1)
        self.conv2    = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=8, stride=1, padding=1)
        self.maxpool  = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        
        self.fc1      = nn.Linear(28816,256)
        self.fc2      = nn.Linear(256,128)
        self.fc3      = nn.Linear(128,3)
        
        self.flag     = True

    def init_data(self,data,cuda):
        if cuda:
            self.inputs_train  = torch.from_numpy(data.x_train).cuda().float()
            self.targets_train = torch.from_numpy(data.y_train_norm).cuda().float()
            
            self.inputs_cross  = torch.from_numpy(data.x_cross).cuda().float()
            self.targets_cross = torch.from_numpy(data.y_cross_norm).cuda().float()
        else:
            self.inputs_train  = torch.from_numpy(data.x_train).float()
            self.targets_train = torch.from_numpy(data.y_train_norm).float()

            self.inputs_cross  = torch.from_numpy(data.x_cross).float()
            self.targets_cross = torch.from_numpy(data.y_cross_norm).float()
            
            
    def forward(self,x):
        out = func.relu(self.conv1(x))
        out = func.relu(self.conv2(out))
        
        out = self.maxpool(out)
        out = out.view(out.size(0),-1)  #flatten for FC
        
        out = func.relu(self.fc1(out))
        out = func.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        
        
    def backprop(self, data, loss, optimizer, n_train, device):
        self.train()
        
        n_total   = len(data.x_train[:,0,0])
        iters     = int(np.floor(n_total/n_train))
        loss_vals = []
        
        if self.flag:
            print('Running #{} iterations per epoch.'.format(iters))
            self.flag = False
        
        start_time = time()
        
        for i in range(iters):
            args_lower = i*n_train 
            args_upper = (i+1)*n_train
            
            
            inputs  = torch.from_numpy(data.x_train[args_lower:args_upper,:,:]).to(device).float()
            targets = torch.from_numpy(data.y_train_norm[args_lower:args_upper,:]).to(device).float()
            
            outputs= self(inputs)        
            obj_val= loss(outputs, targets)
            
            optimizer.zero_grad()
            obj_val.backward()
            optimizer.step()
            
            loss_vals.append(obj_val.item())
        
        if n_total % n_train != 0:
            #final step:    ##account for leftover indices in input array
            args_lower = (i+1)*n_train 
            args_upper = n_total
            
            
            inputs  = torch.from_numpy(data.x_train[args_lower:args_upper,:,:]).to(device).float()
            targets = torch.from_numpy(data.y_train_norm[args_lower:args_upper,:]).to(device).float()
            
            outputs= self(inputs)        
            obj_val= loss(outputs, targets)
            
            optimizer.zero_grad()
            obj_val.backward()
            optimizer.step()
            
            loss_vals.append(obj_val.item())
    
        ellapsed_time = (time() - start_time)/60
    
        return np.mean(loss_vals), ellapsed_time
            
    def cross(self, data, loss, device):
        inputs  = torch.from_numpy(data.x_cross[:,:,:]).to(device).float()
        targets = torch.from_numpy(data.y_cross_norm[:,:]).to(device).float()
        self.eval()
        with torch.no_grad():
            outputs   = self(inputs)
            cross_val = loss(outputs, targets)
            
        return cross_val.item()    
    
    def model_predictions(self, data, n_train, device):
        self.eval()
        with torch.no_grad():
            n_total   = len(data.x_test[:,0,0])
            iters     = int(np.floor(n_total/n_train))
            loss_vals = []
            
            print('Running #{} iterations for test data.'.format(iters))
            
            start_time = time()
            
            for i in range(iters):
                args_lower = i*n_train 
                args_upper = (i+1)*n_train
                
                
                inputs  = torch.from_numpy(data.x_test[args_lower:args_upper,:,:]).to(device).float()
                outputs= self(inputs) 
                
                if i == 0:
                    predicted_target_array = outputs.cpu().numpy()
                else:
                    predicted_target_array = np.concatenate((predicted_target_array, outputs.cpu().numpy()), axis=0)
            
            if n_total % n_train != 0:
                #final step:    ##account for leftover indices in input array
                args_lower = (i+1)*n_train 
                args_upper = n_total
                
                
                inputs  = torch.from_numpy(data.x_test[args_lower:args_upper,:,:]).to(device).float()
                outputs= self(inputs)
                
                predicted_target_array = np.concatenate((predicted_target_array, outputs.cpu().numpy()), axis=0)
        
        print('Target prediction ellapsed time: {:.4f}m'.format( (time()-start_time)/60) )
        
        return predicted_target_array
        
    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.maxpool.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        
        
    
