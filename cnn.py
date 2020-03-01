import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

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
        Output        (1x3)             [Linear]
    '''
    
    def __init__(self):
        super(StarNet, self).__init__()
        
        self.conv1    = nn.Conv2d(in_channels=1, out_channels=4,  kernel_size=8, stride=1, padding=1)
        self.conv2    = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=1, padding=1)
        self.maxpool  = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        
        self.fc1      = nn.Linear(256,128)
        self.fc2      = nn.Linear(128,3)
        

    def init_data(self,data,cuda):
        if cuda:
            self.inputs_train  = torch.from_numpy(data.x_train).cuda()
            self.targets_train = torch.from_numpy(data.y_train).cuda().long()
            
            self.inputs_test  = torch.from_numpy(data.x_test).cuda()
            self.targets_test = torch.from_numpy(data.y_test).cuda().long()
        else:
            self.inputs_train  = torch.from_numpy(data.x_train)
            self.targets_train = torch.from_numpy(data.y_train).long()

            self.inputs_test  = torch.from_numpy(data.x_test)
            self.targets_test = torch.from_numpy(data.y_test).long()
            
            
    def forward(self,x):
        out = func.relu(self.conv1(x))
        out = func.relu(self.conv2(out))
        
        out = self.maxpool(out)
        out = out.view(out.size(0),-1)  #flatten for FC
        
        out = func.relu(self.fc1(out))
        out = self.fc2(out)
        return out
        
        
    def backprop(self, loss, optimizer, n_train):
        self.train()
        
        args_batch = np.random.randint(0, len(self.inputs_train)-n_train)
        
        outputs= self(self.inputs_train[args_batch: args_batch+n_train])        
        obj_val= loss(outputs, self.targets_train[args_batch: args_batch+n_train])
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()
        
        
    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.maxpool.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
        
    