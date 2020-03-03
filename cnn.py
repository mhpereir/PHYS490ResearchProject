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
        FC_3          (1x3)             [Linear]
    '''
    
    def __init__(self):
        super(StarNet, self).__init__()
        
        self.conv1    = nn.Conv1d(in_channels=1, out_channels=4,  kernel_size=8, stride=1, padding=1)
        self.conv2    = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=8, stride=1, padding=1)
        self.maxpool  = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        
        self.fc1      = nn.Linear(179088,256)
        self.fc2      = nn.Linear(256,128)
        self.fc3      = nn.Linear(128,3)
        

    def init_data(self,data,cuda):
        if cuda:
            self.inputs_train  = torch.from_numpy(data.x_train).cuda().double()
            self.targets_train = torch.from_numpy(data.y_train).cuda().double()
            
            self.inputs_test  = torch.from_numpy(data.x_test).cuda().double()
            self.targets_test = torch.from_numpy(data.y_test).cuda().double()
        else:
            self.inputs_train  = torch.from_numpy(data.x_train).double()
            self.targets_train = torch.from_numpy(data.y_train).double()

            self.inputs_test  = torch.from_numpy(data.x_test).double()
            self.targets_test = torch.from_numpy(data.y_test).double()
            
            
    def forward(self,x):
        out = func.relu(self.conv1(x))
        out = func.relu(self.conv2(out))
        
        out = self.maxpool(out)
        out = out.view(out.size(0),-1)  #flatten for FC
        
        out = func.relu(self.fc1(out))
        out = func.relu(self.fc2(out))
        out = self.fc3(out)
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
        
    def test(self, loss):
        self.eval()
        with torch.no_grad():
            outputs= self(self.inputs_test)
            cross_val= loss(outputs, self.targets_test)  #self.forward(inputs)
        return cross_val.item()    
    
        
    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.maxpool.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
        
    