import json, argparse, torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np

import torch.nn as nn

from time import time
from cnn import StarNet
from data_utils import Data
from processing_utils import post_processing

if __name__ == '__main__':
    start_time = time()
    # Command line arguments (taken from tutorial script (lesson 5))
    parser = argparse.ArgumentParser(description='Final Project: CNN script')
    parser.add_argument('--data_path', metavar='path',
                        help='path to data directory')
    parser.add_argument('--train', metavar='str',
                        help='which training dataset to use: real or synth')
    parser.add_argument('--test', metavar='str',
                        help='which testing dataset to use: real or synth')
    parser.add_argument('-o', metavar='results', default='results/',
                        help='path to results')
    parser.add_argument('-v', type=int, default=2, metavar='N',
                        help='verbosity (default: 2)')
    parser.add_argument('-c', type=int, default=0, metavar='N',
                       help='cuda indicator (default: 0 = OFF)')
    args = parser.parse_args()
    
    data_file_path   = args.data_path
    train_data       = args.train
    test_data        = args.test
    output_path      = args.o
    verb             = args.v
    cuda_input       = args.c
    
    params_file_path = str(data_file_path) + '/params.json'
    
    
    with open(params_file_path) as paramfile:
        param_file = json.load(paramfile)
    
    data  = Data(data_file_path, train_data, test_data, param_file['n_cross'])
    model = StarNet().float()
    
    model.init_data(data,cuda_input)
    
    # Define an optimizer and the loss function
    optimizer  = optim.Adam(model.parameters(), lr=param_file['lr'])
    loss       = torch.nn.MSELoss()
    
    obj_vals   = []
    cross_vals = []
    num_epochs = int(param_file['n_epoch'])
    
    #model = nn.DataParallel(model)
    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_val = model.backprop(loss, optimizer, n_train=param_file['n_mini_batch'])
        obj_vals.append(train_val)
        
        cross_val = model.cross(loss)
        cross_vals.append(cross_val)
        
        # High verbosity report in output stream
        if args.v>=2:
            if not ((epoch + 1) % int(param_file['n_epoch_v'])):
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTest Loss: {:.4f}'.format(cross_val))
    
    # Low verbosity final report
    if args.v>=1:
        print('Final training loss: {:.4f}'.format(obj_vals[-1]))
        print('Final test loss: {:.4f}'.format(cross_vals[-1]))
    
    print('Ellapsed time: {}m'.format( (time()-start_time)/60) )
    
    # Plot saved in results folder
    fig,ax = plt.subplots()
    ax.plot(range(num_epochs), obj_vals, label= "Training loss", color="blue")
    ax.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
    ax.legend()
    fig.savefig('results/loss.pdf')
    plt.close()
    
    respath = 'results'
    trialname = 'Trial1'
    model_targets = data.y_test
    fake_SN = np.random.randint(0,300,len(model_targets))
    fake_N = (np.random.choice(np.array([1.,-1.]),len(model_targets))*fake_SN/300.).reshape(len(model_targets),1)
    fake_N = np.concatenate((fake_N,fake_N,fake_N),axis=1)
    model_results = model_targets +  fake_N
    model_targets = np.concatenate((model_targets,fake_SN.reshape(len(model_targets),1)),axis=1)
    pp = post_processing(model_results, model_targets, respath, trialname)
    pp.plotResults()
    #model.predict_test()
    
