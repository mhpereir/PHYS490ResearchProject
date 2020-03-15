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
    
    # Load in the training datasets
    data  = Data(data_file_path, train_data, test_data)
    data.load_train(param_file['n_cross'])
    
    # Initialize the model
    model = StarNet().float()
    
    # Define an optimizer and the loss function
    optimizer  = optim.Adam(model.parameters(), lr=param_file['lr'])
    
    reduce_lr_factor         = 0.5
    reduce_lr_patience       = 2
    reduce_lr_min            = 0.00008
    reduce_lr_threshold      = 0.0009        #assuming this is the same as reduce_lr_epsilon in original code....???? 
    reduce_lr_threshold_mode = 'abs'
    
    early_stop_patience = 4
    early_stop_min_diff = 0.0001
    
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=reduce_lr_factor, 
                                                    patience=reduce_lr_patience, 
                                                    min_lr=reduce_lr_min, 
                                                    #eps=reduce_lr_epsilon,
                                                    threshold=reduce_lr_threshold,
                                                    threshold_mode=reduce_lr_threshold_mode,
                                                    verbose=True)
    loss       = torch.nn.MSELoss()
        
    obj_vals     = []
    cross_vals   = []
    num_epochs   = int(param_file['n_epoch'])
    num_epochs_v = int(param_file['n_epoch_v'])
    n_train      = int(param_file['n_mini_batch'])
    early_stop_condition = False
    
    if torch.cuda.is_available() and cuda_input == 1:
        device = 'cuda'
        print('Running with CUDA')
    else:
        device = 'cpu'
        print('Running with CPU')
    
    model.to(device)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_val,time_epoch = model.backprop(data, loss, optimizer, n_train, device)
        obj_vals.append(train_val)
        
        
        cross_val = model.cross(data, loss, device)
        cross_vals.append(cross_val)
        
        scheduler.step(train_val)
        
        # High verbosity report in output stream
        if args.v>=2:
            if not ((epoch) % num_epochs_v):
                print('Epoch [{}/{}]'.format(epoch, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTest Loss: {:.4f}'.format(cross_val)+\
                       '\tEllapsed time: {:.4f}m'.format(time_epoch))
    
        early_stop_condition = model.check_early_stop(cross_vals, patience=early_stop_patience, min_diff=early_stop_min_diff)
    
        if early_stop_condition:
            print('Early stop condition met. Breaking at epoch: {}'.format(epoch))
            break
    
    # Low verbosity final report
    if args.v>=1:
        print('Final training loss: {:.4f}'.format(obj_vals[-1]))
        print('Final test loss: {:.4f}'.format(cross_vals[-1]))
    
    print('Training time ellapsed: {:.4f}m'.format( (time()-start_time)/60) )
    
    # Plot saved in results folder
    fig,ax = plt.subplots()
    ax.plot(range(epoch), obj_vals, label= "Training loss", color="blue")
    ax.plot(range(epoch), cross_vals, label= "Test loss", color= "green")
    ax.legend()
    fig.savefig('results/loss.pdf')
    plt.close()
    

    data.close('train')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ## beginning of post processing 
    
    data.load_test()
    predicted_targets = model.model_predictions(data, n_train, device)
    predicted_targets = data.re_normalize_targets(predicted_targets)
    real_targets = data.y_test
    snr = data.snr
    real_targets = np.concatenate((real_targets,snr),axis=1)
    data.close('test')
    
    respath = 'results'
    trialname = 'Trial1'
    pp = post_processing(predicted_targets, real_targets, respath, trialname)
    pp.plotResults()
    
    
