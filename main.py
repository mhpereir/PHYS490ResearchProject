#!/usr/bin/env python3


import json, argparse, torch, os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from cnn import StarNet
from data_utils import Data
from processing_utils import post_processing


def get_args():
    ''' Command line argument parsing '''

    parser = argparse.ArgumentParser(
        description='STARNET: CNN regressor to predict solar properties from stellar spectra',
        allow_abbrev=True)

    parser.add_argument('-d', '--data_path', type=str, metavar='path/to/data',
        required=True, help='Path to directory containing all datasets and hyperparameter files')
    parser.add_argument('--train', required=True, choices=['real', 'synthetic'],
        type=str, help='Which training dataset to use: real or synthetic?')
    parser.add_argument('--test', required=True, choices=['real', 'synthetic'],
        type=str, help='Which testing dataset to use: real or synthetic?')
    parser.add_argument('-o', '--output_path', type=str, metavar='path/to/results',
        default='results/', help='Path to store plots and results')
    parser.add_argument('-v', '--verbose', type=str2bool, default=False, metavar='True',
        help='Boolean flag to specify verbose output (default: False)')
    parser.add_argument('-c', '--cuda', type=str2bool, default=False, metavar='True',
        help='Boolean flag to specify to use CUDA (default: False)')
    parser.add_argument('-s', '--save', type=str2bool, default=True, metavar='True',
        help='Boolean flag to specify whether to save the model\'s learned \
        parameters to the output_path (default: True)')
    parser.add_argument('-m', '--max_cpu', type=str2bool, default=False, metavar='True',
        help='Boolean flag to specify whether to use all CPU cores if CUDA is not \
        in use or is unavailable. Note that this will restrict other system processes \
        from running. Recommended to only enable on servers. (default: False, \
        resorts to using PyTorch\'s get_num_threads implementation, which usually \
        maximizes use across 50%% of cores)')

    return parser.parse_args()


def str2bool(s):
    ''' Boolean argparse typecaster, since in-built bool acts unexpectedly '''
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'y', 'true', 't', '1'):
        return True
    elif s.lower() in ('no', 'none', 'n', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_main():
    # Load and extract CLI arguments
    args = get_args()

    data_path        = args.data_path
    train_data       = args.train
    test_data        = args.test
    output_path      = args.output_path
    verbose          = args.verbose
    cuda_input       = args.cuda
    save_model       = args.save
    max_cpu          = args.max_cpu

    # Load hyperparameters from file
    params_path = os.path.join(data_path, 'params.json')
    with open(params_path) as paramfile:
        params = json.load(paramfile)
    num_epochs   = int(params['n_epoch'])
    num_epochs_v = int(params['n_epoch_v'])
    n_train      = int(params['n_mini_batch'])
    n_rank_max   = int(params['n_rank'])
    n_cross      = int(params['n_cross'])

    # Load in the training datasets
    data  = Data(data_path, train_data, test_data, n_rank_max, n_cross)

    # CUDA usage
    device = torch.device('cuda' if (torch.cuda.is_available() and cuda_input) else 'cpu')
    print('\nRunning on {}\n'.format(device))

    # Initialize the model
    model = StarNet().float()
    # model.init_data(data, device)
    model.to(device)
    if device == 'cpu':
        if max_cpu and os.cpu_count > 4:  # Maximize the CPU
            num_processes = os.cpu_count()
        else:  # Preserve the CPU for other activities
            num_processes = torch.get_num_threads()
        torch.set_num_threads(num_processes)  # Naive maximization of CPU usage

    # Define an optimizer and the loss function
    optimizer  = optim.Adam(model.parameters(), lr=params['lr'])
    loss       = torch.nn.MSELoss()

    # Define LR scheduler and its parameters
    reduce_lr_factor         = 0.5
    reduce_lr_patience       = 2
    reduce_lr_min            = 0.00008
    reduce_lr_threshold      = 0.0009
    reduce_lr_threshold_mode = 'abs'

    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=reduce_lr_factor,
                                                      patience=reduce_lr_patience,
                                                      min_lr=reduce_lr_min,
                                                      threshold=reduce_lr_threshold,
                                                      threshold_mode=reduce_lr_threshold_mode,
                                                      verbose=verbose)

    early_stop_patience = 4
    early_stop_min_diff = 0.0001
    early_stop_condition = False

    obj_vals     = []
    cross_vals   = []

    # Training loop
    flag = True
    print('Training over {} epochs:'.format(num_epochs))
    with tqdm(total=num_epochs, dynamic_ncols=True) as pbar:
        for epoch in range(1, num_epochs + 1):
            # Training
            if epoch != 1 and data.train_flag == 1:
                flag = False
            train_val, time_epoch = model.backprop(data, loss, optimizer, n_train, device, flag)
            obj_vals.append(train_val)

            # Cross-validation evaluation
            cross_val = model.cross(data, loss, device)
            cross_vals.append(cross_val)

            scheduler.step(train_val)

            # Update progress bar
            pbar.update()

            # High verbosity report in output stream
            if verbose:
                if not ((epoch) % num_epochs_v):
                    print('Epoch [{}/{}]'.format(epoch, num_epochs)+\
                          '\tTraining Loss: {:.4f}'.format(train_val)+\
                          '\tTest Loss: {:.4f}'.format(cross_val)+\
                          '\tEpoch duration: {:.4f}m'.format(time_epoch))

            early_stop_condition = model.check_early_stop(cross_vals, patience=early_stop_patience, min_diff=early_stop_min_diff)

            if early_stop_condition:
                print('Early stopping condition met. Breaking at epoch: {}'.format(epoch))
                break

    # Low verbosity final report
    print('Final training loss: {:.4f}'.format(obj_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))

    # Plot loss and save to file
    plt.plot(range(1, epoch+1), obj_vals, label= 'Training loss', color='blue')
    plt.plot(range(1, epoch+1), cross_vals, label= 'CV loss', color= 'green')
    plt.title('Cross-Validation Loss over %i Training Epochs' % num_epochs)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plt.savefig(os.path.join(output_path, 'loss.png'), dpi=400)

    if save_model:
        torch.save(model.state_dict(), os.path.join(output_path, 'weights_{}_{}.pt'.format(train_data, epoch+1)))

    data.close('train')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ## beginning of post processing

    data.load_test()
    predicted_targets = model.model_predictions(data, n_train, device)
    predicted_targets = data.re_normalize_targets(predicted_targets)
    real_targets = data.y_test
    snr = data.snr
    real_targets = np.concatenate((real_targets, snr), axis=1)
    data.close('test')

    trialname = 'Trial1'
    pp = post_processing(predicted_targets, real_targets, output_path, trialname)
    pp.plotResults()


if __name__ == '__main__':
    run_main()
