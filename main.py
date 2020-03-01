import json, argparse, torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch

from time import time
from cnn import StarNet
from data_utils import Data

if __name__ == '__main__':
    start_time = time()
    # Command line arguments (taken from tutorial script (lesson 5))
    parser = argparse.ArgumentParser(description='Assignment 2: CNN script')
    parser.add_argument('--train_path', metavar='inputs',
                        help='training data file name (hdf5)')
    parser.add_argument('--test_path', metavar='inputs',
                        help='test data file name (hdf5)')
    parser.add_argument('--params_path', metavar='params',
                        help='hyper params file name (json)')
    parser.add_argument('--output_path', metavar='results',
                        help='path to results')
    parser.add_argument('-v', type=int, default=2, metavar='N',
                        help='verbosity (default: 2)')
    parser.add_argument('-cuda', type=int, default=1, metavar='N',
                       help='cuda indicator (default: 1 = ON)')
    args = parser.parse_args()
    
    train_file_path  = args.train_path
    test_file_path   = args.test_path
    params_file_path = args.params_path
    output_file_path = args.output_path
    cuda_input       = args.cuda
    
    with open(params_file_path) as paramfile:
        param_file = json.load(paramfile)
    
    data  = Data(train_file_path, test_file_path)
    model = StarNet()
    
    model.init_data(data,cuda_input)
    
    # Define an optimizer and the loss function
    optimizer  = optim.Adam(model.parameters(), lr=param_file['lr'])
    loss       = torch.nn.MSELoss()
    
    obj_vals   = []
    cross_vals = []
    num_epochs = int(param_file['n_epoch'])
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_val = model.backprop(loss, optimizer, n_train=param_file['n_mini_batch'])
        obj_vals.append(train_val)
        
        
        test_val = model.test(loss)
        cross_vals.append(test_val)
        
        # High verbosity report in output stream
        if args.v>=2:
            if not ((epoch + 1) % int(param_file['n_epoch_v'])):
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTest Loss: {:.4f}'.format(test_val))
    
    # Low verbosity final report
    if args.v:
        print('Final training loss: {:.4f}'.format(obj_vals[-1]))
        print('Final test loss: {:.4f}'.format(cross_vals[-1]))
    
    print('Ellapsed time: {}m'.format( (time()-start_time)/60) )
    
    # Plot saved in results folder
    fig,ax = plt.subplots()
    ax.plot(range(num_epochs), obj_vals, label= "Training loss", color="blue")
    ax.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
    ax.legend()
    ax.set_ylim([0,1])
    fig.savefig(args.output_path + 'fig.pdf')
    plt.close()
    
    model.predict_test()
    