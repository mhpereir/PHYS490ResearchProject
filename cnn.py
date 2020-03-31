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
        Convolution_1 (8  @ 1x7214)     [ReLU]
        Convolution_2 (8 @ 1x7214)      [ReLU]
        Max Pooling   (8 @ 1x1801)
        FC_1          (1x256)           [ReLU]
        FC_2          (1x128)           [ReLU]
        FC_3          (1x3)             [Linear]
    '''

    def __init__(self):
        super(StarNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LayerNorm(7214),
            nn.Conv1d(in_channels=8, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LayerNorm(7214),
            nn.Dropout(0.2, inplace=True),
            nn.Conv1d(in_channels=1, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LayerNorm(7214),
            nn.Conv1d(in_channels=8, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LayerNorm(7214),
            nn.AvgPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(14408, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3, inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        out = self.cnn(x)
        return out

    def backprop(self, data, loss, optimizer, n_train, device, flag):
        self.train()
        loss_vals = []

        start_time = time()

        for n in range(0, data.n_rank_max_train):

            print('Rank: [{}/{}]'.format(n+1, data.n_rank_max_train))
            if flag:
                data.load_train(n)
                #self.init_data(data, device)
            else:
                pass

            n_total = len(data.x_train[:, 0, 0])
            iters = int(n_total // n_train)

            for i in range(iters):
                args_lower = i*n_train
                args_upper = (i+1)*n_train

                inputs = torch.from_numpy(data.x_train[args_lower:args_upper, :, :]).float().to(
                    device)
                targets = torch.from_numpy(data.y_train_norm[args_lower:args_upper, :]).float().to(
                    device)

                outputs = self(inputs)
                obj_val = loss(outputs, targets)

                optimizer.zero_grad()
                obj_val.backward()
                optimizer.step()

                loss_vals.append(obj_val.item())

            if n_total % n_train != 0:
                #final step:    ##account for leftover indices in input array
                args_lower = (i+1)*n_train
                args_upper = n_total

                inputs = torch.from_numpy(data.x_train[args_lower:args_upper, :, :]).float().to(
                    device)
                targets = torch.from_numpy(data.y_train_norm[args_lower:args_upper, :]).float().to(
                    device)

                outputs = self(inputs)
                obj_val = loss(outputs, targets)

                optimizer.zero_grad()
                obj_val.backward()
                optimizer.step()

                loss_vals.append(obj_val.item())

        ellapsed_time = (time() - start_time)/60

        return np.mean(loss_vals), ellapsed_time

    def cross(self, data, loss, device):
        inputs = torch.from_numpy(data.x_cross[:, :, :]).float().to(
                device)
        targets = torch.from_numpy(data.y_cross_norm[:, :]).float().to(
                device)

        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
            cross_val = loss(outputs, targets)

        return cross_val.item()

    def model_predictions(self, data, n_train, device):
        self.eval()
        with torch.no_grad():
            n_total = len(data.x_test[:, 0, 0])
            iters = int(np.floor(n_total/n_train))
            loss_vals = []

            print('Running #{} iterations for test data.'.format(iters))

            start_time = time()

            for i in range(iters):
                args_lower = i*n_train
                args_upper = (i+1)*n_train

                inputs = torch.from_numpy(
                    data.x_test[args_lower:args_upper, :, :]).to(device).float()
                outputs = self(inputs)

                if i == 0:
                    predicted_target_array = outputs.cpu().numpy()
                else:
                    predicted_target_array = np.concatenate(
                        (predicted_target_array, outputs.cpu().numpy()), axis=0)

            if n_total % n_train != 0:
                #final step:    ##account for leftover indices in input array
                args_lower = (i+1)*n_train
                args_upper = n_total

                inputs = torch.from_numpy(
                    data.x_test[args_lower:args_upper, :, :]).to(device).float()
                outputs = self(inputs)

                predicted_target_array = np.concatenate(
                    (predicted_target_array, outputs.cpu().numpy()), axis=0)

        print('Target prediction ellapsed time: {:.4f}m'.format(
            (time()-start_time)/60))

        return predicted_target_array

    def check_early_stop(self, loss_cross, patience, min_diff):

        if len(loss_cross) < 2:
            self.counter = 0
        elif loss_cross[-1] > loss_cross[-2] or abs(loss_cross[-1] - loss_cross[-2]) <= min_diff:
            self.counter += 1
        elif loss_cross[-1] <= loss_cross[-2]:
            self.counter = 0
        else:
            pass

        if len(loss_cross) >= patience:
            if (self.counter == patience):
                return True

    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.maxpool.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
