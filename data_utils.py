import h5py
import os
import numpy as np


class Data():
    def __init__(self, data_file_path, train, test, n_rank_max, n_cross): #Set data paths and flags for different cases

        self.train_flag = 0
        self.test_flag = 0
        self.n_cross = int(n_cross)
        self.n_rank_max_train = int(n_rank_max)

        training_data = os.path.join(data_file_path, 'training_data.h5')
        test_data     = os.path.join(data_file_path, 'test_data.h5')
        synth_data    = os.path.join(data_file_path, 'ASSET.h5')

        if 'r' in train:
            self.train_file_name  = training_data
            self.train_flag       = 1
            self.n_rank_max_train = 1
        else:
            self.train_file_name  = synth_data
            self.train_flag       = 2

        if 'r' in test:
            self.test_file_name   = test_data
            self.test_flag        = 1
        else:
            self.test_file_name   = synth_data
            self.test_flag        = 2

        self.find_normalize()

    def find_normalize(self): #Find normalization constants for targets

        with h5py.File(self.train_file_name, 'r') as f5:

            if self.train_flag == 1:
                fe_h = np.array(f5['FE_H'], dtype=np.float)
                logg = np.array(f5['LOGG'], dtype=np.float)
                teff = np.array(f5['TEFF'], dtype=np.float)

                y_train = np.concatenate(
                    (fe_h, logg, teff), axis=1).reshape(-1, 3)

                self.mean_labels = np.mean(y_train, axis=0)
                self.std_labels  = np.std(y_train, axis=0)

                fe_h    = None
                logg    = None
                teff    = None
                y_train = None

            elif self.train_flag == 2:

                mean_list = []
                std_list  = []

                for var in ['FE_H', 'LOGG', 'TEFF']:

                    array = np.array(
                        f5['ASSET {} train'.format(var)], dtype=np.float)

                    mean_list.append(np.mean(array))
                    std_list.append(np.std(array))

                self.mean_labels = np.array(mean_list)
                self.std_labels  = np.array(std_list)

                mean_list = None
                std_list  = None

    def load_train(self, n_rank): #Manage how to load training data
        with h5py.File(self.train_file_name, 'r') as f5:

            if self.train_flag == 1: #Training on real, can load entire dataset set at once
                fe_h  = np.array(f5['FE_H'], dtype=np.float)
                logg  = np.array(f5['LOGG'], dtype=np.float)
                teff  = np.array(f5['TEFF'], dtype=np.float)
                spect = np.array(f5['spectrum'], dtype=np.float)

                n = len(spect[0, :])

                x_train = spect/(np.median(spect, axis=1)[:,None])
                x_train = x_train.reshape(-1,1,n)
                y_train = np.concatenate(
                    (fe_h, logg, teff), axis=1).reshape(-1, 3)

                self.x_cross = x_train[0:self.n_cross, :, :]
                self.y_cross = y_train[0:self.n_cross, :]

                self.x_train = x_train[self.n_cross:, :, :]
                self.y_train = y_train[self.n_cross:, :]

                fe_h  = None
                logg  = None
                teff  = None
                spect = None

            elif self.train_flag == 2: #Training on synth, load dataset in segments called `ranks'
                n_init = self.n_cross
                rank_len = np.floor(
                    (len(f5['ASSET FE_H train']) - self.n_cross) / self.n_rank_max_train)

                if n_rank < self.n_rank_max_train-1:
                    nLower = int(n_init + n_rank*rank_len)
                    nUpper = int(n_init + (n_rank+1)*rank_len)
                elif n_rank == self.n_rank_max_train-1:
                    nLower = int(n_init + n_rank*rank_len)
                    nUpper = -1
                else:
                    print('Error, n_rank = ', n_rank)
                    nLower = 0
                    nUpper = 0

                fe_h = np.array(f5['ASSET FE_H train']
                                [nLower:nUpper, :], dtype=np.float)
                logg = np.array(f5['ASSET LOGG train']
                                [nLower:nUpper, :], dtype=np.float)
                teff = np.array(f5['ASSET TEFF train']
                                [nLower:nUpper, :], dtype=np.float)
                spect = np.array(f5['ASSET spectrum train']
                                 [nLower:nUpper, :], dtype=np.float)

                n = len(spect[0, :])

                self.x_train = spect/(np.median(spect, axis=1)[:,None])
                self.x_train = self.x_train.reshape(-1,1,n)
                self.y_train = np.concatenate(
                    (fe_h, logg, teff), axis=1).reshape(-1, 3)

                #generate SNR and add to spectra
                SNRs = np.random.randint(20, 250, np.shape(self.x_train)[0])
                self.add_SNR(SNRs, self.x_train, n)

                if n_rank == 0:  # loading in cross
                    fe_h = np.array(f5['ASSET FE_H train']
                                    [0:self.n_cross, :], dtype=np.float)
                    logg = np.array(f5['ASSET LOGG train']
                                    [0:self.n_cross, :], dtype=np.float)
                    teff = np.array(f5['ASSET TEFF train']
                                    [0:self.n_cross, :], dtype=np.float)
                    spect = np.array(f5['ASSET spectrum train']
                                     [0:self.n_cross, :], dtype=np.float)

                    n = len(spect[0, :])

                    self.x_cross = spect/(np.median(spect, axis=1)[:,None])
                    self.x_cross = self.x_cross.reshape(-1,1,n)
                    self.y_cross = np.concatenate(
                        (fe_h, logg, teff), axis=1).reshape(-1, 3)
                    
                    #generate SNR and add to spectra
                    SNRs = np.random.randint(
                        20, 250, np.shape(self.x_cross)[0])
                    self.add_SNR(SNRs, self.x_cross, n)

                fe_h  = None
                logg  = None
                teff  = None
                spect = None
                SNRs  = None

        self.normalize_targets()

    def close(self, which): #Empty arrays to save memory
        if which == 'train':
            self.x_cross = None
            self.y_cross = None
            self.x_train = None
            self.y_train = None
        elif which == 'test':
            self.x_test = None
            self.y_test = None

    def load_test(self): #Load test data arrays
        with h5py.File(self.test_file_name, 'r') as f5:

            if self.test_flag == 1:
                fe_h = np.array(f5['FE_H'], dtype=np.float)
                logg = np.array(f5['LOGG'], dtype=np.float)
                teff = np.array(f5['TEFF'], dtype=np.float)
                spect = np.array(f5['spectrum'], dtype=np.float)

                self.snr = np.array(f5['combined_snr'], dtype=np.float)

                n = len(spect[0, :])
                self.x_test = spect/(np.median(spect, axis=1)[:,None])
                self.x_test = self.x_test.reshape(-1,1,n)[:,:,:]

            elif self.test_flag == 2:
                fe_h = np.array(f5['ASSET FE_H test'], dtype=np.float)
                logg = np.array(f5['ASSET LOGG test'], dtype=np.float)
                teff = np.array(f5['ASSET TEFF test'], dtype=np.float)
                spect = np.array(f5['ASSET spectrum test'], dtype=np.float)

                n = len(spect[0, :])
                self.x_test = spect/(np.median(spect, axis=1)[:,None])
                self.x_test = self.x_test.reshape(-1,1,n)[:,:,:]

                #generate SNR and add to spectra
                SNRs = np.random.randint(20, 250, np.shape(self.x_test)[0])
                self.add_SNR(SNRs, self.x_test, n)
                self.snr = SNRs.reshape(-1, 1)

        self.y_test = np.concatenate(
            (fe_h, logg, teff), axis=1).reshape(-1, 3)[:, :]

        fe_h = None
        logg = None
        teff = None
        spect = None

    def normalize_targets(self): #Normalize each target array
        self.y_train_norm = np.add(
            self.y_train, -self.mean_labels)/self.std_labels
        self.y_cross_norm = np.add(
            self.y_cross, -self.mean_labels)/self.std_labels

    def re_normalize_targets(self, trained_targets): #Undo normalization of target array
        return np.add(trained_targets*self.std_labels, self.mean_labels)

    def add_SNR(self, SNR, spec, n): #Gen Gauss. random noise and add to spectra
        Psignal = np.mean(spec, axis=2)[:, 0]
        #Pnoise = Psignal/SNR
        for i in range(0, len(SNR)):
            Pnoise = Psignal[i]/SNR[i]
            noise = np.random.normal(0, np.sqrt(Pnoise), n)  # n = len of spectra
            spec[i] = np.copy(spec[i]) + noise
