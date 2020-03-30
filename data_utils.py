import h5py
import os
import numpy as np


class Data():
    def __init__(self, data_file_path, train, test):

        self.train_flag = 0
        self.test_flag = 0

        training_data = os.path.join(data_file_path, 'training_data.h5')
        test_data = os.path.join(data_file_path, 'test_data.h5')
        synth_data = os.path.join(data_file_path, 'ASSET.h5')

        if 'r' in train:
            self.train_file_name = training_data
            self.train_flag = 1
        else:
            self.train_file_name = synth_data
            self.train_flag = 2

        if 'r' in test:
            self.test_file_name = test_data
            self.test_flag = 1
        else:
            self.test_file_name = synth_data
            self.test_flag = 2

    def load_train(self, n_cross):
        with h5py.File(self.train_file_name, 'r') as f5:

            if self.train_flag == 1:
                fe_h = np.array(f5['FE_H'], dtype=np.float)
                logg = np.array(f5['LOGG'], dtype=np.float)
                teff = np.array(f5['TEFF'], dtype=np.float)
                spect = np.array(f5['spectrum'], dtype=np.float)

            elif self.train_flag == 2:
                fe_h = np.array(f5['ASSET FE_H train'], dtype=np.float)
                logg = np.array(f5['ASSET LOGG train'], dtype=np.float)
                teff = np.array(f5['ASSET TEFF train'], dtype=np.float)
                spect = np.array(f5['ASSET spectrum train'], dtype=np.float)

        n = len(spect[0, :])

        x_train = spect.reshape(-1, 1, n)
        y_train = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1, 3)

        self.x_cross = x_train[0:n_cross, :, :]
        self.y_cross = y_train[0:n_cross, :]

        self.x_train = x_train[n_cross:, :, :]
        self.y_train = y_train[n_cross:, :]

        fe_h = None
        logg = None
        teff = None
        spect = None

        self.normalize_targets()

    def close(self, which):
        if which == 'train':
            self.x_cross = None
            self.y_cross = None
            self.x_train = None
            self.y_train = None
        elif which == 'test':
            self.x_test = None
            self.y_test = None

    def load_test(self):
        with h5py.File(self.test_file_name, 'r') as f5:

            if self.test_flag == 1:
                fe_h = np.array(f5['FE_H'], dtype=np.float)
                logg = np.array(f5['LOGG'], dtype=np.float)
                teff = np.array(f5['TEFF'], dtype=np.float)
                spect = np.array(f5['spectrum'], dtype=np.float)

                self.snr = np.array(f5['combined_snr'], dtype=np.float)

            elif self.test_flag == 2:
                fe_h = np.array(f5['ASSET FE_H test'], dtype=np.float)
                logg = np.array(f5['ASSET LOGG test'], dtype=np.float)
                teff = np.array(f5['ASSET TEFF test'], dtype=np.float)
                spect = np.array(f5['ASSET spectrum test'], dtype=np.float)

                self.snr = None

        n = len(spect[0, :])

        self.x_test = spect.reshape(-1, 1, n)[:, :, :]
        self.y_test = np.concatenate(
            (fe_h, logg, teff), axis=1).reshape(-1, 3)[:, :]

        fe_h = None
        logg = None
        teff = None
        spect = None

    def normalize_targets(self):
        self.mean_labels = np.mean(self.y_train, axis=0)
        self.std_labels = np.std(self.y_train, axis=0)

        self.y_train_norm = np.add(
            self.y_train, -self.mean_labels)/self.std_labels
        self.y_cross_norm = np.add(
            self.y_cross, -self.mean_labels)/self.std_labels

    def re_normalize_targets(self, trained_targets):
        return np.add(trained_targets*self.std_labels, self.mean_labels)
