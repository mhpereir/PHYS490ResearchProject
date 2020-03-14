import h5py
import numpy as np


class Data():
    def __init__(self, data_file_path, train, test):
        
        self.train_flag = 0
        self.test_flag  = 0
        
        if train == 'real' or train == 'r':
            self.train_file_name = data_file_path + '/training_data.h5'
            self.train_flag = 1
        elif train == 'synth' or train == 's':
            self.train_file_name = data_file_path + '/ASSET.h5'
            self.train_flag = 2
        else:
            print('Input for either --train is not recognized. \\Expecting "real" or "synth".')
            
        if test == 'real' or test == 'r':
            self.test_file_name = data_file_path + '/test_data.h5'
            self.test_flag = 1
        elif test == 'synth' or test == 's':
            self.test_file_name = data_file_path + '/ASSET.h5'
            self.test_flag = 2
        else:
            print('Input for either --test is not recognized. \\Expecting "real" or "synth".')
        
    def load_train(self, n_cross):
        with h5py.File(self.train_file_name, 'r') as f5:            
            
            if self.train_flag == 1:
                fe_h  = np.array(f5['FE_H'], dtype=float)
                logg  = np.array(f5['LOGG'], dtype=float)
                teff  = np.array(f5['TEFF'], dtype=float)
                spect = np.array(f5['spectrum'], dtype=float)
            
            elif self.train_flag == 2:
                fe_h  = np.array(f5['ASSET FE_H train'], dtype=float)
                logg  = np.array(f5['ASSET LOGG train'], dtype=float)
                teff  = np.array(f5['ASSET TEFF train'], dtype=float)
                spect = np.array(f5['ASSET spectrum train'], dtype=float)
            
            n = len(spect[0,:])
            
            x_train = spect.reshape(-1,1,n)
            y_train = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1,3)
            
            self.x_cross = x_train[0:n_cross,:,:]
            self.y_cross = y_train[0:n_cross,:]
            
            self.x_train = x_train[n_cross:,:,:]
            self.y_train = y_train[n_cross:,:]
            
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
                fe_h  = np.array(f5['FE_H'], dtype=float)
                logg  = np.array(f5['LOGG'], dtype=float)
                teff  = np.array(f5['TEFF'], dtype=float)
                spect = np.array(f5['spectrum'], dtype=float)
            
            elif self.test_flag == 2:
                fe_h  = np.array(f5['ASSET FE_H test'], dtype=float)
                logg  = np.array(f5['ASSET LOGG test'], dtype=float)
                teff  = np.array(f5['ASSET TEFF test'], dtype=float)
                spect = np.array(f5['ASSET spectrum test'], dtype=float)
            
            n = len(spect[0,:])
            
            self.x_test = spect.reshape(-1,1,n)[:,:,:]
            self.y_test = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1,3)[:,:]
            
            fe_h = None
            logg = None
            teff = None
            spect = None
            
            
    def normalize_targets(self):
        self.mean_labels = np.mean(self.y_train, axis=0)
        self.std_labels  = np.std(self.y_train, axis=0)
        
        self.y_train_norm = np.add(self.y_train,-self.mean_labels)/self.std_labels
        self.y_cross_norm = np.add(self.y_cross,-self.mean_labels)/self.std_labels
        
    def re_normalize_targets(self, trained_targets):
        self.y_train_renorm = np.add(trained_targets,self.mean_labels)*self.std_labels
        
        
            