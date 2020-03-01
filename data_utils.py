import h5py
import numpy as np


class Data():
    def __init__(self, train_file_name, test_file_name):
        
        with h5py.File(train_file_name, 'r') as f5:            
            a_group_keys = list(f5.keys())
            
            
            fe_h  = np.array(f5['FE_H'], dtype=float)
            logg  = np.array(f5['LOGG'], dtype=float)
            teff  = np.array(f5['TEFF'], dtype=float)
            spect = np.array(f5['spectrum'], dtype=float)
            
            self.x_train = spect
            self.y_train = np.concatenate((fe_h, logg, teff), axis=1)
            
            fe_h = None
            logg = None
            teff = None
            spect = None
            
            
            
            
        with h5py.File(test_file_name, 'r') as f5:            
            a_group_keys = list(f5.keys())
            
            
            fe_h  = np.array(f5['FE_H'], dtype=float)
            logg  = np.array(f5['LOGG'], dtype=float)
            teff  = np.array(f5['TEFF'], dtype=float)
            spect = np.array(f5['spectrum'], dtype=float)
            
            self.x_test = spect
            self.y_test = np.concatenate((fe_h, logg, teff), axis=1)
            
            fe_h = None
            logg = None
            teff = None
            spect = None