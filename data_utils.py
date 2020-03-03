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
            
            n = len(spect[0,:])
            
            self.x_train = spect.reshape(-1,1,n)
            self.y_train = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1,3)
            
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
            
            n = len(spect[0,:])
            
            self.x_test = spect.reshape(-1,1,n)            
            self.y_test = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1,3)
            
            fe_h = None
            logg = None
            teff = None
            spect = None
            