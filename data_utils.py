import h5py
import numpy as np


class Data():
    def __init__(self, data_file_path, train, test, n_rank_max, n_cross):
        
        self.train_flag = 0
        self.test_flag  = 0
        self.n_cross    = int(n_cross)
        self.n_rank_max_train = int(n_rank_max)
        
        if train == 'real' or train == 'r':
            self.train_file_name  = data_file_path + '/training_data.h5'
            self.train_flag       = 1
            self.n_rank_max_train = 1
        elif train == 'synth' or train == 's':
            self.train_file_name = data_file_path + '/ASSET.h5'
            self.train_flag = 2
        else:
            print('Input for either --train is not recognized. \\Expecting "real" or "synth".')
            
        if test == 'real' or test == 'r':
            self.test_file_name  = data_file_path + '/test_data.h5'
            self.test_flag       = 1
        elif test == 'synth' or test == 's':
            self.test_file_name = data_file_path + '/ASSET.h5'
            self.test_flag = 2
        else:
            print('Input for either --test is not recognized. \\Expecting "real" or "synth".')
    
        self.find_normalize()
    
    def find_normalize(self):
        
        with h5py.File(self.train_file_name, 'r') as f5:
            
            if self.train_flag == 1:
                fe_h  = np.array(f5['FE_H'], dtype=float)
                logg  = np.array(f5['LOGG'], dtype=float)
                teff  = np.array(f5['TEFF'], dtype=float)
            
                y_train = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1,3)
                
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
                    
                    array = np.array(f5['ASSET {} train'.format(var)], dtype=float)
                    
                    mean_list.append(np.mean(array))
                    std_list.append(np.std(array))
                    
                self.mean_labels = np.array(mean_list)
                self.std_labels  = np.array(std_list)
                
                mean_list = None
                std_list  = None
                
            
    def load_train(self, n_rank):
        with h5py.File(self.train_file_name, 'r') as f5:            
            
            if self.train_flag == 1:
                fe_h  = np.array(f5['FE_H'], dtype=float)
                logg  = np.array(f5['LOGG'], dtype=float)
                teff  = np.array(f5['TEFF'], dtype=float)
                spect = np.array(f5['spectrum'], dtype=float)
            
                n = len(spect[0,:])
            
                x_train = spect.reshape(-1,1,n)
                y_train = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1,3)
                
                self.x_cross = x_train[0:self.n_cross,:,:]
                self.y_cross = y_train[0:self.n_cross,:]
                
                self.x_train = x_train[self.n_cross:,:,:]
                self.y_train = y_train[self.n_cross:,:]
                
                fe_h  = None
                logg  = None
                teff  = None
                spect = None
            
            
            
            elif self.train_flag == 2:
                n_init   = self.n_cross
                rank_len = np.floor( (len(f5['ASSET FE_H train']) - self.n_cross) / self.n_rank_max_train )
                
                if n_rank < self.n_rank_max_train-1:
                    nLower   = int(n_init + n_rank*rank_len)
                    nUpper   = int(n_init + (n_rank+1)*rank_len)
                elif n_rank == self.n_rank_max_train-1:
                    nLower   = int(n_init + n_rank*rank_len)
                    nUpper   = -1
                else:
                    print('Error, n_rank = ', n_rank)
                    nLower = 0
                    nUpper = 0
                                
                fe_h  = np.array(f5['ASSET FE_H train'][nLower:nUpper,:], dtype=float)
                logg  = np.array(f5['ASSET LOGG train'][nLower:nUpper,:], dtype=float)
                teff  = np.array(f5['ASSET TEFF train'][nLower:nUpper,:], dtype=float)
                spect = np.array(f5['ASSET spectrum train'][nLower:nUpper,:], dtype=float)
            
                n = len(spect[0,:])
                
                self.x_train = spect.reshape(-1,1,n)
                self.y_train = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1,3)
                
                #generate SNR and add to spectra
                SNRs = np.random.randint(20,250,np.shape(self.x_train)[0])
                self.add_SNR(SNRs,self.x_train,n)
                
                if n_rank == 0: #loading in cross
                    fe_h  = np.array(f5['ASSET FE_H train'][0:self.n_cross,:], dtype=float)
                    logg  = np.array(f5['ASSET LOGG train'][0:self.n_cross,:], dtype=float)
                    teff  = np.array(f5['ASSET TEFF train'][0:self.n_cross,:], dtype=float)
                    spect = np.array(f5['ASSET spectrum train'][0:self.n_cross,:], dtype=float)
                
                    n = len(spect[0,:])
                    
                    self.x_cross = spect.reshape(-1,1,n)
                    self.y_cross = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1,3)
                    
                    #generate SNR and add to spectra
                    SNRs = np.random.randint(20,250,np.shape(self.x_cross)[0])
                    self.add_SNR(SNRs,self.x_cross,n)
                
                
                fe_h  = None
                logg  = None
                teff  = None
                spect = None
                SNRs  = None
            
        self.normalize_targets() 
    
    def close(self, which):
        if which == 'train':
            self.x_cross = None
            self.y_cross = None
            self.x_train = None
            self.y_train = None
        elif which == 'test':
            self.x_test  = None
            self.y_test  = None
            
    def load_test(self):
        with h5py.File(self.test_file_name, 'r') as f5: 
            
            if self.test_flag == 1:
                fe_h  = np.array(f5['FE_H'], dtype=float)
                logg  = np.array(f5['LOGG'], dtype=float)
                teff  = np.array(f5['TEFF'], dtype=float)
                spect = np.array(f5['spectrum'], dtype=float)
                
                self.snr = np.array(f5['combined_snr'], dtype=float)
                
                n = len(spect[0,:])
                self.x_test = spect.reshape(-1,1,n)[:,:,:]
            
            elif self.test_flag == 2:
                fe_h  = np.array(f5['ASSET FE_H test'], dtype=float)
                logg  = np.array(f5['ASSET LOGG test'], dtype=float)
                teff  = np.array(f5['ASSET TEFF test'], dtype=float)
                spect = np.array(f5['ASSET spectrum test'], dtype=float)
                
                n = len(spect[0,:])
                self.x_test = spect.reshape(-1,1,n)[:,:,:]
            
                #generate SNR and add to spectra
                SNRs = np.random.randint(20,250,np.shape(self.x_test)[0])
                self.add_SNR(SNRs,self.x_test,n)
                self.snr = SNRs.reshape(-1,1)
            

            self.y_test = np.concatenate((fe_h, logg, teff), axis=1).reshape(-1,3)[:,:]
            
            fe_h  = None
            logg  = None
            teff  = None
            spect = None
            
            
    def normalize_targets(self):       
        self.y_train_norm = np.add(self.y_train,-self.mean_labels)/self.std_labels
        self.y_cross_norm = np.add(self.y_cross,-self.mean_labels)/self.std_labels
        
    def re_normalize_targets(self, trained_targets):
        return np.add(trained_targets*self.std_labels,self.mean_labels)
    
    def add_SNR(self,SNR,spec,n):
        Psignal = np.mean(spec,axis=2)[:,0]
        #Pnoise = Psignal/SNR
        for i in range(0,len(SNR)):
            Pnoise = Psignal[i]/SNR[i]
            noise = np.random.normal(0,np.sqrt(Pnoise),n) #n = len of spectra 
            spec[i] = np.copy(spec[i]) + noise
        