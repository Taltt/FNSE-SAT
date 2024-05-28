from random import random
import soundfile as sf
import librosa
import torch
import os
import numpy as np
from scipy import signal
#import matplotlib.pyplot as plt
import pandas as pd
import random


root_path='/data/ssd2/SpeechDatabase/'

class Dataset(torch.utils.data.Dataset):
    def __init__(self, fs=16000, length_in_seconds=8, num_data_per_epoch =20000 ,random_start_point=False, train=True, DATABASE=None):
        #print(DATABASE)
        if DATABASE == 'DNS3':# g8 /data/ssd2/SpeechDatabase/DNS3_pairs_2000hrs
            self.noisy_database_train = librosa.util.find_files(root_path+'DNS3_pairs_2000hrs/noisy_train', ext='wav')
            self.noisy_database_valid = librosa.util.find_files(root_path+'DNS3_pairs_2000hrs/noisy_valid', ext='wav')
        if DATABASE == 'DNS3_1000hrs':
            self.noisy_database_train = librosa.util.find_files(root_path+'DNS3_pairs_2000hrs/noisy_train', ext='wav')[:360000]
            self.noisy_database_valid = librosa.util.find_files(root_path+'DNS3_pairs_2000hrs/noisy_valid', ext='wav')[:360000]
        if DATABASE == 'Realrec':
            self.noisy_database_train = librosa.util.find_files(root_path+'Realrec_pairs_1000hrs_DNS3noise/noisy_train', ext='wav')
            self.noisy_database_valid = librosa.util.find_files(root_path+'Realrec_pairs_1000hrs_DNS3noise/noisy_valid', ext='wav')

        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.num_data_per_epoch = num_data_per_epoch
        self.train = train
        
    def sample(self):
        #print(self.noisy_database_train, self.num_data_per_epoch)
        self.noisy_data_train = random.sample(self.noisy_database_train, self.num_data_per_epoch)

    def __getitem__(self, idx):
        if self.train:
            noisy_list = random.sample(self.noisy_database_train, self.num_data_per_epoch)
        else:
            noisy_list = self.noisy_database_valid


        if self.random_start_point:
            Begin_S = int(np.random.uniform(0,10 - self.length_in_seconds)) * self.fs
            noisy, _ = sf.read(noisy_list[idx], dtype='float32',start= Begin_S,stop = Begin_S + self.L)
            clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32',start= Begin_S,stop = Begin_S + self.L)

        else:
            noisy, _ = sf.read(noisy_list[idx], dtype='float32',start= 0,stop = self.L) 
            clean, _ = sf.read(noisy_list[idx].replace('noisy', 'clean'), dtype='float32',start= 0,stop = self.L)

            
        if self.train:
            return noisy.astype(np.float32), clean.astype(np.float32)
        else:
            return noisy.astype(np.float32), clean.astype(np.float32),noisy_list[idx].split('/')[-1].split('.wav')[0]

    def __len__(self):
        if self.train:
            return self.num_data_per_epoch
        else:
            return len(self.noisy_database_valid)



if __name__=='__main__':
    dataset = Dataset(length_in_seconds=8, random_start_point=False, train=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    train_loader.dataset.sample()
    for i, (noisy,clean,_) in enumerate(train_loader):
        print(noisy.shape)# [8, 128000]
        break
