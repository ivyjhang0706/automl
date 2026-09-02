import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.fftpack import fft
import pywt


##-------------For regression model--------------
class Regression_ECGDataset(Dataset):
    def __init__(self, dir_path, used_feature_array, type='Normal', method='raw'):
        self.dir_path = os.path.abspath(dir_path)
        self.method = method

        if type == 'Normal':
            normalLabels, normalSigs = self.read_files('Normal', used_feature_array)
        elif type == 'High':
            normalLabels, normalSigs = self.read_files('High', used_feature_array)
        else:
            normalLabels, normalSigs = self.read_files('Low', used_feature_array)

        self.Signals = normalSigs
        self.Labels = normalLabels

    def __getitem__(self, index):
        signal = self.Signals[index]
        label = self.Labels[index]

        return signal, label

    def __len__(self):
        return len(self.Labels)

    ## Parsing files in folder
    def read_files(self, foldername, used_feature_array):

        Sig = []
        Label = []
        self.file_list = []
        self.channel = 0

        for filename in os.listdir(os.path.join(self.dir_path, foldername)):
            file_path = os.path.join(self.dir_path, foldername, filename)

            Sig.append(self.load_data(file_path, used_feature_array))

            self.file_list.append(filename)

            string_array = filename.split('_')
            value_string = (string_array[-1]).split('.')
            value = int(value_string[0])
            Label.append(value)

        Sig = torch.FloatTensor(Sig)
        Label = torch.FloatTensor(Label)

        if self.channel == 1:
            Sig = Sig.unsqueeze(1)

        return Label, Sig

    ## Read data in files & Preprocessing
    def load_data(self, filepath, used_feature_array):

        values = np.genfromtxt(filepath, delimiter='')
        values = values[:, 1]  ##取得第2個column的數值
        feature_indices = np.where(used_feature_array == 1)[0]
        sig = values[feature_indices]
        sig.reshape(len(feature_indices), 1)

        self.channel = 1

        if self.method == 'raw' or self.method == 'meta':
            # 'meta' 是訓練管線名稱，讀特徵與 raw 相同
            return sig.tolist()

        if self.method == 'time':
            sig = self.normalize1(sig)
            return sig.tolist()

        if self.method == 'freq':
            sig = self.normalize1(sig)
            #sig = self.FFT(sig)
            sig = self.DWT(sig)
            self.channel = len(sig)
            return sig.tolist()

        if self.method == 'combine':
            sig = self.normalize1(sig)
            x1 = sig.tolist()
            #x2 = self.FFT(sig).tolist()
            x3 = self.DWT(sig).tolist()
            #Comb = [x1, x2, x3]
            x3.append(x1)
            self.channel = len(x3)
            return x3

        raise ValueError(f"Unknown feature load method: {self.method!r}")

    

    def normalize1(self, x):
        x_max = np.max(x)
        x_min = np.min(x)
        x_norm = (x - x_min) / (x_max - x_min + 1)
        return x_norm

    
