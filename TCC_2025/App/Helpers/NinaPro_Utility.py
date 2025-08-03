import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.io import loadmat
import os
from tqdm import tqdm


def get_data(path,file):
    mat = loadmat(os.path.join(path,file))
    data = pd.DataFrame(mat['emg'])
    data['stimulus'] = mat['restimulus'] 
    data['repetition'] = mat['repetition']

    return data

def normalise(data, train_reps):
    
    x = [np.where(data.values[:, 13] == rep) for rep in train_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    train_data = data.iloc[indices, :].reset_index(drop=True)

    
    scaler = StandardScaler(with_mean=True, with_std=True, copy=False).fit(train_data.iloc[:, :12])
    scaled = scaler.transform(data.iloc[:, :12])
    
    normalised = pd.DataFrame(scaled, columns=data.columns[:12])
    normalised['stimulus'] = data['stimulus']  
    normalised['repetition'] = data['repetition']  
    return normalised


def filter_data(data, f, butterworth_order = 4, btype = 'lowpass'):
    emg_data = data.values[:,:12]
    
    f_sampling = 2000
    nyquist = f_sampling/2
    if isinstance(f, int):
        fc = f/nyquist
    else:
        fc = list(f)
        for i in range(len(f)):
            fc[i] = fc[i]/nyquist
            
    b,a = signal.butter(butterworth_order, fc, btype=btype)
    transpose = emg_data.T.copy()
    
    for i in range(len(transpose)):
        transpose[i] = signal.filtfilt(b, a, transpose[i])
    
    filtered = pd.DataFrame(transpose.T)
    filtered['stimulus'] = data['stimulus']
    filtered['repetition'] = data['repetition']
    
    return filtered

def windowing(data, reps, gestures, win_len, win_stride):
    if reps:
        x = [np.where(data.values[:,13] == rep) for rep in reps]
        indices = np.squeeze(np.concatenate(x, axis = -1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)
        
    if gestures:
        x = [np.where(data.values[:,12] == move) for move in gestures]
        indices = np.squeeze(np.concatenate(x, axis = -1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)
        
    idx=  [i for i in range(win_len, len(data), win_stride)]
    
    X = np.zeros([len(idx), win_len, len(data.columns)-2])
    y = np.zeros([len(idx), ])
    reps = np.zeros([len(idx), ])
    
    for i,end in enumerate(idx):
        start = end - win_len
        X[i] = data.iloc[start:end, 0:12].values
        y[i] = data.iloc[end, 12]
        reps[i] = data.iloc[end, 13]
        
    return X, y, reps


def get_categorical(y):
    return pd.get_dummies(pd.Series(y)).values

def feature_extractor(features, shape, data):
    l = pd.DataFrame()
    for i, function in enumerate(tqdm(features)):
        feature = []
        print("Extracting feature....{}".format(str(function)))
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                feature.append(function(data[i][:, j]))
        feature = np.reshape(feature, shape)
        l = pd.concat([l, pd.DataFrame(feature)], axis=1)
        print("Done extracting feature....{}".format(str(function)))
        print()
    return l
