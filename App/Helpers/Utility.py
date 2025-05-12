import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.io import loadmat
import os
from tqdm import tqdm


def get_data(path, file):
    mat = loadmat(os.path.join(path, file))
    data = pd.DataFrame(mat['emg'])
    data['stimulus'] = mat['restimulus']
    data['repetition'] = mat['repetition']
    return data


def normalise(data, train_reps, n_channels, rep_col):
    x = [np.where(data.values[:, rep_col] == rep) for rep in train_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    train_data = data.iloc[indices, :].reset_index(drop=True)

    scaler = StandardScaler(with_mean=True, with_std=True, copy=False).fit(train_data.iloc[:, :n_channels])
    scaled = scaler.transform(data.iloc[:, :n_channels])

    normalised = pd.DataFrame(scaled, columns=data.columns[:n_channels])
    normalised['stimulus'] = data['stimulus']
    normalised['repetition'] = data['repetition']
    return normalised


def filter_data(data, f, sampling_rate, n_channels, butterworth_order=4, btype='lowpass'):
    emg_data = data.values[:, :n_channels]
    nyquist = sampling_rate / 2

    if isinstance(f, int):
        fc = f / nyquist
    else:
        fc = [freq / nyquist for freq in f]

    b, a = signal.butter(butterworth_order, fc, btype=btype)
    transpose = emg_data.T.copy()

    for i in range(len(transpose)):
        transpose[i] = signal.filtfilt(b, a, transpose[i])

    filtered = pd.DataFrame(transpose.T)
    filtered['stimulus'] = data['stimulus']
    filtered['repetition'] = data['repetition']
    return filtered


def windowing(data, reps, gestures, win_len, win_stride, n_channels, stim_col, rep_col):
    if reps:
        x = [np.where(data.values[:, rep_col] == rep) for rep in reps]
        indices = np.squeeze(np.concatenate(x, axis=-1))
        data = data.iloc[indices, :].reset_index(drop=True)

    if gestures:
        x = [np.where(data.values[:, stim_col] == move) for move in gestures]
        indices = np.squeeze(np.concatenate(x, axis=-1))
        data = data.iloc[indices, :].reset_index(drop=True)

    idx = [i for i in range(win_len, len(data), win_stride)]

    X = np.zeros([len(idx), win_len, n_channels])
    y = np.zeros([len(idx), ])
    reps = np.zeros([len(idx), ])

    for i, end in enumerate(idx):
        start = end - win_len
        X[i] = data.iloc[start:end, 0:n_channels].values
        y[i] = data.iloc[end, stim_col]
        reps[i] = data.iloc[end, rep_col]

    return X, y, reps


def get_categorical(y):
    return pd.get_dummies(pd.Series(y)).values


def feature_extractor(features, shape, data):
    l = pd.DataFrame()
    for i, function in enumerate(tqdm(features)):
        feature = []
        print(f"Extracting feature....{function}")
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                feature.append(function(data[i][:, j]))
        feature = np.reshape(feature, shape)
        l = pd.concat([l, pd.DataFrame(feature)], axis=1)
        print(f"Done extracting feature....{function}\n")
    return l
