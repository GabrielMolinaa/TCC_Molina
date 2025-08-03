import numpy as np
import os
from scipy.signal import welch
import pywt

diretorio_janelado = r"D:\Stash\Datasets\db5_janelado\5g" 
diretorio_saida_features = r"D:\Stash\Datasets\db5_features_all\5g"
fs = 200  


os.makedirs(diretorio_saida_features, exist_ok=True)


def mean_absolute_value(signal):
    return np.mean(np.abs(signal))

def root_mean_square(signal):
    return np.sqrt(np.mean(signal**2))

def zero_crossings(signal, threshold=0.01):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings[np.abs(signal[zero_crossings]) > threshold])

def slope_sign_changes(signal, threshold=0.01):
    diff_signal = np.diff(signal)
    slope_changes = np.where(np.diff(np.sign(diff_signal)))[0]
    return len(slope_changes[np.abs(diff_signal[slope_changes]) > threshold])

def waveform_length(signal):
    return np.sum(np.abs(np.diff(signal)))

def median_frequency(signal, fs):
    freqs, psd = welch(signal, fs=fs)
    cumulative_power = np.cumsum(psd)
    if cumulative_power[-1] == 0:
        return 0.0
    median_freq = freqs[np.where(cumulative_power >= cumulative_power[-1] / 2)[0][0]]
    return median_freq

def mean_frequency(signal, fs):
    freqs, psd = welch(signal, fs=fs)
    total_power = np.sum(psd)
    if total_power == 0:
        return 0.0  
    return np.sum(freqs * psd) / total_power

def mdwt(signal):
    coeffs = pywt.wavedec(signal, 'db7', level=3)
    return [np.sum(np.square(c)) for c in coeffs]

def spec(signal):
    f, psd = welch(signal, fs=200, nperseg=len(signal))
    return -np.sum(psd * np.log(psd + 1e-12)) 

def extract_features_from_window(window, fs):
    features = []
    
    for channel in range(window.shape[1]):
        signal = window[:, channel]
        features.extend([
            mean_absolute_value(signal),
            root_mean_square(signal),
            zero_crossings(signal),
            slope_sign_changes(signal),
            waveform_length(signal),
            median_frequency(signal, fs),
            mean_frequency(signal, fs),
            spec(signal)
        ] + mdwt(signal))
    return features


arquivos = [f for f in os.listdir(diretorio_janelado) if f.endswith('.npz')]

X_train_total, y_train_total = [], []
X_test_total, y_test_total = [], []

for arquivo in arquivos:
    path_arquivo = os.path.join(diretorio_janelado, arquivo)
    dados = np.load(path_arquivo)

    sujeito = arquivo.split('_')[0] 
    tipo = "train" if "train" in arquivo else "test"  

    print(f"Extraindo features de {sujeito} ({tipo})...")

    X_windows = dados[f'X_{tipo}']
    y_labels = dados[f'y_{tipo}']

    for i in range(X_windows.shape[0]):
        window = X_windows[i]
        features = extract_features_from_window(window, fs)

        if tipo == "train":
            X_train_total.append(features)
            y_train_total.append(y_labels[i])
        else:
            X_test_total.append(features)
            y_test_total.append(y_labels[i])


X_train_total = np.array(X_train_total)
y_train_total = np.array(y_train_total)
X_test_total = np.array(X_test_total)
y_test_total = np.array(y_test_total)

np.savez_compressed(os.path.join(diretorio_saida_features, "features_train.npz"), X=X_train_total, y=y_train_total)
np.savez_compressed(os.path.join(diretorio_saida_features, "features_test.npz"), X=X_test_total, y=y_test_total)

print("Extração de features concluída para treino e teste!")
