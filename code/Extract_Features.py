import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft

def extract_features(emg_signal):
    # (Root Mean Square)
    rms = np.sqrt(np.mean(emg_signal**2))
    
    # MAV (Mean Absolute Value)
    mav = np.mean(np.abs(emg_signal))
    
    # Variância
    variance = np.var(emg_signal)
    
    #  Waveform Length
    wl = np.sum(np.abs(np.diff(emg_signal)))
    
    # Zero Crossing
    zero_crossings = np.where(np.diff(np.sign(emg_signal)))[0]
    zero_crossings_count = len(zero_crossings)
    
    # Slope Sign Changes (SSC)
    ssc = np.sum(np.diff(np.sign(np.diff(emg_signal))) != 0)


    fft_values = np.abs(fft(emg_signal))
    dominant_frequency = np.argmax(fft_values)
    
    return [rms, mav, variance, wl, zero_crossings_count, ssc, dominant_frequency]


def extract_features_from_dataset(emg_signals):
    features = []
    for signal in emg_signals:
    
        feature_vector = np.concatenate([extract_features(signal[:, i]) for i in range(signal.shape[1])])
        features.append(feature_vector)
    return np.array(features)

