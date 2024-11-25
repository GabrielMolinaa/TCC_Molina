import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

def extract_features(emg_signal, feature_set=1):

    features = []
    
 
    rms = np.sqrt(np.mean(emg_signal**2))            
    mav = np.mean(np.abs(emg_signal))               
    variance = np.var(emg_signal)                   
    wl = np.sum(np.abs(np.diff(emg_signal)))        
    zero_crossings_count = len(np.where(np.diff(np.sign(emg_signal)))[0]) 
    ssc = np.sum(np.diff(np.sign(np.diff(emg_signal))) != 0)  
    

    if feature_set == 1:
        features.extend([rms, mav, variance, wl, zero_crossings_count, ssc])
    

    elif feature_set == 2:
        fft_values = np.abs(fft(emg_signal))
        freqs = fftfreq(len(emg_signal), d=1/1000)  
        dominant_frequency = freqs[np.argmax(fft_values)] 
        features.extend([rms, mav, variance, wl, zero_crossings_count, ssc, dominant_frequency])
    
    
    elif feature_set == 3:
        fft_values = np.abs(fft(emg_signal))
        freqs = fftfreq(len(emg_signal), d=1/1000)  
        dominant_frequency = freqs[np.argmax(fft_values)]  
        mean_frequency = np.sum(freqs * fft_values) / np.sum(fft_values)
        median_frequency = freqs[np.where(np.cumsum(fft_values) >= np.sum(fft_values) / 2)[0][0]]
        features.extend([rms, mav, variance, wl, zero_crossings_count, ssc, dominant_frequency, mean_frequency, median_frequency])
    
    return features

def extract_features_from_dataset(emg_signals, feature_set=1):
    features = []
    for signal in emg_signals:

        feature_vector = np.concatenate([extract_features(signal[:, i], feature_set) for i in range(signal.shape[1])])
        features.append(feature_vector)
    return np.array(features)

def normalize_features(features):

    scaler = StandardScaler()
    return scaler.fit_transform(features)


def process_emg_signals(emg_signals, feature_set=1):
    features = extract_features_from_dataset(emg_signals, feature_set)
    features_normalized = normalize_features(features)

    # outliers = np.abs(features_normalized) > 3
    # if np.any(outliers):
    #     print("Outliers detectados. Recomenda-se revisar os dados ou aplicar técnicas de filtragem.")
    
    return features_normalized
