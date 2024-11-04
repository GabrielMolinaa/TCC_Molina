from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow.compat.v1 as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
from scipy.io import loadmat
import os
from functools import reduce
from scipy import signal
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as K
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional, InputLayer
from tensorflow.keras.models import Sequential, Model, load_model
from keras import Sequential, optimizers, callbacks
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import nina_funcs as nf
import NinaPro_Utility as ninau
import Extract_Features as ef

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

data = nf.get_data("C:/Users/PC/Desktop/TCC_Folders/TCC_code/TCC_molina/code/Databases/NinaproDB2/DB2_s1/","S1_E1_A1.mat")

train_reps = [1,3,4,6]  
test_reps = [2,5]

data = nf.normalise(data, train_reps)

emg_band = nf.filter_data(data=data, f=(20,500), butterworth_order=4, btype='bandpass') 

np.unique(data.stimulus)

gestures = [i for i in range(1,18)]

win_len = 600            
win_stride = 20

X_train, y_train, r_train = nf.windowing(emg_band, train_reps, gestures, win_len, win_stride)
X_test, y_test, r_test = nf.windowing(emg_band, test_reps, gestures, win_len, win_stride)

y_train = nf.get_categorical(y_train)
y_test = nf.get_categorical(y_test)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

if len(y_train.shape) > 1 and y_train.shape[1] > 1:
    y_train = np.argmax(y_train, axis=1)
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)

X_train_features = ef.extract_features_from_dataset(X_train)
X_test_features = ef.extract_features_from_dataset(X_test)

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train_features, y_train)


y_pred = rf_model.predict(X_test_features)
train_acc = accuracy_score(y_train, rf_model.predict(X_train_features))
test_acc = accuracy_score(y_test, y_pred)
print(f'Acurácia no treino: {train_acc:.3f}, Acurácia no teste: {test_acc:.3f}')


cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
cmd.plot(cmap='Blues')
plt.title("Matriz de Confusão - Random Forest")
plt.show()

