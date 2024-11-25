from concurrent.futures import ThreadPoolExecutor
import tensorflow.compat.v1 as tf  # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from models.cnn_model import train_cnn
import nina_funcs as nf  # type: ignore
import helpers.Extract_Features as ef  
from train_and_evaluate import train_and_evaluate_model
from models.random_forest_model import get_random_forest_model
from models.svm_model import get_svm_model
from models.lda_model import get_lda_model
from optimize.cnn_optmize import bayesian_optimization_cnn
import time

print("Iniciando o carregamento dos dados processados...")


data_path = "C:/Users/PC/Desktop/TCC_Folders/TCC_code/code/processed_data/"
data2_path = r"E:\db5/"


X_train = np.load(data2_path + "S5E2_X_train.npy")
X_test = np.load(data2_path + "S5E2_X_test.npy")
y_train = np.load(data2_path + "S5E2_y_train.npy")
y_test = np.load(data2_path + "S5E2_y_test.npy")

print("Dados carregados com sucesso!")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


if len(y_train.shape) > 1 and y_train.shape[1] > 1:
    y_train = np.argmax(y_train, axis=1)
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)


print("Extraindo e normalizando as features...")
X_train_features = ef.process_emg_signals(X_train, feature_set=1)
X_test_features = ef.process_emg_signals(X_test, feature_set=1)



def train_random_forest():
    start_time = time.time()  
    print("Iniciando a otimização bayesiana para Random Forest...")
    rf_model, rf_params, rf_score = get_random_forest_model(X_train_features, y_train)
    print('SCORE:',rf_score)
    train_and_evaluate_model(rf_model, "Random Forest", X_train_features, y_train, X_test_features, y_test)
    elapsed_time = time.time() - start_time  
    print(f"Random Forest treinado em {elapsed_time:.2f} segundos.")

def train_svm():
    start_time = time.time()  
    print("Iniciando a otimização bayesiana para SVM...")
    svm_model, svm_params = get_svm_model(X_train_features, y_train)
    train_and_evaluate_model(svm_model, "SVM", X_train_features, y_train, X_test_features, y_test)
    elapsed_time = time.time() - start_time  
    print(f"SVM treinado em {elapsed_time:.2f} segundos.")

def train_lda():
    start_time = time.time() 
    print("Treinando e avaliando o modelo LDA...")
    lda_model = get_lda_model()
    train_and_evaluate_model(lda_model, "LDA", X_train_features, y_train, X_test_features, y_test)
    elapsed_time = time.time() - start_time  
    print(f"LDA treinado em {elapsed_time:.2f} segundos.")

def train_cnn_wrapper():
    train_cnn(X_train, y_train, X_test, y_test)


#bestmodel= bayesian_optimization_cnn(X_train, y_train, X_test, y_test)


with ThreadPoolExecutor(max_workers=1) as executor:
    futures = [
         executor.submit(train_random_forest),
         #executor.submit(train_svm),
        # executor.submit(train_lda)
        #executor.submit(train_cnn_wrapper)
    ]

    for future in futures:
        future.result()

print("Treinamento paralelo concluído!")