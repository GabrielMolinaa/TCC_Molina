from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
from models.random_forest_model import get_random_forest_model
from train_and_evaluate import train_and_evaluate_model
import helpers.Extract_Features as ef

print("Iniciando o treinamento para todos os indivíduos...")

data2_path = r"E:\db3/"


subjects = [f"S{i}" for i in range(1, 12)]  

def train_model_for_individual(subject_id):
    try:
        print(f"Iniciando o treinamento para {subject_id}...")

        # Carregar os dados do indivíduo
        X_train = np.load(data2_path + f"{subject_id}E1_X_train.npy")
        X_test = np.load(data2_path + f"{subject_id}E1_X_test.npy")
        y_train = np.load(data2_path + f"{subject_id}E1_y_train.npy")
        y_test = np.load(data2_path + f"{subject_id}E1_y_test.npy")

        print(f"Dados carregados para {subject_id}. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")


        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train = np.argmax(y_train, axis=1)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)


        print(f"Extraindo e normalizando as features para {subject_id}...")
        X_train_features = ef.process_emg_signals(X_train, feature_set=3)
        X_test_features = ef.process_emg_signals(X_test, feature_set=3)

        start_time = time.time()
        rf_model, rf_params, rf_score = get_random_forest_model(X_train_features, y_train)
        train_and_evaluate_model(rf_model, f"Random Forest - {subject_id}", X_train_features, y_train, X_test_features, y_test, rf_params)
        elapsed_time = time.time() - start_time

        print(f"Modelo treinado para {subject_id} em {elapsed_time:.2f} segundos.")
    except FileNotFoundError:
        print(f"Arquivos para {subject_id} não encontrados. Pulando...")
    except Exception as e:
        print(f"Erro ao treinar para {subject_id}: {str(e)}")


for subject in subjects:
    train_model_for_individual(subject)

print("Treinamento concluído para todos os indivíduos!")
