from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time
import os
from models.random_forest_model import get_random_forest_model
from models.svm_model import get_svm_model
from models.lda_model import get_lda_model
from train_and_evaluate import train_and_evaluate_model
import Extract_Features as ef

def train_model_for_feature_set(feature_set, data2_path, results_dir):
    print(f"Iniciando o treinamento para Feature Set {feature_set}...")
    
    # Lista de IDs dos indivíduos
    subjects = [f"S{i}" for i in range(1, 11)]  # Supondo 10 indivíduos

    # Listas para armazenar os dados de todos os indivíduos
    X_train_all = []
    X_test_all = []
    y_train_all = []
    y_test_all = []

    # Carregar os dados de todos os indivíduos
    for subject_id in subjects:
        try:
            print(f"Carregando dados para {subject_id}...")
            X_train = np.load(data2_path + f"{subject_id}E1_X_train.npy")
            X_test = np.load(data2_path + f"{subject_id}E1_X_test.npy")
            y_train = np.load(data2_path + f"{subject_id}E1_y_train.npy")
            y_test = np.load(data2_path + f"{subject_id}E1_y_test.npy")

            # Adicionar aos conjuntos globais
            X_train_all.append(X_train)
            X_test_all.append(X_test)
            y_train_all.append(y_train)
            y_test_all.append(y_test)
        except FileNotFoundError:
            print(f"Arquivos para {subject_id} não encontrados. Pulando...")

    # Concatenar os dados de todos os indivíduos
    X_train_combined = np.concatenate(X_train_all, axis=0)
    X_test_combined = np.concatenate(X_test_all, axis=0)
    y_train_combined = np.concatenate(y_train_all, axis=0)
    y_test_combined = np.concatenate(y_test_all, axis=0)

    # Ajustar as labels, se necessário
    if len(y_train_combined.shape) > 1 and y_train_combined.shape[1] > 1:
        y_train_combined = np.argmax(y_train_combined, axis=1)
    if len(y_test_combined.shape) > 1 and y_test_combined.shape[1] > 1:
        y_test_combined = np.argmax(y_test_combined, axis=1)

    # Extração de features
    print(f"Extraindo e normalizando as features para Feature Set {feature_set}...")
    X_train_features = ef.process_emg_signals(X_train_combined, feature_set=feature_set)
    X_test_features = ef.process_emg_signals(X_test_combined, feature_set=feature_set)

    # Treinamento do Random Forest
    print("Treinando o modelo Random Forest...")
    rf_model, rf_params = get_random_forest_model(X_train_features, y_train_combined)
    train_and_evaluate_model(
        rf_model,
        f"Random Forest - Feature Set {feature_set}",
        X_train_features,
        y_train_combined,
        X_test_features,
        y_test_combined,
        results_dir
    )

    # Treinamento do SVM
    print("Treinando o modelo SVM...")
    svm_model, svm_params = get_svm_model(X_train_features, y_train_combined)
    train_and_evaluate_model(
        svm_model,
        f"SVM - Feature Set {feature_set}",
        X_train_features,
        y_train_combined,
        X_test_features,
        y_test_combined,
        results_dir
    )

    # Treinamento do LDA
    print("Treinando o modelo LDA...")
    lda_model = get_lda_model()
    train_and_evaluate_model(
        lda_model,
        f"LDA - Feature Set {feature_set}",
        X_train_features,
        y_train_combined,
        X_test_features,
        y_test_combined,
        results_dir
    )

    print(f"Treinamento para Feature Set {feature_set} concluído.")

if __name__ == "__main__":
    
    data2_path = r"E:\datatcc/"
    results_dir = "results_feature_sets"
    os.makedirs(results_dir, exist_ok=True)

    feature_sets = [1, 2, 3] 

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(train_model_for_feature_set, feature_set, data2_path, results_dir)
            for feature_set in feature_sets
        ]
        for future in futures:
            future.result()

    print("Treinamento paralelo concluído para todos os Feature Sets.")
