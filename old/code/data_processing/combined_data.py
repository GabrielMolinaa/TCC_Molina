import numpy as np


data2_path = r"E:\db5/"


X_train_all = []
X_test_all = []
y_train_all = []
y_test_all = []

for subject in range(1, 41):  
    
    subject_id = f"S{subject}"
    print(f'Processando: {subject_id}')
    
    train_file = f"{data2_path}{subject_id}E1_X_train.npy"
    test_file = f"{data2_path}{subject_id}E1_X_test.npy"
    y_train_file = f"{data2_path}{subject_id}E1_y_train.npy"
    y_test_file = f"{data2_path}{subject_id}E1_y_test.npy"
    
    try:
        
        X_train = np.memmap(train_file, dtype='float32', mode='r')
        X_test = np.memmap(test_file, dtype='float32', mode='r')
        y_train = np.memmap(y_train_file, dtype='float32', mode='r')
        y_test = np.memmap(y_test_file, dtype='float32', mode='r')

        X_train_all.append(X_train)
        X_test_all.append(X_test)
        y_train_all.append(y_train)
        y_test_all.append(y_test)
    except FileNotFoundError:
        print(f"Arquivos para {subject_id} não encontrados. Ignorando.")

X_train_all = [x.astype(np.float32) for x in X_train_all]
X_test_all = [x.astype(np.float32) for x in X_test_all]
X_train_combined = np.concatenate(X_train_all, axis=0)
X_test_combined = np.concatenate(X_test_all, axis=0)

print(f"Tamanho do conjunto combinado: {X_train_combined.nbytes / (1024**3):.2f} GiB")

X_train_combined = np.concatenate(X_train_all, axis=0)
X_test_combined = np.concatenate(X_test_all, axis=0)
y_train_combined = np.concatenate(y_train_all, axis=0)
y_test_combined = np.concatenate(y_test_all, axis=0)

print(f"X_train_combined shape: {X_train_combined.shape}")
print(f"X_test_combined shape: {X_test_combined.shape}")
print(f"y_train_combined shape: {y_train_combined.shape}")
print(f"y_test_combined shape: {y_test_combined.shape}")



output_path = r"E:\datatcc_combined/"


import os
os.makedirs(output_path, exist_ok=True)

np.save(os.path.join(output_path, "X_train_combined.npy"), X_train_combined)
np.save(os.path.join(output_path, "X_test_combined.npy"), X_test_combined)
np.save(os.path.join(output_path, "y_train_combined.npy"), y_train_combined)
np.save(os.path.join(output_path, "y_test_combined.npy"), y_test_combined)

print("Conjuntos concatenados salvos com sucesso!")