
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from skopt.plots import plot_convergence, plot_gaussian_process
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import numpy as np
import pywt
from joblib import parallel_backend

import matplotlib.pyplot as plt


# Diretório dos dados
output_dir = "db5_combined/"

# Identificação do sujeito e exercício
subject_id = "3"
exercise_id = "1"

# Carregar os dados
X_train = np.load(f"{output_dir}/all_X_train_features.npy")
y_train = np.load(f"{output_dir}/all_y_train.npy")
X_test = np.load(f"{output_dir}/all_X_test_features.npy")
y_test = np.load(f"{output_dir}/all_y_test.npy")

# # Extração de features para treino
# X_train_features = np.array([extract_features(window) for window in X_train])
# X_test_features = np.array([extract_features(window) for window in X_test])

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir o espaço de busca
search_space = [
    Real(0.001, 1000, name="C"),        # Hiperparâmetro C da SVM
    Real(0.001, 10, name="gamma")      # Hiperparâmetro gamma da SVM
]

# Função objetivo para otimização
@use_named_args(search_space)
def objective(C, gamma):
    model = SVC(C=C, gamma=gamma, kernel="rbf", random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    return -acc  # Minimizar a perda (-acurácia)

# Otimizar usando gp_minimize
print("Iniciando a otimização Bayesiana...")
with parallel_backend("loky", n_jobs=12): 
  result = gp_minimize(objective, search_space, n_calls=50, random_state=42,n_random_starts=10)


plot_convergence(result)
plt.title("Convergência da Otimização")
plt.show()

# Plotar o Processo Gaussiano
#plot_gaussian_process(result, n_calls=50)
#plt.title("Processo Gaussiano Durante a Otimização")
#plt.show()

# Melhor resultado
best_C = result.x[0]
best_gamma = result.x[1]
best_accuracy = -result.fun
print(f"Melhores hiperparâmetros encontrados: C={best_C}, gamma={best_gamma}")
print(f"Acurácia com os melhores hiperparâmetros: {best_accuracy:.2f}")

# Treinar modelo com os melhores hiperparâmetros
best_model = SVC(C=best_C, gamma=best_gamma, kernel="rbf", random_state=42)
best_model.fit(X_train_scaled, y_train)


# Previsão no conjunto de teste
y_pred = best_model.predict(X_test_scaled)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
plt.figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title(f"Matriz de Confusão - Acurácia: {best_accuracy:.2f}")
plt.show()
# Reduzir os dados para 2 dimensões com PCA (visualização)
# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)

# n_classes = len(np.unique(y_train))
# custom_cmap = ListedColormap(plt.cm.tab10.colors[:n_classes])


# # Plotar as fronteiras de decisão
# def plot_decision_boundary_with_trained_model(model, X_2d, X_original, y, title):
#     x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
#     y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                          np.arange(y_min, y_max, 0.1))
#     grid_points = np.c_[xx.ravel(), yy.ravel()]
#     grid_points_original = pca.inverse_transform(grid_points)

#     Z = model.predict(grid_points_original)
#     Z = Z.reshape(xx.shape)
    
#     plt.figure(figsize=(10, 8))
#     plt.contourf(xx, yy, Z, alpha=0.8, cmap=custom_cmap)  # Usar o colormap ajustado
#     scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolor='k', cmap=custom_cmap)
#     plt.title(title)
#     plt.xlabel("Componente Principal 1")
#     plt.ylabel("Componente Principal 2")
#     plt.colorbar(scatter, ticks=range(n_classes), label="Gestos")  # Ajustar ticks para as classes
#     plt.grid()
#     plt.show()

# # Plotar a fronteira de decisão para o modelo otimizado
# plot_decision_boundary_with_trained_model(best_model, X_train_pca, X_train_scaled, y_train, 
#                                           "Fronteira de Decisão (Treinamento - Modelo Otimizado)")

# plot_decision_boundary_with_trained_model(best_model, X_test_pca, X_test_scaled, y_test, 
#                                           "Fronteira de Decisão (Teste - Modelo Otimizado)")

