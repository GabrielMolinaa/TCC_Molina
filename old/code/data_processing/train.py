from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Configuração do conjunto de dados e modelo
set = 3
model_type = "SVM"  # Alterar entre "SVM", "RF" ou "LDA"

# Diretório dos dados
output_dir = f"DB2/ex2/feature_set_{set}"
output2 = f"db5_combined"

# Carregar os dados
X_train = np.load(f"{output_dir}/db2_ex2_all_X_train_features_{set}.npy")
y_train = np.load(f"{output_dir}/db2_ex2_all_y_train_{set}.npy")
X_test = np.load(f"{output_dir}/db2_ex2_all_X_test_features_{set}.npy")
y_test = np.load(f"{output_dir}/db2_ex2_all_y_test_{set}.npy")

# X_train = np.load(f"{output2}/all_X_train_features.npy")
# y_train = np.load(f"{output2}/all_y_train.npy")
# X_test = np.load(f"{output2}/all_X_test_features.npy")
# y_test = np.load(f"{output2}/all_y_test.npy")


# imputer = SimpleImputer(strategy="mean")  # Use "median" ou "constant" se preferir
# X_train = imputer.fit_transform(X_train)
# X_test = imputer.transform(X_test)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir e treinar o modelo com base na escolha
if model_type == "SVM":
    model = SVC(kernel='rbf', random_state=42)
elif model_type == "RF":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_type == "LDA":
    model = LinearDiscriminantAnalysis(solver='svd')
else:
    raise ValueError("Modelo inválido! Escolha entre 'SVM', 'RF' ou 'LDA'.")

model.fit(X_train, y_train)

# Previsão no conjunto de teste
y_pred = model.predict(X_test)

# Métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
classification_rep = classification_report(y_test, y_pred, target_names=[str(label) for label in np.unique(y_test)])

# Criar diretório para salvar os resultados
results_dir = os.path.join(output_dir, f"results_{model_type.lower()}")
os.makedirs(results_dir, exist_ok=True)

# Salvar métricas em CSV
results = {
    "Accuracy": [accuracy],
    "F1-Score": [f1],
    "Recall": [recall],
    "Precision": [precision]
}
results_df = pd.DataFrame(results)
metrics_csv_path = os.path.join(results_dir, "model_metrics.csv")
results_df.to_csv(metrics_csv_path, index=False)

# Salvar matriz de confusão
cm = confusion_matrix(y_test, y_pred)
cm_fig_path = os.path.join(results_dir, "confusion_matrix.png")
plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title(f"Matriz de Confusão - Acurácia: {accuracy:.2f}")
plt.savefig(cm_fig_path)
plt.close()

# Salvar o relatório de classificação em um arquivo de texto
classification_rep_path = os.path.join(results_dir, "classification_report.txt")
with open(classification_rep_path, "w") as f:
    f.write(classification_rep)

# Exibir um resumo do processo
print(f"Métricas salvas em: {metrics_csv_path}")
print(f"Matriz de confusão salva em: {cm_fig_path}")
print(f"Relatório de classificação salvo em: {classification_rep_path}")
