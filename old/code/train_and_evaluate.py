import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


os.makedirs("results/plots", exist_ok=True)

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test,params,results_file="results/accuracy_results.txt"):
    print(f"Treinando e avaliando o modelo {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    train_f1 = f1_score(y_train, y_pred_train, average='weighted') 
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    with open(results_file, "a") as f:
        f.write(f'{model_name} - Acurácia no treino: {train_acc:.3f}, Acurácia no teste: {test_acc:.3f}\n')
        f.write(f'{model_name} - F1-Score no treino: {train_f1:.3f}, F1-Score no teste: {test_f1:.3f}\n')
        f.write(f'Parâmetros: {params}\n\n')

    cm = confusion_matrix(y_test, y_pred)
    unique_labels = np.unique(y_test)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)


    fig, ax = plt.subplots()
    cmd.plot(cmap='Blues', ax=ax)
    plt.title(f"Matriz de Confusão - {model_name}")
    fig.savefig(f"results/plots/{model_name}_confusion_matrix.png")
    plt.close(fig)  
