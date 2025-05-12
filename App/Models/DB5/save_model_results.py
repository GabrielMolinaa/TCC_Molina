import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def create_model_folder(base_path, model_name):
    folder_path = os.path.join(base_path, model_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_metrics(folder_path, acc, elapsed_time):
    with open(os.path.join(folder_path, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Training Time (s): {elapsed_time:.2f}\n")

def save_best_params(folder_path, best_params):
    with open(os.path.join(folder_path, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

def save_search_space(folder_path, search_space):
    with open(os.path.join(folder_path, "search_space.json"), "w") as f:
        json.dump(search_space, f, indent=4)

def save_confusion_matrix(folder_path, y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "confusion_matrix.png"))
    plt.close()

def save_accuracy_barplot(folder_path, model_name, acc):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[model_name], y=[acc])
    plt.title(f"Accuracy - {model_name}")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "accuracy_barplot.png"))
    plt.close()

def save_classification_report_csv(folder_path, y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(folder_path, "classification_report.csv"))
