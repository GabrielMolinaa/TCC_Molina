import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import optuna

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from save_model_results import *

# === BASES A SEREM TREINADAS ===
datasets = {

    # "DB3_5_9Feats": {
    #     "train": r"D:\Stash\Datasets\db3_features_all\5g\features_train.npz",
    #     "test":  r"D:\Stash\Datasets\db3_features_all\5g\features_test.npz"
    # },
    # "DB3_10_3Feats": {
    #     "train": r"D:\Stash\Datasets\db3_features_3\10g\features_train.npz",
    #     "test":  r"D:\Stash\Datasets\db3_features_3\10g\features_test.npz"
    # },
    # "DB2_5_3FEATS": {
    #     "train": r"D:\Stash\Datasets\db2_features_3\5g\features_train.npz",
    #     "test":  r"D:\Stash\Datasets\db2_features_3\5g\features_test.npz"
    # },
    # "DB2_5_9FEATS": {
    #     "train": r"D:\Stash\Datasets\db2_features_all\5g\features_train.npz",
    #     "test":  r"D:\Stash\Datasets\db2_features_all\5g\features_test.npz"
    # },
    # "DB2_10_3FEATS": {
    #     "train": r"D:\Stash\Datasets\db2_features_3\10g\features_train.npz",
    #     "test":  r"D:\Stash\Datasets\db2_features_3\10g\features_test.npz"
    # },
    # "DB2_10_9FEATS": {
    #     "train": r"D:\Stash\Datasets\db2_features_all\10g\features_train.npz",
    #     "test":  r"D:\Stash\Datasets\db2_features_all\10g\features_test.npz"
    # }
    "DB5_5_3FEATS":{
        "train": r"D:\Stash\Datasets\db5_features_3\5g\features_train.npz",
        "test":  r"D:\Stash\Datasets\db5_features_3\5g\features_test.npz"
    },
    "DB5_5_9FEATS":{
        "train": r"D:\Stash\Datasets\db5_features_all\5g\features_train.npz",
        "test":  r"D:\Stash\Datasets\db5_features_all\5g\features_test.npz"
    },
    "DB5_10_3FEATS":{
        "train": r"D:\Stash\Datasets\db5_features_3\10g\features_train.npz",
        "test":  r"D:\Stash\Datasets\db5_features_3\10g\features_test.npz"
    },
    "DB5_10_9FEATS":{
        "train": r"D:\Stash\Datasets\db5_features_all\10g\features_train.npz",
        "test":  r"D:\Stash\Datasets\db5_features_all\10g\features_test.npz"
    },
}

results_global = []

USE_OPTIMIZATION = True

for dataset_name, paths in datasets.items():
    print(f"\n📁 Treinando na base: {dataset_name}")

    # === SETUP ===
    results_dir = f"DB5_Results_New_{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)

    # === LOAD ===
    train_data = np.load(paths["train"])
    test_data = np.load(paths["test"])
    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]


    # === NORMALIZAR ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # === MODELOS ===
    models = {
        "SVM": SVC,
        "RandomForest": RandomForestClassifier,
        "KNN": KNeighborsClassifier,
        "LogisticRegression": LogisticRegression,
        "LDA": LinearDiscriminantAnalysis
    }

# === ESPAÇOS DE BUSCA ===
    search_spaces = {
        "SVM": {
            "C": lambda trial: trial.suggest_float("C", 1e-2, 1e1, log=True),
            "gamma": lambda trial: trial.suggest_float("gamma", 1e-3, 1e3, log=True)
        },
        "RandomForest": {
            "n_estimators": lambda trial: trial.suggest_int("n_estimators", 100, 600),
            "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 20)
        },
        "KNN": {
            "n_neighbors": lambda trial: trial.suggest_int("n_neighbors", 1, 30)
        },
        "LogisticRegression": {
            "C": lambda trial: trial.suggest_float("C", 1e-2, 100, log=True)
        },
        "LDA": {} 
    }

    # === RESULTADOS ===
    results_summary = []

    # === LOOP DE MODELOS ===
    for name, ModelClass in models.items():
        print(f"\n🔍 Otimizando modelo: {name}")
        start_time = time.time()

        def objective(trial):
            params = {key: suggest(trial) for key, suggest in search_spaces.get(name, {}).items()}
            model = ModelClass(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)

        if USE_OPTIMIZATION and search_spaces.get(name):
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30)
            best_params = study.best_params
        else:
            best_params = {}
            study = None

        # Treinamento final
        final_model = ModelClass(**best_params)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        elapsed_time = time.time() - start_time

        print(f"✅ {name}: acc={acc:.4f} | f1={f1:.4f}")


        model_dir = create_model_folder(results_dir, name)
        pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(os.path.join(model_dir, "predictions.csv"), index=False)
        save_metrics(model_dir, acc, elapsed_time)
        save_confusion_matrix(model_dir, y_test, y_pred, name)
        save_accuracy_barplot(model_dir, name, acc)
        save_classification_report_csv(model_dir, y_test, y_pred)
        if study:
            save_best_params(model_dir, best_params)
            try:
                import optuna.visualization.matplotlib as optuna_viz
                fig = optuna_viz.plot_optimization_history(study)
                fig.savefig(os.path.join(model_dir, "optuna_history.png"))
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ Não foi possível salvar histórico do Optuna para {name}: {e}")

        if name == "RandomForest":
            importances = final_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            num_features = len(importances)
            
            # Gerar nomes reais das features
            num_channels = X_train.shape[1] // 12  # Cada canal tem 12 features
            base_features = [
                "MAV", "RMS", "ZC", "SSC", "WL",
                "MedianFreq", "MeanFreq", "Spec",
                "MDWT_L1", "MDWT_L2", "MDWT_L3", "MDWT_L4"
            ]
            feature_names = []
            for ch in range(num_channels):
                feature_names.extend([f"Ch{ch+1}_{feat}" for feat in base_features])

            plt.figure(figsize=(12, 8))
            plt.title("Feature Importances")
            plt.bar(range(num_features), importances[indices], align="center")
            plt.xticks(range(num_features), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "feature_importances.png"))
            plt.close()

            feature_importance_df = pd.DataFrame({
                "Feature": [feature_names[i] for i in range(num_features)],
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            feature_importance_df.to_csv(os.path.join(model_dir, "feature_importances.csv"), index=False)

        results_summary.append({
            "Model": name,
            "Accuracy": acc,
            "F1-score": f1
        })
        results_global.append({
            "Dataset": dataset_name,
            "Model": name,
            "Accuracy": acc,
            "F1-score": f1
        })

    # === SALVAR RESUMO ===
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(os.path.join(results_dir, "results_summary.csv"), index=False)

    # === GRÁFICOS ===
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Model", y="Accuracy")
    plt.title("Accuracy Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Model", y="F1-score")
    plt.title("Macro F1-score Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "f1_score_comparison.png"))
    plt.close()



# === SALVAR RESUMO GLOBAL ===
df_global = pd.DataFrame(results_global)
df_global.to_csv("results_all_bases_summary.csv", index=False)

# === GRÁFICO COMPARATIVO GERAL ===
plt.figure(figsize=(12, 6))
sns.barplot(data=df_global, x="Model", y="Accuracy", hue="Dataset")
plt.title("Accuracy por Modelo e Dataset")
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("global_accuracy_comparison.png")
plt.close()
