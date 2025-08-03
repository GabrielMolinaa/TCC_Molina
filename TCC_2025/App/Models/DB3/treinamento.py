import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from save_model_results import *
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===

features_train_path = r"D:\Stash\Datasets\db3_features_all\5g\features_train.npz"
features_test_path = r"D:\Stash\Datasets\db3_features_all\5g\features_test.npz"



use_bayesian_optimization = False  # << FLAG to enable/disable optimization
results_dir = "New_OB_Results_DB3_5_AllFeats"
os.makedirs(results_dir, exist_ok=True)

# === LOAD AND NORMALIZE ===
train_data = np.load(features_train_path)
test_data = np.load(features_test_path)
X_train, y_train = train_data["X"], train_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

test_fold = np.full(len(X_train), -1)  # -1 = treino
ps = PredefinedSplit(test_fold)

# === MODELS ===
models = {
   "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True,random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=3000),
    "LDA": LinearDiscriminantAnalysis()
}

# === SEARCH SPACES ===
search_spaces = {
    "Random Forest": {
        "n_estimators": Integer(50, 500),
        "max_depth": Integer(2, 10),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 5),
    },
    "SVM": {
        "C": Real(1e-3, 1e3, prior='log-uniform'),
        "gamma": Real(1e-4, 1e3, prior='log-uniform')
    },
    "KNN": {
        "n_neighbors": Integer(1, 20)
    },
    "Logistic Regression": {
        "C": Real(1e-3, 1e2, prior='log-uniform'),
        "penalty": Categorical(['l1', 'l2']),
        "solver": Categorical(['liblinear', 'saga'])
    }
}

# === TRAINING AND EVALUATION ===
#cv = StratifiedKFold(n_splits=6)
results_summary = []

for name, model in models.items():
    print(f"\nTraining model: {name}")
    start_time = time.time()

    if use_bayesian_optimization and name in search_spaces:
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces[name],
            n_iter=30,
            scoring='accuracy',
           #cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=2

        )
        bayes_search.fit(X_train, y_train)
        final_model = bayes_search.best_estimator_
        best_params = bayes_search.best_params_
    else:
        final_model = model.fit(X_train, y_train)
        best_params = model.get_params()

    elapsed_time = time.time() - start_time
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "index": np.arange(len(y_test)),
    })
    
    
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f} | F1-score (macro): {f1:.4f}")

    # SAVE RESULTS
    model_dir = create_model_folder(results_dir, name.replace(" ", "_"))
    predictions_df.to_csv(os.path.join(model_dir, "predictions.csv"), index=False)
    save_metrics(model_dir, acc, elapsed_time)
    save_confusion_matrix(model_dir, y_test, y_pred, name)
    save_accuracy_barplot(model_dir, name, acc)
    save_classification_report_csv(model_dir, y_test, y_pred)

    if use_bayesian_optimization and name in search_spaces:
        save_best_params(model_dir, best_params)
        

    results_summary.append({
        "Model": name,
        "Accuracy": acc,
        "F1-score": f1
    })

# === FINAL SUMMARY ===
results_df = pd.DataFrame(results_summary)
results_df.to_csv(os.path.join(results_dir, "results_summary.csv"), index=False)


plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Accuracy Comparison Between Models")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"))
plt.close()


plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="F1-score")
plt.title("Macro F1-score Comparison Between Models")
plt.ylim(0, 1)
plt.ylabel("F1-score (macro)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "f1_score_comparison.png"))
plt.close()
