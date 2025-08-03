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

# === CONFIGURATION ===
#features_train_path = r"C:\\Users\\PC\\Desktop\\TCC_2025\\App\\Data\\db2_features\\features_train.npz"
#features_test_path = r"C:\\Users\\PC\\Desktop\\TCC_2025\\App\\Data\\db2_features\\features_test.npz"

features_train_path = r"D:\Stash\Datasets\db2_features_all\10g\features_train.npz"
features_test_path = r"D:\Stash\Datasets\db2_features_all\10g\features_test.npz"



use_bayesian_optimization = True  # << FLAG to enable/disable optimization
results_dir = "OB_Results_DB2_10_AllFeats"
os.makedirs(results_dir, exist_ok=True)

# === LOAD AND NORMALIZE ===
train_data = np.load(features_train_path)
test_data = np.load(features_test_path)
X_train, y_train = train_data["X"], train_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === MODELS ===
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=5000),
    "LDA": LinearDiscriminantAnalysis()
}

# === SEARCH SPACES ===
search_spaces = {
    "Random Forest": {
        "n_estimators": Integer(100, 500),
        "max_depth": Integer(2, 10),
    },
    "SVM": {
        "C": Real(1e-3, 1e3, prior='log-uniform'),
        "gamma": Real(1e-4, 1e0, prior='log-uniform')
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
cv = StratifiedKFold(n_splits=5)
results_summary = []

for name, model in models.items():
    print(f"\nTraining model: {name}")
    start_time = time.time()

    if use_bayesian_optimization and name in search_spaces:
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces[name],
            n_iter=50,
            scoring='accuracy',
            cv=cv,
            random_state=42,
            n_jobs=-1
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
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f} | F1-score (macro): {f1:.4f}")

    # SAVE RESULTS
    model_dir = create_model_folder(results_dir, name.replace(" ", "_"))
    save_metrics(model_dir, acc, elapsed_time)
    save_confusion_matrix(model_dir, y_test, y_pred, name)
    save_accuracy_barplot(model_dir, name, acc)
    save_classification_report_csv(model_dir, y_test, y_pred)

    if use_bayesian_optimization and name in search_spaces:
        save_best_params(model_dir, best_params)
        #save_search_space(model_dir, search_spaces[name])

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
