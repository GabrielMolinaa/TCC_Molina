
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dados organizados
data = [
    # DB2
    ("DB2", "5 Gestures", "3 Features", "Random Forest", 0.846),
    ("DB2", "5 Gestures", "3 Features", "SVM", 0.694),
    ("DB2", "5 Gestures", "3 Features", "KNN", 0.790),
    ("DB2", "5 Gestures", "3 Features", "Logistic Regression", 0.485),
    ("DB2", "5 Gestures", "3 Features", "LDA", 0.471),
    ("DB2", "5 Gestures", "All Features", "Random Forest", 0.850),
    ("DB2", "5 Gestures", "All Features", "SVM", 0.796),
    ("DB2", "5 Gestures", "All Features", "KNN", 0.791),
    ("DB2", "5 Gestures", "All Features", "Logistic Regression", 0.537),
    ("DB2", "5 Gestures", "All Features", "LDA", 0.519),
    ("DB2", "10 Gestures", "3 Features", "Random Forest", 0.816),
    ("DB2", "10 Gestures", "3 Features", "SVM", 0.570),
    ("DB2", "10 Gestures", "3 Features", "KNN", 0.740),
    ("DB2", "10 Gestures", "3 Features", "Logistic Regression", 0.423),
    ("DB2", "10 Gestures", "3 Features", "LDA", 0.393),
    ("DB2", "10 Gestures", "All Features", "Random Forest", 0.823),
    ("DB2", "10 Gestures", "All Features", "SVM", 0.729),
    ("DB2", "10 Gestures", "All Features", "KNN", 0.741),
    ("DB2", "10 Gestures", "All Features", "Logistic Regression", 0.459),
    ("DB2", "10 Gestures", "All Features", "LDA", 0.429),

    # DB3
    ("DB3", "5 Gestures", "3 Features", "Random Forest", 0.713),
    ("DB3", "5 Gestures", "3 Features", "SVM", 0.614),
    ("DB3", "5 Gestures", "3 Features", "KNN", 0.649),
    ("DB3", "5 Gestures", "3 Features", "Logistic Regression", 0.423),
    ("DB3", "5 Gestures", "3 Features", "LDA", 0.427),
    ("DB3", "5 Gestures", "All Features", "Random Forest", 0.719),
    ("DB3", "5 Gestures", "All Features", "SVM", 0.675),
    ("DB3", "5 Gestures", "All Features", "KNN", 0.653),
    ("DB3", "5 Gestures", "All Features", "Logistic Regression", 0.485),
    ("DB3", "5 Gestures", "All Features", "LDA", 0.488),
    ("DB3", "10 Gestures", "3 Features", "Random Forest", 0.534),
    ("DB3", "10 Gestures", "3 Features", "SVM", 0.429),
    ("DB3", "10 Gestures", "3 Features", "KNN", 0.455),
    ("DB3", "10 Gestures", "3 Features", "Logistic Regression", 0.279),
    ("DB3", "10 Gestures", "3 Features", "LDA", 0.269),
    ("DB3", "10 Gestures", "All Features", "Random Forest", 0.537),
    ("DB3", "10 Gestures", "All Features", "SVM", 0.496),
    ("DB3", "10 Gestures", "All Features", "KNN", 0.447),
    ("DB3", "10 Gestures", "All Features", "Logistic Regression", 0.327),
    ("DB3", "10 Gestures", "All Features", "LDA", 0.318),

    # DB5
    ("DB5", "5 Gestures", "3 Features", "Random Forest", 0.707),
    ("DB5", "5 Gestures", "3 Features", "SVM", 0.677),
    ("DB5", "5 Gestures", "3 Features", "KNN", 0.686),
    ("DB5", "5 Gestures", "3 Features", "Logistic Regression", 0.615),
    ("DB5", "5 Gestures", "3 Features", "LDA", 0.604),
    ("DB5", "5 Gestures", "All Features", "Random Forest", 0.710),
    ("DB5", "5 Gestures", "All Features", "SVM", 0.669),
    ("DB5", "5 Gestures", "All Features", "KNN", 0.572),
    ("DB5", "5 Gestures", "All Features", "Logistic Regression", 0.618),
    ("DB5", "5 Gestures", "All Features", "LDA", 0.606),
    ("DB5", "10 Gestures", "3 Features", "Random Forest", 0.636),
    ("DB5", "10 Gestures", "3 Features", "SVM", 0.592),
    ("DB5", "10 Gestures", "3 Features", "KNN", 0.603),
    ("DB5", "10 Gestures", "3 Features", "Logistic Regression", 0.520),
    ("DB5", "10 Gestures", "3 Features", "LDA", 0.508),
    ("DB5", "10 Gestures", "All Features", "Random Forest", 0.639),
    ("DB5", "10 Gestures", "All Features", "SVM", 0.580),
    ("DB5", "10 Gestures", "All Features", "KNN", 0.452),
    ("DB5", "10 Gestures", "All Features", "Logistic Regression", 0.518),
    ("DB5", "10 Gestures", "All Features", "LDA", 0.502),
]

# DataFrame
df = pd.DataFrame(data, columns=["Database", "Gestures", "Features", "Model", "Accuracy"])

# Abreviações
model_abbr = {
    "Random Forest": "RF",
    "SVM": "SVM",
    "KNN": "KNN",
    "Logistic Regression": "LR",
    "LDA": "LDA"
}
df["Model Abbr"] = df["Model"].map(model_abbr)
df["Config (EN)"] = df["Gestures"] + " - " + df["Features"]

# Função para salvar heatmap por base
def save_db_heatmap(df, db_name, cmap_name):
    subset = df[df["Database"] == db_name]
    pivot = subset.pivot_table(index="Config (EN)", columns="Model Abbr", values="Accuracy")

    plt.figure(figsize=(8, 4))
    sns.set(style="whitegrid")
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap_name,
        linewidths=0.5,
        cbar_kws={'label': 'Accuracy'},
        annot_kws={"size": 9}
    )
    ax.set_title(f"{db_name} - Classification Accuracy", fontsize=14, pad=16)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Configuration", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{db_name}_heatmap_accuracy.png", dpi=300)
    plt.close()

# Gerar gráficos
save_db_heatmap(df, "DB2", "YlGnBu")
save_db_heatmap(df, "DB3", "OrRd")
save_db_heatmap(df, "DB5", "PuBuGn")
