from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from skopt import BayesSearchCV
import tensorflow.compat.v1 as tf  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import nina_funcs as nf  # type: ignore
import helpers.Extract_Features as ef  


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    print(f"Treinando e avaliando o modelo {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    print(f'{model_name} - Acurácia no treino: {train_acc:.3f}, Acurácia no teste: {test_acc:.3f}')


    cm = confusion_matrix(y_test, y_pred)
    unique_labels = np.unique(y_test)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    cmd.plot(cmap='Blues')
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.show()


print("Iniciando o carregamento dos dados...")
data = nf.get_data("C:/Users/PC/Desktop/TCC_Folders/TCC_code/TCC_molina/code/Databases/NinaproDB2/DB2_s1/", "S1_E2_A1.mat")
train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
print("Normalizando os dados de treino...")
data = nf.normalise(data, train_reps)
print("Aplicando filtragem nos dados EMG...")
emg_band = nf.filter_data(data=data, f=(20, 500), butterworth_order=4, btype='bandpass')
gestures = [i for i in range(18, 40)]
win_len = 600
win_stride = 20
print("Aplicando janelamento nos dados...")
X_train, y_train, _ = nf.windowing(emg_band, train_reps, gestures, win_len, win_stride)
X_test, y_test, _ = nf.windowing(emg_band, test_reps, gestures, win_len, win_stride)
print("Codificando as classes...")
y_train = nf.get_categorical(y_train)
y_test = nf.get_categorical(y_test)
print("Ajustando o formato das labels...")
if len(y_train.shape) > 1 and y_train.shape[1] > 1:
    y_train = np.argmax(y_train, axis=1)
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)
print("Extraindo e normalizando as features...")
X_train_features = ef.process_emg_signals(X_train)
X_test_features = ef.process_emg_signals(X_test)


print("Iniciando a otimização bayesiana para Random Forest...")
rf_param_grid = {
    'n_estimators': (10, 100),
    'max_depth': (5, 15),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5)
}
rf_bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=33),
    search_spaces=rf_param_grid,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    random_state=33
)
rf_bayes_search.fit(X_train_features, y_train)
print("Random Forest - Melhores hiperparâmetros:", rf_bayes_search.best_params_)
train_and_evaluate_model(rf_bayes_search.best_estimator_, "Random Forest", X_train_features, y_train, X_test_features, y_test)

# 2. SVM
print("Iniciando a otimização bayesiana para SVM...")
svm_param_grid = {
    'C': (0.1, 10.0, 'log-uniform'),
    'gamma': (0.001, 1.0, 'log-uniform'),  
    'kernel': ['linear', 'rbf']
}
svm_bayes_search = BayesSearchCV(
    estimator=SVC(random_state=33),
    search_spaces=svm_param_grid,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    random_state=33
)
svm_bayes_search.fit(X_train_features, y_train)
print("SVM - Melhores hiperparâmetros:", svm_bayes_search.best_params_)
train_and_evaluate_model(svm_bayes_search.best_estimator_, "SVM", X_train_features, y_train, X_test_features, y_test)

print("Treinando e avaliando o modelo LDA...")
lda_model = LinearDiscriminantAnalysis()
train_and_evaluate_model(lda_model, "LDA", X_train_features, y_train, X_test_features, y_test)

print("Processo completo.")
