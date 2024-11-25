from keras_tuner import HyperModel
from keras_tuner import BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization,Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



def plot_confusion_matrix(y_test, y_pred, classes):
    # Gerar a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # Plotar e salvar a matriz de confusão
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.savefig("results/confusion_matrix.png") 

def plot_training_history(history):
    # Plotando a perda
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid()
    plt.title('Evolução da Perda')

    # Plotando a acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid()
    plt.title('Evolução da Acurácia')

    plt.tight_layout()
    plt.savefig("results/training_history.png")  



def save_metrics(history, filename="results/metrics.csv"):
    metrics = pd.DataFrame(history.history)
    metrics['epoch'] = range(1, len(metrics) + 1) 
    metrics.to_csv(filename, index=False)
    print(f"Métricas salvas em {filename}")



class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential([
            # 1ª Camada Convolucional
            Conv2D(
                filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
                kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
                activation='relu',
                input_shape=self.input_shape
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), padding='same'),
            
            # 2ª Camada Convolucional
            Conv2D(
                filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
                kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
                activation='relu',
                padding='same'
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), padding='same'),
            Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)),

            # Camadas Densas
            Flatten(),
            Dense(
                units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
                activation='relu'
            ),
            Dropout(rate=hp.Float('dropout_rate_dense', min_value=0.2, max_value=0.5, step=0.1)),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compilar o modelo
        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model



def bayesian_optimization_cnn(X_train, y_train, X_test, y_test):
    # Adiciona o canal para as entradas da CNN
    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    # Codificar as labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Converter labels para One-Hot Encoding
    num_classes = len(np.unique(y_train_encoded))
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)

    # Define o hipermodelo
    hypermodel = CNNHyperModel(input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], 1), num_classes=num_classes)

    # Configura o tuner para otimização Bayesiana
    tuner = BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,  # Número de combinações de hiperparâmetros a serem testadas
        directory='bayesian_cnn_tuning',
        project_name='cnn_optimization'
    )

    # Executa a busca pelos melhores hiperparâmetros
    tuner.search(
        X_train_cnn, y_train_categorical,
        validation_data=(X_test_cnn, y_test_categorical),
        epochs=20,
        batch_size=128,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
    )

    # Melhor conjunto de hiperparâmetros
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Melhores hiperparâmetros:", best_hyperparameters.values)

    # Treinar o modelo com os melhores hiperparâmetros
    best_model = tuner.hypermodel.build(best_hyperparameters)
    history = best_model.fit(
        X_train_cnn, y_train_categorical,
        validation_data=(X_test_cnn, y_test_categorical),
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
    )

    
    test_loss, test_accuracy = best_model.evaluate(X_test_cnn, y_test_categorical, verbose=1)
    print(f"Acurácia da CNN no teste com melhores hiperparâmetros: {test_accuracy:.2f}")

    
    best_model.save("best_cnn_model.keras")
    plot_training_history(history)

    return best_model 