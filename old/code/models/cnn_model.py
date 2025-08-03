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

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.savefig("results/confusion_matrix.png")  

def plot_training_history(history):

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.title('Evolução da Perda')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.title('Evolução da Acurácia')

    plt.tight_layout()
    plt.savefig("results/training_history.png")  



def save_metrics(history, filename="results/metrics.csv"):
    metrics = pd.DataFrame(history.history)
    metrics['epoch'] = range(1, len(metrics) + 1) 
    metrics.to_csv(filename, index=False)
    print(f"Métricas salvas em {filename}")

def train_cnn(X_train, y_train, X_test, y_test, save_path="cnn_model.h5"):
    start_time = time.time()  
    print("Iniciando o treinamento da CNN...")


    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]


    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    num_classes = len(np.unique(y_train_encoded))
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)


    model = Sequential([


        Conv2D(16, (3, 3), activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        

        Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),


        Conv2D(64, kernel_size=5, strides=1, activation='relu', padding='same', use_bias=False),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        

        Conv2D(64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same'),
        Dropout(0.5),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  
    ])


    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    early_stopping = EarlyStopping(
        monitor='val_loss',        # Métrica a ser monitorada
        patience=5,                # Número de épocas sem melhoria antes de parar
        restore_best_weights=True  # Restaura os pesos da melhor época
    )


    history = model.fit(
        X_train_cnn, y_train_categorical,
        validation_data=(X_test_cnn, y_test_categorical),
        epochs=100,  
        batch_size=128,
        verbose=1,
        callbacks=[early_stopping]  
    )
    

    test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_categorical, verbose=1)
    print(f"Acurácia da CNN no teste: {test_accuracy:.2f}")

    model.save(save_path)
    print(f"Modelo CNN salvo como {save_path}")

    elapsed_time = time.time() - start_time 
    print(f"CNN treinada em {elapsed_time:.2f} segundos.")

    plot_training_history(history)

  
    y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
    y_test_labels = np.argmax(y_test_categorical, axis=1)
    classes = [f"Classe {i}" for i in range(num_classes)]
    plot_confusion_matrix(y_test_labels, y_pred, classes)
    save_metrics(history)
