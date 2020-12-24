from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, MaxPooling2D, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import seaborn as sns
import numpy as np


def model_creation():
        input_shape = (64, 64, 3)
        model = Sequential()

        # 1st conv layer
        model.add(Conv2D(64, (2, 2), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())

        # 2nd conv layer
        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())

        # 3rd conv layer
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())

        # flatten output and feed it into dense layer
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        # output layer
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        METRICS = [keras.metrics.BinaryAccuracy(name='accuracy'),
                   keras.metrics.AUC(name='auc'),
                   keras.metrics.TruePositives(name='tp'),
                   keras.metrics.FalsePositives(name='fp'),
                   keras.metrics.TrueNegatives(name='tn'),
                   keras.metrics.FalseNegatives(name='fn'),
                   ]
        optimiser = keras.optimizers.Adam(learning_rate=0.00003)
        model.compile(optimizer=optimiser,
                      loss='binary_crossentropy',
                      metrics=METRICS)
        return model

def model_eval(model, X_train, X_test, y_train, y_test):
        history = model.fit(X_train, y_train, batch_size=64, epochs=150, validation_data=(X_test, y_test))
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


        auc = history.history['auc']
        val_auc = history.history['val_auc']
        plt.plot(epochs, auc, 'y', label='Training acc')
        plt.plot(epochs, val_auc, 'r', label='Validation acc')
        plt.title('Training and validation auc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


        # Оценка модели
        _, acc, auc, tp, fp, tn, fn = model.evaluate(X_test, y_test)
        print("Accuracy = ", (acc * 100.0), "%")
        print("AUC = ", auc)
        tp = int(tp)
        tn = int(tn)
        fp = int(fp)
        tn = int(tn)
        confusion_matrix = np.array([[tp, fn],[fp, tn]], np.int32)
        print(confusion_matrix)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        sns.heatmap(confusion_matrix, cmap=plt.cm.Blues, annot=True)
        plt.show()

        # Сохранение модели
        model.save("Nn_model_1")