import os
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split


def preProcessData(imgPath, Dataset = [], Label = [], dataClass = 0):
    image_directory = imgPath
    dataset = Dataset
    label = Label
    images = os.listdir(image_directory)
    for image_name in images:
        image = cv2.imread(image_directory + image_name)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(image)
        image = image.resize((64, 64))
        image = np.array(image)
        dataset.append(image)
        label.append(dataClass)

    print("Обработка класса {} завершена".format(dataClass))
    print(len(label))

    return dataset, label

def splitData(Dataset, Label):
    dataset = np.array(Dataset)
    label = np.array(Label)
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3, random_state=1)
    X_train = normalize(X_train, axis=1)
    #X_train = X_train[..., np.newaxis]
    X_test = normalize(X_test, axis=1)
    #X_test = X_test[..., np.newaxis]
    print("Разделение данных завершено")

    return X_train, X_test, y_train, y_test