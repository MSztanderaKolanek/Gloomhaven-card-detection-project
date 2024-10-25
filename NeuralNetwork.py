from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd


class CardDetector:
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


def load_images_from_folder(folder):
    # images = []
    # for filename in os.listdir(f"generated_data/{folder}"):
    #     img = cv2.imread(f"generated_data/{folder}/{filename}")
    #     if img is not None:
    #         images.append(img)
    # return images
    return [cv2.imread(f"generated_data/{folder}/{file}") for file in os.listdir(f"generated_data/{folder}")]


def create_images_and_labels_arrays(classes, images_for_class):
    images = []
    for x in range(len(classes)):
        current_images = load_images_from_folder(classes[x])
        for i in range(images_for_class):
            images.append(current_images[i])
    images = np.array(images)
    labels = []
    for v in range(len(classes)):
        for z in range(images_for_class):
            label = []
            for _ in range(len(classes)):
                label.append(0)
            label[v] = 1
            labels.append(label)
    labels = np.array(labels)
    return np.array(images), np.array(labels)


class_names = ['brak_karty', 'ciemnosc_nocy', 'empatyczne_natarcie',
               'gryzaca_horda', 'pasozytniczy_wplyw', 'ped',
               'przerazajace_ostrze', 'przewrotne_ostrze',
               'slabosc_umyslu', 'sprzezenie_zwrotne', 'uleglosc']

images, labels = create_images_and_labels_arrays(class_names, 20)
X_train, X_test, Y_train, Y_test = train_test_split(images, labels)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


model = Sequential([
    Flatten(input_shape=(290, 217, 3)),
    Dense(units=80, activation='relu'),
    Dense(units=80, activation='relu'),
    Dense(units=11, activation='softmax')
])

model.compile(
    optimizer='adam',
    # loss='sparse_categorical_crossentropy',
    loss='binary_crossentropy',
    )

historia = model.fit(X_train, Y_train,
                     epochs=4,
                     validation_data=(X_test, Y_test),
                     verbose=1
                     # callbacks=[earlyStopping]
                     )

images_to_predict = X_test
labels_to_predict = Y_test
prediction = model.predict(images_to_predict)
good_predictions = 0
for x in range(len(list(prediction))):
    real_value = list(labels_to_predict[x]).index(max(labels_to_predict[x]))
    # predict = list(prediction[x]).index(max(prediction[x]))
    difference_array = np.absolute(prediction[x] - 1)
    predict = difference_array.argmin()
    # print(f"wartosc prawdziwa {real_value}, predykcja {predict}")
    if real_value == predict:
        good_predictions += 1
print(f"Accuracy: {good_predictions/len(list(prediction))} %")


model.evaluate(X_test, Y_test)
pd.DataFrame(historia.history).plot()
model.summary()

Y_pred = np.argmax(model.predict(X_test), axis=1)
new_Y_test = []
for prediction in Y_test:
    for i in range(len(prediction)):
        if prediction[i] == 1:
            new_Y_test.append(i)
            break
Y_test = new_Y_test

cm = confusion_matrix(Y_test, Y_pred, labels=np.unique(Y_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=np.unique(Y_test))
disp.plot()
plt.show()

