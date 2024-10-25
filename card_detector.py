from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


class CardDetector:
    def __init__(self):
        self.model = Sequential([Flatten(input_shape=(290, 217, 3)),
                                 Dense(units=80, activation='relu'),
                                 Dense(units=80, activation='relu'),
                                 Dense(units=11, activation='softmax')])
        self.historical_values = None

    def display_historical_values(self):
        pd.DataFrame(self.historical_values.history).plot()
        plt.show()

    def save(self):
        model_pkl_file = "gloomhaven_classifier_model.pkl"
        with open(model_pkl_file, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self):
        model_pkl_file = "gloomhaven_classifier_model.pkl"
        with open(model_pkl_file, 'rb') as file:
            self.model = pickle.load(file)

    def train(self, train_data, train_labels):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy')
        self.historical_values = self.model.fit(train_data,
                                                train_labels,
                                                epochs=4,
                                                validation_data=(train_data, train_labels),
                                                verbose=1)

    def classify(self, predict_data, predict_labels):
        prediction = self.model.predict(predict_data)
        accurate_predictions = 0
        for x in range(len(list(prediction))):
            real_value = list(predict_labels[x]).index(max(predict_labels[x]))
            current_prediction = list(prediction[x]).index(max(prediction[x]))
            difference_array = np.absolute(prediction[x] - 1)
            predict = difference_array.argmin()
            print(f"wartosc prawdziwa {real_value}, predykcja {current_prediction}")
            if real_value == predict:
                accurate_predictions += 1
        print(f"Accuracy: {accurate_predictions / len(list(prediction))} %")

        self.model.evaluate(predict_labels)
        pd.DataFrame(historia.history).plot()
        self.model.summary()

        Y_pred = np.argmax(self.model.predict(predict_data), axis=1)
        new_Y_test = []
        for prediction in predict_labels:
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
