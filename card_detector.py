from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
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
        self.train_values = None
        self.prediction = None

    def display_historical_values(self):
        if self.train_values:
            pd.DataFrame(self.train_values.history).plot()
            plt.show()
        else:
            print("No data to display")

    def display_confusion_matrix(self, predict_labels):
        if self.prediction is not None:
            predictions = np.argmax(self.prediction, axis=1)
            transformed_labels = [list(label).index(1) for label in predict_labels]
            cm = confusion_matrix(transformed_labels, predictions, labels=np.unique(transformed_labels))
            display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                             display_labels=np.unique(transformed_labels))
            display.plot()
            plt.show()
        else:
            print("No prediction done, cannot display confusion matrix")

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
        self.train_values = self.model.fit(train_data,
                                           train_labels,
                                           epochs=4,
                                           validation_data=(train_data, train_labels),
                                           verbose=1)

    def classify(self, predict_data, predict_labels):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy')
        self.prediction = self.model.predict(predict_data)
        accurate_predictions = 0
        for i in range(len(list(self.prediction))):
            real_value = list(predict_labels[i]).index(max(predict_labels[i]))
            difference_array = np.absolute(self.prediction[i] - 1)
            current_prediction = difference_array.argmin()
            if real_value == current_prediction:
                accurate_predictions += 1
        print(f"Accuracy: {accurate_predictions / len(list(self.prediction))} %")
