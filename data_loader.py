import numpy as np
import cv2
import os


class DataLoader:
    def __init__(self):
        self.data = []
        self.labels = []

    @staticmethod
    def load_data_from_folder(folder):
        # TODO DataGenerator should also remove none files so directories are clean
        return [cv2.imread(f"generated_data/{folder}/{file}") for file in os.listdir(f"generated_data/{folder}")]

    def create_data_and_labels_arrays(self, classes, images_for_class):
        # TODO instead of two lists dictionary could be better
        self.data = np.array([file for current_class in classes for file in self.load_data_from_folder(current_class)])
        for x in range(len(classes)):
            label = [0]*9
            label[x] = 1
            self.labels.extend([label for _ in range(images_for_class)])
        self.labels = np.array(self.labels)
        return self.data, self.labels
