from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import cv2
import os
import glob


class DataGenerator:

    @staticmethod
    def images_generate(classes, number_of_originals, images_to_generate):

        for class_ in classes:
            for generating_index in range(number_of_originals):
                card = img_to_array(load_img(f'OriginalData/{class_}/{class_}_{str(generating_index)}.jpg'))
                card = cv2.resize(card, dsize=(217, 290))
                card = np.expand_dims(card, axis=0)
                imgen = ImageDataGenerator(rotation_range=20,
                                           brightness_range=(0.5, 1),
                                           shear_range=3.0,
                                           zoom_range=[.9, 1.1], )
                count = 0
                for _ in imgen.flow(card,
                                    batch_size=1,
                                    save_to_dir=f'generated_data/{class_}',
                                    save_format='jpg'):
                    count += 1
                    if count == images_to_generate:
                        break

    @staticmethod
    def deleting_all_generated_data():
        files = glob.glob('generated_data/ped')
        for f in files:
            os.remove(f)
