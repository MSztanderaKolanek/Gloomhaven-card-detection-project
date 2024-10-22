from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import cv2
import os
import glob


def images_generator(classes, number_of_originals, images_to_generate):
    for class_ in classes:
        for generating_index in number_of_originals:
            card = img_to_array(load_img(f'OriginalData/{class_}/{class_}_{str(generating_index)}.jpg'))
            card = cv2.resize(card, dsize=(217, 290))
            card = np.expand_dims(card, axis=0)
            imgen = ImageDataGenerator(rotation_range=20,
                                       brightness_range=(0.5, 1),
                                       shear_range=3.0,
                                       zoom_range=[.9, 1.1], )
            count = 0
            for batch in imgen.flow(card,
                                    batch_size=1,
                                    save_to_dir=f'GeneratedData/{class_}',
                                    save_format='jpg'):
                count += 1
                if count == images_to_generate:
                    break
    return 0


def deleting_all_generated_data():
    files = glob.glob('GeneratedData/ped')
    for f in files:
        os.remove(f)


class_names = ['brak_karty', 'ciemnosc_nocy', 'empatyczne_natarcie',
               'gryzaca_horda', 'pasozytniczy_wplyw', 'ped',
               'przerazajace_ostrze', 'przewrotne_ostrze',
               'slabosc_umyslu', 'sprzezenie_zwrotne', 'uleglosc']

indexes_to_generate = [i for i in range(11)]
# images_generator(class_names, indexes_to_generate, 100)
