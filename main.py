from card_detector import CardDetector
from data_loader import DataLoader


CLASS_NAMES = ['brak_karty', 'ciemnosc_nocy', 'empatyczne_natarcie',
               'gryzaca_horda', 'pasozytniczy_wplyw', 'ped',
               'przerazajace_ostrze', 'przewrotne_ostrze',
               'slabosc_umyslu', 'sprzezenie_zwrotne', 'uleglosc']
IMAGES_FOR_CLASS = 3
TRAIN = False
PREDICT = True
DISPLAY_CONFUSION_MATRIX = True
DISPLAY_HISTORICAL_DATA_FIT = True


def main():
    # Prepare data of images and corresponding labels
    dataloader = DataLoader()
    dataloader.create_data_and_labels_arrays(classes=CLASS_NAMES, images_for_class=IMAGES_FOR_CLASS)

    # Setup model
    mindthief_cards_detector = CardDetector()
    if TRAIN:
        mindthief_cards_detector.train(dataloader.data, dataloader.labels)
        mindthief_cards_detector.save()
    if PREDICT:
        mindthief_cards_detector.load()
        mindthief_cards_detector.classify(dataloader.data, dataloader.labels)
        if DISPLAY_CONFUSION_MATRIX:
            mindthief_cards_detector.display_confusion_matrix(dataloader.labels)
        if DISPLAY_HISTORICAL_DATA_FIT:
            mindthief_cards_detector.display_historical_values()


if __name__ == '__main__':
    main()
