from card_detector import CardDetector
from data_loader import DataLoader


CLASS_NAMES = ['brak_karty', 'ciemnosc_nocy', 'empatyczne_natarcie',
               'gryzaca_horda', 'pasozytniczy_wplyw', 'ped',
               'przerazajace_ostrze', 'przewrotne_ostrze',
               'slabosc_umyslu', 'sprzezenie_zwrotne', 'uleglosc']
IMAGES_FOR_CLASS = 3


def main():
    # prepare data of images and corresponding labels
    dataloader = DataLoader()
    dataloader.create_data_and_labels_arrays(classes=CLASS_NAMES, images_for_class=IMAGES_FOR_CLASS)
    # setup model
    mindthief_cards_detector = CardDetector()
    # mindthief_cards_detector.train(dataloader.data, dataloader.labels)
    # mindthief_cards_detector.save()
    mindthief_cards_detector.load()
    mindthief_cards_detector.classify(dataloader.data, dataloader.labels)


if __name__ == '__main__':
    main()
