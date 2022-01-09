"""
TP 3 - Transfer Learning et Fine-Tuning
Implémentation des fonctions
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from data_gen import PascalVOCDataGenerator
import numpy as np
from sklearn.metrics import average_precision_score

default_batch_size = 200
default_data_dir = "D:\VOCdevkit\VOC2007"


def evaluate(model, subset, batch_size=default_batch_size, data_dir=default_data_dir, verbose=0):
    """evaluate
    Compute the mean Average Precision metrics on a subset with a given model

    :param model: the model to evaluate
    :param subset: the data subset
    :param batch_size: the batch which will be use in the data generator
    :param data_dir: the directory where the data is stored
    :param verbose: display a progress bar or not, default is no (0)
    """
    # disable_tqdm = (verbose == 0)

    # Create the generator on the given subset
    data_generator = PascalVOCDataGenerator(subset, data_dir)
    steps_per_epoch = int(len(data_generator.id_to_label) / batch_size) + 1

    # Get the generator
    generator = data_generator.flow(batch_size=batch_size)

    y_all = []
    y_pred_all = []
    for i in range(steps_per_epoch):
        # Get the next batch
        X, y = next(generator)
        y_pred = model.predict(X)
        # We concatenate all the y and the prediction
        for y_sample, y_pred_sample in zip(y, y_pred):
            y_all.append(y_sample)
            y_pred_all.append(y_pred_sample)
    y_all = np.array(y_all)
    y_pred_all = np.array(y_pred_all)

    # Now we can compute the AP for each class
    AP = np.zeros(data_generator.nb_classes)
    for cl in range(data_generator.nb_classes):
        AP[cl] = average_precision_score(y[:, cl], y_pred[:, cl])

    return AP
