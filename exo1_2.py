"""
TP 3 - Transfer Learning et Fine-Tuning
Exercice 1 : Modèle ResNet-50 avec Keras | Exercice 2 : Extraction de « Deep Features »
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.optimizers import SGD
from data_gen import PascalVOCDataGenerator

# Création du modèle
model = ResNet50(include_top=True, weights='imagenet')

model.layers.pop()

model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Vérification de l'architecture du modèle
model.summary()

# Compilation du modèle
learning_rate = 0.1
sgd = SGD(learning_rate)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])

# Chargement des données de la base
data_dir = "D:\VOCdevkit\VOC2007"
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)

batch_size = 32
generator = data_generator_train.flow(batch_size=batch_size)

Nb_images = len(data_generator_train.images_ids_in_subset)
# Initialisation des matrices contenant les Deep Features et les labels
X_train = np.zeros((len(data_generator_train.images_ids_in_subset), 2048))
Y_train = np.zeros((len(data_generator_train.images_ids_in_subset), 20))
# Calcul du nombre de batchs
nb_batches = int(len(data_generator_train.images_ids_in_subset) / batch_size) + 1

for i in range(nb_batches):
    # Extraction les images d'entrée X et les labels y pour chaque batch
    X, y = next(generator)
    # Récupération des Deep Feature
    y_pred = model.predict(X)
    X_train[i * batch_size:(i + 1) * batch_size, :] = y_pred
    Y_train[i * batch_size:(i + 1) * batch_size, :] = y

data_generator_test = PascalVOCDataGenerator('test', data_dir)
generator = data_generator_test.flow(batch_size=batch_size)
Nb_images = len(data_generator_test.images_ids_in_subset)
# Extraction des images et des deep features
# Initialisation des matrices contenant les Deep Features et les labels
X_test = np.zeros((len(data_generator_test.images_ids_in_subset), 2048))
Y_test = np.zeros((len(data_generator_test.images_ids_in_subset), 20))
# Calcul du nombre de batchs
nb_batches = int(len(data_generator_test.images_ids_in_subset) / batch_size) + 1

for i in range(nb_batches):
    # Extraction les images d'entrée X et les labels y pour chaque batch
    X, y = next(generator)
    # Récupération des Deep Feature
    y_pred = model.predict(X)
    X_test[i * batch_size:(i + 1) * batch_size, :] = y_pred
    Y_test[i * batch_size:(i + 1) * batch_size, :] = y

# Sauvegardre des Deep Features et labels de la manière suivante
outfile = 'DF_ResNet50_VOC2007'
np.savez(outfile, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
