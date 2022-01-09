"""
TP 3 - Transfer Learning et Fine-Tuning
Exercice 4 : Fine-tuning sur VOC 2007
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Model
from sklearn.metrics import average_precision_score

from data_gen import PascalVOCDataGenerator

# Chargement de l'architecture ResNet50 et ses poids

model = ResNet50(include_top=True, weights='imagenet')
model.layers.pop()

data_dir = 'D:\VOCdevkit\VOC2007'
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)

# Modification des couches
x = model.layers[-2].output
x = Dense(data_generator_train.nb_classes, activation='sigmoid')(x)
model = Model(inputs=model.input, outputs=x)

# Apprentissage des paramètres de l’ensemble du réseau
for i in range(len(model.layers)):
    model.layers[i].trainable = True

# Compilation du modèle
lr = 0.1
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr), metrics=['binary_accuracy'])

# Entrainement du modèle
batch_size = 32
nb_epochs = 10
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)
steps_per_epoch_train = int(len(data_generator_train.id_to_label) / batch_size) + 1
model.fit_generator(data_generator_train.flow(batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=nb_epochs,
                    verbose=1)

# Evaluation du modèle sur la base test
data_generator_test = PascalVOCDataGenerator('test', data_dir)
batch_size = 32
generator = data_generator_test.flow(batch_size=batch_size)
# Nombre d'images
Nb_images = len(data_generator_test.images_ids_in_subset)

# Calcul du nombre de batchs
nb_batches = int(len(data_generator_test.images_ids_in_subset) / batch_size) + 1
scores = np.array([0., 0.])
for i in range(nb_batches):
    # Pour chaque batch, on extrait les images d'entrée X et les labels y
    X, y = next(generator)
    scores += model.evaluate(X, y, verbose=0)
print("%s TEST: %.2f%%" % (model.metrics_names[0], scores[0] * 100 / nb_batches))
print("%s TEST: %.2f%%" % (model.metrics_names[1], scores[1] * 100 / nb_batches))

# Calcul du MAP
data_generator_train = PascalVOCDataGenerator('train', data_dir)
data_generator_test = PascalVOCDataGenerator('test', data_dir)
batch_size = 32
generator_test = data_generator_test.flow(batch_size=batch_size)
generator_train = data_generator_train.flow(batch_size=batch_size)
# Nombre d'images
Nb_images = len(data_generator_test.images_ids_in_subset)

# Calcul du nombre de batchs
nb_batches = int(len(data_generator_test.images_ids_in_subset) / batch_size) + 1

AP_train = np.zeros(20)
AP_test = np.zeros(20)
y_pred_train = np.array([])
y_pred_test = np.array([])
Y_test = np.array([])
Y_train = np.array([])
for i in range(nb_batches):
    # Pour chaque batch, on extrait les images d'entrée X et les labels y
    # Données d'entraînement
    X_train, y_train = next(generator_train)
    # Données test
    X_test, y_test = next(generator_test)

    if i == 0:
        Y_test = y_test
        Y_train = y_train
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
    else:
        Y_test = np.concatenate((Y_test, y_test), axis=0)
        Y_train = np.concatenate((Y_train, y_train), axis=0)
        y_pred_test = np.concatenate((y_pred_test, model.predict(X_test)), axis=0)
        y_pred_train = np.concatenate((y_pred_train, model.predict(X_train)), axis=0)

for c in range(20):
    AP_train[c] = average_precision_score(Y_train[:, c], y_pred_train[:, c])
    AP_test[c] = average_precision_score(Y_test[:, c], y_pred_test[:, c])
print("MAP TRAIN =", AP_train.mean() * 100)
print("MAP TEST =", AP_test.mean() * 100)
