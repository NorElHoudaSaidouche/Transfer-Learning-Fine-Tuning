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

from data_gen import PascalVOCDataGenerator

# Chargement de l'architecture ResNet50 et ses poids
from divers import evaluate

model = ResNet50(include_top=True, weights='imagenet')
model.layers.pop()

data_dir = 'D:\VOCdevkit\VOC2007'
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)

# Modification des couches
x = model.layers[-1].output
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
LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
          'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

subset = 'test'
AP = evaluate(model, subset, batch_size=batch_size, data_dir=data_dir, verbose=0)

print('\n')
mAP = np.mean(AP, axis=0)
for i in range(len(LABELS)):
    print("%s : mAP = %f" % (LABELS[i], AP[i]))
print('Total -----------')
print("Mean Average Precision (MAP) = %f" % mAP)
