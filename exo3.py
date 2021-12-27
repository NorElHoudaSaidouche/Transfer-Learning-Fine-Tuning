"""
TP 3 - Transfer Learning et Fine-Tuning
Exercice 3 : Transfert sur VOC 2007
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import average_precision_score

# Récupération du dataset
outfile = 'DF_ResNet50_VOC2007.npz'
npzfile = np.load(outfile)
X_train = npzfile['X_train']
Y_train = npzfile['Y_train']
X_test = npzfile['X_test']
Y_test = npzfile['Y_test']
print("X_train=", X_train.shape, "Y_train=", Y_train.shape, " X_test=", X_test.shape, "Y_train=", Y_test.shape)

# Création du modèle
model = Sequential()
model.add(Dense(20, input_dim=2048, name='fc1', activation='sigmoid'))
model.summary()

# Compilation du modèle
learning_rate = 0.1
sgd = SGD(learning_rate)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])

# Entrainement du modèle
batch_size = 32
nb_epoch = 20
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluation du modèle
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s TEST: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s TEST: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Evaluation finale
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
AP_train = np.zeros(20)
AP_test = np.zeros(20)
for c in range(20):
    AP_train[c] = average_precision_score(Y_train[:, c], y_pred_train[:, c])
    AP_test[c] = average_precision_score(Y_test[:, c], y_pred_test[:, c])

print("MAP TRAIN =", AP_train.mean() * 100)
print("MAP TEST =", AP_test.mean() * 100)
