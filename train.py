import tensorflow as tf
import keras
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import random
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense,Input, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import cohen_kappa_score, f1_score

LR = 1e-3
N_NEURONS = (128, 128, 32)
N_EPOCHS = 100
BATCH_SIZE = 64
DROPOUT = 0.2

os.getcwd()

if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train.zip")
    os.system("unzip train.zip")

DATA_DIR = os.getcwd() + "/train/"
RESIZE_TO = 50
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

x, y = [], []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    with open(DATA_DIR + path[:-4] + ".txt", "r") as s:
        label = s.read()
    y.append(label)
x, y = np.array(x), np.array(y)


le = LabelEncoder()
le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
y = le.transform(y)
print(x.shape, y.shape)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, labels, random_state=SEED, test_size=0.2, stratify=y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)


model = Sequential([Dense(N_NEURONS[0], input_dim=7500, kernel_initializer=weight_init),Activation("relu"),Dropout(DROPOUT), BatchNormalization()])

for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, activation="relu", kernel_initializer=weight_init))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())

# Adds a final output layer with softmax to map to the 4 classes
model.add(Dense(4, activation="softmax", kernel_initializer=weight_init))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# Training MLP, while printing validation loss and metrics at each epoch
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),callbacks=[ModelCheckpoint("mlp_srijon.hdf5", monitor="val_loss", save_best_only=True)])

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")

