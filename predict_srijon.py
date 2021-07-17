import numpy as np
from keras.models import load_model
import cv2
import os

def predict(list_path):
    # input will be a list of paths containing images to be classified
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    x=[]
    RESIZE_TO=50
    for path in list_path:
        x.append(cv2.resize(cv2.imread(path), (RESIZE_TO, RESIZE_TO)))
    x=np.array(x)
    x = x.reshape(len(x), -1)
    x = x / 255
    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('mlp_srijon.hdf5')
    # If using more than one model to get y_pred, they need to be named as "mlp_ajafari1.hdf5", ""mlp_ajafari2.hdf5", etc.
    y_pred = np.argmax(model.predict(x), axis=1)
    return y_pred, model

