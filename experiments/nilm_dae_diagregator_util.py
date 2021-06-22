
import random
import sys

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout
from tensorflow.keras.utils import plot_model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.legacy.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore


def create_model(sequence_len):
    '''Creates the Auto encoder module described in the paper
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(8, 4, activation="linear", input_shape=(sequence_len, 1), padding="same", strides=1))
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dropout(0.2))
    model.add(Dense((sequence_len - 0) * 8, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense((sequence_len - 0) * 8, activation='relu'))

    model.add(Dropout(0.2))

    # 1D Conv
    model.add(Reshape(((sequence_len - 0), 8)))
    model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

    model.compile(loss='mse', optimizer='adam')
    plot_model(model, to_file='model.png', show_shapes=True)

    return model

