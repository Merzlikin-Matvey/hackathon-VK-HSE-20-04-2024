import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Flatten

import pandas as pd
import numpy as np

import os

if os.path.exists('data/data.csv'):
    DATA = pd.read_csv('data/data.csv')
else:
    DATA = pd.read_csv('../data/data.csv')

DATA_SIZE = len(DATA)

class NeuralNetwork:
    def __init__(self, model=None):
        if model is None:
            self.model = Sequential()
            self.model.add(Flatten())
            self.model.add(Dense(50))  # RNN layer with 50 units
            self.model.add(Dense(50, activation='relu'))  # Change this line,
            self.model.add(Dense(4, activation='softmax'))
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')  # And this line
        else:
            self.model = model

    def summary(self):
        self.model.summary()

    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y, *args, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path="../model.keras"):
        self.model.save(path)

    @classmethod
    def load(cls, path="../model.keras"):
        model = tf.keras.models.load_model(path)
        return cls(model)