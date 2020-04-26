# -*- coding: utf-8 -*-

import keras

INPUT_SHAPE = [78, 64]
MAP_NUM_LAYER1 = 10
MAP_NUM_LAYER2 = 50
BOTTLENECK_SIZE = 100

class CNNModel(keras.Model):

    def __init__(self, dropout=0.5):
        super(CNNModel, self).__init__(name='cnn')
        self.conv1 = keras.layers.Conv2D(MAP_NUM_LAYER1, [1, 64], input_shape=INPUT_SHAPE, activation='relu')
        self.conv2 = keras.layers.Conv2D(MAP_NUM_LAYER2, [13, 1], strides=[13, 1], activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(BOTTLENECK_SIZE, activation='relu')
        self.dropout = keras.layers.Dropout(dropout)
        self.out = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.out(x)


