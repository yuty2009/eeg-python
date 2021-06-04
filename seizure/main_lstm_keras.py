# -*- coding: utf-8 -*-

import os
import keras
import matplotlib.pyplot as plt
from keras import layers
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from common.utils import *
from seizure.eegreader_csv import *

datapath = 'e:/eegdata/seizure/Epileptic_Seizure_Recognition/data.csv'
data, labels = read_csvdata(datapath)
data = np.expand_dims(data, axis=-1)
labels = onehot(labels, 5)

model = Sequential()
model.add(layers.Conv1D(64, 15, strides=2,
                        input_shape=(178, 1), use_bias=False))
model.add(layers.ReLU())
model.add(layers.Conv1D(64, 3))
model.add(layers.Conv1D(64, 3, strides=2))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Conv1D(64, 3))
model.add(layers.Conv1D(64, 3, strides=2))
model.add(layers.BatchNormalization())
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.LSTM(32))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation="softmax"))
model.summary()

save_path = './keras_model3.h5'

if os.path.isfile(save_path):
    model.load_weights(save_path)
    print('reloaded.')

adam = keras.optimizers.adam()
model.compile(optimizer=adam,
              loss="categorical_crossentropy", metrics=["acc"])


# 计算学习率
def lr_scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的0.5
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)


lrate = LearningRateScheduler(lr_scheduler)

history = model.fit(data, labels, epochs=400,
                    batch_size=128, validation_split=0.2,
                    verbose=2, callbacks=[lrate])

model.save_weights(save_path)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
