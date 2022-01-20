from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
import random
import numpy as np
import os

path_0 = "dataSet/0/"
path_1 = "dataSet/1/"
x = []
y = []

for photo in os.listdir(path_0):
    x.append(cv2.imread(f"{path_0}/{photo}"))
    y.append(0)

for photo in os.listdir(path_1):
    x.append(cv2.imread(f"{path_1}/{photo}"))
    y.append(1)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=69)


x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)
model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 input_shape=(1080, 1920, 3),
                 activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 activation='relu',
                 strides=(2, 3)))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=4e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), batch_size=2,
          callbacks=[EarlyStopping(monitor='val_accuracy',
                                   mode='max',
                                   restore_best_weights=True,
                                   patience=5)])
model.save("model.h5")
