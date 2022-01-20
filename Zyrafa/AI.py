from keras.datasets import mnist, cifar10
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from VisualiseKerasLayers import Visualise
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
import os
import cv2
x = []
y = []
for i in os.listdir("data2"):
    a = cv2.imread(f"data2/{i}", cv2.IMREAD_GRAYSCALE)
    x.append(a/255.0)
    y.append(0)
for i in os.listdir("data"):
    a = cv2.imread(f"data/{i}", cv2.IMREAD_GRAYSCALE)
    x.append(a/255.0)
    y.append(1)

datagen = ImageDataGenerator(rotation_range=35, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=False)

xtrain = np.array(x)
xtrain = xtrain.reshape(xtrain.shape + (1,))
ytrain_dum = pd.get_dummies(y)

bob = datagen.flow(xtrain,ytrain_dum)
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=xtrain.shape[1:]))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(filters=16, kernel_size=(3, 3)))
model.add(MaxPool2D((2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['acc'])

model.fit(bob, epochs=100, callbacks=[EarlyStopping(patience=50, monitor="acc"), ModelCheckpoint(filepath="model.h5", save_best_only=True, monitor="acc", verbose=1)])
model = load_model("model.h5")
x2 = cv2.imread(f"siema.png", cv2.IMREAD_GRAYSCALE)/255.0
xtest = np.array(x2)
xtest = xtest.reshape((1,) + xtest.shape + (1,))
print(xtest.shape)
preds = model.predict(xtest)
print(preds)
#preds_arg = preds.argmax(axis=1)
# acc = accuracy_score(ytest, preds_arg)
# print(acc)
