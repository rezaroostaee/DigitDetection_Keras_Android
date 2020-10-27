import glob
import os
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential

# set parameters of your keras model. Here is a simple structure for clarifying the flow.
# You can change it to your specific keras model.
img_rows = 28
img_cols = 28
nb_filters = 32
batch_size = 128
nb_epoch = 2
nb_classes = 10

pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

model = Sequential()
model.add(Reshape((img_rows, img_cols, 1), input_shape=[img_rows * img_cols], name="input"))
model.add(Convolution2D(nb_filters, kernel_size, activation='relu'))

model.add(Convolution2D(nb_filters, kernel_size, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(2 * nb_filters, kernel_size, activation='relu'))

model.add(Convolution2D(2 * nb_filters, kernel_size, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(2 * nb_filters, kernel_size, activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_classes, activation='softmax', name="output"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.summary()

base_path = 'data'
folders = os.listdir(base_path)
# N is the total number of your data (train + test) of all classes
N = 73228
X_train = []
Y_train = np.zeros((N, 10))
y_cnt = 0
n_cnt = 0


for f in folders:
    data_path = os.path.join(base_path + "/" + f, '*.png')
    files = glob.glob(data_path)

    if files.__len__() == 0:
        data_path = os.path.join(base_path + "/" + f, '*.jpg')
        files = glob.glob(data_path)

    for f1 in files:
        # If your data need to preprocess to train, you can handle that here by cv2 lib

        img = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_rows, img_cols))
        X_train.append(img)
        Y_train[n_cnt, y_cnt] = 1
        n_cnt += 1

    y_cnt += 1
    print('load ' + f + " class")

ind_list = [i for i in range(N)]
random.shuffle(ind_list)
Y_train = Y_train[ind_list, ]
X_train = np.array(X_train)[ind_list, ]

X_train = X_train.reshape(N, img_rows * img_cols)

# You can split your data to train and test by number
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# X_test = X_train[:1000]
# X_train = X_train[1000:]
# Y_test = Y_train[:1000]
# Y_train = Y_train[1000:]

# Create Dir for save trained model and checkpoints during the training
if not os.path.exists('model/checkpoints'):
    os.makedirs('model/checkpoints')

filepath="model/checkpoints/tmp__weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=callbacks_list)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save('model/custom_keras_model.h5')

# If you need to save trained weights you can do. (But for deploying the model to your Android app, it is not necessary)
# model.save_weights('model/custom_keras_model_weights.h5')
