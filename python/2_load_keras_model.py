import glob
import os
import random

import cv2
import numpy as np
from tensorflow.keras.models import load_model

new_model = load_model('model/custom_keras_model.h5')

new_model.summary()

base_path = 'data'
folders = os.listdir(base_path)
# N is the total number of your data (train + test) of all classes
N = 10586
img_rows = 32
img_cols = 32
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

    # files = files[:1000]

    for f1 in files:
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

X_test = X_train[:1000]
X_train = X_train[1000:]
Y_test = Y_train[:1000]
Y_train = Y_train[1000:]

score = new_model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
