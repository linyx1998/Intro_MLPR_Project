import numpy as np
import pickle
import os

def read_data(shuffle):
    data_folder = "../cnn/cifar10_data/cifar-10-batches-py/"
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for file in os.listdir(data_folder):
        if file.startswith('data'):
            with open(os.path.join(data_folder,file), 'rb') as f:
                imgs = pickle.load(f, encoding='latin1')
                temp_X = imgs['data'].astype("float")
                temp_y = np.array(imgs['labels'])
                X_train.append(temp_X)
                y_train.append(temp_y)
        if file.startswith('test'):
            with open(os.path.join(data_folder,file), 'rb') as f:
                imgs = pickle.load(f, encoding='latin1')
                X_test = imgs['data'].astype("float")
                y_test = np.array(imgs['labels'])
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    # print(X_test.shape, y_test.shape)

    if shuffle:
        randomize = np.arange(X_train.shape[0])
        np.random.shuffle(randomize)
        X_train = X_train[randomize]
        y_train = y_train[randomize]
    return X_train, y_train, X_test, y_test

read_data(True)