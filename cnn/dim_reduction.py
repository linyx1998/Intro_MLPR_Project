import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
import pickle
import os
import torch
from torchvision import transforms

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
    
    X_train = reshape_image(X_train, 32)
    X_test = reshape_image(X_test, 32)

    # X_train = X_train[:500]
    # y_train = y_train[:500]
    # X_test = X_test[:500]
    # y_test = y_test[:500]

    transform = transforms.Normalize([0.5,0.5,0.5],\
        [0.5,0.5,0.5])
    X_train = transform(torch.from_numpy(X_train)).numpy()
    X_test = transform(torch.from_numpy(X_test)).numpy()
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test


def reshape_image(X, width):
    # print(X.shape)
    length = X.shape[0]
    X = X.reshape((length, 3, width, width))
    X_new = []
    for i in range(X.shape[0]):
        temp = []
        for j in range(X[i].shape[0]):
            temp.append(np.pad(X[i][j], ((0, 32-width), (0, 32-width)), 'constant', \
            constant_values=0))
        X_new.append(temp)
    return np.array(X_new)

def pca(width, remain):
    X_train, y_train, X_test, y_test = read_data(True)
    print('PCA')
    print('remain width %d' % width)
    pca = PCA(remain)
    X_train_ = pca.fit_transform(X_train)
    X_test_ = pca.fit_transform(X_test)

    X_train_ = reshape_image(X_train_, width)
    X_test_ = reshape_image(X_test_, width)
    print(X_train_.shape)

    # X_train_ = X_train_[:1000]
    # y_train = y_train[:1000]
    # X_test_ = X_test_[:1000]
    # y_test = y_test[:1000]

    return X_train_, y_train, X_test_, y_test

def kernel_pca(width, remain):
    X_train, y_train, X_test, y_test = read_data(True)

    # X_train = X_train[:4000]
    # y_train = y_train[:4000]
    # X_test = X_test[:4000]
    # y_test = y_test[:4000]

    print('KernelPCA')
    print('remain width %d' % width)
    pca = KernelPCA(kernel="rbf", n_components=remain)
    X_train_ = pca.fit_transform(X_train)
    X_test_ = pca.fit_transform(X_test)

    print(X_train_.shape)

    X_train_ = reshape_image(X_train_, width)
    X_test_ = reshape_image(X_test_, width)
    # print(X_train_.shape)

    print(X_train_.shape)
    print(X_test_.shape)

    return X_train_, y_train, X_test_, y_test

def lle(width, remain):
    X_train, y_train, X_test, y_test = read_data(True)

    # X_train = X_train[:4000]
    # y_train = y_train[:4000]
    # X_test = X_test[:4000]
    # y_test = y_test[:4000]

    print('LLE')
    print('remain width %d' % width)
    X_train_ = LLE(n_neighbors=30, n_components=remain, method='standard').\
        fit_transform(X_train)
    X_test_ = LLE(n_neighbors=30, n_components=remain, method='standard').\
        fit_transform(X_test)
    
    X_train_ = reshape_image(X_train_, width)
    X_test_ = reshape_image(X_test_, width)
    # print(X_train_.shape)

    X_train_ = X_train_[:500]
    y_train = y_train[:500]
    X_test_ = X_test_[:500]
    y_test = y_test[:500]

    print(X_train_.shape)
    print(X_test_.shape)

    return X_train_, y_train, X_test_, y_test

# pca(0.2, 36*3)